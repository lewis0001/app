"""Exercise AnthropicLLM's real request path against a local fake API server.

No key or network is needed: the fake server records the request body and
returns a Message whose text is JSON matching the requested schema. This
verifies the request shape (beta parse, structured output, fallbacks, cache
control, effort) and the response handling, without spending money.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from workhouse.config import Settings
from workhouse.review import ReviewOutput


REVIEW_PAYLOAD = {
    "default_ai_would_have_produced": "a generic listicle",
    "differentiation_claim_holds": True,
    "quality": 0.8, "originality": 0.7, "impact": 0.6,
    "verdict": "approve", "feedback": "solid", "required_changes": [], "praise": "specific",
}


class FakeAPI(BaseHTTPRequestHandler):
    requests: list[dict] = []
    # The next response the fake server sends; tests override it.
    next_response: dict | None = None

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        FakeAPI.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        msg = FakeAPI.next_response or {
            "id": "msg_test", "type": "message", "role": "assistant", "model": body.get("model"),
            "content": [{"type": "text", "text": json.dumps(REVIEW_PAYLOAD)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 200, "cache_read_input_tokens": 800, "cache_creation_input_tokens": 0},
        }
        FakeAPI.next_response = None
        data = json.dumps(msg).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_server(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeAPI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        monkeypatch.delenv(var, raising=False)
    FakeAPI.requests.clear()
    FakeAPI.next_response = None
    yield server
    server.shutdown()


def test_anthropic_llm_request_shape_and_parsing(fake_server):
    from workhouse.llm import AnthropicLLM

    llm = AnthropicLLM(Settings(mode="live", model="claude-opus-5"))
    out = llm.complete("SYSTEM PROMPT", "review this", ReviewOutput, effort="xhigh", web_search=True, label="t")
    assert isinstance(out, ReviewOutput) and out.verdict.value == "approve"
    assert llm.usage.calls == 1 and llm.usage.cache_read_tokens == 800 and llm.usage.cost_usd > 0
    req = FakeAPI.requests[-1]
    body = req["body"]
    assert "/v1/messages" in req["path"]
    assert body["model"] == "claude-opus-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "xhigh"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "verdict" in body["output_config"]["format"]["schema"]["properties"]
    assert body["fallbacks"] == "default"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][0]["type"] == "web_search_20260209"
    assert "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")
    # sampling params must not be sent (rejected on Claude Opus 5)
    assert "temperature" not in body and "top_p" not in body


def test_gate_schema_enumerates_every_check(fake_server):
    """Structured outputs collapse free-form dicts to an empty object, so the
    gate's check results must be enumerated fields, one per check key."""
    from workhouse.llm import AnthropicLLM
    from workhouse.originality import CHECKS, GateOutput

    llm = AnthropicLLM(Settings(mode="live"))
    FakeAPI.next_response = None
    try:
        llm.complete("s", "u", GateOutput, label="gate")
    except RuntimeError:
        pass  # the default fake payload is a review, not a gate output; only the request matters here
    schema = FakeAPI.requests[-1]["body"]["output_config"]["format"]["schema"]
    node = schema["properties"]["check_results"]
    if "$ref" in node:  # nested models may be emitted as $defs references
        node = schema["$defs"][node["$ref"].split("/")[-1]]
    assert set(node["properties"]) == {c.key for c in CHECKS}
    assert node.get("additionalProperties") is False


def test_refusal_and_truncation_still_record_usage(fake_server):
    from workhouse.llm import AnthropicLLM, LLMRefusal

    llm = AnthropicLLM(Settings(mode="live"))
    FakeAPI.next_response = {
        "id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "content": [{"type": "text", "text": "I can't help with that."}],
        "stop_reason": "refusal", "stop_details": {"type": "refusal", "category": "cyber", "explanation": "policy"}, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    with pytest.raises(LLMRefusal):
        llm.complete("s", "u", ReviewOutput)
    assert llm.usage.calls == 1 and llm.usage.cost_usd > 0
    FakeAPI.next_response = {
        "id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "content": [{"type": "text", "text": "{\"verdict\": \"appr"}],
        "stop_reason": "max_tokens", "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 16000},
    }
    with pytest.raises(RuntimeError):
        llm.complete("s", "u", ReviewOutput)
    assert llm.usage.calls == 2


def test_narration_blocks_around_web_search_are_tolerated_and_priced(fake_server):
    from workhouse.llm import AnthropicLLM

    llm = AnthropicLLM(Settings(mode="live"))
    FakeAPI.next_response = {
        "id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "content": [
            {"type": "text", "text": "Let me look that up."},
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "x402"}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
            {"type": "text", "text": json.dumps(REVIEW_PAYLOAD)},
        ],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 300, "cache_creation_input_tokens": 2000, "server_tool_use": {"web_search_requests": 3}},
    }
    out = llm.complete("s", "u", ReviewOutput, web_search=True)
    assert out.verdict.value == "approve"
    # 3 searches at $0.01 plus a 2k-token cache write at 1.25x input price are counted
    assert llm.usage.cost_usd > 0.03 + 2000 * 5.0 * 1.25 / 1_000_000


def test_missing_credentials_fail_fast(monkeypatch):
    from workhouse.llm import AnthropicLLM, MockLLM, build_llm

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_PROFILE", "definitely-not-a-profile")
    with pytest.raises(Exception):
        AnthropicLLM(Settings(mode="live"))
    assert isinstance(build_llm(Settings(mode="live")), MockLLM)
