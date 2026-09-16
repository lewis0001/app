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


class FakeAPI(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        FakeAPI.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        payload = {
            "default_ai_would_have_produced": "a generic listicle",
            "differentiation_claim_holds": True,
            "quality": 0.8, "originality": 0.7, "impact": 0.6,
            "verdict": "approve", "feedback": "solid", "required_changes": [], "praise": "specific",
        }
        msg = {
            "id": "msg_test", "type": "message", "role": "assistant", "model": body.get("model"),
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 200, "cache_read_input_tokens": 800, "cache_creation_input_tokens": 0},
        }
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
    assert body["fallbacks"] == "default"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][0]["type"] == "web_search_20260209"
    assert "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")
    # sampling params must not be sent (rejected on Claude Opus 5)
    assert "temperature" not in body and "top_p" not in body
