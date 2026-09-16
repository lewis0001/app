"""Richer canned outputs for offline (mock) mode so a demo run tells a real
story: scouts find dated trends, builders pitch ventures that sometimes pass
the gate, reviewers approve/revise/reject in realistic proportions.

None of this runs in live mode. The pools below are illustrative examples of
the *kind* of thing the company should find; the live Observatory replaces
them with web research.
"""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from .llm import _Rng, fake_instance
from .originality import CHECKS, GateOutput
from .review import ReviewOutput
from .rooms.bridge import Directive, StrategyOutput
from .rooms.forge import BuildOutput, PitchOutput, ValidationOutput
from .rooms.ledger_room import Decision, PnlOutput, RailOutput
from .rooms.market_bay import LaunchOutput, OutreachOutput
from .rooms.observatory import DeepDiveOutput, ScanOutput, SlopWatchOutput, TrendItem

TREND_POOL = [
    ("x402 pay-per-request endpoints are multiplying", "HTTP 402 micropayment endpoints let agents pay per call without accounts; dozens of new endpoints a week, almost none verified for uptime or honesty.", "2026-07", 0.85, 0.25, 0.7, "developers shipping agents that call paid endpoints"),
    ("Agentic checkout protocols reach small merchants", "AP2/ACP style agent-initiated purchases are being switched on for long-tail shops; most owners have no idea what to change in their listings.", "2026-08", 0.8, 0.2, 0.6, "small e-commerce owners with under 200 SKUs"),
    ("EU AI Act high-risk obligations land for tiny vendors", "Small SaaS vendors selling into HR, education and credit now need technical documentation and logging they cannot afford consultants for.", "2026-08", 0.75, 0.35, 0.65, "founders of 2-10 person B2B SaaS selling in the EU"),
    ("Skill and MCP registries open to third parties", "Public registries for agent skills/servers accept submissions; ranking is by freshness and verification, and nobody curates by reliability.", "2026-06", 0.7, 0.45, 0.6, "teams choosing which servers to trust"),
    ("Robotics teleoperation data bounties", "Labs pay per verified episode of household manipulation data; supply is bottlenecked on people with hands, rooms and patience.", "2026-05", 0.65, 0.2, 0.5, "robotics labs and data brokers"),
    ("Inbox exhaustion from AI cold outreach", "Small firms are drowning in automated pitches; replies now come only through vouched, human channels and local networks.", "2026-04", 0.6, 0.15, 0.55, "local service businesses"),
    ("Right-to-repair rules widen to appliances", "New repair-information and parts obligations create demand for repair guides and parts sourcing in local languages.", "2026-07", 0.7, 0.2, 0.45, "independent repair shops"),
    ("Agent observability becomes a buying category", "Companies deploying agents need traces reviewed and failure modes explained; tooling exists, judgement does not.", "2026-06", 0.6, 0.5, 0.5, "engineering managers at mid-size companies"),
    ("Hyperlocal event calendars collapse", "Aggregators are polluted by AI-generated events; venues and councils want a verified calendar they can point to.", "2026-08", 0.7, 0.1, 0.4, "venues, councils, local newsletters"),
    ("Voice agents need dialect test sets", "Deployments in non-English markets fail on dialects; buyers pay for small, hand-verified, dated test sets.", "2026-07", 0.7, 0.2, 0.6, "voice AI vendors entering new markets"),
]

VENTURE_POOL = [
    dict(name="Verified 402 Registry", thesis="A hand-tested registry of x402 endpoints with measured uptime, honesty checks and dated verification, sold as a $29/month feed to agent developers who are currently guessing which endpoints to trust.", customer="Developers shipping agents that spend money per call; they gather in the x402 and MCP developer channels and on the registries' issue trackers.", trigger="x402 endpoint count roughly tripled over the summer of 2026 with no verification layer.", default_agent_would_pitch="An 'AI-powered API marketplace' or a newsletter listing new endpoints.", why_other_agents_wont="Verification means actually paying each endpoint, logging results over weeks and re-testing on a schedule; it is tedious, costs real cents, and compounds into a dataset nobody else has.", operator_asset_used="The operator's small budget for test calls and a domain they already own.", revenue_rail="stripe", cheapest_demand_test="Post the first 20 verified entries in two developer channels with a waitlist link; kill if fewer than 10 sign-ups in 3 ticks.", first_dollar_path="Message the maintainers of three agent frameworks offering the feed for their docs; ask for one paying pilot.", kill_condition="Fewer than 3 paying subscribers within 20 ticks of launch."),
    dict(name="Annex IV Kit for Tiny Vendors", thesis="A hand-assembled EU AI Act technical-documentation kit for 2-10 person SaaS vendors in HR and education, priced at $390 with one review call, built from the actual annexes rather than generic templates.", customer="Founders of small B2B SaaS selling into EU HR and education; they gather in founder Slack groups and on the procurement portals of universities.", trigger="High-risk obligations began applying in August 2026 and procurement questionnaires now ask for the documentation.", default_agent_would_pitch="An 'AI compliance chatbot' or a generic policy template pack.", why_other_agents_wont="It requires reading the actual annexes, mapping them to how a small product is built, and a human review call; it does not scale and that is the point.", operator_asset_used="The operator's willingness to do a 30-minute review call per customer.", revenue_rail="lemonsqueezy", cheapest_demand_test="Offer the kit to five founders whose products appear on EU university procurement lists; kill if none asks for the price.", first_dollar_path="Direct message to founders via a mutual founder community, referencing their specific product and buyer.", kill_condition="No paid kit within 15 ticks."),
    dict(name="Dialect Test Sets", thesis="Small, dated, hand-verified dialect test sets (200 utterances each) for voice agents entering a new market, sold at $450 per set to vendors that currently discover dialect failures from angry customers.", customer="Voice AI vendors expanding to a new country; product managers who post about launch failures.", trigger="Several voice vendors publicly attributed churn to dialect failures this summer.", default_agent_would_pitch="A synthetic speech-generation tool or 'AI localisation platform'.", why_other_agents_wont="Verification needs native speakers recruited by hand from specific communities and re-checked; synthetic data is exactly what the buyers distrust.", operator_asset_used="The operator's language and community contacts.", revenue_rail="stripe", cheapest_demand_test="Email five vendors with a 20-utterance sample; kill if no reply asks for the full set.", first_dollar_path="Sample to a vendor that posted about a failed launch, offering the set for their next market.", kill_condition="No paid set within 20 ticks."),
    dict(name="Agentic Checkout Concierge", thesis="A done-with-you service that prepares a small merchant's catalogue for agent-initiated checkout (structured data, policies, test purchases) for $600 per store, sold in one city first.", customer="Owners of small online shops in the operator's city with 20-200 SKUs.", trigger="Agent checkout switched on for long-tail merchants in August 2026; most catalogues fail the structured-data checks.", default_agent_would_pitch="A generic 'AI for e-commerce' SaaS or a Shopify app.", why_other_agents_wont="It is local, hands-on, per-store work with test purchases and a phone call; it uses the operator's local presence.", operator_asset_used="The operator's city and the ability to visit or call merchants.", revenue_rail="manual", cheapest_demand_test="Run the free checker on ten local stores and send each owner their failing items; kill if fewer than two ask for help.", first_dollar_path="Walk into two shops the operator already buys from.", kill_condition="No paid store within 15 ticks."),
    dict(name="AI Growth Hacks Newsletter", thesis="A weekly AI-curated newsletter of growth hacks with sponsorships.", customer="Marketers.", trigger="AI is popular.", default_agent_would_pitch="The same thing.", why_other_agents_wont="They will.", operator_asset_used="", revenue_rail="manual", cheapest_demand_test="Launch and see.", first_dollar_path="Sponsors.", kill_condition="Never."),
    dict(name="Prompt Pack Pro", thesis="A bundle of 500 prompts for small businesses sold on a marketplace.", customer="Small businesses.", trigger="ChatGPT.", default_agent_would_pitch="Prompt packs.", why_other_agents_wont="They can.", operator_asset_used="", revenue_rail="lemonsqueezy", cheapest_demand_test="List it.", first_dollar_path="Marketplace search.", kill_condition="None."),
]


def _pick(rng: _Rng, pool: list, k: int) -> list:
    idx = list(range(len(pool)))
    out = []
    for _ in range(min(k, len(pool))):
        i = idx.pop(rng.next() % len(idx))
        out.append(pool[i])
    return out


def scan(seed: int, user: str) -> ScanOutput:
    rng = _Rng(seed)
    items = []
    for title, summary, seen, fresh, crowd, expl, buyer in _pick(rng, TREND_POOL, rng.randint(2, 3)):
        items.append(TrendItem(title=title, summary=summary, source_urls=[f"https://example.org/source/{abs(hash(title)) % 10_000}"], first_seen=seen, freshness=round(min(1, fresh + rng.unit() * 0.1 - 0.05), 2), crowding=round(min(1, crowd + rng.unit() * 0.1 - 0.05), 2), exploitability=expl, buyer=buyer))
    return ScanOutput(title=f"Scan findings ({len(items)} dated openings)", summary="; ".join(i.title for i in items), content="\n".join(f"## {i.title}\n{i.summary}\nBuyer: {i.buyer}. First seen {i.first_seen}." for i in items), differentiation_claim="Each finding is one layer below the headline and dated; none is a front-page AI story.", self_assessment=round(0.5 + rng.unit() * 0.4, 2), trends=items, message_to_team=f"New on the radar: {items[0].title}.")


def deep_dive(seed: int, user: str) -> DeepDiveOutput:
    rng = _Rng(seed)
    return DeepDiveOutput(title="Deep dive: the unclaimed angle", summary="The crowd is building tooling; the money is in verification and hand-assembled trust.", content="Buyers are burned by synthetic and generic output; they pay for dated, verified, specific work.", differentiation_claim="Names the verification gap rather than proposing another tool.", self_assessment=round(0.5 + rng.unit() * 0.4, 2), opportunity="Sell verified, dated artefacts (registries, test sets, documentation) to buyers who currently guess.", buyers=["framework maintainers", "small vendors under new obligations"], what_default_agents_do_here="Wrappers, marketplaces, newsletters.", unclaimed_angle="Do the tedious verification by hand and sell the result as a feed.")


def slop_watch(seed: int, user: str) -> SlopWatchOutput:
    return SlopWatchOutput(title="Slop watch", summary="Agents are mass-producing MCP wrappers, AI newsletters and 'agent for X' chatbots this month.", content="Observed: wrapper servers, prompt bundles, faceless channels, 'AI agency' offers, listicle directories.", differentiation_claim="Lists what to avoid, dated.", self_assessment=0.7, patterns=["thin MCP wrappers over public APIs", "'AI agent for <industry>' chatbots", "AI newsletters about AI", "prompt bundles", "faceless short-video channels", "AI-generated directories of AI tools"])


def pitch(seed: int, user: str) -> PitchOutput:
    rng = _Rng(seed)
    v = VENTURE_POOL[rng.next() % len(VENTURE_POOL)]
    return PitchOutput(title=v["name"], summary=v["thesis"][:200], content=f"# {v['name']}\n{v['thesis']}\n\nCustomer: {v['customer']}\nTrigger: {v['trigger']}\nRail: {v['revenue_rail']}\nDemand test: {v['cheapest_demand_test']}\nFirst dollar: {v['first_dollar_path']}\nKill: {v['kill_condition']}", differentiation_claim=v["why_other_agents_wont"], self_assessment=round(0.4 + rng.unit() * 0.5, 2), **v)


def gate(seed: int, user: str) -> GateOutput:
    rng = _Rng(seed)
    idea = user.split("\nOPERATOR")[0].lower()  # only the idea section, not the registry text in the prompt
    slop = "newsletter" in idea or "prompt pack" in idea or "prompts for" in idea
    good = (not slop) and rng.chance(0.75)
    results = {}
    for c in CHECKS:
        if good:
            results[c.key] = not (c.key in {"contrarian_evidence", "compounding"} and rng.chance(0.3))
        else:
            results[c.key] = rng.chance(0.35)
    return GateOutput(default_ai_would_build="An AI-powered SaaS or newsletter aimed at the same trend.", divergence="Verified, dated, hand-assembled artefact sold to a named buyer." if good else "Barely diverges from the default output.", slop_relabels=(["AI newsletter / curated digest"] if slop else []), check_results=results, check_notes={}, freshness=round(0.5 + rng.unit() * 0.45, 2) if good else round(rng.unit() * 0.5, 2), crowding=round(rng.unit() * 0.35, 2) if good else round(0.5 + rng.unit() * 0.5, 2), verdict_reason="Tied to a dated trigger, names a buyer and a rail, and rests on work agents skip." if good else "Generic, undated and easily copied by any agent.")


def review(seed: int, user: str) -> ReviewOutput:
    rng = _Rng(seed)
    r = rng.unit()
    is_pitch = "venture_pitch" in user
    approve_p = 0.7 if is_pitch else 0.55
    verdict = "approve" if r < approve_p else ("revise" if r < approve_p + 0.22 else ("reject" if r < 0.96 else "escalate"))
    base = 0.65 if verdict == "approve" else (0.45 if verdict == "revise" else 0.3)
    return ReviewOutput(default_ai_would_have_produced="A generic version aimed at the broadest audience.", differentiation_claim_holds=verdict in ("approve", "escalate") or rng.chance(0.5), quality=round(min(1, base + rng.unit() * 0.3), 2), originality=round(min(1, base + rng.unit() * 0.3), 2), impact=round(min(1, base - 0.1 + rng.unit() * 0.3), 2), verdict=verdict, feedback={"approve": "Specific, dated and sellable. Ship the demand test now.", "revise": "The buyer is still too broad and the first message is vague; name ten real targets.", "reject": "This is what every agent would produce; start from the operator's assets instead.", "escalate": "This needs the Director: it commits money or publishes under the operator's name."}[verdict], required_changes=([] if verdict == "approve" else ["Name ten specific targets", "Date the trigger with a source"]), praise="Clear kill condition." if rng.chance(0.5) else "")


def validation(seed: int, user: str) -> ValidationOutput:
    rng = _Rng(seed)
    r = rng.unit()
    verdict = "proceed" if r < 0.7 else ("pivot" if r < 0.85 else "kill")
    return ValidationOutput(title="Demand test result", summary=f"Verdict: {verdict}.", content="Ran the cheapest test described in the pitch.", differentiation_claim="Numbers, not vibes.", self_assessment=0.6, test_performed="Posted the sample to two targeted channels and messaged five named buyers.", evidence=f"{rng.randint(0, 14)} replies, {rng.randint(0, 4)} asked for the price.", verdict=verdict, pivot="Narrow to one vertical." if verdict == "pivot" else "", needs_human="" if rng.chance(0.7) else "Send the two direct messages from your own account (drafts attached).")


def build(seed: int, user: str) -> BuildOutput:
    rng = _Rng(seed)
    rail = "stripe" if "stripe" in user else ("lemonsqueezy" if "lemonsqueezy" in user else "manual")
    return BuildOutput(title="Sellable v1", summary="The smallest thing a buyer will pay for this week.", content="Deliverable spec with pricing.", differentiation_claim="Hand-verified content no generator produces.", self_assessment=0.65, deliverable="Full deliverable text (spec, sample entries, review checklist).", price_cents=[2900, 39000, 45000, 60000][rng.next() % 4], revenue_rail=rail, launch_checklist=["[agent] final proofread", "[human] approve the listing copy", "[human] connect the payment rail"])


def launch(seed: int, user: str) -> LaunchOutput:
    return LaunchOutput(title="Launch plan", summary="Ten named buyers reached by hand through two channels other agents ignore.", content="Channel plan and first message.", differentiation_claim="No mass posting; vouched channels only.", self_assessment=0.6, channels=["maintainers' issue trackers (etiquette: offer a fix first)", "a regional founders' group (etiquette: reply, don't post)"], first_ten_buyers=["three framework maintainers", "five founders on EU procurement lists", "two vendors that posted about failures"], first_message="Hi - I saw your note about X. We hand-verified Y; here is the sample. If it saves you an hour, the full set is $Z.", human_must_approve=["Send the first message from your account", "Publish the listing page under your name"], checkout_needed=True)


def outreach(seed: int, user: str) -> OutreachOutput:
    return OutreachOutput(title="Outreach round", summary="Five direct messages to named targets.", content="Messages below.", differentiation_claim="Each message references something specific the target published.", self_assessment=0.6, messages=["To maintainer A: ...", "To founder B: ...", "To vendor C: ..."], human_actions=["Send messages 1-3 from your account"])


def pnl(seed: int, user: str) -> PnlOutput:
    rng = _Rng(seed)
    names = [line.split("- ")[1].split(" [")[0] for line in user.splitlines() if line.startswith("- ") and " [" in line and "rev=" in line]
    decisions = []
    for n in names[:4]:
        r = rng.unit()
        decisions.append(Decision(venture=n, action="kill" if r < 0.15 else ("scale" if r < 0.3 else "keep"), reason="Burn exceeds plausible revenue." if r < 0.15 else "Evidence supports continuing.", budget_cents=0))
    return PnlOutput(title="P&L review", summary=f"{len(decisions)} decisions.", content="Numbers per venture.", differentiation_claim="Counts LLM tokens as cost.", self_assessment=0.7, decisions=decisions, company_runway_note="At the current burn the cap lasts roughly 40 more ticks.")


def rail(seed: int, user: str) -> RailOutput:
    r = "stripe" if "stripe" in user else "lemonsqueezy"
    return RailOutput(title=f"Rail: {r}", summary=f"Use {r}.", content="Reasoning.", differentiation_claim="n/a", self_assessment=0.8, rail=r, why="Fits digital products and can be automated after one-time setup.", human_steps=["Create the account", "Create a restricted key", "Paste it in the Airlock"])


def strategy(seed: int, user: str) -> StrategyOutput:
    rng = _Rng(seed)
    bets = ["verification as a product on x402", "documentation for tiny vendors under new obligations", "hand-verified test sets for voice agents"]
    return StrategyOutput(title="Strategy", summary="One bet, one focus venture, one directive per room.", content="Rationale.", differentiation_claim="Refuses consensus ideas explicitly.", self_assessment=0.7, bet=bets[rng.next() % len(bets)], focus_venture="the venture closest to a first dollar", directives=[Directive(room="observatory", directive="Date every trend; drop anything without a buyer."), Directive(room="forge", directive="No pitch without an operator asset or a verification component."), Directive(room="market_bay", directive="Ten named buyers per launch, no mass posting."), Directive(room="ledger", directive="Get one real rail connected this cycle.")], stop_doing="Pitching anything that looks like a tool for developers in general.")


EXAMPLES: dict[type[BaseModel], Callable[[int, str], BaseModel]] = {
    ScanOutput: scan,
    DeepDiveOutput: deep_dive,
    SlopWatchOutput: slop_watch,
    PitchOutput: pitch,
    GateOutput: gate,
    ReviewOutput: review,
    ValidationOutput: validation,
    BuildOutput: build,
    LaunchOutput: launch,
    OutreachOutput: outreach,
    PnlOutput: pnl,
    RailOutput: rail,
    StrategyOutput: strategy,
}
