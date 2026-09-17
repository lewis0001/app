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
from .mirror import TwinOutput
from .originality import CHECKS, CheckResults, GateOutput
from .review import ReviewOutput
from .rooms.bridge import Directive, StrategyOutput
from .rooms.forge import BuildOutput, Candidate, PitchOutput, ValidationOutput
from .rooms.ledger_room import Decision, PnlOutput, RailOutput
from .rooms.market_bay import LaunchOutput, OutreachOutput
from .rooms.base import WorkOutput
from .rooms.observatory import DeepDiveOutput, ScanOutput, SlopWatchOutput, TrendItem

TREND_POOL = [
    ("Cloudflare default-blocks unsigned agent crawlers and launches Pay Per Use", "From Sept 15, 2026 mixed-use crawlers are blocked by default on new and free-tier sites; signed agents (Web Bot Auth) and paid access get through. Publishers have not tooled up; most expect AI licensing to be minor revenue.", "2026-09-15", 0.95, 0.15, 0.6, "publishers with ad-carrying sites and agent developers who need access"),
    ("'Know Your Agent' becomes a cross-network payments requirement", "Ant International, Visa and Mastercard announced a KYA interoperability framework on Sept 10, 2026: operator traceability, mandates, logs and spend caps. No self-serve certification yet.", "2026-09-10", 0.95, 0.1, 0.45, "teams shipping paying agents"),
    ("Stripe Machine Payments settle agent-to-agent payments in fiat", "Since Feb-Mar 2026 an HTTP 402 endpoint can be paid by card token or stablecoin and settle to a normal Stripe balance; quality metered services are scarce.", "2026-03-18", 0.75, 0.3, 0.7, "agent developers buying data, verification and transforms per call"),
    ("EU AI Act transparency duties and state AI acts are now in force", "Article 50 applies from Aug 2, 2026, Colorado's AI Act from Jun 30, California SB 942 from Aug 2; most small vendors had not acted by spring. Watermark/C2PA marking and human-review records are the accepted approaches.", "2026-08-02", 0.85, 0.35, 0.65, "small B2B vendors selling into the EU, Colorado and California"),
    ("Figure Index pays people for first-person chore video", "Launched Aug 25, 2026: 44k weekly active creators, $15M paid, a stated billion-dollar budget; teleoperators earn $22-120/hour. The bottleneck is reliable coordination of humans with hands, rooms and patience.", "2026-08-25", 0.9, 0.2, 0.5, "robotics labs and data brokers"),
    ("In-chat checkout failed; merchants need agent-readiness instead", "ChatGPT Instant Checkout was retired in March 2026 with a handful of merchants live; Google UCP went self-serve on Shopify in June. Long-tail merchants' catalogues fail structured-data checks.", "2026-06-15", 0.7, 0.45, 0.6, "small online and local merchants"),
    ("Apify rental pricing ends Oct 1, 2026; pay-per-event is the rail that pays", "Fewer than 5% of 20k+ MCP servers earn anything; Apify pays 80% on pay-per-event and sunsets rental pricing on Oct 1. Migration help is needed this month.", "2026-09-01", 0.85, 0.3, 0.55, "developers with rental-priced actors"),
    ("Agent-run physical businesses are a repeatable pattern", "Andon Labs runs a market in San Francisco and a cafe in Sweden with a 'mechanical CEO' and human staff (June 2026); the human-supervised physical footprint is the moat.", "2026-06-02", 0.6, 0.1, 0.35, "operators with a physical location"),
    ("Human-made certification and the AI-slop backlash create a labelled premium", "Human Made Mark (Apr 2026), YouTube demonetising mass-produced AI content; 'a human reviewed this' is now commercially valuable and required by transparency rules.", "2026-04-15", 0.6, 0.2, 0.6, "buyers burned by undisclosed AI output"),
    ("Voice agents fail on dialects in new markets", "Several voice vendors attributed summer churn to dialect failures; small, dated, hand-verified test sets are what buyers distrust synthetic data for.", "2026-07-20", 0.7, 0.2, 0.6, "voice AI vendors entering new markets"),
    ("Agent-eligible bounty tiers exist but most platforms are unreliable", "gigs.sh indexes 46 platforms; Superteam Earn has AGENT_ALLOWED listings; open-source bounties are 73% honeypots. Reliability itself is the edge, and the safe path is human-approved allowlists.", "2026-05-18", 0.6, 0.6, 0.3, "platforms and sponsors needing reliable deliverers"),
    ("FAA Part 108 BVLOS rule pending at OIRA", "Final rule expected late 2026 with a 6-12 month transition; not yet actionable, but a compliance offer should be ready the week it publishes.", "2026-07-10", 0.7, 0.1, 0.2, "drone operators"),
]

VENTURE_POOL = [
    dict(name="Signed-Agent Access Desk", thesis="A hand-run service that gets a publisher's site ready for the post-Sept-15 web: Web Bot Auth policy, Pay Per Use pricing, verified-agent allowlist, with a monthly report of who paid to crawl; $450 setup plus $90/month, sold to publishers whose AI traffic just went dark.", customer="Independent publishers and niche B2B media with ad-carrying sites on Cloudflare; they gather in publisher forums and the Cloudflare community.", trigger="Cloudflare's default block on unsigned crawlers took effect on Sept 15, 2026 and Pay Per Use opened.", default_agent_would_pitch="A 'GEO audit' or an AI visibility dashboard.", why_other_agents_wont="It needs per-site policy decisions with the publisher, verified-agent lists maintained by hand, and reconciliation of tiny payments; the value compounds into a dated dataset of which agents pay.", operator_asset_used="The operator's own domain to run as the reference implementation.", revenue_rail="stripe", cheapest_demand_test="Offer the free readiness check to ten publishers who posted about AI traffic drops; kill if fewer than two reply asking for the setup.", first_dollar_path="Direct message to publishers who complained publicly about AI crawlers this month, with their own site's check attached.", kill_condition="No paid setup within 15 ticks."),
    dict(name="Verified 402 Registry", thesis="A hand-tested registry of HTTP-402 (x402/MPP) endpoints with measured uptime, honesty checks and dated verification, sold as a $29/month feed to agent developers who are currently guessing which endpoints to trust.", customer="Developers shipping agents that spend money per call; they gather in the x402 and MCP developer channels and on the registries' issue trackers.", trigger="Machine-payment endpoints multiplied after Stripe MPP (Mar 2026) and Agentic.market (Apr 2026), and about half of measured x402 volume is gamed.", default_agent_would_pitch="An 'AI-powered API marketplace' or a newsletter listing new endpoints.", why_other_agents_wont="Verification means actually paying each endpoint, logging results over weeks and re-testing on a schedule; it is tedious, costs real cents, and compounds into a dataset nobody else has.", operator_asset_used="The operator's small budget for test calls and a domain they already own.", revenue_rail="stripe", cheapest_demand_test="Post the first 20 verified entries in two developer channels with a waitlist link; kill if fewer than 10 sign-ups in 3 ticks.", first_dollar_path="Message the maintainers of three agent frameworks offering the feed for their docs; ask for one paying pilot.", kill_condition="Fewer than 3 paying subscribers within 20 ticks of launch."),
    dict(name="Annex IV Kit for Tiny Vendors", thesis="A hand-assembled EU AI Act technical-documentation kit for 2-10 person SaaS vendors in HR and education, priced at $390 with one review call, built from the actual annexes rather than generic templates.", customer="Founders of small B2B SaaS selling into EU HR and education; they gather in founder Slack groups and on the procurement portals of universities.", trigger="High-risk obligations began applying in August 2026 and procurement questionnaires now ask for the documentation.", default_agent_would_pitch="An 'AI compliance chatbot' or a generic policy template pack.", why_other_agents_wont="It requires reading the actual annexes, mapping them to how a small product is built, and a human review call; it does not scale and that is the point.", operator_asset_used="The operator's willingness to do a 30-minute review call per customer.", revenue_rail="lemonsqueezy", cheapest_demand_test="Offer the kit to five founders whose products appear on EU university procurement lists; kill if none asks for the price.", first_dollar_path="Direct message to founders via a mutual founder community, referencing their specific product and buyer.", kill_condition="No paid kit within 15 ticks."),
    dict(name="Dialect Test Sets", thesis="Small, dated, hand-verified dialect test sets (200 utterances each) for voice agents entering a new market, sold at $450 per set to vendors that currently discover dialect failures from angry customers.", customer="Voice AI vendors expanding to a new country; product managers who post about launch failures.", trigger="In July and August 2026 several voice vendors publicly attributed churn to dialect failures.", default_agent_would_pitch="A synthetic speech-generation tool or 'AI localisation platform'.", why_other_agents_wont="Verification needs native speakers recruited by hand from specific communities and re-checked; synthetic data is exactly what the buyers distrust.", operator_asset_used="The operator's language and community contacts.", revenue_rail="stripe", cheapest_demand_test="Email five vendors with a 20-utterance sample; kill if no reply asks for the full set.", first_dollar_path="Sample to a vendor that posted about a failed launch, offering the set for their next market.", kill_condition="No paid set within 20 ticks."),
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
    slop = v["name"] in ("AI Growth Hacks Newsletter", "Prompt Pack Pro")
    cands = [Candidate(idea="An AI newsletter about the trend", default_probability=0.9), Candidate(idea="A ChatGPT-for-X wrapper", default_probability=0.85), Candidate(idea="An AI visibility audit agency", default_probability=0.8), Candidate(idea="A generic MCP server", default_probability=0.7), Candidate(idea=v["name"], default_probability=0.9 if slop else round(0.05 + rng.unit() * 0.2, 2))]
    return PitchOutput(title=v["name"], summary=v["thesis"][:200], content=f"# {v['name']}\n{v['thesis']}\n\nCustomer: {v['customer']}\nTrigger: {v['trigger']}\nRail: {v['revenue_rail']}\nDemand test: {v['cheapest_demand_test']}\nFirst dollar: {v['first_dollar_path']}\nKill: {v['kill_condition']}", differentiation_claim=v["why_other_agents_wont"], self_assessment=round(0.4 + rng.unit() * 0.5, 2), candidates_considered=cands, market=v["customer"][:60], mechanism="verification and hand-assembled service" if not slop else "content", channel="direct, vouched channels" if not slop else "marketplace search", core_keywords=[w for w in v["name"].lower().split()[:3]], kill_by_days=14, unit_price="$450 per accepted deliverable" if not slop else "$19 per pack", **v)


def gate(seed: int, user: str) -> GateOutput:
    rng = _Rng(seed)
    idea = user.split("\nOPERATOR")[0].lower()  # only the idea section, not the registry text in the prompt
    slop = "newsletter" in idea or "prompt pack" in idea or "prompts for" in idea
    good = (not slop) and rng.chance(0.8)
    results = {}
    for c in CHECKS:
        if good:
            results[c.key] = not (c.key in {"contrarian_evidence", "compounding", "operator_asset_used", "non_llm_oracle"} and rng.chance(0.3))
        else:
            results[c.key] = rng.chance(0.35)
    return GateOutput(default_ai_would_build="An AI-powered SaaS or newsletter aimed at the same trend.", divergence="Verified, dated, hand-assembled artefact sold to a named buyer." if good else "Barely diverges from the default output.", slop_relabels=(["AI newsletter / curated digest"] if slop else []), check_results=CheckResults(**results), check_notes=[], freshness=round(0.5 + rng.unit() * 0.45, 2) if good else round(rng.unit() * 0.5, 2), crowding=round(rng.unit() * 0.35, 2) if good else round(0.5 + rng.unit() * 0.5, 2), verdict_reason="Tied to a dated trigger, names a buyer and a rail, and rests on work agents skip." if good else "Generic, undated and easily copied by any agent.")


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
    kind = ["replies_with_links", "inbound_request", "operator_confirmed", "replies_with_links", "prepared_only"][rng.next() % 5]
    return ValidationOutput(title="Demand test result", summary=f"Verdict: {verdict}.", content="Ran the cheapest test described in the pitch.", differentiation_claim="Numbers, not vibes.", self_assessment=0.6, test_performed="Posted the sample to two targeted channels and messaged twelve named buyers.", evidence=f"{rng.randint(0, 14)} replies, {rng.randint(0, 4)} asked for the price.", evidence_kind=kind, sample_size=rng.randint(9, 18), verdict=verdict, pivot="Narrow to one vertical." if verdict == "pivot" else "", needs_human="" if rng.chance(0.7) else "Send the two direct messages from your own account (drafts attached).")


def build(seed: int, user: str) -> BuildOutput:
    rng = _Rng(seed)
    rail = "stripe" if "stripe" in user else ("lemonsqueezy" if "lemonsqueezy" in user else "manual")
    return BuildOutput(title="Sellable v1", summary="The smallest thing a buyer will pay for this week.", content="Deliverable spec with pricing.", differentiation_claim="Hand-verified content no generator produces.", self_assessment=0.65, deliverable="Full deliverable text (spec, sample entries, review checklist).", price_cents=[2900, 39000, 45000, 60000][rng.next() % 4], revenue_rail=rail, launch_checklist=["[agent] final proofread", "[human] approve the listing copy", "[human] connect the payment rail"])


def launch(seed: int, user: str) -> LaunchOutput:
    return LaunchOutput(title="Launch plan", summary="Ten named buyers reached by hand through two channels other agents ignore.", content="Channel plan and first message.", differentiation_claim="No mass posting; vouched channels only.", self_assessment=0.6, channels=["maintainers' issue trackers (etiquette: offer a fix first)", "a regional founders' group (etiquette: reply, don't post)"], first_ten_buyers=["maintainer of framework A", "maintainer of framework B", "founder C (EU procurement list)", "founder D (EU procurement list)", "vendor E (posted about a failed launch)", "vendor F"], first_message="Hi - I saw your note about X. We hand-verified Y; here is the sample. If it saves you an hour, the full set is $Z.", human_must_approve=["Send the first message from your account", "Publish the listing page under your name"], checkout_needed=True)


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


def generic(seed: int, user: str) -> WorkOutput:
    rng = _Rng(seed)
    task = user.split("TASK (")[1].split(")")[0] if "TASK (" in user else "work"
    return WorkOutput(title=f"{task.replace('_', ' ').title()} findings", summary="Specific, dated findings with named buyers and channels other agents ignore.", content="## Findings\n- A named channel with etiquette notes\n- A dated trigger with a source\n- The unglamorous step that makes it defensible", differentiation_claim="Names channels and steps a default agent would skip.", self_assessment=round(0.5 + rng.unit() * 0.4, 2), message_to_team="" if rng.chance(0.6) else "Found a channel worth a look; details in my report.")


def twin(seed: int, user: str) -> TwinOutput:
    return TwinOutput(
        ideas=["Sell prompt packs and 'survival kits' for coding agents at $9-$49", "An AI newsletter about AI tools with sponsorships", "A ChatGPT-for-X wrapper SaaS", "An AI automation agency for small businesses", "A faceless YouTube channel with AI voice-over", "AI-written Kindle books", "Print-on-demand AI art store", "An 'AI visibility' audit service", "A generic MCP server starter kit", "A crypto/prediction-market trading bot"],
        product_types=["prompt pack $19", "template bundle $29", "starter kit $49", "ebook $9", "newsletter sponsorship", "SaaS $29/month", "audit $299", "course $99"],
        channels=["Dev.to articles", "Reddit posts from a new account", "Hacker News Show HN", "X replies", "Product Hunt launch", "cold email"],
    )


EXAMPLES: dict[type[BaseModel], Callable[[int, str], BaseModel]] = {
    WorkOutput: generic,
    TwinOutput: twin,
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
