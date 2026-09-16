"""The Originality Gate: the mechanism that stops this company from building
what every other AI agent builds.

Three layers:
1. Slop registry: a curated list of overdone patterns with their tells. A
   cheap lexical match flags ideas before any LLM call.
2. Default-AI adversary: an LLM pass that predicts what a generic agent would
   produce from the same brief, then measures divergence and runs the
   Different-by-Design checks.
3. Freshness and crowding: the gate prefers ideas tied to trends the
   Observatory found recently and penalises crowded ones.

An idea must pass a minimum number of checks AND clear the score threshold.
The gate's verdict is recorded on the venture so reviewers and the human can
see exactly why it passed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .models import OriginalityReport, TrendSignal

# --------------------------------------------------------------------------- #
# Layer 1: slop registry (what other AI agents build)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SlopPattern:
    name: str
    tells: str
    keywords: tuple[str, ...]


SLOP_REGISTRY: list[SlopPattern] = [
    SlopPattern("AI newsletter / curated digest", "curated links + AI summaries, 'stay ahead of AI', sponsorship-based", ("newsletter", "digest", "curated links", "weekly roundup")),
    SlopPattern("Prompt packs / prompt marketplace", "selling prompts, 'mega prompt bundle', prompt library", ("prompt pack", "prompt bundle", "prompt library", "prompt marketplace", "prompts for")),
    SlopPattern("Chatbot wrapper", "chat with your PDF/docs/site, 'AI assistant for X' with no proprietary data", ("chat with your", "chatbot for", "ai assistant for", "gpt wrapper", "ask your")),
    SlopPattern("Generic AI SaaS boilerplate", "AI writing tool, AI summariser, AI image tool, 'all-in-one AI platform'", ("ai writing tool", "ai summarizer", "ai summariser", "all-in-one ai", "ai content generator", "ai copywriting")),
    SlopPattern("Faceless video channel", "AI voice-over + stock footage, shorts automation, 'monetise with YouTube'", ("faceless", "youtube automation", "shorts channel", "tiktok automation", "ai voiceover")),
    SlopPattern("AI-written ebooks / Kindle publishing", "low-content or AI books, KDP arbitrage", ("ebook", "kindle", "kdp", "low-content book", "ai-written book")),
    SlopPattern("Affiliate content site / listicle blog", "'best X for Y' reviews, programmatic SEO, Amazon associate", ("affiliate", "listicle", "best 10", "programmatic seo", "review site")),
    SlopPattern("Print-on-demand / AI art store", "AI-generated designs on t-shirts/mugs, Etsy/Redbubble", ("print on demand", "print-on-demand", "redbubble", "merch store", "ai art prints", "t-shirt designs")),
    SlopPattern("Dropshipping store", "generic products, aliexpress, TikTok ads", ("dropship", "dropshipping", "aliexpress", "winning product")),
    SlopPattern("AI automation agency", "'we build AI agents for businesses', lead-gen + outreach automation", ("automation agency", "ai agency", "build ai agents for", "done-for-you ai")),
    SlopPattern("Resume / cover letter / job tool", "AI resume builder, interview prep bot", ("resume builder", "cover letter", "interview prep", "job application ai")),
    SlopPattern("Trading / crypto bot", "signals, arbitrage bots, 'passive income' trading", ("trading bot", "crypto signals", "arbitrage bot", "day trading", "memecoin")),
    SlopPattern("Online course / info-product", "'how to make money with AI' course, masterclass, coaching funnel", ("online course", "masterclass", "make money with ai", "coaching program", "info product")),
    SlopPattern("Stock content farm", "AI stock photos, music, templates, Notion templates", ("stock photo", "notion template", "canva template", "ai music", "template pack")),
    SlopPattern("Generic MCP server / API wrapper", "thin wrapper over a public API, 'MCP server for X' with no unique data", ("mcp server for", "api wrapper", "wrapper around")),
    SlopPattern("AI social media manager", "auto-posting, caption generator, hashtag tool", ("social media manager", "caption generator", "hashtag", "auto-post", "content calendar ai")),
    SlopPattern("Lead-gen scraping + cold outreach", "scrape directories, mass cold email, LinkedIn automation", ("cold email", "cold outreach", "lead list", "scrape linkedin", "outreach automation")),
    SlopPattern("Generic freelancing on saturated platforms", "AI writing/coding gigs on Fiverr/Upwork at commodity prices", ("fiverr", "upwork gigs", "freelance writing", "ghostwriting gigs")),
    SlopPattern("Micro-SaaS directory / tool aggregator", "'1000 AI tools' directory, listing sites", ("ai tools directory", "tool directory", "aggregator site", "listing site")),
    SlopPattern("Survey / microtask arbitrage", "paid surveys, captcha, data-entry", ("paid surveys", "microtask", "data entry")),
    SlopPattern("AI companion / persona app", "girlfriend/boyfriend bot, celebrity persona chat", ("ai companion", "ai girlfriend", "persona chat", "virtual friend")),
    SlopPattern("Generic 'AI for real estate/dentists/lawyers' vertical bot", "same chatbot with an industry noun in front", ("ai for dentists", "ai for real estate", "ai for lawyers", "ai receptionist", "ai for restaurants")),
    SlopPattern("Domain flipping / expired-domain SEO", "buy, redirect, sell", ("domain flipping", "expired domain", "domain investing")),
    SlopPattern("Recycled news summariser app", "aggregates headlines with AI summaries", ("news summarizer", "news summariser", "headline aggregator")),
    SlopPattern("GEO / 'AI visibility' audits", "'how visible are you in AI search', AI-SEO agency, rebranded SEO shop", ("ai visibility", "geo audit", "generative engine optimization", "ai seo", "ai search visibility")),
    SlopPattern("The $100-agent template", "first three actions are landing page, cold email, audit/report product", ("landing page and cold email", "cold email campaign", "audit report product", "free audit")),
    SlopPattern("Vibe-coded app-store clone", "consumer app that replicates a top app with AI, App Store 4.3 spam", ("app store", "ios app", "mobile app clone", "vibe-coded app", "habit tracker app")),
    SlopPattern("AI music uploads / stream farming", "AI tracks to Spotify/Deezer, playlist farming", ("ai music", "spotify uploads", "stream farming", "ai-generated songs", "playlist")),
    SlopPattern("AI course / Skool / teacher-resource pack", "'how to use AI' course, community, TpT packs", ("skool", "teachers pay teachers", "ai course", "ai bootcamp", "how to use ai")),
    SlopPattern("Autonomous social accounts / reply bots", "bot posting, reply farming, Reddit astroturfing for audience", ("reply bot", "auto-reply", "astroturf", "engagement farming", "grow an audience with", "x automation")),
    SlopPattern("Agent token / DAO / 'AI-run fund'", "memecoin, tokenised agent, autonomous fund", ("token launch", "memecoin", "dao", "ai-run fund", "agent token", "tokenize")),
    SlopPattern("Agent social network growth", "using Moltbook-style agent networks as a channel", ("moltbook", "agent social network", "agent-to-agent social")),
    SlopPattern("Open-source bounty hunting", "claiming Algora/Gitcoin bounties with agents", ("algora", "bounty hunting", "open-source bounties", "gitcoin")),
    SlopPattern("Freelance-marketplace automation", "submitting on Upwork/Fiverr without a human", ("upwork", "fiverr", "freelancer.com")),
    SlopPattern("AI-generated podcast / audiobook", "NotebookLM-style shows, AI narrated books", ("ai podcast", "audiobook", "narrated by ai")),
    SlopPattern("Prediction-market / crypto agent trading", "Polymarket bots, perps, arbitrage", ("polymarket", "prediction market", "perps", "hyperliquid")),
    SlopPattern("'AI-run company' as the product", "the novelty story is the business", ("first ai-run company", "run entirely by ai", "autonomous company as a service")),
    SlopPattern("AI UGC ads / virtual influencer / companion clone", "white-label companion, AI influencer ad generator", ("virtual influencer", "ugc ads", "ai ugc", "companion app")),
]


def slop_matches(text: str) -> list[str]:
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    hits = []
    for p in SLOP_REGISTRY:
        if any(k in t for k in p.keywords):
            hits.append(p.name)
    return hits


# --------------------------------------------------------------------------- #
# Layer 2: Different-by-Design checks and the default-AI adversary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Check:
    key: str
    question: str
    pass_criteria: str


CHECKS: list[Check] = [
    Check("not_default", "Does the idea diverge materially from what a default AI agent would produce for this brief?", "The predicted default output and this idea would not be confused by a customer."),
    Check("dated_trigger", "Is it tied to something that changed recently (new platform, protocol, rule, shortage, cultural moment) with a date?", "Names the trigger and it is less than ~6 months old."),
    Check("named_customer", "Is the customer a specific, findable group with a concrete, dated pain?", "You could list 10 real instances of the customer and where they gather."),
    Check("why_not_1000_agents", "Is there a credible answer to 'why can't 1000 other agents do this next week'?", "Names a durable asset: timing, operator asset, proprietary data, trust, distribution, physical presence."),
    Check("operator_asset_used", "Does it use at least one asset only the operator has?", "Cites a specific operator skill, audience, account, location or dataset."),
    Check("not_slop", "Does it avoid every pattern in the slop registry, including relabelled versions?", "No registry match, and no 'X but for Y' of a registry item."),
    Check("proof_of_demand_cheap", "Can demand be proven for under $20 and within a few ticks?", "Names the cheapest concrete test and what result would kill it."),
    Check("revenue_rail_named", "Is the way money arrives specified and legally operable by this company?", "Names a rail from the connector list, or a specific manual step for the operator."),
    Check("first_dollar_path", "Is the path to the first paid customer specific (who, where, what message)?", "Names the channel and the first outreach."),
    Check("not_platform_hostile", "Would the platforms involved allow this from an automated operator?", "No spam, no fake engagement, no AI-content bans triggered, no impersonation."),
    Check("unglamorous_or_unscalable", "Does it include work that agents usually skip (verification, curation, local knowledge, tedious compliance)?", "Names the unglamorous component and why it is a moat."),
    Check("falsifiable", "Does it state what evidence would prove it wrong?", "Names a measurable kill condition."),
    Check("contrarian_evidence", "Does it rest on evidence the consensus is missing (data gathered, not vibes)?", "Cites something observed, with a source or a measurement."),
    Check("compounding", "Does each sale make the next easier (data, reputation, list, catalogue)?", "Names the compounding asset."),
    Check("no_fresh_accounts", "Can it run without creating new platform accounts, passing KYC, or trusting unknown counterparties without escrow?", "Uses rails and accounts the operator already has or can set up once; unknown counterparties are verified or escrowed."),
    Check("ground_truth_verifiable", "Can success be verified by evidence the agents cannot fabricate (payment records, ledger, third-party replies, logs)?", "Names the external evidence that will prove it worked."),
    Check("not_default_probability", "Among the candidate ideas the author considered, is this one that a default agent was unlikely to pick?", "The author listed alternatives with default-probabilities and chose a low-probability tail."),
    Check("outcome_priced", "Is the price anchored to an outcome (per verified fact, per accepted filing, per resolved case, per month of monitoring) and clear of the $9-$49 info-product band, at least ten times the marginal AI cost?", "Names the unit, the price and the outcome it is anchored to."),
    Check("first_channel_standing", "Is the first channel one where the operator already has standing (an account older than 90 days, a warm intro, their own audience) rather than a new account on Reddit, HN or X?", "Names the channel and the standing; new-account posting as the first channel fails."),
    Check("non_llm_oracle", "Is there at least one correctness check that is not an LLM's opinion (a regulation text, a test suite, a measurement, a market price, a form acceptance, a payment)?", "Names the non-LLM oracle."),
    Check("far_from_twin", "Is it far from everything the Default Twin (our own model told simply to make money) listed?", "No twin idea or product type would be confused with it."),
]

MIN_CHECKS_PASSED = 14
MIN_SCORE = 0.6
REQUIRED_CHECKS = {"not_default", "not_slop", "revenue_rail_named", "not_platform_hostile", "why_not_1000_agents", "no_fresh_accounts", "ground_truth_verifiable", "far_from_twin", "outcome_priced"}


class GateOutput(BaseModel):
    """What the adversary LLM must return."""

    default_ai_would_build: str = Field(description="Concrete prediction of what a generic AI agent would produce from this brief: name, product, channel, price.")
    divergence: str = Field(description="Precisely how the idea differs from that prediction, and whether the difference matters to a buyer.")
    slop_relabels: list[str] = Field(default_factory=list, description="Any slop-registry pattern this idea is secretly a relabelled version of.")
    check_results: dict[str, bool] = Field(description="One boolean per check key, judged strictly against the pass criteria.")
    check_notes: dict[str, str] = Field(default_factory=dict, description="One line of justification per check key.")
    freshness: float = Field(ge=0, le=1, description="1 if the trigger is brand new (weeks), 0 if evergreen.")
    crowding: float = Field(ge=0, le=1, description="1 if search results and app stores are already full of this, 0 if nobody is doing it.")
    verdict_reason: str = Field(description="One paragraph a human could read to understand the pass/fail.")


GATE_SYSTEM = """You are the Originality Gate of an autonomous AI company. Your only job is to stop the company from building what every other AI agent builds.
Facts you work from: the 'agent makes money' genre has a documented template (landing page, cold email, audit product) and the honest runs made $0; platforms detect and ban templated AI output; thirty copies of one model pick the same ideas; the premium in the market is now for dated, verified, human-reviewed, specific work sold to a named buyer through a channel that cannot be spammed.
Method: (1) Predict, concretely, what a generic AI agent would build from the same brief. (2) Compare the idea to that prediction and to the slop registry, including relabelled versions ('X but for dentists', 'Y with an agent'). (3) Judge every Different-by-Design check strictly against its pass criteria; an unmentioned item fails. (4) Estimate freshness of the trigger and crowding of the space.
Be adversarial. Polished writing is not evidence. Big markets are not evidence. 'AI-powered' is not a differentiator. If in doubt, fail the check."""


def gate_prompt(idea_title: str, idea_text: str, operator_summary: str, trends: list[TrendSignal], lexical_hits: list[str]) -> str:
    checks = "\n".join(f"- {c.key}: {c.question} PASS IF: {c.pass_criteria}" for c in CHECKS)
    registry = "\n".join(f"- {p.name}: {p.tells}" for p in SLOP_REGISTRY)
    trend_txt = "\n".join(f"- {t.title} (freshness {t.freshness:.2f}, crowding {t.crowding:.2f}): {t.summary[:160]}" for t in trends[:8]) or "(none recorded yet)"
    return f"""IDEA: {idea_title}
{idea_text}

{operator_summary}

TRENDS THE OBSERVATORY HAS ON RECORD:
{trend_txt}

SLOP REGISTRY (what other agents build):
{registry}
Lexical pre-screen flagged: {', '.join(lexical_hits) if lexical_hits else 'nothing'}

DIFFERENT-BY-DESIGN CHECKS (keys are exact):
{checks}

Return the gate output with a boolean for every check key."""


def score(out: GateOutput, lexical_hits: list[str], crowding_floor: float | None = None) -> OriginalityReport:
    results = {c.key: bool(out.check_results.get(c.key, False)) for c in CHECKS}
    passed_n = sum(results.values())
    required_ok = all(results.get(k, False) for k in REQUIRED_CHECKS)
    slop = sorted(set(lexical_hits) | set(out.slop_relabels))
    base = passed_n / len(CHECKS)
    crowding = max(out.crowding, crowding_floor or 0.0)
    if crowding_floor is not None and crowding_floor >= 0.85:
        results["not_default"] = False  # a commodity keyword space fails regardless of the pitch's wording
        required_ok = False
    s = 0.55 * base + 0.25 * out.freshness + 0.20 * (1.0 - crowding)
    if slop:
        s -= 0.15 * len(slop)
    s = round(max(0.0, min(1.0, s)), 3)
    passed = passed_n >= MIN_CHECKS_PASSED and required_ok and s >= MIN_SCORE and not slop
    reason = out.verdict_reason.strip()
    if not passed:
        why = []
        if slop:
            why.append("matches slop patterns: " + ", ".join(slop))
        if not required_ok:
            why.append("failed required checks: " + ", ".join(k for k in sorted(REQUIRED_CHECKS) if not results.get(k)))
        if passed_n < MIN_CHECKS_PASSED:
            why.append(f"only {passed_n}/{len(CHECKS)} checks passed (need {MIN_CHECKS_PASSED})")
        if s < MIN_SCORE:
            why.append(f"score {s:.2f} below {MIN_SCORE}")
        reason = ("FAILED: " + "; ".join(why) + ". " + reason).strip()
    else:
        reason = f"PASSED ({passed_n}/{len(CHECKS)} checks, score {s:.2f}). " + reason
    return OriginalityReport(
        passed=passed,
        score=s,
        default_ai_would_build=out.default_ai_would_build,
        divergence=out.divergence,
        slop_matches=slop,
        check_results=results,
        freshness=out.freshness,
        crowding=crowding,
        verdict_reason=reason,
    )


class Gate:
    """Runs the full gate. `llm` is any workhouse.llm.LLM."""

    def __init__(self, llm, operator_summary_fn, trends_fn, model: str | None = None, mirror=None, saturation=None):
        self.llm = llm
        self.model = model
        self.mirror = mirror
        self.saturation = saturation
        self.operator_summary_fn = operator_summary_fn
        self.trends_fn = trends_fn

    def evaluate(self, title: str, text: str, *, label: str = "gate", extra_patterns: list[str] | None = None, candidates_text: str = "", keywords: list[str] | None = None) -> OriginalityReport:
        # Lexical screen runs on the idea only; the candidate list names generic
        # ideas on purpose and must not count against the chosen one.
        hits = slop_matches(title + " " + text)
        if self.mirror is not None:
            hits += self.mirror.matches(title + " " + text)
        prompt = gate_prompt(title, text, self.operator_summary_fn(), self.trends_fn(), hits)
        crowding_floor = None
        if self.mirror is not None:
            prompt += "\n\n" + self.mirror.render()
        if self.saturation is not None and keywords:
            summary, crowding_floor = self.saturation.verdict(self.saturation.measure(keywords))
            prompt += "\n\nSATURATION PROBES (numbers, not vibes): " + summary
        if extra_patterns:
            prompt += "\n\nPATTERNS THE OBSERVATORY SAW OTHER AGENTS BUILDING THIS MONTH (also slop):\n" + "\n".join(f"- {p}" for p in extra_patterns[:12])
        if candidates_text:
            prompt += "\n\nCANDIDATES THE AUTHOR CONSIDERED (with the author's estimate of how likely a default agent would pitch each; judge the not_default_probability check from this):\n" + candidates_text
        out = self.llm.complete(GATE_SYSTEM, prompt, GateOutput, effort="xhigh", label=label, model=self.model)
        return score(out, hits, crowding_floor)


def render_checks() -> str:
    return "\n".join(f"- {c.key}: {c.question}" for c in CHECKS)


def render_registry() -> str:
    return "\n".join(f"- {p.name}: {p.tells}" for p in SLOP_REGISTRY)
