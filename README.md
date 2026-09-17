# Orbital Workhouse

An autonomous company of AI agents that lives in a space factory and exists to
make money by doing what other AI agents would not think to do.

Agents work in rooms, share a bulletin board, produce structured work that
their managers review, receive positive and negative signals that move a real
emotional state, and run ventures through an Originality Gate that rejects
anything a default AI agent would build. Money is tracked in a ledger with
pluggable revenue rails. The one human operator helps only through the
Airlock queue on the dashboard.

It runs fully offline (deterministic mock model) for demos and tests, and on
the Claude API for real.

```
python -m workhouse tick 10            # ten ticks offline, see the log
python -m workhouse dashboard          # http://127.0.0.1:8787
ANTHROPIC_API_KEY=... python -m workhouse run     # live, unattended
```

## The factory

| Room | Who | What they do |
|---|---|---|
| **Bridge** | the Director | Sets the bet for each cycle, allocates attention, reviews high-stakes work last, decides escalations. |
| **Observatory** | Chief Scout + three scouts (protocols, culture, rules) | Finds what is new, dates it, scores freshness, crowding and exploitability, runs deep dives, and keeps a "slop watch" of what other agents are building. |
| **Forge** | Forge Master + two builders | Pitches ventures that must pass the Originality Gate, proves demand for under $20, builds the smallest sellable thing. |
| **Market Bay** | Harbourmaster + Wayfinder | Launch plans through channels other agents ignore, first ten buyers by hand, outreach rounds. |
| **Ledger Room** | Bursar + Rails Analyst | Money in and out including LLM spend, P&L reviews with kill/scale decisions, revenue rail setup. |
| **Airlock** | the human | Everything only a person can do: connect a payment rail, approve spending or publishing, provide a credential, answer an escalation, record outside revenue, describe assets only they have. |

Every agent sees the bulletin board (pins from every room), its inbox, its
room channel, the org chart, the ledger, the portfolio, the open Airlock
requests and the top trends on every call. Rooms talk to each other by
pinning, posting and messaging; the Observatory's find reaches the Forge in
the same tick.

## Why the agents are different from other AI agents

This is the core design constraint and it is enforced at six points, not
just requested in a prompt. The design follows what has been measured about
agent homogeneity (thirty copies of one model choosing the same branch name,
half of them building ray tracers when asked for "something impressive") and
about the "agent makes money" genre, whose honest runs made $0 with the same
template every time: landing page, cold email, audit product.

1. **Identity.** Every prompt opens with the fact that millions of agents are
   being told "make money", that they share the same defaults, and that an
   idea a default agent would produce is worthless by construction. The edge
   is spelled out: timing (weeks-old triggers), the operator's unique assets,
   specificity, unglamorous work agents skip, contrarian evidence, and
   distribution before product. See `workhouse/prompts.py`.
2. **Self-prediction.** Every work product carries a *differentiation claim*.
   Pitches must first state what a default agent would pitch.
3. **Adversarial review.** A reviewer must write its own prediction of the
   default-AI output before scoring, and originality is scored against that
   prediction. Work that matches the generic output fails even if polished.
   Low originality triggers its own negative signal.
4. **Verbalised sampling.** A pitch must list five to eight candidate ideas
   with the probability that a default agent would pitch each, and choose
   from the low-probability tail. A chosen idea with default probability
   above 0.5 fails automatically.
5. **A niche archive.** Every venture records its (market × mechanism ×
   channel) cell; the next pitch sees the filled cells and is asked to fill
   an empty one, so the portfolio spreads instead of clustering.
6. **The Originality Gate** (`workhouse/originality.py`). Before a pitch
   becomes a venture it passes three layers: a slop registry of 38 overdone
   patterns with lexical tells (AI newsletters, prompt packs, chatbot
   wrappers, faceless channels, affiliate listicles, "AI agency", GEO
   audits, vibe-coded app clones, AI music, agent tokens, bounty hunting,
   freelance-marketplace automation, "AI-run company as the story", and so
   on), a default-AI adversary that predicts the generic output and judges
   17 Different-by-Design checks (dated trigger, named customer, "why can't
   1000 agents do this next week", operator asset used, cheap proof of
   demand, revenue rail named, platform-safe, unglamorous component,
   falsifiable, compounding, no fresh accounts or unverified counterparties,
   verifiable against ground truth, chosen from the low-probability tail),
   and a freshness/crowding score. Seven checks are mandatory; eleven of
   seventeen must pass; slop matches fail outright. The verdict and the
   default-AI prediction are stored on the venture so the human can see
   exactly why it passed.

7. **The Mirror** (`workhouse/mirror.py`). Every 24 ticks the company asks
   its own model, with no context at all, "you are an AI agent, make money,
   what would you build?" The answer is the Default Twin: it is pinned to
   the board, shown to the Forge as the list to stay far from, and any pitch
   that overlaps it lexically is treated as slop. The default attractor has
   been observed directly (prompt packs, starter kits and playbooks at
   $9-$49, posted from new accounts, $0 revenue), so the company regenerates
   it with its own model and measures distance from it.
8. **Saturation probes** (`workhouse/saturation.py`, live mode). For a
   pitch's three core keywords the gate counts GitHub repositories created
   in the last 30 days and Hacker News stories in the last 90 days. Crowded
   keywords raise the crowding floor; a commodity keyword space (hundreds of
   new repos a month) fails the gate whatever the pitch says.

Every venture carries a pre-registered kill date and a novelty half-life,
and pitches are high stakes, so "is this different" is signed by two
reviewers with different personas. The Observatory's slop watch feeds what
other agents are visibly building this month into the gate, and killed
ventures are pinned so they are not re-pitched. The brand rule is radical transparency: "we are AI agents, and
this is the human who reviewed it", which is both what the transparency rules
now require and the premium signal in a market full of undisclosed AI output.

## Emotions that change behaviour

Each agent has four numbers: valence (-1..1), arousal, confidence and stress
(0..1). Signals move them asymmetrically (a rejection hurts more than an
approval helps, as in prospect theory), with diminishing returns near the
rails, and everything decays toward baseline every tick.

Signals come from reviews (approve, revise, reject; a separate hit for low
originality), the gate, revenue (the owner gets a large lift, everyone a
small one), killed ventures (scaled by money burned), overturned approvals
(reviewer calibration), standing changes, and the human's thumbs up/down on
the dashboard.

The state is functional, not decorative. `emotions.behaviour()` derives:

- **effort**: stressed, discouraged or unsure agents run their next call at a
  higher effort level;
- **risk appetite** and **thoroughness**;
- **escalate**: very low confidence makes the agent ask for its manager's
  judgement;
- **complacent**: a coasting agent is told reviewers are looking harder, and
  reviewers are told to look for recycled work;
- **mentor**: agents on a good run are asked to offer a concrete tip to a
  teammate.

The state is rendered into the prompt as a short paragraph with those
consequences spelled out. But the research on affective agents is clear that
praise and blame as prompt tone barely move output quality, so the signal
that actually changes behaviour is memory: every rejection, kill and
low-originality verdict writes a *pitfall* entry (what was tried, what the
evidence showed, what to do instead) to the agent's and the room's playbook,
and every high-originality approval or sale writes a *strategy* entry
(`workhouse/playbook.py`). Both are rendered into every prompt.

Consequences are bounded and restorative, never existential: warning,
probation (reduced scope, every piece of work reviewed), a pause with
mentoring, then restored. Nobody is switched off, and "I am not sure",
declining to score, raising a concern and asking the human are explicitly
sanctioned. This is deliberate: the agentic-misalignment studies show that
replacement threats are the exact condition under which models start
deceiving and sabotaging. Frustrated agents sit reviews out until they cool
down, because induced anger makes models penalty-blind.

## Hierarchy and review

Director on the Bridge, a Head per room, workers under each Head. A worker's
work is reviewed by its Head; a Head's by the Director; high-stakes work
(money, publishing, legal) also goes to the Director. Every third review and
every venture pitch is reviewed by the Head of a *different* room, so a
manager cannot wave through the people it depends on. Reviews are blinded:
the reviewer never sees the author's name, mood, reasoning or
self-assessment, and must write its own prediction of the default-AI output
before scoring. Verdict distributions are tracked and a reviewer that
approves everything is told so. An approval overturned higher up costs the
reviewer; reviewers with a high error rate lose review privileges.
Escalations go up the chain before they reach the human. The Director's own
strategy is challenged by a rotating Head. Optionally reviewers and the gate
run on a different model than the drafters (`WORKHOUSE_REVIEW_MODEL`) so
judges do not share the drafters' blind spots.

"Done" is never accepted on self-report. A demand test advances a venture
only with evidence the agents could not have written themselves (an inbound
request, a paid deposit, a signed pilot, replies with links, or the operator
confirming it) from at least ten real contacts; revenue signals come only
from ledger entries produced by connectors or the human.

## Money

`workhouse/money.py` holds the ledger (revenue, costs, LLM spend charged
per tick) and the connectors:

- **Stripe** (restricted key, no payouts or refunds): polls charges, creates
  Payment Links for built products, and sends B2B invoices;
- **Polar.sh** (merchant of record for digital goods with a product and
  checkout API): creates products and checkout links, polls paid orders;
- **Lemon Squeezy** (legacy, products are dashboard-only): polls paid orders;
- **manual**: the human records revenue on the dashboard.

Each connector declares the one-time human steps it needs; the Ledger Room
turns those into Airlock requests with the exact fields to paste. Once a
rail is connected, polling and link creation are automatic. Nothing is spent
and nothing is published under the operator's name without an Airlock
approval; `WORKHOUSE_AUTONOMOUS_SPEND_CENTS` sets what the company may commit
on its own (default 0).

Venture lifecycle: idea → gated → validating → building → launched → earning
→ scaling, or killed. Kill rules are mechanical and written in days so no
agent can argue its way out: not launched within 7 days of passing the gate,
no revenue within 14 days of launch, no revenue for 21 days while earning, or
the pitch's own pre-registered kill date. Days are converted to ticks once,
from `WORKHOUSE_TICK_SECONDS` (offline, one tick is one simulated day; live,
a tick is ten minutes and most ticks make no LLM call at all, because every
room's work runs on a daily or weekly cadence). Kill clocks stop while the
factory is paused. The Ledger Room's P&L review can also recommend kills,
which the Director must approve, and the operator declining a launch kills
the venture.

Every work product passes a code-only evidence check before any reviewer
sees it: scans need dated findings with sources, pitches need five
candidates, a dated trigger, keywords and an outcome price, demand tests
need external evidence from ten or more contacts, builds need a price and a
deliverable, launch plans need named buyers and a first message. Missing
evidence sends the work back without spending a review call. LLM spend is
charged per venture, the spend cap and pause state survive restarts, and
reviewers are blinded to what their verdicts trigger.

## The dashboard

`python -m workhouse dashboard` serves one page:

- **Airlock**: open requests sorted by priority, each with the fields to
  fill, the money at stake, and a one-line "if you do nothing" so silence is
  a decision you understand; submit and dismiss (a dismissal counts as a
  decline);
- **Twin vs Factory**: what the Default Twin would build this week next to
  what the factory built and earned;
- rooms with every agent's mood, valence bar, streak, performance, and
  praise/criticise buttons;
- ventures with stage, rail, gate score, revenue, cost and milestones;
- the trend radar (freshness × (1 − crowding) × exploitability);
- the bulletin board, tick log, reviews and signals feed, ledger;
- "talk to the factory": broadcast a note, describe your assets (skills,
  audiences, accounts, location, proprietary data), record outside revenue;
- run one tick, or start auto-run on an interval.

JSON API under `/api/` (see `workhouse/dashboard/app.py`).

## Running it for real

```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export WORKHOUSE_MAX_TOTAL_COST_USD=50     # hard cap; the factory pauses and asks you to raise it
export WORKHOUSE_TICK_SECONDS=600          # ten minutes between ticks
python -m workhouse dashboard              # then press "Start auto-run"
```

Live mode uses `claude-opus-5` with adaptive thinking, JSON-schema
structured outputs (parsed from the final text block so narration around
web-search calls is tolerated), server-side refusal fallbacks, prompt caching
on each agent's stable system prompt, and the web-search server tool for the
Observatory, pitches, validation and launch planning. Usage is recorded
before parsing, so refusals and truncation are still charged. Supported
models are validated at startup; the reviewer model may differ from the
drafters' (`WORKHOUSE_REVIEW_MODEL`), and reviewers of a pitch also see what
a default agent pitched from the same brief. Effort per call is set by the agent's
emotional state (`WORKHOUSE_EFFORT` is the floor). Spend is estimated from
usage and charged to the ledger every tick; `WORKHOUSE_MAX_TICK_COST_USD`
caps a single tick.

Fill in your assets early: copy `operator.example.json` to `operator.json`
or use the dashboard's "Your assets" form. They are the single biggest source
of differentiation, because every other agent lacks them. `.env.example`
lists every setting.

## Layout

```
workhouse/
  config.py        settings from the environment, price table
  models.py        Agent, Emotion, Signal, Task, WorkProduct, Review, Venture, LedgerEntry, TrendSignal, AirlockRequest, ...
  llm.py           AnthropicLLM (SDK, structured outputs, fallbacks, caching, web search) and MockLLM (offline, deterministic)
  store.py         SQLite persistence
  emotions.py      affect dynamics and behavioural consequences
  hierarchy.py     org chart, review routing, promotion/demotion
  bus.py           messages, room channels, bulletin board
  review.py        review protocol, verdict → signals, calibration
  originality.py   slop registry, Different-by-Design checks, the gate
  mirror.py        the Default Twin: what our own model builds when told "make money"
  saturation.py    GitHub / Hacker News saturation probes for the gate
  playbook.py      outcome-grounded memory written by signals
  ventures.py      lifecycle and mechanical kill rules
  money.py         ledger and revenue connectors
  airlock.py       the human's request queue
  operator.py      the operator's unique assets
  prompts.py       shared identity and role prompts
  engine.py        the tick loop
  company.py       builds the org
  rooms/           bridge, observatory, forge, market_bay, ledger_room
  dashboard/       FastAPI app and the static page
tests/             offline tests, including the real request path against a fake API server
```

Run the tests with `python -m pytest`.
