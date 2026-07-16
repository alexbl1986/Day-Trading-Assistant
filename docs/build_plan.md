# Certification Challenge — Trading Assistant: Consolidated Design & Build Plan

## Context

Building an **Agentic RAG trading assistant**: a signal-following retail options day-trader on IBKR who manually reconciles his book against his own exposure rules and his desk's Hebrew daily/weekly market reviews. Task 1 (problem/audience) is done (`docs/task1_problem_scope.md`).
This plan is the output of a full `/grill-with-docs` design session — all decisions below
are user-confirmed; ADRs 0001–0006 and `CONTEXT.md` are already written in
`Certification_Challenge/`.

**Headline value:** automate the *synthesis* (pain H) — reconcile positions vs. exposure
policy vs. desk thesis and answer "exactly what should I change?" Live IB feed is out of
scope (ADR-0001).

## Settled architecture (references)

- **Data split (ADR-0002):** IBKR CSVs → purpose-built deterministic pandas tools (never
embedded). Desk-review PDFs → RAG corpus. External data mirrors the split: yfinance
(numeric, behind a `QuoteProvider` interface, fail-loud) + Tavily (narrative).
- **Two CSV types, one upload control (format-sniffed, no user dropdown):**
(a) **Tactical book export** (flat Flex table: Symbol, CurrencyPrimary, FXRateToBase,
AssetClass, Strike/Expiry/Put-Call, Quantity, MarkPrice, PositionValue, CostBasisPrice,
PercentOfNAV; options in OCC symbology; dates DD/MM/YYYY) — uploaded several times a day
→ **positions snapshot** (exposure, scale-out, list_positions). (b) **YTD Activity
Statement** (27-section) — occasional → **trades ledger** (get_trades, performance
attribution); its Open Positions never overwrite the fresher tactical snapshot. Each
store keeps its own `uploaded_at`. Router sniffs line 1 (`"Symbol","CurrencyPrimary"…`
vs `Statement,Header…`); unrecognized CSV → reject-and-advise.
**Data-verified rules:** `CostBasisPrice` is 0 on ALL tactical rows (stocks and options)
→ **ledger upload is a prerequisite for scale-out accuracy** — cost basis comes from the
statement; `scan_scaleout` refuses (with an "upload a recent statement" message) when no
ledger is present, and warns on ledger staleness. `PercentOfNAV` is percent of the
*asset class's* NAV, not the book → exposure math computes its own percentages from
`PositionValue × FXRateToBase ÷ total NAV` and ignores that column.
- **Runtime corpus = exactly 2 docs:** latest daily + latest weekly, replace-on-upload
(warn if older date). Stance-shift history (task1 eval Q4) deferred to Demo Day.
**Per-user corpus** via Qdrant `user_id` payload filter (ADR-0005).
- **Ingestion (ADR-0004):** PyMuPDF + deterministic RTL-repair pass (Hebrew corpus;
pypdf output = ground truth for repair unit tests). Adaptive structure-aware chunking:
per-doc modal font = body, ≥ body+~2pt = headings clustered per-doc; chunk = between
headings; guardrails (sub-split long, merge tiny). Metadata: review_date, doc_type,
section, tickers (whitelist-validated, ALSO verbatim in text for BM25), page.
*(Amended 2026-07-14: tickers metadata dropped — never wired to a whitelist in prod and
unused downstream; layout tables now serialized row-wise at extraction — ADR-0004
amendment.)*
Scanned/image PDFs rejected with "attach it in chat instead" advice (session-scoped read).
- **Rules (ADR-0003, scope expanded by ADR-0007 from real-usage transcripts):** exposure
policy = typed record in persistent LangGraph Store, seeded from default config; writes
only via confirmation-gated `update_policy` tool (LangGraph `interrupt()`); audit trail.
Trader owns and can edit his thresholds. Record fields = rules a route reads: sizing
(options 1% / stocks 3% new / 6% existing-holding cap), 20% max offensive exposure,
options ≤10% NAV, **hedge ratio = put value ÷ call value 10–15%** (his precise formula —
confirm cross-hedge denominator with trader before Day 3), cross-hedge, scale-out,
DTE exit matrix (4 tiers), IV-shield threshold (manual-check reminder), moonshot +150%/+50% (labeled proxy). NOT machine-checked: roll rules (no free
Greeks), roll-chained campaigns, screenshot audits (redirect to re-upload).
- **Agent shape (ADR-0006):** structured single-context LangGraph — intake/scoper
(structured output, multi-label intent + entities + hypothetical flag, 1 clarification
round max) → deterministic pre-fetch by route (union) → tool agent (iteration cap) →
per-route typed synthesis (copies numbers verbatim) → deterministic audit node (numbers
must match tool outputs; 1 bounce then warning banner) → interrupt-gated policy writes.
Subagents rejected. Daily supersedes weekly on conflict (prompt + eval case).
**Two routes added by ADR-0007:** `trade_signal_eval` (shorthand signal → parse w/
chain-verified expiry, size from policy, DTE-tier exit plan, desk bias+tier cross-ref
incl. Tier-2 sizing cap, inventory conflict check, fail-loud IV rank → typed TradePlan;
built LAST behind a clean seam, first cut if schedule slips) and `daily_briefing`
(composition: exposure + scale-out + hedge ratio + desk summary + index
technicals → DailyBriefing type).
**Cold-start contract:** snapshot/ledger/corpus reads distinguish "never uploaded" from
"empty result" — tools return typed `MissingData(store, remedy)` instead of an empty
frame (an empty snapshot must not audit-pass as "0% exposure, within policy");
synthesis answers what it can + explicit "upload X" line; audit blocks portfolio numbers
whose backing store was never uploaded. market_regime + policy routes work with zero
uploads (policy record seeded at first login).
- **Tool roster:** search_desk_reviews · list_positions / get_trades (**default =
current active campaign**: entry, scales, net basis, realized-so-far, house-money
status; full history on explicit request; campaign = fills in same contract while
continuously open, rolls don't chain in v1) / check_exposure / scan_scaleout (incl.
moonshot ≥+150% proxy flag) (Postgres snapshot) · performance-attribution tools (Tier 1:
realized P/L by symbol/class/month, win rate, cost drag, hedge cost, scale-out proxy —
from IBKR's own Realized P/L column) · get_quote / get_option_chain /
get_technical_snapshot (50/200d MA, RSI, swing hi/lo) · search_web (Tavily) ·
update_policy (gated) · remember/segment-membership.
Deliberate no-s (ADR-0006/0007): chart patterns, counterfactual backtests, statistical
"validity", desk-signal attribution, IV-rank computation, machine-graded rolls,
path-dependent moonshot, screenshot compliance audits.
- **Stack:** LangGraph ≥1.0 server (Docker on **Railway**, session-9 recipe: port 8080)
  - forked **agent-chat-ui** on **Vercel** (login page, upload control, policy panel).
  **Vercel AI Gateway** (user's credits; static AI_GATEWAY_API_KEY; OpenAI-compat
  endpoint; default `openai/gpt-5.4-mini` — verify slug day 1). Embeddings direct OpenAI
  `text-embedding-3-large`. Qdrant Cloud free (vectors), Railway Postgres (checkpointer +
  Store + parsed snapshot). LangSmith EU + gateway dashboard. Simple credential login
  (2 users: alex-demo / real-user), user_id scopes everything.
- **Data policy (ADR-0005):** real statements & desk PDFs never in git; synthetic fixtures
(statement + 1 daily + 1 weekly Hebrew reviews) committed; deployed endpoint gated;
tracing behind env flag; committed eval artifacts = questions/ground truth/summary tables.



## Retrieval & eval protocol (final, user-decided)

- **Base prototype (Task 4) ships the full pipeline: dense(3-large) → +BM25 w/ RRF →
+parent-child recovery** (child = theme block, parent = full section).
*(Amended 2026-07-14: parent-child recovery removed — chunks retrieved directly, verbatim
and traceable to `docs/chunk_preview`; see ADR-0006 amendment.)*
- **Task 6.1 advanced addition: Cohere reranking** on top. Before/after table = base
pipeline vs. +rerank (LangSmith experiments).
- **Task 6.3 "other piece": embedder A/B** (3-large vs 3-small, optionally Cohere
multilingual), same eval set.
- **Harness (Task 5), three layers:** (1) pytest golden values for deterministic tools
(synthetic statement); (2) retrieval metrics (hit-rate/MRR) with **exhaustive answer
key** (~60–100 chunks, I draft / user spot-checks); harness can toggle pipeline stages
for an ablation grid; (3) RAGAS faithfulness/relevancy + synthesis rubric (cites right
rule; refuses on missing quote; daily-vs-weekly collision case; cold-start case — fresh
account asks "am I within policy?", correct behavior names the missing upload and
invents no numbers; signal-eval case — shorthand in → exact contracts/SL/TP out, IV rank
stated as manual-check; campaign-history goldens — current campaign vs full history). Seed = task-1 questions
(minus deferred Q4) + performance-review Qs + paraphrase variants; synthetic-corpus twins.



## Build order (4 days)

**Day 1 — foundations + smoke tests:** repo scaffold (uv project, `app/` packages);
day-1 smoke list: gateway slug + OpenAI-compat call from Python, agent-chat-ui interrupt
rendering, Qdrant Cloud + Railway Postgres provisioning. **Two IBKR parsers + sniffing
router**: tactical book export (flat Flex table → positions snapshot) and Activity
Statement (27-section CSV → trades ledger) + synthetic fixtures for both + pytest goldens.
**IBKR↔Yahoo symbol mapping layer** (build-risk, reduced: tactical options use standard
OCC symbology; equities `HPS.A→HPS-A.TO`, `SIVE→SIVE.ST`; fallback = statement close
marked as-of-date).

**Day 2 — ingestion + tools:** PyMuPDF extraction + RTL-repair (tested vs pypdf ground
truth) + adaptive chunker (tested on all 6 archive PDFs) + Qdrant indexing w/ replace-on-
upload + reject-and-advise guardrail. Deterministic tools incl. Tier-1 attribution +
technical snapshot + **campaign grouping for get_trades (house-money derived field) +
moonshot proxy flag** (pytest goldens for each). Retrieval pipeline
(dense + BM25/RRF + parent-child). *(Amended 2026-07-14: parent-child removed.)*

**Day 3 — agent + UI + deploy:** LangGraph graph (intake → pre-fetch → agent → synthesis →
audit → interrupt), policy record + update_policy, memory wiring, cold-start contract
(`MissingData` return type through pre-fetch/synthesis/audit), `daily_briefing` route
(composition of existing fetches). Fork agent-chat-ui: login, upload, policy panel.
Deploy Railway + Vercel; end-to-end on phone. **Last (clean seam, first cut if
slipping): `trade_signal_eval` route** — signal-parse intake schema + TradePlan
synthesis type + DTE-tier lookup.

**Day 4 — evals + Task 6 + submission:** answer-key labeling, run harness (3 layers),
add Cohere rerank, before/after + embedder A/B tables, Task 7 write-up, README written
document addressing every deliverable, infra + agent-workflow + "today" diagrams
(mermaid), Loom ≤10 min (use-case → live demo on phone → eval tables), final repo hygiene
(verify no real data staged; push to origin).

## Deliverables mapping

- Task 1 ✅ done (Q4 row annotated deferred). Task 2: one-liner + infra diagram (stack
above, one sentence each) + agent workflow diagram + 1–2 paragraphs. Task 3: chunking
write-up (adaptive structure-aware + why; template-drift evidence) + data sources
(IBKR CSV + desk PDFs + yfinance + Tavily, how they interact). Task 4: deployed
prototype (Railway+Vercel, gated). Task 5: harness + conclusions. Task 6: +rerank table
  - embedder A/B + improvement narrative. Task 7: keep/change reflection (keep:
  architecture, harness; change/Demo Day: archive corpus + stance-shift, research library,
  OCR ingestion, TradingView rating/widget, multi-agent research mode, official market
  data, live IB feed via IBKR API replacing CSV uploads, Discord signal bridge
  (push → auto pre-trade eval), paid options data unlocking honest IV rank + Greeks/roll
  grading + path-dependent moonshot, roll-aware campaign chaining — plain-language gap
  list for the trader in gitignored docs/PROTOTYPE_GAPS.md).



## Verification

- pytest suite green (parser, RTL repair, chunker on 6 fixtures, tools, symbol mapping,
audit node).
- LangSmith experiment links for: base pipeline, +rerank, embedder A/B, layer-3 rubric.
- End-to-end on deployed URL from a phone browser: login → upload synthetic statement +
reviews → "am I within policy?" (typed report w/ audit pass) → "raise my options cap to
12%" (interrupt → confirm → policy panel updates) → scanned-PDF upload rejected with
advice → performance question answers with cost-drag numbers → paste "AAOI 150 NEXT
WEEK 3.1" (TradePlan: contracts, SL/TP tiers, desk tier, IV-rank manual-check line) →
"morning briefing" (DailyBriefing renders).
- Repo check before push: no real statements/PDFs staged; ADRs + CONTEXT.md current.



## Post-approval bookkeeping

Update auto-memory (`cert-challenge-grilling-design.md`) with the Q21 resolution (base =
full pipeline, Task 6 = +rerank) and grilling-complete status.