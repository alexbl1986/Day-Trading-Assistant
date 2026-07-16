# Structured single-context agent graph; subagents rejected

The agent is one reasoning context wrapped in deterministic scaffolding, not a plain
tool-calling loop and not a multi-agent system. The LangGraph shape:

1. **Intake/scoper node** — structured output: **multi-label** intent ({status_check,
   desk_question, rebalance_advice, trade_history, performance_review, policy_change,
   market_regime, trade_signal_eval, daily_briefing, off_topic} — the last two added by
   ADR-0007 from the trader's real-usage transcripts) plus extracted entities and a
   `hypothetical` flag ("if I raised my cap…" must analyze, not write). Reads thread history; ambiguity gets one
   clarification round, then best-effort with stated assumptions; off_topic is refused.
2. **Deterministic pre-fetch by route** — the calls that must happen are graph code, not
   LLM discretion: status/rebalance always runs exposure + scale-out + positions + quotes;
   desk questions always retrieve. Multi-label intent fetches the union.
3. **Agent node** with the full tool roster for the long tail, under an iteration cap.
   Web-search results are demoted to quoted evidence (prompt-injection surface).
4. **Synthesis node** — per-route typed outputs (an AssessmentReport with per-rule
   pass/breach, actions, and evidence refs for status/rebalance; simpler types for e.g.
   trade history). Synthesis copies numbers verbatim from the evidence table — it never
   recomputes.
5. **Deterministic audit node** — code, not LLM: every number in the answer must match a
   tool output, every cited rule must exist in the policy record, no price may appear if
   the quote was unavailable, and no portfolio number may appear if its backing store was
   never uploaded. One bounce-back, then deliver with a warning banner.
6. **policy_change routes through a LangGraph `interrupt()`** — the human-approval gate
   required by ADR-0003.

**Cold-start contract.** Snapshot, ledger, and corpus reads must distinguish *"never
uploaded"* from *"empty result"*. A pandas query over a never-populated snapshot returns
an empty DataFrame — which `check_exposure` would read as "options at 0%, fully within
policy": a confident, audit-passing, wrong answer. So tools and pre-fetch return a typed
`MissingData(store, remedy)` value instead of an empty frame when the backing store was
never uploaded (generalizing the existing `scan_scaleout`-refuses-without-ledger rule);
synthesis renders what *is* answerable and adds an explicit "to answer the rest, upload
X" line rather than refusing the whole question; the audit rule above enforces it.
Routes needing no uploads work on day one: `market_regime` (live quotes/Tavily) and
policy questions/edits (record is seeded from default config at first login).

**Subagents-as-tools were considered and rejected.** The product's value is the synthesis
across portfolio, policy, quotes, and desk thesis; subagent boundaries cut exactly at that
seam — the supervisor would reason over summaries-of-summaries and lose the specific
contracts and tickers that make a recommendation actionable. None of the conditions that
justify subagents hold here (small tool outputs, convergent not parallel questions, ~10
tools). Multi-agent remains the Demo Day evolution if a large research-library corpus
lands (context-bloat condition would then hold).

## Amendment (2026-07-12) — after the first working slices and a trace review

* **Step 2 pre-fetch is unconditional for the book tools, gated only for retrieval.**
  The deterministic CSV tools (exposure, scale-out, P/L, positions, performance;
  campaigns when a ticker is named) run on every turn that reaches pre-fetch: they cost
  milliseconds, and gating them by intent created a route-mapping bug class (a route
  forgotten in the fetch table silently became a stub — observed with
  `performance_review`). Intent still gates what needs gating: desk retrieval (it
  requires a query — the question itself, or the standing briefing query; discretionary
  or unconditional retrieval would respectively risk answer-from-priors or pollute the
  audit grounding) and which `MissingData` items surface as upload asks, so a desk-only
  question is never nagged about the statement it doesn't need. A misroute now costs at
  worst a missing nag line, never a missing answer.
* **Consolidating the CSV tools into one function was considered and rejected.** They are
  not LLM-called in the hot path (pre-fetch is one node making plain function calls — no
  round-trips to save), and separateness is load-bearing: per-store cold-start
  granularity, per-tool evidence→audit traceability, per-tool goldens, and small focused
  outputs for the step-3 agent loop.
* **Clarification is ask-and-end, not an in-run `interrupt()`.** The scoper emits its one
  clarifying question as a normal assistant message and the run ends; the reply re-enters
  as the next turn. An interrupt-style clarify node (session-04 pattern) suits a one-shot
  pipeline with no conversation to return to; here the scoper is the first node (nothing
  in-flight to preserve), the chat thread is the natural reply channel, and a user who
  ignores the question is handled gracefully as just the next message. The "one round
  max" rule is a code guarantee (a `pending_clarification` flag read by the next turn's
  scoper), not a prompt hope.
* **Per-turn state resets are part of the intake contract.** A server thread persists
  state across runs; everything derived from one turn (repair-attempt counter, audit
  feedback, evidence table, pending policy proposal) is reset by the scope node at the
  start of every run. Trace-proven failure otherwise: a later turn started at the repair
  cap with a previous turn's audit feedback injected into its prompt.
* **A `policy_change` with co-intents continues into pre-fetch after the gate**, so
  "raise my cap to 12% — am I within policy?" answers the status half against the
  just-decided policy instead of silently dropping it.

## Amendment (2026-07-13) — one answering agent; retrieval as a tool; no truncation

A trace review of a real briefing run ("how is my book doing compared to the
daily/weekly brief?") against the trader's actual co-pilot transcripts exposed three
structural problems, and the fix collapses steps 3 and 4 into one node:

* **Steps 3+4 merge into a single answering agent.** The split (a tool agent forbidden
  to answer, feeding a synthesis node forbidden to fetch) required a private scratch
  transcript, a `collect` adapter converting ToolMessages into evidence items, and a
  double `build_facts` call — pure plumbing tax of the two-persona design. Now the model
  that writes the answer runs the bounded tool loop itself **on the user-visible
  `messages` thread** (the standard LangGraph shape, cf. course sessions 02/03), so tool
  calls and results persist and follow-ups can refer back to them. The three original
  guarantees survive: numbers come pre-formatted from the deterministic digest; the
  final (no-tool-call) response is stored as a **draft** and gated by the audit before
  entering the thread (grounding = digest + every tool result on the thread, so
  tool-fetched figures are audit-backed automatically); the iteration cap and the
  audit-bounce repair both run on the UNBOUND model — they can write, never fetch. The
  scoper filters tool traffic out of its own view of the thread.
* **Desk retrieval is a tool (`search_desk_reviews`), not a pre-fetch.** The seam is now
  clean: pre-fetch = pure functions over the uploaded CSVs; anything touching an index
  or the network is a tool the agent calls. A briefing MUST call the desk search (prompt
  rule, test-enforced) with the standing query as fallback; the cold-start "upload your
  reviews" ask travels as the tool's empty-corpus return value. The tool also appends a
  deterministic **held-names footer** (book symbols word-matched against the retrieved
  text, with base-currency values) — the book↔brief cross-reference the trace review
  showed the split design never produced.
* **No truncation of retrieved chunks, anywhere.** The old formatting layer silently
  passed only 3 of the 5 retrieved sections, each cut at 600 chars — the trace showed it
  dropping exactly the chunk that mentioned a held name (PENG) and cutting off the
  weekly's "core stays positive" bottom line, skewing the desk read bearish. Chunk size
  is bounded at ingest by the chunker; a second cap at prompt time was never a decision,
  only a leak. Web excerpts keep their caps (scraped pages are unbounded; our chunks are
  not).
* **Output sections are composed in code from the multi-label intents** instead of a
  binary template pick. Trace-proven failure: `daily_briefing + performance_review` chose
  the briefing template, which had no performance slot, so fetched realized-P/L evidence
  silently never reached the answer. Headings now assemble from ALL intents (briefing
  headings + "Your book vs the brief" + a Performance heading when asked for).

**Deliberate exclusions** (each considered, each a permanent or scope-level "no"):
chart-pattern recognition in the synthesis path (subjective; vision-LLM chart reading is
confidently wrong — dangerous in a trading tool); counterfactual backtests of the trading
rules (no free historical option data; approximating from the underlying produces
misleading results); statistical "validity" claims about the strategy (six months of one
trader's trades cannot support them — the agent states this caveat instead); desk-signal
profitability attribution (no trade-to-signal log exists); and, per ADR-0007: IV-rank
computation (no free 52-week IV history — fail-loud manual-check reminder instead),
machine-graded rolls (no free Greeks; a computed-from-stale-IV delta is false
confidence), path-dependent moonshot tracking (proxy check on current gain only), and
compliance verdicts from screenshots (redirect to re-uploading the tactical export).

## Amendment (2026-07-14) — traceable chunk-level retrieval

A trace-vs-source review of the briefing run found the retrieval output untraceable —
whole parent sections under a bare `[doc_type date · section]` header, with no chunk id,
source, or score to grep against the ingest preview — and the per-name desk calls
unreadable inside them (the extraction-side fix is ADR-0004's table amendment). Three
changes, superseding this ADR's earlier "returns sections in full" wording:

* **Parent-child recovery removed.** `retrieve()` is now dense + BM25 → RRF top-k over
  the chunks themselves, returned verbatim. With table rows serialized at extraction,
  "whole section for context" no longer justified the indirection; every retrieved text
  is byte-identical to `docs/chunk_preview/` (regenerated by the committed
  `scripts/preview_chunks.py`) and to its Qdrant point. The unused `tickers` metadata
  and the parent-era `start_index` were dropped from the pipeline.
* **`search_desk_reviews` returns the course format** (sessions 01–03): one block per
  chunk — `[Source N: source, doc_type=…, review_date=…, section=…, chunk_id=…,
  pages=…, score=…]` header + the full chunk text — blocks joined by a blank line;
  the held-names footer and the empty-corpus upload ask are unchanged. `doc_type` /
  `review_date` are model-facing decision inputs (daily-vs-weekly attribution,
  freshness); `chunk_id`/`pages`/`source` are the in-thread paper trail back to the
  preview and the PDF; `Source N` numbering enables inline citations.
* **Retrieval flows through a `BaseRetriever` adapter** (`DeskReviewRetriever` in
  `tools.py`), so LangSmith auto-emits a `run_type="retriever"` child run rendered as
  per-chunk Documents — the same zero-instrumentation observability the course stack
  inherits from `as_retriever()`; the hand-rolled hybrid retriever alone was invisible
  to the tracer. The tool formats its string from the very Documents the trace shows,
  so trace and model provably see identical content.

## Amendment (2026-07-14) — policy reads, and the write gate must not eat the question

A studio run of "what's my current policy?" exposed a compounding failure: the scoper
tagged it `status_check + policy_change` (while its own recorded assumption said "not
requesting a change"), the gate correctly refused to parse a change — and then ended the
run with "which rule do you want to change?", dropping the status half entirely. The
values were in state the whole time and never shown. Three fixes:

* **A failed policy parse no longer dead-ends co-intents.** The earlier "continues into
  pre-fetch after the gate" rule now covers the parse-failure path too: when other
  intents remain, `policy_prepare` records its ask as a `policy_note` (reset per turn)
  that the answering agent folds into the answer, instead of a terminal counter-question.
  Only a pure `policy_change` message still ends at the gate's question. This restores
  the invariant "a misroute costs at worst a stray line, never the answer" for the one
  route where it didn't hold.
* **The scoper prompt separates reads from writes** — `policy_change` now requires a
  stated new value; "what are my rules?" is explicitly a `status_check`. The semantic
  assertion joins the Task 5 layer-3 routing rubric (see R2 in task6 refinements).
* **The full rulebook rides in the evidence digest** (`policy_rules`, unconditional —
  policy exists from day one, seeded defaults, no upload). Previously only the limits
  embedded in exposure checks reached the model, so a policy read was only half
  answerable; now every rule value is citable and, being digest text, audit-backed.
