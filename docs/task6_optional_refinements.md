# Task 6 — Optional Refinements to Consider

> A running list of refinements that surfaced during the build, each considered
> but **not** settled. These are candidate improvements — some may land before
> submission, some become Demo-Day items, some are deliberately declined. Each
> entry states the current behavior, the proposed change, the trade-off, and a
> status so the decision is explicit rather than implicit.

---

## R1. Deterministic ticker extraction in the scoper

**Status:** open — not yet decided. *(Premise update 2026-07-14:
`app/rag/chunk.py::_extract_tickers` no longer exists — the RAG-layer tickers metadata
was removed as unused. The refinement stands on its own merits, but the regex+whitelist
extractor would need to be (trivially) recreated rather than reused.)*

**Current behavior.** The intake/scoper node (`app/graphs/trading_assistant/scope.py`)
asks the LLM to produce the whole `Scope` in one structured-output call, including
`tickers`. So the *symbols* are extracted by the model alongside the intent.

**Observation.** The symbols the scoper returns drive real, consequential downstream
work: `trade_history` calls `get_trades(trades, ticker)` per ticker, and
`trade_signal_eval` sizes a position off the parsed symbol. A wrong or missed ticker
there is not a cosmetic error — it queries the wrong campaign or sizes the wrong name.
Meanwhile we **already have** deterministic ticker extraction in the RAG layer:
`app/rag/chunk.py::_extract_tickers` (regex `\b[A-Z]{1,5}\b` intersected with a
ticker whitelist). Symbol recognition is exactly the kind of narrow, verifiable task
that code does more reliably than a language model.

**Proposed refinement.** Split the responsibility: keep the **LLM for intent**
(fuzzy language → one of the 10 routes, the hypothetical flag, clarification), but
extract **symbols in code** — regex + a whitelist of the user's actual holdings /
known desk-universe symbols — either replacing or cross-checking the model's
`tickers`. This mirrors the graph's guiding principle (ADR-0006): the LLM interprets
language; deterministic code owns anything safety- or correctness-critical.

**Trade-off / nuance (why it isn't a clean win).**
- A pure regex+whitelist can't resolve a *company name* to a ticker ("Taiwan Semi" →
  `TSMC`) or an *implied* holding, which the LLM handles for free.
- The whitelist has to come from somewhere current — the parsed snapshot/ledger
  symbols plus the desk-review ticker set — so it's per-user state, not a static list.
- A hybrid (code extracts + validates against the whitelist; LLM fills gaps for
  names not matched) captures most of the reliability gain without losing name
  resolution, at the cost of a second reconciliation step.

**Recommendation.** Defer past the base agent build; revisit alongside the Task 5
eval harness. The routing rubric will quantify how often the LLM actually mis-extracts
a ticker on real phrasings — if that rate is low, the added machinery isn't worth it;
if it's material (especially on `trade_signal_eval`), do the hybrid version. Decide
with data, not a priori.

---

## R2. Automated semantic-accuracy check for scoper routing

**Status:** planned (folds into Task 5), noted here for completeness.

**Current behavior.** The scoper's output *shape* is guaranteed by the Pydantic schema
(`with_structured_output(Scope)` — intents can only be the 10 `Literal` values), and
the one safety-critical semantic rule is enforced deterministically (`_normalize_scope`
strips `policy_change` from a hypothetical). But whether the LLM *labels correctly*
(e.g. "sell everything and buy dogecoin" → should be `off_topic`, was observed as
`rebalance_advice`) has **no assertion** — the unit tests use a stub model and check
only the graph wiring.

**Proposed refinement.** The Task 5 eval layer-3 routing rubric (paraphrase variants,
daily-vs-weekly collision, cold-start, off-topic) becomes the semantic check. This is
already in the plan; the note here is to make sure the routing cases explicitly cover
misroutes like the dogecoin example and any ticker mis-extraction from R1.

*(Case added 2026-07-14, from a studio run:)* "what's my current policy?" was tagged
`status_check + policy_change` even though the model's own assumption said "not
requesting a change" — a policy READ must never carry `policy_change`. The prompt now
has an explicit read-vs-write rule (pinned by a unit test), and the graph survives the
misroute regardless (failed parse with co-intents continues to prefetch), but the
rubric should assert the label directly.

**Recommendation.** Build with Task 5; no separate work.

---

## R3. Backlogged policy rules: IV shield and max offensive exposure

**Status:** deliberately out of scope for the prototype (trader decision, 2026-07-15).

**Context.** The policy record's charter (ADR-0007) is "nothing enters the record that
no code reads." After the fills-based scale-out ladder (which removed the moonshot
fields) and the sizing-only `trade_signal_eval` (which gave the two sizing percentages
their consumer), two fields remained with no reader and were removed from
`PolicyRecord`/`FIELDS`:

- **`iv_shield` (was 70%).** Returns with the FULL `trade_signal_eval` route as the
  fail-loud manual reminder ADR-0007 prescribes ("IV rank ≥ 70% mandates a spread —
  check manually"); it is never a computation (no free IV-rank history; a verdict from
  stale IV is the false-confidence failure the app exists to prevent). Until then, the
  sizing tool's NOT-CHECKED list names IV rank explicitly.
- **`max_offensive_exposure` (was 20%).** No consumer was ever designed and "active
  offensive exposure" was never defined (all long calls? calls + growth-stock
  positions? everything that isn't a hedge?). Precondition for revival: the trader
  defines the position set; then it is one more `ExposureCheck` inside
  `check_exposure`. If no definition materializes, remove permanently.

**Consequence accepted:** "what's my current policy?" recites only the 8 enforced
rules; these two live in the trader's own rulebook documents, not in the app.
