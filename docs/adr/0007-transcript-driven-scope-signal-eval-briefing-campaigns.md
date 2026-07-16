# Transcript-driven scope: pre-trade signal eval, daily briefing, campaign-scoped history

> **Superseded in part (2026-07-11):** the `scan_deadlegs` tool and `dead_leg_floor`
> ($0.50) named below were dropped after the trader confirmed the Thursday-flush rule
> wasn't worth building as a machine check. The rest of this ADR stands.

Reviewing the trader's real transcripts with his previous chat-agent setup revealed that
his highest-frequency use case — evaluating an incoming trade signal — was missing from
the route list, and that his rulebook is larger and more precise than the policy record
we had scoped. The transcripts also validated the architecture twice: the agent read a
14,700 SEK position as USD and ordered an immediate trim ("confidently wrong compliance
verdict from LLM arithmetic" — the exact failure ADR-0002 exists to prevent), and its
per-question search of the 9,489-line annual statement was slow enough that the trader
asked it to stop (the parse-once ledger, justified in his own words).

## Decisions

**1. New route: `trade_signal_eval` (in cert scope, built last behind a clean seam —
first cut if the schedule slips).** Input: shorthand signal pasted into chat ("AAOI 150
NEXT WEEK 3.1"). v1 scope: intake parses ticker/strike/type/premium (defaults per the
trader's own protocol: call if unspecified, nearest weekly/monthly **verified against
the live yfinance option chain**); sizing from the policy record (options 1% / stocks 3%
of NAV → max contracts at the signal premium); exit plan from the DTE-tier matrix;
desk bias + tier cross-ref via RAG (the desk's Tier-2 "high-beta optionality" label caps
sizing at standard — a deterministic rule keyed on a retrieved label); existing-inventory
conflict check against the snapshot; **fail-loud on IV rank** ("not available from free
data — check manually; if >70% your rules mandate a spread" — today's IV from the chain
is shown, the 52-week rank cannot be computed without paid IV history). Output: a typed
`TradePlan`. Out even within this route: IV-rank computation, automated roll grading,
Discord ingestion (paste-only for cert).

**2. New route: `daily_briefing` (in — a composition, not a feature).** Trigger phrases
("morning briefing", "start my day"); pre-fetch = union of existing calls: exposure
check + scale-out scan + dead-leg scan (Thursday-aware) + hedge ratio + desk-summary
retrieval over the latest daily & weekly + index technical snapshot. One synthesis type
(`DailyBriefing`): book status → rule flags → desk's read → what to watch. Replaces the
auto-boot sequence the trader hand-built in his previous setup.

**3. `get_trades` defaults to the current active campaign.** Campaign v1 = all ledger
fills in the same contract (symbol+strike+expiry) while the position has been
continuously open; closes when quantity returns to zero; stocks analogously (fills since
last flat). Default answer: entry date/premium, partial scales, net cost basis,
realized-so-far, house-money status. Full history (closed campaigns) only on explicit
request — an intake flag, not a separate tool. Note: his latency complaint doesn't apply
to a Postgres ledger (both queries are milliseconds); we adopt his preferred *answer
shape*, not a performance trade-off. Roll chains do NOT merge campaigns in v1.

**4. Policy record scope: rules a cert route reads, plus three cheap checks.** In:
sizing caps (options 1% / stocks 3% new / **6% existing-holding cap**), 20% max active
offensive exposure, options ≤10% NAV, **hedge ratio = put value ÷ call value, target
10–15%** (the precise formula from his rules — not a %-of-NAV figure; whether the
cross-hedge ~15% uses the same ratio logic must be confirmed with the trader before
Day 3), cross-hedge, scale-out (+100%/+200%), the DTE exit matrix (0–2 / 3–7 / 8–30 /
31+: size tier, SL, TP, scale/roll clauses), IV-shield threshold (70% — enforced as a
manual-check reminder), `dead_leg_floor` ($0.50, commission-aware) powering a
`scan_deadlegs` tool, and moonshot thresholds (+150% trigger / +50% stop) as a
**labeled proxy** flag in the scale-out scan (true rule is path-dependent — "ever
touched +150%" — which needs option price history that free data doesn't have and the
app is upload-blind between snapshots). House-money status is a derived field of
campaign history (descriptive, not a rule check). **Excluded from machine checking:**
roll rules (0.50-delta target and 2×-cash: no free Greeks; computing delta = hand-rolled
Black-Scholes over yfinance IV that is often stale/zero on thin strikes — a
precise-looking wrong delta feeding a compliance verdict is the false-confidence failure
this app exists to prevent), roll-chained campaigns, post-trade **screenshot** audits
(compliance verdicts never hang off pixels; the synthesis prompt redirects to re-upload
the tactical export — the trader's existing habit — after which `status_check` on the
fresh snapshot IS the post-trade audit). An untested rule field that looks
machine-enforced is a liability, so nothing enters the record that no code reads; the
record carries a `version` field for Demo Day extension.

Demo Day items this creates: Discord signal bridge (push → auto-eval), IBKR API replacing
CSV uploads, paid options data (unlocks honest IV rank, Greeks/roll grading, historical
option prices), roll-aware campaign chaining. The full plain-language gap list for the
trader lives in the gitignored `docs/PROTOTYPE_GAPS.md`.

## Amendment (2026-07-15) — fills-based scale-out ladder; moonshot fields removed

The "moonshot thresholds as a labeled proxy" decision above is superseded. Two insights
from a design review with the trader: (1) the rulebook contains two DIFFERENT rule
types that the old single-label, current-gain-only classification wrongly forced onto
one axis — sale rungs (sell contract 1 at +100%, contract 2 at +200%), which are keyed
on how many sales were already made, and a protection rule (+150% arms a hard stop at
+50% on the remainder), which is an alarm, not a sale; (2) the campaign's recorded
sell fills ARE the ladder state, so no threshold reordering is needed and the scan can
be honest about what was already done.

`scan_scaleout` is now stateful: `Campaign.scale_outs` counts closing-direction fills
(events, not contracts) in the open campaign; 0 sales + gain ≥ `scale_out_first` →
first tranche due; 1 sale + gain ≥ `scale_out_second` → second tranche due; 2+ sales →
**moonshot runner**, reported at ANY current gain as inventory (a melted runner stays
on the report), never as an action. This ended the scan's re-nagging (acting on a flag
is itself the state change that clears it) and, on the real book, corrected a live
verdict: IGV at +223% was flagged "moonshot" by the old proxy, but the ledger shows one
scale-out already taken — the true next action is the second sale, which the new scan
reports.

**`moonshot_trigger` and `moonshot_stop` are removed from the policy record** (trader
decision, applying this ADR's own charter: an unenforced field that looks
machine-managed is a liability). The runner's endgame is explicitly manual: the stop
rule is path-dependent ("ever touched +150%"), snapshots cannot see the path, and the
digest's runner line says so ("exit is manual — path not tracked"). Known bound,
accepted: the sales count is only as old as the statement window, so a pre-window
scale-out reads as rung 0 — the same visibility bound the cost-basis fallback already
has.

## Amendment (2026-07-15) — sizing-only trade_signal_eval; policy record trimmed to enforced rules

The trader's #1 route ships in its honest first cut. `size_trade_signal` (a tool on the
answering agent's roster, mandatory on the `trade_signal_eval` intent via a prompt rule
carrying the verbatim `signal_text`): the LLM's only job is parsing the pasted shorthand
into typed args; the money math is code (`app.trading.sizing.size_new_position` — budget
= NAV × the policy's per-entry percentage, whole units floored, option multiplier
applied), so every figure is audit-backed. The output adds two book cross-checks from
the same snapshot the exposure check reads (already-held inventory; options-cap headroom
after the entry) and ends with a mandatory NOT-CHECKED list — chain existence/liquidity,
IV rank, DTE exit plan, desk view — the fail-loud boundary of the cut. Missing NAV
returns the cold-start upload ask instead of a zero-sized plan.

With the sizing fields now consumed, the policy record is trimmed to EXACTLY the rules
code enforces (8 fields): `iv_shield` and `max_offensive_exposure` are removed
(trader decision, applying this ADR's charter). Backlog disposition (task6 refinements
R3): the IV shield returns with the full trade_signal_eval as the fail-loud manual
reminder this ADR already prescribed; `max_offensive_exposure` needs the trader to
define "offensive exposure" before any code could read it — define it or drop it for
good. The DTE exit matrix and chain verification remain the rest of the full route.
