# Trading Assistant

The domain language for an Agentic RAG assistant that helps a signal-following retail
options day-trader keep his book aligned with his own rules and his desk's market thesis.
This file is a glossary only — no implementation detail.

## Language

**Book**:
The trader's live set of positions taken as a whole — the thing he keeps "in balance".
_Avoid_: account (that's the IBKR entity), portfolio (ambiguous with the portfolio CSV file)

**Position**:
One held instrument — a stock or an option — with its quantity and cost.

**Desk review**:
The dated PDF in which the trading desk publishes its market read.
_Avoid_: report, note, newsletter

**Thesis**:
The desk's current directional stance for a name, theme, or market regime.
_Avoid_: view, opinion, call

**Signal**:
A shorthand trade idea arriving on the trader's signal channel (Discord), e.g.
"AAOI 150 NEXT WEEK 3.1" — ticker, strike, expiry hint, premium. He follows these;
the assistant evaluates them against his rules and the desk's thesis.
_Avoid_: alert, tip

**Trade plan**:
The assistant's typed answer to a signal: max size under the sizing rules, the DTE-tier
exit levels, desk bias/tier on the name, and any inventory conflicts.

**Campaign**:
The full story of one trade in one contract — entry, partial scale-outs, what's still
open — from first fill until quantity returns to zero. History questions default to the
current campaign. Rolls start a new campaign (v1).

**House money**:
A campaign state: the contracts already sold brought in more cash than the whole trade
cost, so the remaining position risks none of the trader's own capital.

**Roll**:
Selling the current call to fund a higher-strike one, staying in the trade while
extracting cash. The assistant discusses roll rules but never machine-grades a roll
(no trustworthy free Greeks).

**Exposure policy**:
The trader's own numeric rules that his book must stay within — sizing per entry
(options 1%, stocks 3% new / 6% held), options ≤10% of NAV, hedge ratio (put value ÷
call value, 10–15%), ~15% cross-hedge, the DTE exit matrix, scale-out levels.
_Avoid_: strategy (broader than the caps), config (an implementation word)

**Scale-out**:
The fixed profit-taking rule: at +100% sell one contract, at +200% sell another, leave
the rest as a "moonshot".

**Cross-hedge**:
The ~15% position taken against a prevailing-sentiment segment (e.g. "AI").

**Segment**:
The set of tickers the trader treats as one theme (e.g. his "AI" names). Membership is
his call, not a market definition.

**Synthesis**:
The judgment the assistant now performs for him — reconciling positions against the
exposure policy and the desk's thesis and returning exactly what to change.
_Avoid_: analysis (undersells it; this is combining, not just reading)

**Portfolio snapshot**:
One IBKR export the trader uploads; the assistant holds it for the session.

**Desk-review corpus**:
The body of desk-review PDFs indexed for retrieval (RAG). The portfolio CSVs are NOT
part of this corpus — they are queried deterministically.
