"""The injected seam the whole graph runs on (house style, cf. `quotes.fetch`).

One graph, three wirings of the SAME `AgentContext`:
  * tests  -> a stub chat model, no network (deterministic, the graded layer);
  * `langgraph dev` / Studio -> the real gateway `ChatOpenAI` + seeded local data;
  * deploy -> gateway model + Postgres/Qdrant-backed loaders.

Real wirings must build the chat model with `tags=["nostream"]`: no LLM call in
this graph is user-facing (delivery is audit-gated; visible messages are
constructed in code), so any streamed model tokens leak scoper JSON / unaudited
drafts into the client's `messages` stream as extra chat bubbles. `dev.py` does
this; the deploy wiring must too.

Per-user DATA (snapshot, ledger, policy) is loaded inside nodes keyed by
`user_id`, not baked into this frozen context — the server compiles the graph
once and serves every user. Those data-access callables land with the pre-fetch
node (Slice 2); the scoper needs only the chat model, so that is all Slice 1
requires here. Desk retrieval is not a field: it enters as a tool in
`agent_tools` (built by `dev.py` / the deploy wiring around the retriever).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.graphs.trading_assistant.policy_model import PolicyRecord
from app.trading.domain import MissingData, Position, Trade
from app.trading.symbols import ContractKey

# Per-user data-access seams. Each returns MissingData when the backing store was
# never uploaded, so an empty book can never be mistaken for a flat one (the
# cold-start contract). `user_id` is passed at call time — the server compiles the
# graph once and serves every user; data is never baked into this frozen context.
PositionsLoader = Callable[[str], "list[Position] | MissingData"]
TradesLoader = Callable[[str], "list[Trade] | MissingData"]
NavLoader = Callable[[str], "float | None"]
# The statement's Period end date — the as-of stamp on every figure it backs
# (NAV, ledger, realized P/L). None when unknown; no MissingData, since the
# date is a freshness label, not a store the answer depends on.
AsOfLoader = Callable[[str], "date | None"]
# The statement's Open-Positions Cost Price per contract key — the entry-price
# fallback for holdings with no opening fill in the YTD ledger (bought
# pre-window). Without it those lines silently vanish from P/L and scale-out.
# Empty when unavailable; it rides the same statement the ledger comes from,
# so a missing statement already surfaces as the ledger's MissingData.
EntryFallbackLoader = Callable[[str], "dict[ContractKey, float]"]
# Policy is read-mostly and edited only through the interrupt-gated write path.
PolicyLoader = Callable[[str], PolicyRecord]
PolicySaver = Callable[[str, PolicyRecord], None]


@dataclass(frozen=True)
class AgentContext:
    """Capabilities the graph is built against; data arrives per-request."""

    chat_model: BaseChatModel
    # The answering agent's tool roster (desk retrieval / quotes / web).
    # None -> the agent answers from the pre-fetch digest alone (offline tests,
    # degraded dev).
    agent_tools: Sequence[BaseTool] | None = None
    load_positions: PositionsLoader | None = None
    load_trades: TradesLoader | None = None
    load_nav: NavLoader | None = None
    load_as_of: AsOfLoader | None = None
    load_entry_fallback: EntryFallbackLoader | None = None
    load_policy: PolicyLoader | None = None
    save_policy: PolicySaver | None = None
    default_user_id: str = "alex-demo"
