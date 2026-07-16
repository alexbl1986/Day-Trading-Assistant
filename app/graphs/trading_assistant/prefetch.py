"""Deterministic pre-fetch (ADR-0006 step 2, revised: CSV computations ONLY).

The book tools (exposure, scale-out, P/L, positions; campaigns when a ticker is
named) run on EVERY turn that reaches this node — they are milliseconds over the
snapshot/ledger, and gating them by intent created a route-mapping bug class
(a route forgotten in the fetch table silently became a stub). The clean seam:
this node does pure functions over the uploaded files; anything that touches an
index or the network (desk retrieval, quotes, web) is a tool the answering
agent calls itself.

Every call's outcome lands in the evidence table, and a store that was never
uploaded surfaces as `MissingData` (cold-start contract) rather than a silent
empty result — so the answer can cover what it can and name the missing upload,
and the audit node can block any number with no backing. Intent gates only
which `MissingData` items surface as upload asks, so a desk-only question is
never nagged about the statement it doesn't need. (The desk-reviews cold-start
ask travels with the `search_desk_reviews` tool, not this node.)
"""

from __future__ import annotations

from typing import Callable, Sequence

from app.graphs.trading_assistant.deps import (
    AgentContext,
    NavLoader,
    PositionsLoader,
    TradesLoader,
)
from app.graphs.trading_assistant.policy_model import DEFAULT_POLICY
from app.graphs.trading_assistant.state import AgentState, EvidenceItem, Scope
from app.trading.domain import MissingData, Position, Trade
from app.trading.exposure import check_exposure
from app.trading.ledger import get_trades
from app.trading.performance import performance_summary
from app.trading.pnl import open_position_pnl
from app.trading.scaleout import scan_scaleout

# Which stores each route actually NEEDS — used only to decide which missing
# stores surface as upload asks, never to decide what runs.
_BOOK_STORES = frozenset({"positions snapshot", "ledger", "statement NAV"})
_STORES_BY_INTENT: dict[str, frozenset[str]] = {
    "status_check": _BOOK_STORES,
    "rebalance_advice": _BOOK_STORES,
    "daily_briefing": _BOOK_STORES,
    "performance_review": frozenset({"ledger"}),
    "trade_history": frozenset({"ledger"}),
    "desk_question": frozenset(),
    "trade_signal_eval": _BOOK_STORES,  # sizing needs NAV; inventory needs the book
    "market_regime": frozenset(),
    "policy_change": frozenset(),
    "off_topic": frozenset(),
}


def make_prefetch_node(context: AgentContext) -> Callable[[AgentState], dict]:
    """Bind the pre-fetch to its data-access seams; returns the node callable."""

    def prefetch_node(state: AgentState) -> dict:
        scope: Scope = state["scope"]
        user_id = state.get("user_id") or context.default_user_id
        intents = set(scope.intents)

        positions = _load(
            context.load_positions,
            user_id,
            MissingData("positions snapshot", "Upload your tactical book export."),
        )
        trades = _load(
            context.load_trades,
            user_id,
            MissingData("ledger", "Upload a recent activity statement."),
        )
        nav = context.load_nav(user_id) if context.load_nav else None
        policy = context.load_policy(user_id) if context.load_policy else DEFAULT_POLICY
        entry_fallback = (
            context.load_entry_fallback(user_id) if context.load_entry_fallback else {}
        )

        # The book always runs; a named ticker always pulls its campaign; only
        # desk retrieval is gated (it needs a query to exist).
        evidence: list[EvidenceItem] = []
        as_of = context.load_as_of(user_id) if context.load_as_of else None
        if as_of is not None:
            evidence.append(EvidenceItem("statement_as_of", ok=True, result=as_of))
        evidence += _portfolio_prefetch(positions, trades, nav, policy, entry_fallback)
        # The rulebook itself is evidence: policy exists from day one (seeded
        # defaults, no upload), and putting the full record in the digest makes
        # a policy read answerable and every cited rule value audit-backed.
        evidence.append(EvidenceItem("policy_rules", ok=True, result=policy))
        # Realized attribution needs only the ledger — a missing snapshot must
        # not block it, so it sits outside the snapshot-gated portfolio batch.
        if isinstance(trades, MissingData):
            evidence.append(EvidenceItem("performance_summary", ok=False, missing=trades))
        else:
            evidence.append(
                EvidenceItem("performance_summary", ok=True, result=performance_summary(trades))
            )
        if scope.tickers:
            evidence += _history_prefetch(trades, scope.tickers)
        elif "trade_history" in intents:
            evidence.append(
                EvidenceItem(
                    "get_trades", ok=False, note="No ticker named; ask which position."
                )
            )

        return {"evidence": evidence, "missing": _relevant_missing(evidence, intents)}

    return prefetch_node


# -- route pre-fetches ----------------------------------------------------


def _portfolio_prefetch(
    positions: list[Position] | MissingData,
    trades: list[Trade] | MissingData,
    nav: float | None,
    policy,
    entry_fallback: dict,
) -> list[EvidenceItem]:
    if isinstance(positions, MissingData):
        # Every portfolio tool needs the snapshot; block them all on it.
        return [EvidenceItem("list_positions", ok=False, missing=positions)] + [
            EvidenceItem(tool, ok=False, missing=positions)
            for tool in ("check_exposure", "scan_scaleout", "open_position_pnl")
        ]

    items = [
        EvidenceItem("list_positions", ok=True, result=tuple(positions)),
        _wrap(
            "check_exposure",
            check_exposure(
                positions,
                nav,
                policy.options_limit,
                hedge_low=policy.hedge_ratio_low,
                hedge_high=policy.hedge_ratio_high,
                position_cap=policy.existing_holding_cap,
            ),
        ),
    ]
    if isinstance(trades, MissingData):
        items += [
            EvidenceItem("scan_scaleout", ok=False, missing=trades),
            EvidenceItem("open_position_pnl", ok=False, missing=trades),
        ]
    else:
        items += [
            _wrap(
                "scan_scaleout",
                scan_scaleout(
                    positions,
                    trades,
                    entry_fallback,
                    first_gain=policy.scale_out_first,
                    second_gain=policy.scale_out_second,
                ),
            ),
            _wrap("open_position_pnl", open_position_pnl(positions, trades, entry_fallback)),
        ]
    return items


def _history_prefetch(
    trades: list[Trade] | MissingData, tickers: Sequence[str]
) -> list[EvidenceItem]:
    if isinstance(trades, MissingData):
        return [EvidenceItem("get_trades", ok=False, missing=trades)]
    # Default = current active campaign per ticker (full history is an intake flag).
    return [
        EvidenceItem(f"get_trades:{ticker}", ok=True, result=tuple(get_trades(trades, ticker)))
        for ticker in tickers
    ]


# -- helpers --------------------------------------------------------------


def _load(loader, user_id: str, if_unconfigured: MissingData):
    return loader(user_id) if loader is not None else if_unconfigured


def _wrap(tool: str, result) -> EvidenceItem:
    if isinstance(result, MissingData):
        return EvidenceItem(tool=tool, ok=False, missing=result)
    return EvidenceItem(tool=tool, ok=True, result=result)


def _relevant_missing(
    evidence: Sequence[EvidenceItem], intents: set[str]
) -> list[MissingData]:
    """Deduped missing stores, narrowed to the ones this question needs.

    The evidence table keeps every blocked tool (cold-start truth); this only
    scopes which "upload X" asks reach the user.
    """
    needed: set[str] = set()
    for intent in intents:
        needed |= _STORES_BY_INTENT.get(intent, frozenset())
    seen: dict[str, MissingData] = {}
    for item in evidence:
        if item.missing is not None and item.missing.store in needed:
            seen.setdefault(item.missing.store, item.missing)
    return list(seen.values())
