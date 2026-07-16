"""Unrealized P/L: the current-holdings scoreboard.

For each open position, marks the book's current price against the average entry
price recovered from the ledger (the tactical book's CostBasisPrice is always 0),
and reports dollar + % unrealized P/L in base USD. Same open-position x ledger
join as scan_scaleout, and the same cold-start refusal: no ledger -> no cost
basis -> MissingData rather than a fabricated zero.

Marks come from the book snapshot (fresh as of the last upload); a live-quote
source can be layered on later without reshaping this seam.
"""

from __future__ import annotations

from app.trading.domain import (
    MissingData,
    Position,
    PositionPnL,
    Trade,
    UnrealizedPnLReport,
)
from app.trading.ledger import group_campaigns
from app.trading.symbols import (
    ContractKey,
    position_contract_key,
    statement_symbol_contract_key,
)


def open_position_pnl(
    positions: list[Position],
    trades: list[Trade],
    fallback_entry: dict[ContractKey, float] | None = None,
) -> UnrealizedPnLReport | MissingData:
    if not trades:
        return MissingData(store="ledger", remedy="Upload a recent activity statement.")

    entry_by_key = {
        statement_symbol_contract_key(c.symbol): c.avg_entry_price
        for c in group_campaigns(trades)
        if c.is_open
    }
    fallback_entry = fallback_entry or {}

    lines: list[PositionPnL] = []
    for pos in positions:
        key = position_contract_key(pos)
        # Prefer the ledger's reconstructed entry; fall back to the statement's
        # Open-Positions Cost Price for holdings with no opening fill this period
        # (pre-window, assigned, or rebought after the statement's cutoff).
        entry = entry_by_key.get(key)
        source = "ledger"
        if entry is None:
            entry = fallback_entry.get(key)
            source = "statement"
        if entry is None:
            continue  # neither source has a cost basis -> genuinely unpriceable
        # position_value already carries size, multiplier and sign; scaling it by
        # (mark - entry)/mark yields the native-currency P/L without re-deriving
        # the contract multiplier. fx_rate_to_base normalizes to USD.
        unrealized_native = pos.position_value * (pos.mark_price - entry) / pos.mark_price
        # The ratio is quoted from the position's side: a short leg gains as the
        # mark falls (the signed position_value already handles the dollars).
        gain = pos.mark_price / entry - 1
        lines.append(
            PositionPnL(
                symbol=pos.symbol,
                avg_entry_price=entry,
                mark_price=pos.mark_price,
                gain=-gain if pos.quantity < 0 else gain,
                unrealized_pl=unrealized_native * pos.fx_rate_to_base,
                cost_basis_source=source,
            )
        )

    total = sum(line.unrealized_pl for line in lines)
    return UnrealizedPnLReport(total_unrealized_pl=total, lines=tuple(lines))
