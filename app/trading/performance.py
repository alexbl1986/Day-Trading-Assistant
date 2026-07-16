"""Tier-1 performance attribution: the ledger -> a PerformanceReport.

Everything sums IBKR's own signed columns (Realized P/L, commission) — nothing
is recomputed from fills, so the numbers match the statement the trader can
open himself. Campaign win rate reuses `group_campaigns` (a campaign closes when
net quantity returns to zero; open runners don't count either way).
"""

from __future__ import annotations

from app.trading.domain import PerformanceReport, TickerPerformance, Trade
from app.trading.ledger import group_campaigns

_TOP_N = 5


def performance_summary(trades: list[Trade]) -> PerformanceReport:
    by_ticker: dict[str, float] = {}
    by_month: dict[str, float] = {}
    for t in trades:
        by_ticker[t.root_ticker] = by_ticker.get(t.root_ticker, 0.0) + t.realized_pl
        month = t.timestamp.strftime("%Y-%m")
        by_month[month] = by_month.get(month, 0.0) + t.realized_pl

    ranked = sorted(by_ticker.items(), key=lambda kv: kv[1], reverse=True)
    winners = tuple(
        TickerPerformance(ticker, pl) for ticker, pl in ranked[:_TOP_N] if pl > 0
    )
    losers = tuple(
        TickerPerformance(ticker, pl) for ticker, pl in reversed(ranked[-_TOP_N:]) if pl < 0
    )

    closed = [c for c in group_campaigns(trades) if not c.is_open]
    return PerformanceReport(
        total_realized_pl=sum(t.realized_pl for t in trades),
        by_month=tuple(sorted(by_month.items())),
        top_winners=winners,
        top_losers=losers,
        closed_campaigns=len(closed),
        winning_campaigns=sum(1 for c in closed if c.realized_pl > 0),
        commission_total=sum(t.commission for t in trades),
    )
