"""Campaign grouping over the parsed trades ledger.

A campaign is a continuously-open run of fills in one contract: it opens at
the first fill, absorbs scale-ins/scale-outs, and closes when net quantity
returns to zero. Contract identity is the raw symbol, so different expiries
(rolls) are separate campaigns.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import groupby

from app.trading.domain import Campaign, Split, Trade


def apply_splits(trades: list[Trade], splits: list[Split]) -> list[Trade]:
    """Restate pre-split fills in post-split terms (quantity x ratio, price / ratio).

    IBKR reports each fill at its historical share count, but the split itself
    lives only in the Corporate Actions section — unapplied, a campaign that
    straddles a split miscounts its net quantity and prices today's shares off
    a pre-split entry. Cash columns (proceeds, commission, basis, realized P/L)
    stay untouched: a split moves no money. Stocks only — option contracts
    carry their own OCC adjustments under their own symbols.
    """
    adjusted = list(trades)
    for split in sorted(splits, key=lambda s: s.effective):
        ratio = split.numerator / split.denominator
        adjusted = [
            replace(t, quantity=t.quantity * ratio, price=t.price / ratio)
            if (
                t.asset_category == "Stocks"
                and t.root_ticker == split.symbol
                and t.timestamp < split.effective
            )
            else t
            for t in adjusted
        ]
    return adjusted


def group_campaigns(trades: list[Trade]) -> list[Campaign]:
    campaigns: list[Campaign] = []
    by_symbol = sorted(trades, key=lambda t: (t.symbol, t.timestamp))
    for symbol, group in groupby(by_symbol, key=lambda t: t.symbol):
        run: list[Trade] = []
        net = 0.0
        for fill in group:
            run.append(fill)
            net += fill.quantity
            if net == 0:  # position flat → this campaign is closed
                campaigns.append(_campaign(symbol, run, net))
                run = []
        if run:  # leftover open position
            campaigns.append(_campaign(symbol, run, net))
    return campaigns


def get_trades(
    trades: list[Trade], ticker: str, full_history: bool = False
) -> list[Campaign]:
    """Campaigns for an underlying. Default = currently-open only."""
    campaigns = [c for c in group_campaigns(trades) if c.root_ticker == ticker]
    if full_history:
        return campaigns
    return [c for c in campaigns if c.is_open]


def _campaign(symbol: str, fills: list[Trade], net: float) -> Campaign:
    return Campaign(
        symbol=symbol,
        root_ticker=fills[0].root_ticker,
        fills=tuple(fills),
        net_quantity=net,
    )
