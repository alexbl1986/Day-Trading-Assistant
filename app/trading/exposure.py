"""Exposure checking: measure the book against % -of-NAV policy limits.

Denominator is the cash-inclusive statement NAV (chosen over the intraday book,
which has no cash line). With no NAV the check refuses (MissingData) rather than
divide by a fabricated zero. Buckets: options <= 10% NAV, plus the hedge-ratio
band (put value / call value, his formula) when a band is passed; the "20%
offensive exposure" bucket waits until 'offensive' is defined.

Band limits arrive as scalars, not a policy record — `app.trading` stays
independent of the graph layer where `PolicyRecord` lives.
"""

from __future__ import annotations

from app.trading.domain import (
    ExposureCheck,
    ExposureReport,
    HedgeCheck,
    MissingData,
    Position,
)

_OPTIONS_LIMIT = 0.10  # options <= 10% of NAV


def check_exposure(
    positions: list[Position],
    nav: float | None,
    options_limit: float = _OPTIONS_LIMIT,
    hedge_low: float | None = None,
    hedge_high: float | None = None,
    position_cap: float | None = None,
) -> ExposureReport | MissingData:
    if nav is None:
        return MissingData(
            store="statement NAV",
            remedy="Upload a recent activity statement (needed for the NAV denominator).",
        )

    options_value = sum(
        p.position_value * p.fx_rate_to_base
        for p in positions
        if p.asset_class == "OPT"
    )
    options = ExposureCheck(
        label="options",
        value_base=options_value,
        pct_of_nav=options_value / nav,
        limit=options_limit,
    )
    checks = (options, *_holding_breaches(positions, nav, position_cap))
    return ExposureReport(
        nav=nav, checks=checks, hedge=_hedge_check(positions, hedge_low, hedge_high)
    )


def _holding_breaches(
    positions: list[Position], nav: float, cap: float | None
) -> list[ExposureCheck]:
    """Names whose whole holding (stock + options, per symbol) exceeds the cap.

    Only breaches are emitted — the point is flagging concentration, not
    enumerating the book. The book snapshot already reduces option symbols to
    their root ticker, so grouping by symbol groups by underlying.
    """
    if cap is None:
        return []
    by_symbol: dict[str, float] = {}
    for p in positions:
        by_symbol[p.symbol] = by_symbol.get(p.symbol, 0.0) + p.position_value * p.fx_rate_to_base
    return [
        ExposureCheck(
            label=f"{symbol} holding",
            value_base=value,
            pct_of_nav=abs(value) / nav,
            limit=cap,
        )
        for symbol, value in by_symbol.items()
        if abs(value) / nav > cap
    ]


def _hedge_check(
    positions: list[Position], low: float | None, high: float | None
) -> HedgeCheck | None:
    if low is None or high is None:
        return None
    # Magnitudes: a short call is still call-side exposure, not negative hedge.
    puts = sum(
        abs(p.position_value * p.fx_rate_to_base)
        for p in positions
        if p.asset_class == "OPT" and p.right == "P"
    )
    calls = sum(
        abs(p.position_value * p.fx_rate_to_base)
        for p in positions
        if p.asset_class == "OPT" and p.right == "C"
    )
    if calls == 0:
        return None  # no denominator; a puts-only book has no meaningful ratio
    return HedgeCheck(
        put_value_base=puts, call_value_base=calls, ratio=puts / calls, low=low, high=high
    )
