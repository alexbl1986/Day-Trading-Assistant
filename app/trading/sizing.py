"""Position sizing for a pasted trade signal (sizing-only trade_signal_eval).

Pure math over scalars — the policy percentages arrive as parameters, keeping
this layer decoupled from the graph's PolicyRecord (same seam style as
check_exposure and scan_scaleout). The option multiplier (x100) is applied
here so every figure the tool renders is a final dollar amount.
"""

from __future__ import annotations

from app.trading.domain import TradeSizing

OPTION_MULTIPLIER = 100


def size_new_position(
    nav: float, *, kind: str, unit_price: float, pct: float
) -> TradeSizing:
    """Budget = NAV x pct; quantity = whole units the budget buys (floored).

    `unit_price` is the option premium as quoted (e.g. 3.10) or the stock
    share price; the contract multiplier is applied here, not by the caller.
    A price the budget can't cover returns quantity 0 (the caller renders it
    as over-budget), but nonsense inputs fail loud.
    """
    if kind not in ("option", "stock"):
        raise ValueError(f"unknown position kind {kind!r} — expected 'option' or 'stock'")
    if nav <= 0:
        raise ValueError(f"NAV must be positive to size against, got {nav}")
    if unit_price <= 0:
        raise ValueError(f"unit price must be positive, got {unit_price}")

    unit_cost = unit_price * (OPTION_MULTIPLIER if kind == "option" else 1)
    budget = nav * pct
    quantity = int(budget // unit_cost)
    return TradeSizing(
        kind=kind,
        pct_of_nav=pct,
        budget=budget,
        unit_cost=unit_cost,
        quantity=quantity,
        cost=quantity * unit_cost,
    )
