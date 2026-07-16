"""Seam: size_new_position — what the per-entry sizing rule buys.

Pure math over scalars (policy percentages arrive as parameters, same seam
style as check_exposure): budget = NAV x pct, quantity = whole units the
budget buys. The option multiplier (x100) is applied here so the tool and
the audit both see final dollar figures.
"""

import pytest

from app.trading.sizing import size_new_position


def test_option_sizing_floors_to_whole_contracts():
    # 1% of $100k = $1,000 budget; $3.10 premium = $310/contract -> 3 contracts.
    sizing = size_new_position(100_000.0, kind="option", unit_price=3.10, pct=0.01)
    assert sizing.budget == pytest.approx(1_000.0)
    assert sizing.unit_cost == pytest.approx(310.0)
    assert sizing.quantity == 3
    assert sizing.cost == pytest.approx(930.0)
    assert sizing.pct_of_nav == 0.01


def test_stock_sizing_uses_share_price_directly():
    # 3% of $100k = $3,000; $40/share -> 75 shares, no option multiplier.
    sizing = size_new_position(100_000.0, kind="stock", unit_price=40.0, pct=0.03)
    assert sizing.unit_cost == pytest.approx(40.0)
    assert sizing.quantity == 75
    assert sizing.cost == pytest.approx(3_000.0)


def test_over_budget_premium_buys_zero_contracts():
    # $15.00 premium = $1,500/contract > the $1,000 budget -> 0, cost 0.
    sizing = size_new_position(100_000.0, kind="option", unit_price=15.0, pct=0.01)
    assert sizing.quantity == 0
    assert sizing.cost == 0.0


def test_invalid_inputs_fail_loud():
    with pytest.raises(ValueError):
        size_new_position(0.0, kind="option", unit_price=3.1, pct=0.01)
    with pytest.raises(ValueError):
        size_new_position(100_000.0, kind="option", unit_price=0.0, pct=0.01)
    with pytest.raises(ValueError):
        size_new_position(100_000.0, kind="future", unit_price=3.1, pct=0.01)
