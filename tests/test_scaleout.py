"""Seam: the fills-based scale-out ladder.

Two sale rungs from the trader's rulebook, keyed on how many scale-out sales
the open campaign has already recorded — the fills ARE the ladder state. No
sales yet: +100% flags the first tranche. One sale: +200% flags the second.
Two or more: the remainder is reported as a moonshot runner at ANY current
gain — inventory, not an action; the runner's endgame is manual. `gain` is a
ratio vs entry: +100% == 1.0.
"""

from datetime import date, datetime

import pytest

from app.trading.domain import MissingData, Position, ScaleOutSignal, Trade
from app.trading.scaleout import classify_scaleout, scan_scaleout


def _entry(symbol, qty, price, ts=datetime(2026, 6, 1, 9, 30)):
    """A statement opening fill (statement symbol format)."""
    return Trade(
        symbol=symbol, root_ticker=symbol.split()[0],
        asset_category="Equity and Index Options", currency="USD",
        timestamp=ts, quantity=float(qty), price=price,
        proceeds=-qty * price * 100, commission=-1.0, basis=qty * price * 100,
        realized_pl=0.0, mtm_pl=0.0, code="O",
    )


def _sell(symbol, qty, price, ts=datetime(2026, 6, 15, 10, 0)):
    """A scale-out sale: a closing-direction fill that leaves the campaign open."""
    return Trade(
        symbol=symbol, root_ticker=symbol.split()[0],
        asset_category="Equity and Index Options", currency="USD",
        timestamp=ts, quantity=-float(qty), price=price,
        proceeds=qty * price * 100, commission=-1.0, basis=-qty * price * 100,
        realized_pl=50.0, mtm_pl=0.0, code="C",
    )


def _opt_position(root, strike, expiry, right, mark, qty=2.0):
    return Position(
        symbol=root, asset_class="OPT", currency="USD", fx_rate_to_base=1.0,
        quantity=qty, mark_price=mark, position_value=mark * qty * 100,
        strike=strike, expiry=expiry, right=right,
    )


@pytest.mark.parametrize(
    "scales_taken, gain, expected",
    [
        (0, -0.769, ScaleOutSignal.NONE),             # deep loss, nothing taken
        (0, 0.99, ScaleOutSignal.NONE),               # just under the first rung
        (0, 1.0, ScaleOutSignal.FIRST_TRANCHE_DUE),   # +100% exactly
        (0, 2.6, ScaleOutSignal.FIRST_TRANCHE_DUE),   # way past both rungs, but the
                                                      # NEXT action is still sale #1
        (1, 1.2, ScaleOutSignal.NONE),                # between rungs: nothing due
        (1, 1.99, ScaleOutSignal.NONE),
        (1, 2.0, ScaleOutSignal.SECOND_TRANCHE_DUE),  # +200% exactly
        (2, 2.6, ScaleOutSignal.MOONSHOT_RUNNER),     # ladder complete: report
        (2, 0.3, ScaleOutSignal.MOONSHOT_RUNNER),     # melted runner STAYS reported
        (3, 1.0, ScaleOutSignal.MOONSHOT_RUNNER),
    ],
)
def test_classify_ladder_by_sales_recorded_then_gain(scales_taken, gain, expected):
    assert classify_scaleout(gain, scales_taken) == expected


def test_classify_thresholds_are_injectable():
    # The rungs belong to the policy record, not this module — an edited
    # trigger must move the lines (the defaults stay the rulebook defaults).
    assert classify_scaleout(1.2, 0, first_gain=1.25) == ScaleOutSignal.NONE
    assert classify_scaleout(1.5, 1, second_gain=1.4) == ScaleOutSignal.SECOND_TRANCHE_DUE


def test_scan_reads_ladder_state_from_campaign_fills():
    expiry = date(2026, 7, 17)
    trades = [
        # ADEA: 3 opened, nothing sold -> first tranche due at +150%.
        _entry("ADEA 17JUL26 35 C", 3, 0.40),
        # BEAM: one scale-out recorded -> second tranche due at +225%.
        _entry("BEAM 17JUL26 50 C", 3, 0.40),
        _sell("BEAM 17JUL26 50 C", 1, 0.90),
        # RIDE: both tranches sold -> runner, reported even though it has
        # melted to +25% (fills-based: the roster never hides the pain).
        _entry("RIDE 17JUL26 20 C", 3, 0.40),
        _sell("RIDE 17JUL26 20 C", 1, 0.80),
        _sell("RIDE 17JUL26 20 C", 1, 1.20, ts=datetime(2026, 6, 20, 11, 0)),
        # DOWN: no sales, under water -> not flagged.
        _entry("DOWN 17JUL26 10 C", 2, 0.40),
    ]
    positions = [
        _opt_position("ADEA", 35.0, expiry, "C", mark=1.00, qty=3),
        _opt_position("BEAM", 50.0, expiry, "C", mark=1.30, qty=2),
        _opt_position("RIDE", 20.0, expiry, "C", mark=0.50, qty=1),
        _opt_position("DOWN", 10.0, expiry, "C", mark=0.30, qty=2),
    ]

    candidates = scan_scaleout(positions, trades)

    by_symbol = {c.symbol: c for c in candidates}
    assert set(by_symbol) == {"ADEA", "BEAM", "RIDE"}
    assert by_symbol["ADEA"].signal == ScaleOutSignal.FIRST_TRANCHE_DUE
    assert by_symbol["ADEA"].scales_taken == 0
    assert by_symbol["BEAM"].signal == ScaleOutSignal.SECOND_TRANCHE_DUE
    assert by_symbol["BEAM"].scales_taken == 1
    assert by_symbol["RIDE"].signal == ScaleOutSignal.MOONSHOT_RUNNER
    assert by_symbol["RIDE"].scales_taken == 2
    assert by_symbol["RIDE"].gain == pytest.approx(0.25)
    assert by_symbol["RIDE"].quantity == 1  # contracts riding, from the snapshot


def test_scan_prices_orphans_from_the_statement_cost_fallback():
    # A held LEAP with no opening fill in the YTD ledger (bought last year).
    # The statement's Open-Positions Cost Price stands in as the entry. Sales
    # taken before the window are equally invisible, so the ladder state is
    # honestly assumed to be rung 0.
    trades = [_entry("ADEA 17JUL26 35 C", 2, 0.40)]  # a ledger exists; no IGV in it
    expiry = date(2027, 1, 15)
    position = _opt_position("IGV", 100.0, expiry, "C", mark=2.05)
    fallback = {("IGV", expiry, 100.0, "C"): 1.00}

    candidates = scan_scaleout([position], trades, fallback_entry=fallback)

    (igv,) = candidates
    assert igv.signal == ScaleOutSignal.FIRST_TRANCHE_DUE
    assert igv.scales_taken == 0
    assert igv.gain == pytest.approx(1.05)


def test_scan_ignores_stocks_the_ladder_is_options_only():
    # The tranche ladder is a contract-selling rule ("sell one, let the rest
    # ride"); the trader confirmed stocks are governed by the holding cap
    # instead. A doubled share position must not be flagged even when its entry
    # is known — via the ledger or via the statement cost fallback.
    stock_fill = Trade(
        symbol="GOOGL", root_ticker="GOOGL", asset_category="Stocks",
        currency="USD", timestamp=datetime(2026, 6, 1, 9, 30), quantity=30.0,
        price=176.89, proceeds=-5306.7, commission=-1.0, basis=5307.7,
        realized_pl=0.0, mtm_pl=0.0, code="O",
    )
    stock = Position(
        symbol="GOOGL", asset_class="STK", currency="USD", fx_rate_to_base=1.0,
        quantity=30.0, mark_price=359.91, position_value=10797.3,
    )
    fallback = {("GOOGL", None, None, None): 176.89}

    assert scan_scaleout([stock], [stock_fill]) == []
    assert scan_scaleout([stock], [stock_fill], fallback_entry=fallback) == []


def test_scan_refuses_without_a_ledger():
    # Cost basis lives in the statement (tactical CostBasisPrice is always 0).
    # With no ledger, refuse loudly rather than imply "no candidates".
    positions = [_opt_position("ADEA", 35.0, date(2026, 7, 17), "C", mark=1.00)]

    result = scan_scaleout(positions, [])

    assert isinstance(result, MissingData)
    assert result.store == "ledger"
    assert "statement" in result.remedy.lower()
