"""Seam: open_position_pnl(positions, trades) -> UnrealizedPnLReport | MissingData.

The current-holdings scoreboard: every open position marked against its average
entry (from the ledger, since the book's CostBasisPrice is always 0), reporting
dollar and % unrealized P/L in base USD. Sibling of scan_scaleout -- same
open-position x ledger-cost-basis join, same cold-start refusal -- but a full
descriptive report (winners and losers), not a filtered action alert.
"""

from datetime import date, datetime
from pathlib import Path

import pytest

from app.trading.domain import MissingData, Position, PositionPnL, Trade, UnrealizedPnLReport
from app.trading.ingest.statement import (
    parse_activity_statement,
    parse_open_positions_cost,
)
from app.trading.ingest.tactical import parse_tactical_book
from app.trading.pnl import open_position_pnl

DATA = Path(__file__).parent.parent / "data" / "book"
BOOK = DATA / "Tactical_Boot.csv"
STMT = DATA / "IBKR YTD Statement.csv"


def _stock_entry(symbol, qty, price):
    """A statement opening fill for a stock (multiplier 1)."""
    return Trade(
        symbol=symbol, root_ticker=symbol, asset_category="Stocks", currency="USD",
        timestamp=datetime(2026, 6, 1, 9, 30), quantity=float(qty), price=price,
        proceeds=-qty * price, commission=-1.0, basis=qty * price,
        realized_pl=0.0, mtm_pl=0.0, code="O",
    )


def _opt_entry(symbol, qty, price):
    """A statement opening fill for an option (statement symbol, multiplier 100)."""
    return Trade(
        symbol=symbol, root_ticker=symbol.split()[0],
        asset_category="Equity and Index Options", currency="USD",
        timestamp=datetime(2026, 6, 1, 9, 30), quantity=float(qty), price=price,
        proceeds=-qty * price * 100, commission=-1.0, basis=qty * price * 100,
        realized_pl=0.0, mtm_pl=0.0, code="O",
    )


def _opt_position(root, strike, expiry, right, mark):
    return Position(
        symbol=root, asset_class="OPT", currency="USD", fx_rate_to_base=1.0,
        quantity=2.0, mark_price=mark, position_value=mark * 200,
        strike=strike, expiry=expiry, right=right,
    )


def _stock_position(symbol, qty, mark, fx=1.0, currency="USD"):
    return Position(
        symbol=symbol, asset_class="STK", currency=currency, fx_rate_to_base=fx,
        quantity=float(qty), mark_price=mark, position_value=mark * qty,
    )


def test_reports_dollar_and_pct_pnl_for_a_stock():
    # Bought 100 @ 2.00, now marked 3.00 -> +50%, +$100 unrealized.
    trades = [_stock_entry("AEHR", 100, 2.00)]
    positions = [_stock_position("AEHR", 100, mark=3.00)]

    report = open_position_pnl(positions, trades)

    assert report == UnrealizedPnLReport(
        total_unrealized_pl=100.0,
        lines=(
            PositionPnL(
                symbol="AEHR",
                avg_entry_price=2.00,
                mark_price=3.00,
                gain=0.5,
                unrealized_pl=100.0,
            ),
        ),
    )


def test_reports_a_losing_option_with_multiplier():
    # Bought 2 contracts @ 2.00, now marked 0.50 -> -75%, and -$300 (the x100
    # multiplier rides in via position_value). Losers appear -- unlike scan_scaleout.
    trades = [_opt_entry("AAPL 17JUL26 200 C", 2, 2.00)]
    positions = [_opt_position("AAPL", 200.0, date(2026, 7, 17), "C", mark=0.50)]

    report = open_position_pnl(positions, trades)

    (line,) = report.lines
    assert line.gain == pytest.approx(-0.75)
    assert line.unrealized_pl == pytest.approx(-300.0)
    assert report.total_unrealized_pl == pytest.approx(-300.0)


def test_short_option_gain_flips_sign():
    # Sold 2 contracts @ 2.00 (premium collected), now marked 0.50: 75% of the
    # premium is captured, so the position is UP +75% / +$300 — the long-side
    # ratio (mark/entry - 1 = -75%) must flip for a short leg. The dollar P/L
    # already rides on the signed position_value.
    trades = [_opt_entry("ZETA 17JUL26 35 C", -2, 2.00)]
    positions = [
        Position(
            symbol="ZETA", asset_class="OPT", currency="USD", fx_rate_to_base=1.0,
            quantity=-2.0, mark_price=0.50, position_value=-100.0,
            strike=35.0, expiry=date(2026, 7, 17), right="C",
        )
    ]

    report = open_position_pnl(positions, trades)

    (line,) = report.lines
    assert line.unrealized_pl == pytest.approx(300.0)
    assert line.gain == pytest.approx(0.75)


# --- contract guards (pass on arrival; the slice-1 formula is already general) ---

def test_foreign_pnl_normalized_and_totalled_across_currencies():
    # A EUR winner (200 EUR x 1.1 = 220 USD) plus the USD winner (+100) -> total
    # is base-USD, proving fx normalization and cross-currency summation.
    trades = [_stock_entry("AEHR", 100, 2.00), _stock_entry("SAP", 100, 10.00)]
    positions = [
        _stock_position("AEHR", 100, mark=3.00),
        _stock_position("SAP", 100, mark=12.00, fx=1.1, currency="EUR"),
    ]

    report = open_position_pnl(positions, trades)

    by_symbol = {line.symbol: line for line in report.lines}
    assert by_symbol["SAP"].unrealized_pl == pytest.approx(220.0)
    assert report.total_unrealized_pl == pytest.approx(320.0)


def test_refuses_without_a_ledger():
    # Cost basis lives in the statement (tactical CostBasisPrice is always 0).
    # No ledger -> refuse, don't imply "$0 unrealized".
    result = open_position_pnl([_stock_position("AEHR", 100, mark=3.00)], [])

    assert isinstance(result, MissingData)
    assert result.store == "ledger"
    assert "statement" in result.remedy.lower()


def test_prices_orphan_from_open_positions_cost_fallback():
    # GOOGL: 30 shares held, but no GOOGL *stock* fill in the ledger (only
    # options). Fall back to the statement's Open-Positions Cost Price. Result
    # must match IBKR's own stated Unrealized P/L (5490.7), and be tagged as
    # sourced from the statement, not the ledger.
    trades = [_stock_entry("AEHR", 100, 2.00)]  # AEHR has a campaign; GOOGL doesn't
    positions = [
        _stock_position("AEHR", 100, mark=3.00),
        _stock_position("GOOGL", 30, mark=359.91),
    ]
    fallback = {("GOOGL", None, None, None): 176.886666667}

    report = open_position_pnl(positions, trades, fallback_entry=fallback)

    by_symbol = {line.symbol: line for line in report.lines}
    assert by_symbol["AEHR"].cost_basis_source == "ledger"
    assert by_symbol["GOOGL"].cost_basis_source == "statement"
    assert by_symbol["GOOGL"].avg_entry_price == 176.886666667
    assert by_symbol["GOOGL"].unrealized_pl == pytest.approx(5490.7)  # IBKR's own figure


@pytest.mark.skipif(
    not (BOOK.exists() and STMT.exists()), reason="real files not present"
)
def test_real_book_pnl_is_internally_consistent():
    statement = STMT.read_text(encoding="utf-8")
    positions = parse_tactical_book(BOOK.read_text(encoding="utf-8"))
    trades = parse_activity_statement(statement)
    fallback = parse_open_positions_cost(statement)

    report = open_position_pnl(positions, trades, fallback_entry=fallback)

    assert isinstance(report, UnrealizedPnLReport)
    # With the Open-Positions fallback, every held position now prices (no orphans).
    assert len(report.lines) == len(positions)
    # Total is exactly the sum of the lines, and each line's gain is consistent.
    assert report.total_unrealized_pl == pytest.approx(sum(l.unrealized_pl for l in report.lines))
    for line in report.lines:
        # Magnitude check: the sign is direction-dependent (the real book holds
        # a short ZETA call) and PositionPnL doesn't carry the quantity.
        assert abs(line.gain) == pytest.approx(abs(line.mark_price / line.avg_entry_price - 1))
        assert line.cost_basis_source in {"ledger", "statement"}
