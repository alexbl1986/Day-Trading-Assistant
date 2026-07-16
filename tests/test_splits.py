"""Seam: apply_splits(trades, splits) -> list[Trade].

Restates pre-split fills in post-split terms so campaigns straddling a stock
split stay correct. IBKR reports each fill at its historical share count; the
split itself lives only in the Corporate Actions section, so without this a
straddling campaign miscounts its net quantity (real POWL: read as closed
while 2 shares remain) and prices today's shares off a pre-split entry (real
SMTOY: a -$684 position reported as the phantom -$19,274).
"""

from datetime import datetime
from pathlib import Path

import pytest

from app.trading.domain import Split, Trade
from app.trading.ingest.statement import (
    parse_activity_statement,
    parse_open_positions_cost,
    parse_splits,
)
from app.trading.ingest.tactical import parse_tactical_book
from app.trading.ledger import apply_splits, get_trades, group_campaigns
from app.trading.pnl import open_position_pnl

DATA = Path(__file__).parent.parent / "data" / "book"
BOOK = DATA / "Tactical_Boot.csv"
STMT = DATA / "IBKR YTD Statement.csv"


def _fill(symbol, when, qty, price, *, category="Stocks"):
    return Trade(
        symbol=symbol, root_ticker=symbol.split()[0], asset_category=category,
        currency="USD", timestamp=when, quantity=float(qty), price=price,
        proceeds=-qty * price, commission=-1.0, basis=qty * price,
        realized_pl=0.0, mtm_pl=0.0, code="O",
    )


_SPLIT_8_FOR_1 = Split(
    symbol="SMTOY", numerator=8, denominator=1,
    effective=datetime(2026, 7, 1, 20, 25),
)


def test_pre_split_fill_restated_in_post_split_terms():
    # 30 shares @ 88.53 bought before an 8-for-1 split are, in today's terms,
    # 240 shares @ 11.06625. Cash moved is unchanged — a split moves no money.
    fill = _fill("SMTOY", datetime(2026, 6, 3, 9, 30), 30, 88.53)

    (adjusted,) = apply_splits([fill], [_SPLIT_8_FOR_1])

    assert adjusted.quantity == 240.0
    assert adjusted.price == pytest.approx(11.06625)
    assert adjusted.proceeds == fill.proceeds
    assert adjusted.basis == fill.basis
    assert adjusted.realized_pl == fill.realized_pl
    assert adjusted.commission == fill.commission


def test_post_split_fill_untouched():
    fill = _fill("SMTOY", datetime(2026, 7, 2, 9, 30), 240, 8.5)
    assert apply_splits([fill], [_SPLIT_8_FOR_1]) == [fill]


def test_other_symbols_and_option_fills_untouched():
    other = _fill("AEHR", datetime(2026, 6, 3, 9, 30), 100, 2.0)
    option = _fill(
        "SMTOY 16JAN26 40 C", datetime(2026, 6, 3, 9, 30), 2, 1.5,
        category="Equity and Index Options",
    )
    assert apply_splits([other, option], [_SPLIT_8_FOR_1]) == [other, option]


def test_sequential_splits_compound():
    # A 2-for-1 then a 3-for-1: a fill predating both ends up x6 / ÷6.
    fill = _fill("FAKE", datetime(2026, 1, 5, 9, 30), 10, 60.0)
    splits = [
        Split("FAKE", 3, 1, effective=datetime(2026, 3, 1, 20, 25)),
        Split("FAKE", 2, 1, effective=datetime(2026, 2, 1, 20, 25)),
    ]

    (adjusted,) = apply_splits([fill], splits)

    assert adjusted.quantity == 60.0
    assert adjusted.price == pytest.approx(10.0)


def test_straddling_campaign_stays_open():
    # The real POWL shape: buy 2 / sell 1 pre-split, 3-for-1, then buy 2 /
    # sell 2 / sell 1. Unadjusted, the ledger counts ...=0 and closes the
    # campaign while 2 shares remain; adjusted, it stays open at net 2 with
    # the campaign-average entry over the opening fills (6@167.79 + 2@194.43).
    split = Split("POWL", 3, 1, effective=datetime(2026, 4, 3, 20, 25))
    fills = [
        _fill("POWL", datetime(2026, 3, 9, 10, 27), 2, 503.37),
        _fill("POWL", datetime(2026, 3, 30, 10, 49), -1, 511.92),
        _fill("POWL", datetime(2026, 4, 7, 11, 59), 2, 194.43),
        _fill("POWL", datetime(2026, 4, 10, 10, 29), -2, 232.37),
        _fill("POWL", datetime(2026, 4, 29, 4, 55), -1, 268.12),
    ]

    (campaign,) = group_campaigns(apply_splits(fills, [split]))

    assert campaign.is_open
    assert campaign.net_quantity == 2.0
    assert campaign.avg_entry_price == pytest.approx(174.45)


@pytest.mark.skipif(
    not (BOOK.exists() and STMT.exists()), reason="real files not present"
)
def test_real_smtoy_and_powl_repair():
    statement = STMT.read_text(encoding="utf-8")
    trades = apply_splits(parse_activity_statement(statement), parse_splits(statement))
    positions = parse_tactical_book(BOOK.read_text(encoding="utf-8"))

    report = open_position_pnl(
        positions, trades, fallback_entry=parse_open_positions_cost(statement)
    )

    # SMTOY: IBKR's own Open Positions unrealized is -684.1; the campaign
    # entry excludes the $1 commission IBKR folds into cost, hence the slack.
    (smtoy,) = [l for l in report.lines if l.symbol == "SMTOY"]
    assert smtoy.unrealized_pl == pytest.approx(-684.1, abs=1.5)
    assert smtoy.gain == pytest.approx(-0.257, abs=0.005)

    # POWL: the campaign straddling its April 3-for-1 split is OPEN (2 shares
    # held per the book), no longer misread as closed.
    (powl,) = get_trades(trades, "POWL")
    assert powl.net_quantity == 2.0
