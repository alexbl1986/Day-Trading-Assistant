"""Seam: get_trades(trades, ticker, full_history=False) -> list[Campaign].

The campaign-history tool. Matches by underlying (root ticker); default
returns only the currently-open campaign(s); full_history includes closed ones.
"""

from datetime import datetime

from app.trading.ledger import get_trades
from app.trading.domain import Trade


def _fill(symbol, qty, when):
    return Trade(
        symbol=symbol,
        root_ticker=symbol.split()[0],
        asset_category="Stocks",
        currency="USD",
        timestamp=when,
        quantity=float(qty),
        price=1.0,
        proceeds=0.0,
        commission=0.0,
        basis=0.0,
        realized_pl=0.0,
        mtm_pl=0.0,
        code="O",
    )


LEDGER = [
    _fill("P4O", 130, datetime(2026, 1, 2, 9, 30)),
    _fill("P4O", -130, datetime(2026, 1, 3, 9, 30)),  # round-trip closed
    _fill("P4O", 100, datetime(2026, 1, 9, 9, 30)),  # reopened, still open
]


def test_default_returns_only_the_active_campaign():
    result = get_trades(LEDGER, "P4O")

    assert len(result) == 1
    assert result[0].net_quantity == 100.0
    assert result[0].is_open is True


def test_full_history_includes_closed_campaigns():
    result = get_trades(LEDGER, "P4O", full_history=True)

    assert len(result) == 2


def test_unknown_ticker_returns_empty():
    assert get_trades(LEDGER, "ZZZ") == []
