"""Seam: to_yahoo_symbol(position) -> the string yfinance expects.

Verified against yfinance 1.5.1 (probe): a real option contractSymbol looks like
`AAPL260713C00205000` (root + YYMMDD + C/P + strike*1000 zero-padded to 8), and
Yahoo resolves `BRK-B`, not `BRK.B` (dot -> dash for share classes).
"""

from datetime import date

import pytest

from app.trading.domain import Position
from app.trading.symbols import to_yahoo_symbol


def _stock(symbol, currency="USD"):
    return Position(
        symbol=symbol, asset_class="STK", currency=currency, fx_rate_to_base=1.0,
        quantity=100.0, mark_price=50.0, position_value=5000.0,
    )


def _option(root, expiry, right, strike, currency="USD"):
    return Position(
        symbol=root, asset_class="OPT", currency=currency, fx_rate_to_base=1.0,
        quantity=2.0, mark_price=1.0, position_value=200.0,
        strike=strike, expiry=expiry, right=right,
    )


def test_plain_us_stock_maps_to_itself():
    assert to_yahoo_symbol(_stock("AEHR")) == "AEHR"


def test_share_class_stock_dot_becomes_dash():
    # Verified: yfinance resolves BRK-B, but BRK.B returns NO DATA.
    assert to_yahoo_symbol(_stock("BRK.B")) == "BRK-B"


def test_option_maps_to_occ_contract_symbol():
    # Ground truth copied verbatim from the yfinance probe (AAPL chain):
    # AAPL / 2026-07-13 / Call / strike 205.0 -> AAPL260713C00205000.
    pos = _option("AAPL", date(2026, 7, 13), "C", 205.0)
    assert to_yahoo_symbol(pos) == "AAPL260713C00205000"


@pytest.mark.parametrize(
    "right, strike, expected",
    [
        ("P", 205.0, "AAPL260713P00205000"),   # a put carries P
        ("C", 2.5, "AAPL260713C00002500"),     # fractional strike pads left
        ("C", 0.5, "AAPL260713C00000500"),     # sub-dollar (FLEX-style) strike
        ("C", 1250.0, "AAPL260713C01250000"),  # 4-digit strike fills the 8 slots
    ],
)
def test_option_strike_and_right_encoding(right, strike, expected):
    # Guard: pins the C/P letter and strike*1000 zero-padding across magnitudes.
    assert to_yahoo_symbol(_option("AAPL", date(2026, 7, 13), right, strike)) == expected


def test_non_us_stock_fails_loud():
    # A CAD/EUR/SEK stock can't be placed on Yahoo without its listing exchange
    # (currency can't disambiguate EUR -> .AS vs .DE). Refuse rather than emit a
    # US-style guess that silently returns the wrong instrument's price.
    with pytest.raises(ValueError, match="listing exchange"):
        to_yahoo_symbol(_stock("SHOP", currency="CAD"))


@pytest.mark.parametrize(
    "exch, suffix",
    [("TSE", ".TO"), ("IBIS", ".DE"), ("AEB", ".AS"), ("SFB", ".ST")],
)
def test_non_us_stock_maps_via_listing_exchange(exch, suffix):
    # Suffixes verified against yfinance (SHOP.TO, SAP.DE, ASML.AS, VOLV-B.ST).
    assert to_yahoo_symbol(_stock("XYZ", currency="EUR"), exch) == "XYZ" + suffix


def test_foreign_share_class_gets_dash_and_suffix():
    # Stockholm B-share: dot -> dash AND the .ST suffix (cf. yfinance VOLV-B.ST).
    assert to_yahoo_symbol(_stock("VOLV.B", currency="SEK"), "SFB") == "VOLV-B.ST"


def test_unknown_listing_exchange_fails_loud():
    # Guard: an exchange we have no verified Yahoo suffix for must refuse, not guess.
    with pytest.raises(ValueError, match="listing exchange"):
        to_yahoo_symbol(_stock("XYZ", currency="JPY"), "SOME_UNKNOWN_EXCH")
