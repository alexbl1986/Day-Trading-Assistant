"""Seam: contract keys that match a tactical Position to a statement Campaign.

Tactical options carry strike/expiry/right in columns; statement options encode
them in the symbol string (TICKER DDMMMYY STRIKE RIGHT). Both must reduce to the
same canonical key so a position can be paired with its campaign's cost basis.
"""

from datetime import date

from app.trading.domain import Position
from app.trading.symbols import (
    parse_statement_option_symbol,
    position_contract_key,
    statement_symbol_contract_key,
)


def _opt_position(symbol, strike, expiry, right):
    return Position(
        symbol=symbol, asset_class="OPT", currency="USD", fx_rate_to_base=1.0,
        quantity=3.0, mark_price=0.45, position_value=135.0,
        strike=strike, expiry=expiry, right=right,
    )


def _stock_position(symbol):
    return Position(
        symbol=symbol, asset_class="STK", currency="USD", fx_rate_to_base=1.0,
        quantity=10.0, mark_price=20.0, position_value=200.0,
    )


def test_parses_statement_option_symbol():
    assert parse_statement_option_symbol("ADEA 17JUL26 35 C") == (
        "ADEA", date(2026, 7, 17), 35.0, "C",
    )


def test_parses_decimal_strike():
    assert parse_statement_option_symbol("NVDA 13FEB26 187.5 C")[2] == 187.5


def test_option_position_and_statement_campaign_share_a_key():
    pos = _opt_position("ADEA", 35.0, date(2026, 7, 17), "C")
    assert position_contract_key(pos) == statement_symbol_contract_key("ADEA 17JUL26 35 C")


def test_stock_position_and_statement_symbol_share_a_key():
    assert position_contract_key(_stock_position("P4O")) == statement_symbol_contract_key("P4O")


def test_different_strike_does_not_match():
    pos = _opt_position("ADEA", 35.0, date(2026, 7, 17), "C")
    assert position_contract_key(pos) != statement_symbol_contract_key("ADEA 17JUL26 40 C")
