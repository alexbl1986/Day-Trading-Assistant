"""Seam: get_quote(position, fetch=...) -> Quote.

Resolves a position to its Yahoo symbol, calls an injected `fetch` (the one
network-touching line, faked here for hermetic tests), and returns a Quote that
always carries WHEN the price is from (as_of + market_state) so a stale close
can't pass for a live price. Verified yfinance shape: `.info` gives
regularMarketPrice / regularMarketTime (epoch secs) / marketState / currency.
"""

import os
from datetime import datetime, timezone

import pytest

from app.trading.domain import Position, Quote
from app.trading.quotes import get_quote


def _stock(symbol, currency="USD"):
    return Position(
        symbol=symbol, asset_class="STK", currency=currency, fx_rate_to_base=1.0,
        quantity=100.0, mark_price=50.0, position_value=5000.0,
    )


# Normalized shape the fetch seam returns (a thin adapter over yfinance .info).
# epoch 1783713600 == 2026-07-10 20:00:00 UTC (independently computed).
def _fetch_ok(symbol):
    return {"price": 72.6, "currency": "USD", "epoch": 1783713600, "market_state": "CLOSED"}


def test_returns_quote_with_price_and_provenance():
    quote = get_quote(_stock("AEHR"), fetch=_fetch_ok)

    assert quote == Quote(
        symbol="AEHR",
        price=72.6,
        currency="USD",
        as_of=datetime(2026, 7, 10, 20, 0, 0, tzinfo=timezone.utc),
        market_state="CLOSED",
    )


def test_no_market_data_fails_loud():
    # A delisted/misspelled symbol yields no quote (fetch returns None). Refuse
    # rather than fabricate a price -- a silent None here is a NO-DATA-as-$0 bug.
    def _fetch_none(symbol):
        return None

    with pytest.raises(ValueError, match="no quote"):
        get_quote(_stock("DELISTED"), fetch=_fetch_none)


def test_currency_mismatch_fails_loud():
    # The SIVE-bug guard: we think this is a USD name, but Yahoo quoted it in
    # CAD -- the symbol resolved to the wrong instrument. Refuse rather than
    # report a foreign-currency price as if it were USD.
    def _fetch_cad(symbol):
        return {"price": 72.6, "currency": "CAD", "epoch": 1783713600, "market_state": "CLOSED"}

    with pytest.raises(ValueError, match="currency"):
        get_quote(_stock("AEHR", currency="USD"), fetch=_fetch_cad)


def test_foreign_position_resolves_via_listing_exchange():
    # A EUR Amsterdam name only resolves with its listing exchange; get_quote
    # must forward it to to_yahoo_symbol (XYZ + AEB -> XYZ.AS) and quote it.
    def _fetch_eur(symbol):
        assert symbol == "XYZ.AS"  # the exchange suffix reached the fetch
        return {"price": 30.0, "currency": "EUR", "epoch": 1783713600, "market_state": "CLOSED"}

    quote = get_quote(_stock("XYZ", currency="EUR"), listing_exch="AEB", fetch=_fetch_eur)

    assert quote.symbol == "XYZ.AS"
    assert quote.currency == "EUR"
    assert quote.price == 30.0


@pytest.mark.skipif(
    os.getenv("RUN_NETWORK_TESTS") != "1",
    reason="hits live Yahoo; opt in with RUN_NETWORK_TESTS=1",
)
def test_real_yahoo_fetch_wires_end_to_end():
    # Exercises the default _yahoo_fetch against live Yahoo. Structural only --
    # no price hardcoded (it moves); proves the adapter maps .info correctly.
    quote = get_quote(_stock("AAPL", currency="USD"))

    assert quote.symbol == "AAPL"
    assert isinstance(quote.price, float) and quote.price > 0
    assert quote.currency == "USD"
    assert isinstance(quote.as_of, datetime) and quote.as_of.tzinfo is not None
    assert quote.market_state in {"REGULAR", "CLOSED", "PRE", "POST", "PREPRE", "POSTPOST"}
