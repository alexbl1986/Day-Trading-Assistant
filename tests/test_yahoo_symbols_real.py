"""Characterization: to_yahoo_symbol over the real (gitignored) tactical book.

Structural invariants only — no symbols or counts hardcoded. Confirms every
position either maps to a well-formed Yahoo string or refuses *because* it's a
non-US stock (never silently emits a guess).
"""

import re
from pathlib import Path

import pytest

from app.trading.ingest.statement import parse_instrument_exchanges
from app.trading.ingest.tactical import parse_tactical_book
from app.trading.symbols import to_yahoo_symbol

DATA = Path(__file__).parent.parent / "data" / "book"
BOOK = DATA / "Tactical_Boot.csv"
STMT = DATA / "IBKR YTD Statement.csv"
FOREIGN_SUFFIXES = (".TO", ".DE", ".AS", ".ST")

OPTION_RE = re.compile(r"^[A-Z]+\d{6}[CP]\d{8}$")   # AAPL260713C00205000
STOCK_RE = re.compile(r"^[A-Z]+(-[A-Z]+)?$")        # AEHR or BRK-B, no dot/space


@pytest.mark.skipif(not BOOK.exists(), reason="real book not present")
def test_maps_or_refuses_every_real_position():
    positions = parse_tactical_book(BOOK.read_text(encoding="utf-8"))
    refusals = 0
    for p in positions:
        try:
            yahoo = to_yahoo_symbol(p)
        except ValueError:
            # The only allowed refusal reason is a non-US stock.
            assert p.asset_class == "STK" and p.currency != "USD"
            refusals += 1
            continue
        if p.asset_class == "OPT":
            assert OPTION_RE.match(yahoo), yahoo
        else:
            assert p.currency == "USD"
            assert STOCK_RE.match(yahoo), yahoo

    # Every non-USD stock refused, and nothing else did.
    expected = sum(
        1 for p in positions if p.asset_class == "STK" and p.currency != "USD"
    )
    assert refusals == expected


@pytest.mark.skipif(
    not (BOOK.exists() and STMT.exists()), reason="real files not present"
)
def test_whole_book_resolves_with_statement_exchange_lookup():
    positions = parse_tactical_book(BOOK.read_text(encoding="utf-8"))
    exchanges = parse_instrument_exchanges(STMT.read_text(encoding="utf-8"))

    foreign_resolved = 0
    for p in positions:
        # With the ledger's exchange lookup, nothing should raise.
        yahoo = to_yahoo_symbol(p, exchanges.get(p.symbol))
        assert yahoo
        if p.asset_class == "STK" and p.currency != "USD":
            assert yahoo.endswith(FOREIGN_SUFFIXES), yahoo
            foreign_resolved += 1

    # The foreign names that refused in the no-lookup test now resolve.
    assert foreign_resolved > 0
