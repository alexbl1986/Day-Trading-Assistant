"""Seam: parse_tactical_book(content) -> list[Position].

Parses the IBKR tactical book export (flat Flex table) into typed Position
records. Header line is verbatim from the real export; data rows are
synthetic (the anonymized real book lives in data/book/).
"""

from datetime import date
from pathlib import Path

import pytest

from app.trading.ingest.tactical import parse_tactical_book
from app.trading.domain import Position

REAL_FILE = Path(__file__).parent.parent / "data" / "book" / "Tactical_Boot.csv"

HEADER = (
    '"Symbol","CurrencyPrimary","FXRateToBase","AssetClass","Strike","Expiry",'
    '"Put/Call","Quantity","MarkPrice","PositionValue","CostBasisPrice","PercentOfNAV"'
)


def test_parses_a_single_stock_row():
    content = (
        HEADER + "\n"
        '"FAKE","EUR","1.1436","STK","","","","5","273.1","1365.5","0","1.65"\n'
    )

    positions = parse_tactical_book(content)

    assert positions == [
        Position(
            symbol="FAKE",
            asset_class="STK",
            currency="EUR",
            fx_rate_to_base=1.1436,
            quantity=5.0,
            mark_price=273.1,
            position_value=1365.5,
            strike=None,
            expiry=None,
            right=None,
        )
    ]


def test_parses_a_long_call_option_row():
    # Padded OCC symbol in the Symbol field; strike/expiry/right also broken
    # out into their own columns (expiry is DD/MM/YYYY).
    content = (
        HEADER + "\n"
        '"FAKE  260717C00035000","USD","1","OPT","35","17/07/2026","C","3","0.45","135","0","2.85"\n'
    )

    positions = parse_tactical_book(content)

    assert positions == [
        Position(
            symbol="FAKE",
            asset_class="OPT",
            currency="USD",
            fx_rate_to_base=1.0,
            quantity=3.0,
            mark_price=0.45,
            position_value=135.0,
            strike=35.0,
            expiry=date(2026, 7, 17),
            right="C",
        )
    ]


def test_preserves_signs_of_a_short_option():
    # A written (sold) call: negative quantity and negative position value.
    # Signs must survive — a short is negative exposure for hedge/exposure math.
    content = (
        HEADER + "\n"
        '"FAKE  260717C00035000","USD","1","OPT","35","17/07/2026","C","-1","0.0104","-1.04","0","100.00"\n'
    )

    positions = parse_tactical_book(content)

    assert positions[0].quantity == -1.0
    assert positions[0].position_value == -1.04


@pytest.mark.skipif(not REAL_FILE.exists(), reason="real (gitignored) book export not present")
def test_parses_the_real_book_export_without_dropping_rows():
    content = REAL_FILE.read_text(encoding="utf-8")
    data_rows = content.strip().splitlines()[1:]  # drop header

    positions = parse_tactical_book(content)

    # No row silently dropped, and every position is a known asset class.
    assert len(positions) == len(data_rows)
    assert {p.asset_class for p in positions} <= {"STK", "OPT"}
    # Every option row carries a parsed expiry; stocks never do.
    for p in positions:
        assert (p.expiry is not None) == (p.asset_class == "OPT")
