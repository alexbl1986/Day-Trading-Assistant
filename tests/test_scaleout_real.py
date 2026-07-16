"""Characterization: scan_scaleout over the real (gitignored) book + statement.

Structural invariants only — no specific holdings or numbers hardcoded.
"""

from pathlib import Path

import pytest

from app.trading.domain import MissingData, ScaleOutSignal
from app.trading.ingest.statement import parse_activity_statement
from app.trading.ingest.tactical import parse_tactical_book
from app.trading.scaleout import scan_scaleout

DATA = Path(__file__).parent.parent / "data" / "book"
BOOK = DATA / "Tactical_Boot.csv"
STMT = DATA / "IBKR YTD Statement.csv"


@pytest.mark.skipif(not (BOOK.exists() and STMT.exists()), reason="real files not present")
def test_scan_on_real_data_is_consistent_and_ledger_gated():
    positions = parse_tactical_book(BOOK.read_text(encoding="utf-8"))
    trades = parse_activity_statement(STMT.read_text(encoding="utf-8"))

    # No ledger -> refuse, regardless of positions.
    assert isinstance(scan_scaleout(positions, []), MissingData)

    # With a ledger -> a list; every candidate is an option (the ladder is
    # options-only) and its flag matches its recorded ladder state + gain.
    candidates = scan_scaleout(positions, trades)
    assert isinstance(candidates, list)
    option_symbols = {p.symbol for p in positions if p.asset_class == "OPT"}
    for c in candidates:
        assert c.symbol in option_symbols
        assert c.signal is not ScaleOutSignal.NONE
        assert c.mark_price / c.avg_entry_price - 1 == pytest.approx(c.gain)
        if c.signal is ScaleOutSignal.FIRST_TRANCHE_DUE:
            assert c.scales_taken == 0 and c.gain >= 1.0
        elif c.signal is ScaleOutSignal.SECOND_TRANCHE_DUE:
            assert c.scales_taken == 1 and c.gain >= 2.0
        else:  # a runner is inventory — reported at any gain
            assert c.signal is ScaleOutSignal.MOONSHOT_RUNNER
            assert c.scales_taken >= 2
