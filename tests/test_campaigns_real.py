"""Characterization: campaign grouping over the real (gitignored) statement.

Asserts structural invariants, not specific holdings — no real P/L hardcoded.
"""

from pathlib import Path

import pytest

from app.trading.ingest.statement import parse_activity_statement
from app.trading.ledger import group_campaigns

REAL_FILE = Path(__file__).parent.parent / "data" / "book" / "IBKR YTD Statement.csv"


@pytest.mark.skipif(not REAL_FILE.exists(), reason="real (gitignored) statement not present")
def test_grouping_conserves_pl_and_keeps_contracts_pure():
    trades = parse_activity_statement(REAL_FILE.read_text(encoding="utf-8"))
    campaigns = group_campaigns(trades)

    # No fill dropped or duplicated: realized P/L is conserved through grouping.
    ledger_pl = sum(t.realized_pl for t in trades)
    campaign_pl = sum(c.realized_pl for c in campaigns)
    assert campaign_pl == pytest.approx(ledger_pl)

    # No fill dropped from the campaigns either.
    assert sum(len(c.fills) for c in campaigns) == len(trades)

    # Each campaign holds exactly one contract; open/closed matches net qty.
    for c in campaigns:
        assert len({f.symbol for f in c.fills}) == 1
        assert c.is_open == (c.net_quantity != 0)
