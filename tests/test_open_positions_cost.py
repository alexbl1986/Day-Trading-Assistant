"""Seam: parse_open_positions_cost(statement) -> {contract_key: cost_price}.

The statement's `Open Positions` section carries IBKR's own per-unit `Cost Price`
for every currently-held position -- the fallback entry price for holdings that
have no opening fill in the YTD Trades ledger (acquired before the window, via
assignment, or rebought after the statement's cutoff). Keyed the same way as
campaigns so it drops into the existing open-position join.
"""

from datetime import date

from app.trading.ingest.statement import parse_open_positions_cost

# Verbatim Open Positions header, then a stock and an option row (synthetic).
SECTION = (
    "Open Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,"
    "Quantity,Mult,Cost Price,Cost Basis,Close Price,Value,Unrealized P/L,Code\n"
    "Open Positions,Data,Summary,Stocks,USD,GOOGL,30,1,176.886666667,5306.6,"
    "359.91,10797.3,5490.7,\n"
    "Open Positions,Data,Summary,Equity and Index Options,USD,ADEA 17JUL26 35 C,"
    "3,100,2.730840833,819.25225,0.45,135,-684.25225,\n"
)


def test_maps_contract_key_to_per_unit_cost_price():
    # Stock -> ('GOOGL', None, None, None); option -> full contract key. Cost
    # Price is per-unit (GOOGL 176.89 x 30 = 5306.6 basis), like avg_entry_price.
    assert parse_open_positions_cost(SECTION) == {
        ("GOOGL", None, None, None): 176.886666667,
        ("ADEA", date(2026, 7, 17), 35.0, "C"): 2.730840833,
    }
