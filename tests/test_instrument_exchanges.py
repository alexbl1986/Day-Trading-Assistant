"""Seam: parse_instrument_exchanges(statement) -> {stock symbol: Listing Exch}.

The statement's `Financial Instrument Information` section is the only place the
listing exchange appears (the tactical book has no exchange column). It carries
two header shapes — stocks and options — in one section; we want stocks only,
since options are always US-listed (OCC).
"""

from app.trading.ingest.statement import parse_instrument_exchanges

# Verbatim IBKR headers (stock shape has "Security ID"; option shape does not).
FII_SECTION = (
    "Financial Instrument Information,Header,Asset Category,Symbol,Description,"
    "Conid,Security ID,Underlying,Listing Exch,Multiplier,Type,Code\n"
    "Financial Instrument Information,Data,Stocks,SHOP,SHOPIFY INC,12345,"
    "CA82509L1076,,TSE,1,COMMON,\n"
    "Financial Instrument Information,Data,Stocks,AEHR,AEHR TEST SYSTEMS,678,"
    "US00767T1088,,NASDAQ,1,COMMON,\n"
    "Financial Instrument Information,Header,Asset Category,Symbol,Description,"
    "Conid,Underlying,Listing Exch,Multiplier,Expiry,Delivery Month,Type,Strike,Code\n"
    "Financial Instrument Information,Data,Equity and Index Options,"
    "FAKE 17JUL26 200 C,desc,999,FAKE,CBOE,100,2026-07-17,2026-07,C,200,\n"
)


def test_parses_stock_listing_exchanges_and_skips_options():
    assert parse_instrument_exchanges(FII_SECTION) == {"SHOP": "TSE", "AEHR": "NASDAQ"}
