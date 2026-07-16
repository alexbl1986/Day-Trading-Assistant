"""Seam: sniff_csv_format(content) -> CsvKind — classifies an uploaded CSV
from its first line (ADR-0002 sniffing router).

Fixtures: header lines are copied verbatim from the real IBKR exports
(data/book/, committed anonymized); data rows are synthetic.
"""

from app.trading.ingest.csv_router import CsvKind, sniff_csv_format

# Real first line of the tactical book export (flat Flex table, fully quoted).
TACTICAL_HEADER = (
    '"Symbol","CurrencyPrimary","FXRateToBase","AssetClass","Strike","Expiry",'
    '"Put/Call","Quantity","MarkPrice","PositionValue","CostBasisPrice","PercentOfNAV"'
)


# Real first line of the YTD Activity Statement. The real file starts with a
# UTF-8 BOM (EF BB BF) — verified on the actual export; the sniffer must not
# let it mask the "Statement" prefix.
STATEMENT_HEADER = "﻿Statement,Header,Field Name,Field Value"


def test_tactical_book_export_is_recognized():
    content = TACTICAL_HEADER + '\n"FAKE.A","CAD","0.7","STK","","","","1","10","10","0","1.0"\n'
    assert sniff_csv_format(content) == CsvKind.TACTICAL


def test_activity_statement_with_bom_is_recognized():
    content = STATEMENT_HEADER + "\nStatement,Data,Title,Activity Statement\n"
    assert sniff_csv_format(content) == CsvKind.ACTIVITY_STATEMENT


def test_unrecognized_csv_is_rejected():
    content = "Date,Open,High,Low,Close\n2026-01-02,100,101,99,100.5\n"
    assert sniff_csv_format(content) == CsvKind.UNKNOWN


def test_empty_upload_is_rejected():
    assert sniff_csv_format("") == CsvKind.UNKNOWN
