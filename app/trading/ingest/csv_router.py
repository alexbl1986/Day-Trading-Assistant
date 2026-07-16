"""Sniffing router for uploaded IBKR CSVs.

Classifies an upload from its first line: tactical book export (flat Flex
table) vs YTD Activity Statement (27-section) vs unrecognized.
"""

from enum import Enum


class CsvKind(Enum):
    TACTICAL = "tactical"
    ACTIVITY_STATEMENT = "activity_statement"
    UNKNOWN = "unknown"


# First-line signatures of the two known IBKR exports (real-file verified).
_TACTICAL_PREFIX = '"Symbol","CurrencyPrimary"'
_STATEMENT_PREFIX = "Statement,Header"


def sniff_csv_format(content: str) -> CsvKind:
    first_line = content.removeprefix("\ufeff").split("\n", 1)[0]
    if first_line.startswith(_TACTICAL_PREFIX):
        return CsvKind.TACTICAL
    if first_line.startswith(_STATEMENT_PREFIX):
        return CsvKind.ACTIVITY_STATEMENT
    return CsvKind.UNKNOWN
