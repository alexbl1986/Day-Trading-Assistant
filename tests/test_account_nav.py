"""Seam: parse_account_nav(statement) -> total NAV (cash-inclusive) or None.

The statement's `Net Asset Value` section is the exposure denominator (the trader's
rules are % of NAV). We take the `Total` row's `Current Total` -- a coherent
point-in-time snapshot that includes cash. A trailing "Time Weighted Rate of
Return" sub-header/row shares the section and must be ignored.
"""

from app.trading.ingest.statement import parse_account_nav

# Verbatim Net Asset Value header from the real statement, then a synthetic
# body (round numbers) plus the trailing TWRR sub-section that must not leak.
NAV_SECTION = (
    "Net Asset Value,Header,Asset Class,Prior Total,Current Long,Current Short,"
    "Current Total,Change\n"
    "Net Asset Value,Data,Cash ,8000,50000,-2000,48000,40000\n"
    "Net Asset Value,Data,Stock,0,90000,0,90000,90000\n"
    "Net Asset Value,Data,Options,600,12100,-100,12000,11400\n"
    "Net Asset Value,Data,Total,8600,152100,-2100,150000,141400\n"
    "Net Asset Value,Header,Time Weighted Rate of Return\n"
    "Net Asset Value,Data,151.720768447%\n"
)


def test_parses_total_current_nav():
    # The 'Total' row's Current Total (cash + stock + options), not any bucket.
    assert parse_account_nav(NAV_SECTION) == 150000.0


def test_missing_section_returns_none():
    # No NAV section (e.g. only a tactical book) -> None, which the exposure
    # tool turns into a cold-start MissingData refusal rather than a bogus 0.
    assert parse_account_nav("Trades,Header,Foo,Bar\n") is None
