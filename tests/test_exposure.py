"""Seam: check_exposure(positions, nav) -> ExposureReport | MissingData.

Measures book exposure against the trader's % -of-NAV limits. NAV is the cash-inclusive
statement Total (chosen denominator); with no NAV the tool refuses (MissingData)
rather than divide by a fabricated zero. First metric: options <= 10% NAV.
The "20% offensive exposure" bucket is deferred until 'offensive' is defined.
"""

from pathlib import Path

import pytest

from app.trading.domain import ExposureCheck, ExposureReport, MissingData, Position
from app.trading.exposure import check_exposure
from app.trading.ingest.statement import parse_account_nav
from app.trading.ingest.tactical import parse_tactical_book

DATA = Path(__file__).parent.parent / "data" / "book"
BOOK = DATA / "Tactical_Boot.csv"
STMT = DATA / "IBKR YTD Statement.csv"


def _stock(symbol, value, fx=1.0, currency="USD"):
    return Position(
        symbol=symbol, asset_class="STK", currency=currency, fx_rate_to_base=fx,
        quantity=100.0, mark_price=value / 100.0, position_value=value,
    )


def _option(symbol, value, fx=1.0, currency="USD", right="C"):
    return Position(
        symbol=symbol, asset_class="OPT", currency=currency, fx_rate_to_base=fx,
        quantity=10.0, mark_price=value / 10.0, position_value=value,
        strike=50.0, right=right,
    )


def test_reports_options_pct_of_nav():
    # Options bucket = 12,000 of 150,000 NAV = 8% (<= 10% -> within policy).
    # The stock line must NOT count toward the options bucket.
    positions = [_stock("AEHR", 90000.0), _option("AAPL", 12000.0)]

    report = check_exposure(positions, nav=150000.0)

    assert report == ExposureReport(
        nav=150000.0,
        checks=(ExposureCheck(label="options", value_base=12000.0, pct_of_nav=0.08, limit=0.10),),
    )
    assert report.checks[0].within_policy is True


# --- contract guards (pass on arrival; they pin behavior slice 2 already covers) ---

def test_foreign_option_value_normalized_to_base_usd():
    # SIVE-adjacent guard: a EUR option must be converted via fx before the ratio,
    # not compared in its native currency. 10,000 EUR x 1.1 = 11,000 USD.
    report = check_exposure([_option("SAP", 10000.0, fx=1.1, currency="EUR")], nav=150000.0)
    assert report.checks[0].value_base == 11000.0


def test_options_over_limit_breach_flagged():
    # 20,000 of 100,000 NAV = 20% > 10% -> not within policy.
    report = check_exposure([_option("AAPL", 20000.0)], nav=100000.0)
    assert report.checks[0].pct_of_nav == 0.20
    assert report.checks[0].within_policy is False


# --- existing-holding cap: no single name above position_cap % of NAV ---


def test_holding_over_cap_is_flagged_and_under_cap_names_are_not():
    # GOOGL at 10,800 of 120,000 NAV = 9% > the 6% cap -> flagged; AEHR at 2.5%
    # stays silent (only breaches surface, the digest must not list every name).
    positions = [_stock("GOOGL", 10800.0), _stock("AEHR", 3000.0)]

    report = check_exposure(positions, nav=120000.0, position_cap=0.06)

    (check,) = [c for c in report.checks if c.label.endswith("holding")]
    assert check.label == "GOOGL holding"
    assert check.pct_of_nav == pytest.approx(0.09)
    assert check.limit == 0.06
    assert check.within_policy is False


def test_holding_cap_aggregates_stock_and_options_per_symbol():
    # A "holding" is the whole name: GOOGL stock + GOOGL options together
    # (6,000 + 2,000 = 8,000 of 100,000 NAV = 8% > 6%).
    positions = [_stock("GOOGL", 6000.0), _option("GOOGL", 2000.0)]

    report = check_exposure(positions, nav=100000.0, position_cap=0.06)

    (check,) = [c for c in report.checks if c.label == "GOOGL holding"]
    assert check.value_base == pytest.approx(8000.0)
    assert check.pct_of_nav == pytest.approx(0.08)


def test_no_position_cap_requested_adds_no_holding_checks():
    report = check_exposure([_stock("GOOGL", 10800.0)], nav=120000.0)
    assert [c.label for c in report.checks] == ["options"]


# --- hedge ratio: put value / call value vs the policy band (his formula) ---


def test_hedge_ratio_within_band():
    # puts 1,200 / calls 10,000 = 12% -> inside the 10-15% band.
    positions = [_option("SPY", 10000.0), _option("SPY", 1200.0, right="P")]
    report = check_exposure(positions, nav=150000.0, hedge_low=0.10, hedge_high=0.15)
    assert report.hedge.ratio == pytest.approx(0.12)
    assert report.hedge.status == "within"


def test_all_calls_book_is_under_hedged():
    # His real failure mode: a book of calls with no puts -> 0% hedge, under band.
    report = check_exposure([_option("AAPL", 10000.0)], nav=150000.0,
                            hedge_low=0.10, hedge_high=0.15)
    assert report.hedge.ratio == 0.0
    assert report.hedge.status == "under"


def test_over_hedged_is_flagged():
    positions = [_option("SPY", 10000.0), _option("SPY", 2000.0, right="P")]
    report = check_exposure(positions, nav=150000.0, hedge_low=0.10, hedge_high=0.15)
    assert report.hedge.status == "over"


def test_no_calls_omits_the_hedge_check():
    # No denominator: a puts-only book gets no ratio rather than a division blowup.
    report = check_exposure([_option("SPY", 2000.0, right="P")], nav=150000.0,
                            hedge_low=0.10, hedge_high=0.15)
    assert report.hedge is None


def test_short_option_lines_count_by_absolute_value():
    # A short call (negative position_value) is still call-side exposure.
    positions = [
        _option("SPY", 10000.0), _option("ZETA", -1000.0), _option("SPY", 1200.0, right="P"),
    ]
    report = check_exposure(positions, nav=150000.0, hedge_low=0.10, hedge_high=0.15)
    assert report.hedge.call_value_base == 11000.0


def test_hedge_band_not_requested_reports_none():
    report = check_exposure([_option("AAPL", 12000.0)], nav=150000.0)
    assert report.hedge is None


def test_no_nav_refuses_with_missing_data():
    # Cold start: only a book uploaded, no statement -> refuse, don't imply 0%.
    result = check_exposure([_option("AAPL", 12000.0)], nav=None)
    assert isinstance(result, MissingData)
    assert result.store == "statement NAV"


@pytest.mark.skipif(
    not (BOOK.exists() and STMT.exists()), reason="real files not present"
)
def test_real_book_options_exposure_is_sane():
    positions = parse_tactical_book(BOOK.read_text(encoding="utf-8"))
    nav = parse_account_nav(STMT.read_text(encoding="utf-8"))

    report = check_exposure(positions, nav)

    assert isinstance(report, ExposureReport)
    assert report.nav == nav and nav > 0
    (options,) = report.checks
    # Structural, no hardcoded pct: a real fraction of NAV, within the 10% rule.
    assert 0.0 < options.pct_of_nav < options.limit
    assert options.within_policy is True
