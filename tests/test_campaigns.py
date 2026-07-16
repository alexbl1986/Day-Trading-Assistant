"""Seam: group_campaigns(trades) -> list[Campaign].

Groups a ledger's fills into campaigns — a campaign is a continuously-open
run of fills in one contract (same raw symbol), from first open until net
quantity returns to zero. Rolls (different expiry = different symbol) do not
chain in v1.
"""

from datetime import datetime

from app.trading.ledger import group_campaigns
from app.trading.domain import Trade


def _fill(symbol, qty, realized=0.0, when=None, **kw):
    """A minimal Trade for campaign tests; only the fields grouping reads."""
    return Trade(
        symbol=symbol,
        root_ticker=symbol.split()[0],
        asset_category=kw.get("asset_category", "Stocks"),
        currency=kw.get("currency", "USD"),
        timestamp=when or datetime(2026, 1, 1, 9, 30, 0),
        quantity=float(qty),
        price=kw.get("price", 1.0),
        proceeds=kw.get("proceeds", 0.0),
        commission=kw.get("commission", 0.0),
        basis=kw.get("basis", 0.0),
        realized_pl=float(realized),
        mtm_pl=0.0,
        code=kw.get("code", "O"),
    )


def test_scaling_into_one_contract_is_a_single_open_campaign():
    trades = [
        _fill("P4O", 130, when=datetime(2026, 1, 2, 9, 30)),
        _fill("P4O", 100, when=datetime(2026, 1, 5, 10, 0)),
    ]

    campaigns = group_campaigns(trades)

    assert len(campaigns) == 1
    c = campaigns[0]
    assert c.symbol == "P4O"
    assert c.net_quantity == 230.0
    assert c.is_open is True
    assert len(c.fills) == 2


def test_close_then_reopen_same_contract_is_two_campaigns():
    # Real P4O pattern: buy 130, sell 130 (round-trip closed), buy 100 (new run).
    # The zero-crossing ends campaign 1; the reopen starts campaign 2.
    trades = [
        _fill("P4O", 130, when=datetime(2026, 1, 2, 9, 30)),
        _fill("P4O", -130, when=datetime(2026, 1, 3, 9, 30)),
        _fill("P4O", 100, when=datetime(2026, 1, 9, 9, 30)),
    ]

    campaigns = group_campaigns(trades)

    assert len(campaigns) == 2
    closed, active = campaigns
    assert closed.net_quantity == 0.0
    assert closed.is_open is False
    assert len(closed.fills) == 2
    assert active.net_quantity == 100.0
    assert active.is_open is True
    assert len(active.fills) == 1


def test_campaign_realized_pl_sums_per_fill_column():
    # Realized P/L comes from IBKR's own per-fill column, not a FIFO engine.
    trades = [
        _fill("P4O", 130, realized=0.0, when=datetime(2026, 1, 2, 9, 30)),
        _fill("P4O", -130, realized=-390.2, when=datetime(2026, 1, 3, 9, 30)),
        _fill("P4O", 100, realized=0.0, when=datetime(2026, 1, 9, 9, 30)),
    ]

    closed, active = group_campaigns(trades)

    assert closed.realized_pl == -390.2
    assert active.realized_pl == 0.0


def test_different_expiries_of_one_underlying_do_not_chain():
    # Two option contracts on AAOI (a "roll"): same underlying, different
    # symbols → two separate campaigns in v1, both tagged root_ticker AAOI.
    trades = [
        _fill("AAOI 16JAN26 40 C", 2, asset_category="Equity and Index Options"),
        _fill("AAOI 06FEB26 50 C", 3, asset_category="Equity and Index Options"),
    ]

    campaigns = group_campaigns(trades)

    assert len(campaigns) == 2
    assert {c.symbol for c in campaigns} == {"AAOI 16JAN26 40 C", "AAOI 06FEB26 50 C"}
    assert {c.root_ticker for c in campaigns} == {"AAOI"}


def test_open_campaign_is_house_money_once_cash_extracted_covers_entry():
    # the trader's rule (from his transcript): cash extracted from sales >= cash paid
    # on entries, with a runner still open. Here: paid 582 in, pulled 600 out,
    # 1 contract still open -> 100% house money.
    trades = [
        _fill("FAKE", 2, proceeds=-582.0, when=datetime(2026, 6, 24, 9, 30),
              asset_category="Equity and Index Options"),
        _fill("FAKE", -1, proceeds=600.0, when=datetime(2026, 6, 25, 9, 30),
              asset_category="Equity and Index Options"),
    ]

    campaign = group_campaigns(trades)[0]

    assert campaign.is_open is True
    assert campaign.house_money is True


def test_open_campaign_not_house_money_before_recouping_entry_cash():
    # Only entries so far: nothing extracted, capital still at risk.
    trades = [
        _fill("FAKE", 2, proceeds=-582.0, when=datetime(2026, 6, 24, 9, 30),
              asset_category="Equity and Index Options"),
    ]

    campaign = group_campaigns(trades)[0]

    assert campaign.house_money is False


def test_closed_campaign_is_never_house_money():
    # A fully-closed round-trip is realized P/L, not a house-money runner.
    trades = [
        _fill("FAKE", 2, proceeds=-582.0, when=datetime(2026, 6, 24, 9, 30)),
        _fill("FAKE", -2, proceeds=1200.0, when=datetime(2026, 6, 25, 9, 30)),
    ]

    campaign = group_campaigns(trades)[0]

    assert campaign.is_open is False
    assert campaign.house_money is False


def test_avg_entry_price_uses_only_opening_direction_fills():
    # FLEX active campaign: opened at 2.91, then a profit-taking sell. The
    # sell must not pull the entry price around.
    trades = [
        _fill("FLEX", 2, price=2.91, when=datetime(2026, 6, 24, 9, 30)),
        _fill("FLEX", -1, price=6.00, when=datetime(2026, 6, 25, 9, 30)),
    ]

    assert group_campaigns(trades)[0].avg_entry_price == 2.91


def test_avg_entry_price_is_quantity_weighted_over_scale_ins():
    # Scale-in: 2 @ 2.00 then 3 @ 3.00 -> (2*2 + 3*3) / 5 = 2.6.
    trades = [
        _fill("FAKE", 2, price=2.00, when=datetime(2026, 1, 2, 9, 30)),
        _fill("FAKE", 3, price=3.00, when=datetime(2026, 1, 3, 9, 30)),
    ]

    assert group_campaigns(trades)[0].avg_entry_price == 2.6
