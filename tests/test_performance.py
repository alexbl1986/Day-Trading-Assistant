"""Seam: performance_summary(trades) -> PerformanceReport.

Tier-1 realized attribution, entirely from IBKR's own signed Realized P/L and
commission columns — never recomputed from fills. Win rate counts CLOSED
campaigns only (an open runner isn't a win yet). Synthetic fixtures throughout.
"""

from datetime import datetime

import pytest

from app.trading.domain import MissingData, Trade
from app.trading.performance import performance_summary


def _fill(symbol, qty, realized=0.0, when=None, commission=0.0):
    return Trade(
        symbol=symbol,
        root_ticker=symbol.split()[0],
        asset_category="Stocks",
        currency="USD",
        timestamp=when or datetime(2026, 1, 15, 9, 30, 0),
        quantity=float(qty),
        price=1.0,
        proceeds=0.0,
        commission=float(commission),
        basis=0.0,
        realized_pl=float(realized),
        mtm_pl=0.0,
        code="O",
    )


def test_total_and_monthly_realized_pl():
    trades = [
        _fill("AAOI", 5, realized=600.0, when=datetime(2026, 1, 10, 10, 0)),
        _fill("AAOI", -5, realized=0.0, when=datetime(2026, 1, 20, 10, 0)),
        _fill("NVDA", 10, realized=-150.0, when=datetime(2026, 2, 5, 10, 0)),
        _fill("NVDA", -10, realized=0.0, when=datetime(2026, 2, 6, 10, 0)),
    ]
    report = performance_summary(trades)
    assert report.total_realized_pl == pytest.approx(450.0)
    assert report.by_month == (("2026-01", 600.0), ("2026-02", -150.0))


def test_win_rate_counts_closed_campaigns_only():
    trades = [
        # AAOI: closed at a profit.
        _fill("AAOI", 5, when=datetime(2026, 1, 2)),
        _fill("AAOI", -5, realized=600.0, when=datetime(2026, 1, 9)),
        # NVDA: closed at a loss.
        _fill("NVDA", 10, when=datetime(2026, 1, 3)),
        _fill("NVDA", -10, realized=-150.0, when=datetime(2026, 1, 10)),
        # TER: still open — must not count either way.
        _fill("TER", 6, when=datetime(2026, 1, 4)),
    ]
    report = performance_summary(trades)
    assert report.closed_campaigns == 2
    assert report.winning_campaigns == 1
    assert report.win_rate == pytest.approx(0.5)


def test_win_rate_is_none_with_no_closed_campaigns():
    report = performance_summary([_fill("TER", 6)])
    assert report.closed_campaigns == 0
    assert report.win_rate is None


def test_winners_and_losers_ranked_by_realized():
    trades = [
        _fill("AAOI", -5, realized=600.0),
        _fill("VECO", -5, realized=200.0),
        _fill("NVDA", -5, realized=-150.0),
        _fill("SMTOY", -5, realized=-900.0),
        _fill("FLAT", -5, realized=0.0),  # zero: neither winner nor loser
    ]
    report = performance_summary(trades)
    assert [t.root_ticker for t in report.top_winners] == ["AAOI", "VECO"]
    assert [t.root_ticker for t in report.top_losers] == ["SMTOY", "NVDA"]


def test_commission_drag_totalled_as_reported():
    trades = [_fill("AAOI", 5, commission=-1.5), _fill("AAOI", -5, commission=-2.0)]
    assert performance_summary(trades).commission_total == pytest.approx(-3.5)


# --- prefetch wiring ------------------------------------------------------


def test_prefetch_runs_performance_unconditionally_and_not_snapshot_blocked():
    # Performance needs only the ledger: a missing SNAPSHOT must not block it.
    from langchain_core.messages import HumanMessage

    from app.graphs.trading_assistant.deps import AgentContext
    from app.graphs.trading_assistant.prefetch import make_prefetch_node
    from app.graphs.trading_assistant.state import Scope

    ctx = AgentContext(
        chat_model=object(),
        load_positions=lambda u: MissingData("positions snapshot", "Upload your book export."),
        load_trades=lambda u: [_fill("AAOI", -5, realized=600.0)],
        load_nav=lambda u: 100_000.0,
        default_user_id="alex",
    )
    out = make_prefetch_node(ctx)(
        {"scope": Scope(intents=["performance_review"]),
         "messages": [HumanMessage(content="how am I doing this year?")], "user_id": "alex"}
    )
    items = {it.tool: it for it in out["evidence"]}
    assert items["performance_summary"].ok is True
    assert items["performance_summary"].result.total_realized_pl == pytest.approx(600.0)


def test_prefetch_blocks_performance_on_missing_ledger():
    from langchain_core.messages import HumanMessage

    from app.graphs.trading_assistant.deps import AgentContext
    from app.graphs.trading_assistant.prefetch import make_prefetch_node
    from app.graphs.trading_assistant.state import Scope

    ctx = AgentContext(
        chat_model=object(),
        load_positions=lambda u: [],
        load_trades=lambda u: MissingData("ledger", "Upload a recent activity statement."),
        load_nav=lambda u: 100_000.0,
        default_user_id="alex",
    )
    out = make_prefetch_node(ctx)(
        {"scope": Scope(intents=["performance_review"]),
         "messages": [HumanMessage(content="win rate?")], "user_id": "alex"}
    )
    items = {it.tool: it for it in out["evidence"]}
    assert items["performance_summary"].ok is False
    assert items["performance_summary"].missing.store == "ledger"
    # performance_review NEEDS the ledger -> the upload ask must surface.
    assert [m.store for m in out["missing"]] == ["ledger"]
