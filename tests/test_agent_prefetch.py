"""Deterministic pre-fetch node: CSV computations only, threaded through the
evidence table with the cold-start `MissingData` contract.

Offline: data-access seams are plain lambdas over synthetic fixtures, so no
network is touched. These tests verify the WIRING (the book tools always run,
retrieval never runs here, and a never-uploaded store propagates) — the tools'
own maths are covered by their own suites.
"""

from datetime import date, datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.graph import build_graph
from app.graphs.trading_assistant.prefetch import make_prefetch_node
from app.graphs.trading_assistant.state import Scope
from app.trading.domain import MissingData, Position, Trade


# --- fixtures ------------------------------------------------------------


def _opt_position() -> Position:
    return Position(
        symbol="AAOI 16JAN26 40 C", asset_class="OPT", currency="USD",
        fx_rate_to_base=1.0, quantity=5, mark_price=4.0, position_value=2000.0,
        strike=40.0, expiry=date(2026, 1, 16), right="C",
    )


def _trade() -> Trade:
    return Trade(
        symbol="AAOI 16JAN26 40 C", root_ticker="AAOI",
        asset_category="Equity and Index Options", currency="USD",
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc), quantity=5, price=2.0,
        proceeds=-1000.0, commission=-1.0, basis=1001.0, realized_pl=0.0,
        mtm_pl=0.0, code="O",
    )


def _context(*, positions="present", trades="present", nav=100_000.0, as_of=None,
             entry_fallback=None, policy=None):
    pos = [_opt_position()] if positions == "present" else positions
    trd = [_trade()] if trades == "present" else trades
    return AgentContext(
        chat_model=object(),  # pre-fetch never calls the model
        load_positions=(lambda u: pos),
        load_trades=(lambda u: trd),
        load_nav=(lambda u: nav),
        load_as_of=(lambda u: as_of),
        load_entry_fallback=(lambda u: entry_fallback or {}),
        load_policy=(lambda u: policy) if policy is not None else None,
        default_user_id="alex",
    )


def _prefetch(scope: Scope, context: AgentContext, *, text="hi", user_id="alex") -> dict:
    node = make_prefetch_node(context)
    return node({"scope": scope, "messages": [HumanMessage(content=text)], "user_id": user_id})


def _by_tool(result: dict) -> dict:
    return {item.tool: item for item in result["evidence"]}


# --- portfolio route -----------------------------------------------------


def test_status_check_runs_all_portfolio_tools():
    out = _prefetch(Scope(intents=["status_check"]), _context())
    tools = _by_tool(out)
    assert {"list_positions", "check_exposure", "scan_scaleout", "open_position_pnl"} <= tools.keys()
    assert all(tools[t].ok for t in ("list_positions", "check_exposure", "scan_scaleout"))
    assert out["missing"] == []


def test_missing_snapshot_blocks_every_portfolio_tool():
    ctx = _context(positions=MissingData("positions snapshot", "Upload your tactical book export."))
    out = _prefetch(Scope(intents=["rebalance_advice"]), ctx)
    tools = _by_tool(out)
    for t in ("list_positions", "check_exposure", "scan_scaleout", "open_position_pnl"):
        assert tools[t].ok is False
        assert tools[t].missing.store == "positions snapshot"
    assert [m.store for m in out["missing"]] == ["positions snapshot"]


def test_missing_ledger_blocks_scaleout_and_pnl_but_not_exposure():
    ctx = _context(trades=MissingData("ledger", "Upload a recent activity statement."))
    out = _prefetch(Scope(intents=["status_check"]), ctx)
    tools = _by_tool(out)
    assert tools["check_exposure"].ok is True  # exposure needs only snapshot + NAV
    assert tools["scan_scaleout"].ok is False and tools["scan_scaleout"].missing.store == "ledger"
    assert tools["open_position_pnl"].ok is False
    assert [m.store for m in out["missing"]] == ["ledger"]


def test_prefetch_prices_orphan_holdings_via_the_statement_cost_fallback():
    # A held stock with no opening fill in the YTD ledger (the real GOOGL
    # shape, bought pre-window): the wired Open-Positions cost map must price
    # its P/L line instead of silently skipping it (which flipped the book's
    # P/L sign). It must NOT reach the scale-out scan: the ladder is
    # options-only; a doubled stock answers to the holding cap instead.
    googl = Position(
        symbol="GOOGL", asset_class="STK", currency="USD", fx_rate_to_base=1.0,
        quantity=30, mark_price=359.91, position_value=10797.3,
    )
    ctx = _context(
        positions=[googl],
        entry_fallback={("GOOGL", None, None, None): 176.886666667},
    )

    out = _prefetch(Scope(intents=["status_check"]), ctx)

    tools = _by_tool(out)
    (line,) = tools["open_position_pnl"].result.lines
    assert line.symbol == "GOOGL"
    assert line.cost_basis_source == "statement"
    assert tools["scan_scaleout"].result == []


def test_statement_as_of_lands_in_evidence():
    # The statement's Period end dates every figure it backs; the digest needs
    # it so "how is my book doing" answers can say WHEN the data is from.
    out = _prefetch(
        Scope(intents=["status_check"]), _context(as_of=date(2026, 7, 3))
    )
    item = _by_tool(out)["statement_as_of"]
    assert item.ok and item.result == date(2026, 7, 3)


def test_no_as_of_adds_no_evidence_item():
    out = _prefetch(Scope(intents=["status_check"]), _context())
    assert "statement_as_of" not in _by_tool(out)


def test_prefetch_applies_the_existing_holding_cap():
    # The fixture's one AAOI holding is 2,000 over a 20,000 NAV = 10%, above the
    # policy's 6% existing-holding cap -> a holding breach lands in the report.
    out = _prefetch(Scope(intents=["status_check"]), _context(nav=20_000.0))
    report = _by_tool(out)["check_exposure"].result
    breaches = [c for c in report.checks if c.label.endswith("holding")]
    assert breaches and all(not c.within_policy for c in breaches)


def test_missing_nav_makes_exposure_report_missing_data():
    out = _prefetch(Scope(intents=["status_check"]), _context(nav=None))
    tools = _by_tool(out)
    assert tools["check_exposure"].ok is False
    assert tools["check_exposure"].missing.store == "statement NAV"


# --- desk route: retrieval is the answering agent's tool, never a pre-fetch ---


def test_prefetch_threads_policy_triggers_into_the_scaleout_scan():
    # Both rung thresholds were once hardcoded in scaleout.py — an approved
    # policy edit silently changed nothing. A +100% winner with no sales
    # recorded (joinable: position symbol is the ROOT, as in the real tactical
    # book) must stop flagging under a raised first trigger, and with one sale
    # recorded a lowered second trigger must flag the second tranche.
    from dataclasses import replace

    from app.graphs.trading_assistant.policy_model import DEFAULT_POLICY, apply_change
    from app.trading.domain import ScaleOutSignal

    winner = Position(
        symbol="AAOI", asset_class="OPT", currency="USD", fx_rate_to_base=1.0,
        quantity=5, mark_price=4.0, position_value=2000.0,
        strike=40.0, expiry=date(2026, 1, 16), right="C",
    )

    raised = apply_change(DEFAULT_POLICY, "scale_out_first", 1.2)
    out = _prefetch(
        Scope(intents=["status_check"]), _context(positions=[winner], policy=raised)
    )
    assert _by_tool(out)["scan_scaleout"].result == []

    scale_sale = replace(
        _trade(), quantity=-1.0, price=3.0, proceeds=300.0,
        timestamp=datetime(2026, 6, 10, tzinfo=timezone.utc), code="C",
    )
    lowered = apply_change(DEFAULT_POLICY, "scale_out_second", 0.8)
    out = _prefetch(
        Scope(intents=["status_check"]),
        _context(positions=[winner], trades=[_trade(), scale_sale], policy=lowered),
    )
    (candidate,) = _by_tool(out)["scan_scaleout"].result
    assert candidate.signal is ScaleOutSignal.SECOND_TRANCHE_DUE
    assert candidate.scales_taken == 1


def test_prefetch_always_carries_the_policy_rulebook():
    # The rulebook is state, not an upload — it exists from day one (seeded
    # defaults) and rides in the evidence on EVERY route, so a policy read is
    # answerable and every cited rule value is audit-backed.
    out = _prefetch(Scope(intents=["desk_question"]), _context())
    item = _by_tool(out)["policy_rules"]
    assert item.ok
    assert item.result.options_limit == 0.10
    assert item.result.version == 1


def test_prefetch_never_retrieves_desk_reviews():
    for intents in (["desk_question"], ["daily_briefing"], ["desk_question", "status_check"]):
        out = _prefetch(Scope(intents=intents), _context())
        assert "search_desk_reviews" not in _by_tool(out)


# --- history route -------------------------------------------------------


def test_trade_history_fetches_campaign_per_ticker():
    out = _prefetch(Scope(intents=["trade_history"], tickers=["AAOI"]), _context())
    assert _by_tool(out)["get_trades:AAOI"].ok is True


def test_trade_history_without_ticker_asks_which_position():
    out = _prefetch(Scope(intents=["trade_history"]), _context())
    item = _by_tool(out)["get_trades"]
    assert item.ok is False and "ticker" in item.note.lower()


def test_trade_history_missing_ledger():
    ctx = _context(trades=MissingData("ledger", "Upload a recent activity statement."))
    out = _prefetch(Scope(intents=["trade_history"], tickers=["AAOI"]), ctx)
    assert _by_tool(out)["get_trades"].missing.store == "ledger"


# --- unconditional book fetch & intent-scoped upload asks ----------------


def test_desk_only_question_still_carries_book_evidence():
    # The book tools are milliseconds and always run, whatever the route.
    out = _prefetch(Scope(intents=["desk_question"]), _context())
    assert "check_exposure" in _by_tool(out)


def test_desk_only_question_hides_book_upload_asks():
    # A no-uploads user asking a pure desk question must not be nagged about
    # the statement/export the answer doesn't need; the evidence still records
    # the blocked tools (cold-start truth), only the asks are scoped.
    ctx = _context(
        positions=MissingData("positions snapshot", "Upload your tactical book export."),
        trades=MissingData("ledger", "Upload a recent activity statement."),
    )
    out = _prefetch(Scope(intents=["desk_question"]), ctx)
    assert _by_tool(out)["check_exposure"].ok is False  # still recorded
    assert out["missing"] == []  # but no book nags for a desk question


def test_market_regime_shows_no_upload_asks():
    ctx = _context(
        positions=MissingData("positions snapshot", "Upload your tactical book export."),
        trades=MissingData("ledger", "Upload a recent activity statement."),
    )
    out = _prefetch(Scope(intents=["market_regime"]), ctx)
    assert out["missing"] == []


def test_named_tickers_fetch_campaigns_regardless_of_intent():
    # "Am I within policy on AAOI?" is a status question, but the named ticker
    # means the campaign context should be on the table too.
    out = _prefetch(Scope(intents=["status_check"], tickers=["AAOI"]), _context())
    assert _by_tool(out)["get_trades:AAOI"].ok is True


# --- multi-label union & routing ----------------------------------------


def test_multi_label_unions_the_upload_asks():
    ctx = _context(
        positions=MissingData("positions snapshot", "Upload your tactical book export."),
        trades=MissingData("ledger", "Upload a recent activity statement."),
    )
    out = _prefetch(Scope(intents=["performance_review", "status_check"]), ctx)
    assert {m.store for m in out["missing"]} == {"positions snapshot", "ledger"}


class _StubModel:
    """Answers the scoper's structured call (Scope) and synthesis's free-text call."""

    def __init__(self, scope, answer="Looks fine."):
        self._scope = scope
        self._answer = answer

    def with_structured_output(self, schema):
        return _ScopeInvoker(self._scope)

    def invoke(self, messages):
        return AIMessage(content=self._answer)


class _ScopeInvoker:
    def __init__(self, scope):
        self._scope = scope

    def invoke(self, messages):
        return self._scope


def _graph_context(scope: Scope) -> AgentContext:
    base = _context()
    return AgentContext(
        chat_model=_StubModel(scope),
        load_positions=base.load_positions,
        load_trades=base.load_trades,
        load_nav=base.load_nav,
        default_user_id="alex",
    )


def test_graph_runs_prefetch_for_actionable_intent():
    ctx = _graph_context(Scope(intents=["status_check"]))
    result = build_graph(ctx).invoke({"messages": [HumanMessage(content="am I ok?")], "user_id": "alex"})
    assert "evidence" in result and result["evidence"]


def test_graph_skips_prefetch_for_off_topic():
    ctx = _graph_context(Scope(intents=["off_topic"]))
    result = build_graph(ctx).invoke({"messages": [HumanMessage(content="weather?")], "user_id": "alex"})
    assert not result.get("evidence")  # the per-turn reset leaves an empty table
