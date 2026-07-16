"""Deterministic audit: figure extraction, backing check, the bounce loop.

Offline throughout. The graph end-to-end test uses a dual stub that answers both
the scoper's structured call and the synthesis free-text call, so the audit's
one-bounce-then-banner behavior is exercised without the network.
"""

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from app.graphs.trading_assistant.audit import (
    audit_answer,
    extract_figures,
    make_audit_node,
)
from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.graph import build_graph
from app.graphs.trading_assistant.state import Scope, SynthesisResult
from app.trading.domain import MissingData, Position


# --- figure extraction ---------------------------------------------------


def test_extract_only_dollar_and_percent_figures():
    figs = extract_figures("options at 7.3% of NAV, entry $2.00, strike 40, year 2026")
    assert figs == [("%", 7.3), ("$", 2.0)]  # 40 and 2026 are bare -> ignored


def test_extract_handles_thousands_and_negatives():
    assert ("$", 14700.0) in extract_figures("NAV is $14,700")
    assert ("$", -1200.0) in extract_figures("realized -$1,200 so far")


def test_extract_handles_sign_after_dollar():
    # The digest's own formatter historically emitted "$-20,186"; the extractor
    # must parse it, or every negative dollar in the grounding is unauditable.
    assert ("$", -20186.0) in extract_figures("total unrealized: $-20,186 across 60")


# --- backing check -------------------------------------------------------


def test_audit_passes_when_every_figure_is_grounded():
    grounding = "options: 7.3% of NAV (limit 10%)\naccount NAV: $120,000"
    result = audit_answer("You're at 7.3% of NAV, within your 10% limit.", grounding)
    assert result.ok and result.violations == ()


def test_audit_flags_a_fabricated_figure():
    grounding = "options: 7.3% of NAV (limit 10%)"
    result = audit_answer("Your NAV is $999,999 and options are 7.3%.", grounding)
    assert not result.ok
    assert result.violations == ("$999,999.00",)


def test_audit_tolerates_rounding_within_evidence():
    # $14,698 backs "$14,700" (tight rounding); a desk "25%" quote backs a 25% mention.
    grounding = "total unrealized: $14,698\ndesk notes 25% growth"
    assert audit_answer("up ~$14,700, desk sees 25%.", grounding).ok


def test_audit_backs_prose_that_drops_the_sign():
    # Trace-replay regression: "down $20,186" was flagged as fabricated although
    # the digest carried the figure as -$20,186. Prose legitimately drops signs
    # ("a loss of $X"), so dollars match on absolute value.
    grounding = "### Unrealized P/L\n- total unrealized: -$20,186 across 60 priced positions"
    assert audit_answer("You're down $20,186 overall.", grounding).ok
    assert audit_answer("unrealized P/L is -$20,186.", grounding).ok


def test_audit_rejects_loose_rounding():
    # 2% tolerance let "$165,000" pass against a $162,528 NAV; 0.5% must not.
    assert not audit_answer("NAV is about $165,000.", "account NAV: $162,528").ok


# --- digest/extractor contract -------------------------------------------


def test_every_digest_figure_is_extractable():
    # The audit can only back what it can parse: every $ the digest formatter
    # emits (negative P/L included) must round-trip through extract_figures.
    from app.graphs.trading_assistant.answer import build_facts
    from app.graphs.trading_assistant.state import EvidenceItem
    from app.trading.domain import PositionPnL, UnrealizedPnLReport

    report = UnrealizedPnLReport(
        total_unrealized_pl=-20186.39,
        lines=(
            PositionPnL(
                symbol="SMTOY", avg_entry_price=88.53, mark_price=8.22,
                gain=-0.907, unrealized_pl=-19274.4,
            ),
        ),
    )
    digest, _ = build_facts(
        [EvidenceItem("open_position_pnl", ok=True, result=report)], []
    )
    assert ("$", -20186.0) in extract_figures(digest)


# --- audit node ----------------------------------------------------------


def _ctx():
    return AgentContext(chat_model=object())  # audit never calls the model


def test_audit_node_delivers_clean_answer():
    state = {
        "synthesis": SynthesisResult("You're at 7.3% of NAV.", "options: 7.3% of NAV", ()),
        "synthesis_attempts": 1,
    }
    out = make_audit_node(_ctx())(state)
    assert out["audit"].ok
    assert out["messages"][-1].content == "You're at 7.3% of NAV."


def test_audit_node_bounces_once_with_feedback():
    state = {
        "synthesis": SynthesisResult("NAV is $999,999.", "options: 7.3% of NAV", ()),
        "synthesis_attempts": 1,
    }
    out = make_audit_node(_ctx())(state)
    assert not out["audit"].ok
    assert "messages" not in out  # bounced, not delivered
    assert "$999,999" in out["audit_feedback"]


def test_audit_node_delivers_with_banner_after_max_attempts():
    state = {
        "synthesis": SynthesisResult("NAV is $999,999.", "options: 7.3% of NAV", ()),
        "synthesis_attempts": 2,
    }
    out = make_audit_node(_ctx())(state)
    assert not out["audit"].ok
    assert "⚠️" in out["messages"][-1].content and "$999,999" in out["messages"][-1].content


# --- graph end-to-end: the bounce loop ----------------------------------


class _DualStub:
    """Answers the scoper's structured call and synthesis's free-text call."""

    def __init__(self, scope: Scope, answer: str):
        self._scope, self._answer = scope, answer

    def with_structured_output(self, schema):
        return _ScopeInvoker(self._scope)

    def invoke(self, messages):
        return AIMessage(content=self._answer)


class _ScopeInvoker:
    def __init__(self, scope):
        self._scope = scope

    def invoke(self, messages):
        return self._scope


def _position():
    # options value 7,300 over a 100,000 NAV -> exactly 7.3% of NAV.
    return Position(
        symbol="AAOI 16JAN26 40 C", asset_class="OPT", currency="USD",
        fx_rate_to_base=1.0, quantity=5, mark_price=4.0, position_value=7300.0,
        strike=40.0, expiry=date(2026, 1, 16), right="C",
    )


def _graph(answer: str):
    ctx = AgentContext(
        chat_model=_DualStub(Scope(intents=["status_check"]), answer),
        load_positions=lambda u: [_position()],
        load_trades=lambda u: MissingData("ledger", "Upload a recent activity statement."),
        load_nav=lambda u: 100_000.0,
        default_user_id="alex",
    )
    return build_graph(ctx)


def test_graph_delivers_grounded_status_answer():
    result = _graph("You're at 7.3% of NAV, within your 10% limit.").invoke(
        {"messages": [HumanMessage(content="am I within policy?")], "user_id": "alex"}
    )
    assert result["audit"].ok
    assert "7.3%" in result["messages"][-1].content


def test_graph_bounces_then_banners_on_persistent_hallucination():
    result = _graph("Your options are 42% of NAV and NAV is $5,000,000.").invoke(
        {"messages": [HumanMessage(content="am I within policy?")], "user_id": "alex"}
    )
    assert result["synthesis_attempts"] == 2  # one bounce happened
    assert not result["audit"].ok
    assert "⚠️" in result["messages"][-1].content
