"""Interrupt-gated policy writes: extraction, the approval gate, and the loop
back into exposure. Offline — a stub model answers the structured extraction and
a MemorySaver lets the interrupt pause and resume without the platform.
"""

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.graph import build_graph
from app.graphs.trading_assistant.policy_model import (
    DEFAULT_POLICY,
    PolicyChange,
    apply_change,
    format_value,
    normalize_value,
)
from app.graphs.trading_assistant.prefetch import make_prefetch_node
from app.graphs.trading_assistant.state import Scope
from app.trading.domain import MissingData, Position


# --- pure model ----------------------------------------------------------


def test_apply_change_sets_field_and_bumps_version():
    updated = apply_change(DEFAULT_POLICY, "options_limit", 0.12)
    assert updated.options_limit == 0.12
    assert updated.version == 2
    assert DEFAULT_POLICY.version == 1  # original untouched


def test_normalize_coerces_percent_as_integer():
    assert normalize_value("options_limit", 12) == 0.12  # nav_fraction
    assert normalize_value("scale_out_second", 200) == 2.0  # gain_ratio
    assert normalize_value("options_limit", 0.12) == 0.12  # already native


def test_format_value_by_kind():
    assert format_value("options_limit", 0.12) == "12.0%"
    assert format_value("scale_out_first", 1.0) == "+100%"


# --- stub model ----------------------------------------------------------


class _Stub:
    """Answers scope (Scope) and policy (PolicyChange) structured calls."""

    def __init__(self, *, scope=None, policy_change=None, answer="ok"):
        self._scope, self._pc, self._answer = scope, policy_change, answer

    def with_structured_output(self, schema):
        if schema is Scope:
            return _Return(self._scope)
        if schema is PolicyChange:
            return _Return(self._pc)
        raise AssertionError(f"unexpected schema {schema}")

    def invoke(self, messages):
        return AIMessage(content=self._answer)


class _Return:
    def __init__(self, value):
        self._value = value

    def invoke(self, messages):
        return self._value


def _ctx(stub, store):
    return AgentContext(
        chat_model=stub,
        load_policy=lambda u: store.get(u, DEFAULT_POLICY),
        save_policy=lambda u, p: store.__setitem__(u, p),
        default_user_id="alex",
    )


def _cap_change_graph(store, recognized=True):
    stub = _Stub(
        scope=Scope(intents=["policy_change"]),
        policy_change=PolicyChange(
            recognized=recognized, field="options_limit", new_value=0.12,
            summary="Set options exposure cap to 12%",
        ),
    )
    return build_graph(_ctx(stub, store), checkpointer=MemorySaver())


def _invoke(graph, text="raise my options cap to 12%"):
    cfg = {"configurable": {"thread_id": "t1"}}
    first = graph.invoke({"messages": [HumanMessage(content=text)], "user_id": "alex"}, cfg)
    return first, cfg


# --- interrupt gate ------------------------------------------------------


def test_policy_change_pauses_for_approval_before_writing():
    store: dict = {}
    graph = _cap_change_graph(store)
    first, _ = _invoke(graph)
    assert "__interrupt__" in first  # paused at the confirm gate
    assert store == {}  # nothing written before approval


def test_policy_change_writes_on_approval():
    store: dict = {}
    graph = _cap_change_graph(store)
    _, cfg = _invoke(graph)
    resumed = graph.invoke(Command(resume=True), cfg)
    assert store["alex"].options_limit == 0.12
    assert store["alex"].version == 2
    assert "Updated" in resumed["messages"][-1].content and "12.0%" in resumed["messages"][-1].content


def test_policy_change_rejected_leaves_policy_unchanged():
    store: dict = {}
    graph = _cap_change_graph(store)
    _, cfg = _invoke(graph)
    resumed = graph.invoke(Command(resume=False), cfg)
    assert store == {}  # no write
    assert "No change made" in resumed["messages"][-1].content


def test_unrecognized_change_asks_which_rule_without_interrupt():
    store: dict = {}
    graph = _cap_change_graph(store, recognized=False)
    first, _ = _invoke(graph, text="make my rules better")
    assert "__interrupt__" not in first
    assert "couldn't tell which rule" in first["messages"][-1].content


# --- co-intents: the non-policy half of the question is still answered ---


def test_policy_co_intents_continue_into_a_grounded_answer():
    # "Raise my options cap to 12% AND am I within policy?" — after the gate,
    # the status half must be answered, against the NEW limit.
    store: dict = {}
    stub = _Stub(
        scope=Scope(intents=["policy_change", "status_check"]),
        policy_change=PolicyChange(
            recognized=True, field="options_limit", new_value=0.12,
            summary="Set options exposure cap to 12%",
        ),
        answer="With the new cap you're within policy.",
    )
    ctx = AgentContext(
        chat_model=stub,
        load_positions=lambda u: [_opt_position()],
        load_trades=lambda u: MissingData("ledger", "Upload a recent activity statement."),
        load_nav=lambda u: 100_000.0,
        load_policy=lambda u: store.get(u, DEFAULT_POLICY),
        save_policy=lambda u, p: store.__setitem__(u, p),
        default_user_id="alex",
    )
    graph = build_graph(ctx, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    first = graph.invoke(
        {
            "messages": [HumanMessage(content="raise my cap to 12% — am I within policy?")],
            "user_id": "alex",
        },
        cfg,
    )
    assert "__interrupt__" in first
    resumed = graph.invoke(Command(resume=True), cfg)

    exposure = next(it for it in resumed["evidence"] if it.tool == "check_exposure")
    assert exposure.result.checks[0].limit == 0.12  # answered against the NEW cap
    replies = [m.content for m in resumed["messages"] if isinstance(m, AIMessage)]
    assert any("Updated" in r for r in replies)  # the write confirmation
    # The answer (plus the legitimate ledger upload-ask for this fixture).
    assert replies[-1].startswith("With the new cap you're within policy.")


def test_unrecognized_change_with_co_intents_still_answers():
    # The observed misroute: "what's my current policy?" tagged policy_change +
    # status_check. The gate can't parse a change (correctly), but that must not
    # dead-end the run — the status half still gets a grounded answer, and the
    # "which rule?" ask rides along as a note instead of replacing the answer.
    store: dict = {}
    stub = _Stub(
        scope=Scope(intents=["policy_change", "status_check"]),
        policy_change=PolicyChange(recognized=False),
        answer="All within policy.",
    )
    ctx = AgentContext(
        chat_model=stub,
        load_positions=lambda u: [_opt_position()],
        load_trades=lambda u: MissingData("ledger", "Upload a recent activity statement."),
        load_nav=lambda u: 100_000.0,
        load_policy=lambda u: store.get(u, DEFAULT_POLICY),
        save_policy=lambda u, p: store.__setitem__(u, p),
        default_user_id="alex",
    )
    graph = build_graph(ctx, checkpointer=MemorySaver())
    out = graph.invoke(
        {"messages": [HumanMessage(content="what's my current policy?")], "user_id": "alex"},
        {"configurable": {"thread_id": "t1"}},
    )
    assert "__interrupt__" not in out  # nothing to approve
    assert store == {}  # and certainly no write
    assert any(it.tool == "check_exposure" for it in out["evidence"])  # prefetch ran
    assert out["messages"][-1].content.startswith("All within policy.")  # answered
    assert "which rule" in (out.get("policy_note") or "")  # the ask rides along


def test_policy_only_intent_still_ends_after_the_gate():
    store: dict = {}
    graph = _cap_change_graph(store)
    _, cfg = _invoke(graph)
    resumed = graph.invoke(Command(resume=True), cfg)
    assert "evidence" not in resumed or not resumed["evidence"]  # no follow-on fetch


# --- loop closure: a cap change reaches the exposure check ---------------


def _opt_position():
    return Position(
        symbol="AAOI 16JAN26 40 C", asset_class="OPT", currency="USD",
        fx_rate_to_base=1.0, quantity=5, mark_price=4.0, position_value=7300.0,
        strike=40.0, expiry=date(2026, 1, 16), right="C",
    )


def test_updated_options_cap_flows_into_exposure():
    store = {"alex": apply_change(DEFAULT_POLICY, "options_limit", 0.06)}
    ctx = AgentContext(
        chat_model=object(),
        load_positions=lambda u: [_opt_position()],
        load_trades=lambda u: MissingData("ledger", "Upload a recent activity statement."),
        load_nav=lambda u: 100_000.0,
        load_policy=lambda u: store.get(u, DEFAULT_POLICY),
        default_user_id="alex",
    )
    out = make_prefetch_node(ctx)(
        {"scope": Scope(intents=["status_check"]), "messages": [HumanMessage(content="ok")], "user_id": "alex"}
    )
    exposure = next(it for it in out["evidence"] if it.tool == "check_exposure")
    assert exposure.result.checks[0].limit == 0.06  # 10% default overridden by the edit
    assert not exposure.result.checks[0].within_policy  # 7.3% now breaches a 6% cap
