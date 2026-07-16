"""Cross-turn state hygiene: per-turn keys must reset at intake.

A LangGraph server thread persists state across runs, so anything derived from
one turn (repair-attempt counter, audit feedback, the pending policy proposal)
leaks into the next turn unless the first node clears it. Trace-proven failure:
turn 2 started with `synthesis_attempts` already at the cap (no repair budget)
and injected turn 1's audit feedback into an unrelated prompt. Offline — stubs
answer both structured calls; a MemorySaver provides the thread.
"""

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.graph import build_graph
from app.graphs.trading_assistant.policy_model import DEFAULT_POLICY, PolicyChange
from app.graphs.trading_assistant.state import Scope
from app.trading.domain import MissingData, Position

_CFG = {"configurable": {"thread_id": "t1"}}


class _Return:
    def __init__(self, value):
        self._value = value

    def invoke(self, messages):
        return self._value


class _PopReturn:
    def __init__(self, values):
        self._values = list(values)

    def invoke(self, messages):
        return self._values.pop(0)


class _ScriptedStub:
    """Fixed Scope for every scoper call; scripted free-text synthesis answers
    (popped in order) with each synthesis prompt captured for assertions;
    optional scripted PolicyChange sequence."""

    def __init__(self, scope, answers=(), policy_changes=()):
        self._scope = scope
        self._answers = list(answers)
        self._policy_changes = list(policy_changes)
        self.synthesis_prompts: list[str] = []

    def with_structured_output(self, schema):
        if schema is Scope:
            return _Return(self._scope)
        if schema is PolicyChange:
            return _PopReturn(self._policy_changes)
        raise AssertionError(f"unexpected schema {schema}")

    def invoke(self, messages):
        self.synthesis_prompts.append(messages[-1].content)
        return AIMessage(content=self._answers.pop(0))


def _position():
    # options value 7,300 over a 100,000 NAV -> exactly 7.3% of NAV.
    return Position(
        symbol="AAOI 16JAN26 40 C", asset_class="OPT", currency="USD",
        fx_rate_to_base=1.0, quantity=5, mark_price=4.0, position_value=7300.0,
        strike=40.0, expiry=date(2026, 1, 16), right="C",
    )


def _status_graph(stub):
    ctx = AgentContext(
        chat_model=stub,
        load_positions=lambda u: [_position()],
        load_trades=lambda u: MissingData("ledger", "Upload a recent activity statement."),
        load_nav=lambda u: 100_000.0,
        default_user_id="alex",
    )
    return build_graph(ctx, checkpointer=MemorySaver())


def test_attempts_and_feedback_reset_between_turns():
    stub = _ScriptedStub(
        Scope(intents=["status_check"]),
        answers=[
            "NAV is $999,999.",           # turn 1, attempt 1 -> audit bounce
            "You're at 7.3% of NAV.",     # turn 1, attempt 2 -> delivered
            "You're at 7.3% of NAV.",     # turn 2, attempt 1 -> must be a fresh start
        ],
    )
    graph = _status_graph(stub)

    turn1 = graph.invoke(
        {"messages": [HumanMessage(content="am I within policy?")], "user_id": "alex"}, _CFG
    )
    assert turn1["synthesis_attempts"] == 2  # the bounce happened in turn 1

    turn2 = graph.invoke(
        {"messages": [HumanMessage(content="am I within policy now?")], "user_id": "alex"}, _CFG
    )
    assert turn2["synthesis_attempts"] == 1  # fresh repair budget, not 3
    assert turn2.get("audit_feedback") is None  # turn 1's feedback did not survive
    assert "AUDIT FEEDBACK" not in stub.synthesis_prompts[2]  # ...nor reach the prompt
    assert turn2["audit"].ok


def test_stale_policy_proposal_never_reaches_the_interrupt():
    stub = _ScriptedStub(
        Scope(intents=["policy_change"]),
        policy_changes=[
            PolicyChange(recognized=True, field="options_limit", new_value=0.12,
                         summary="Set options exposure cap to 12%"),
            PolicyChange(recognized=False),  # turn 2: can't map the request
        ],
    )
    store: dict = {}
    ctx = AgentContext(
        chat_model=stub,
        load_policy=lambda u: store.get(u, DEFAULT_POLICY),
        save_policy=lambda u, p: store.__setitem__(u, p),
        default_user_id="alex",
    )
    graph = build_graph(ctx, checkpointer=MemorySaver())

    first = graph.invoke(
        {"messages": [HumanMessage(content="raise my options cap to 12%")], "user_id": "alex"}, _CFG
    )
    assert "__interrupt__" in first
    approved = graph.invoke(Command(resume=True), _CFG)
    assert store["alex"].options_limit == 0.12
    assert "Updated" in approved["messages"][-1].content

    # Turn 2 is unrecognized: it must ask which rule, NOT re-fire the interrupt
    # with turn 1's leftover proposal.
    second = graph.invoke(
        {"messages": [HumanMessage(content="make my rules better")], "user_id": "alex"}, _CFG
    )
    assert "__interrupt__" not in second
    assert "couldn't tell which rule" in second["messages"][-1].content
    assert store["alex"].version == 2  # no second write happened
