"""Intake/scoper node: routing, entity extraction, and the one clarify round.

Offline: a stub structured model stands in for the gateway chat model, so the
graph's routing wiring is tested deterministically without the network. The
scoper's job is only to turn a message into a `Scope`; it must not answer.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.scope import make_scope_node
from app.graphs.trading_assistant.state import Scope


class StubModel:
    """Minimal chat-model stand-in: returns a canned Scope, records what it saw."""

    def __init__(self, scope: Scope) -> None:
        self._scope = scope
        self.seen: list | None = None

    def with_structured_output(self, schema):  # noqa: ANN001 - duck-typed seam
        assert schema is Scope
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.seen = messages
        return self._scope


def _run(scope: Scope, text: str = "hi") -> dict:
    # Test the scoper node in isolation — the full graph would run on to synthesis.
    node = make_scope_node(AgentContext(chat_model=StubModel(scope)))
    return node({"messages": [HumanMessage(content=text)]})


def test_scope_node_classifies_single_intent():
    result = _run(Scope(intents=["status_check"]))
    assert result["scope"].intents == ["status_check"]


def test_scope_node_is_multi_label_with_entities():
    result = _run(
        Scope(intents=["status_check", "desk_question"], tickers=["TSMC"]),
        text="am I within policy and what does the desk think of TSMC?",
    )
    assert set(result["scope"].intents) == {"status_check", "desk_question"}
    assert result["scope"].tickers == ["TSMC"]


def test_hypothetical_is_analysis_not_a_policy_write():
    result = _run(Scope(intents=["status_check"], hypothetical=True))
    assert result["scope"].hypothetical is True
    assert "policy_change" not in result["scope"].intents


def test_hypothetical_policy_change_is_stripped_to_analysis():
    # The observed real-model failure: hypothetical=True but still tagged a write.
    result = _run(Scope(intents=["status_check", "policy_change"], hypothetical=True))
    assert "policy_change" not in result["scope"].intents
    assert "status_check" in result["scope"].intents


def test_real_policy_change_is_preserved():
    result = _run(Scope(intents=["policy_change"], hypothetical=False))
    assert result["scope"].intents == ["policy_change"]


def test_scoper_prompt_separates_policy_reads_from_writes():
    # Observed misroute: "what's my current policy?" tagged policy_change (while
    # the model's own assumption said "not requesting a change"). The semantic
    # assertion belongs to the Task 5 routing rubric; this pins the prompt rule.
    from app.graphs.trading_assistant.scope import SCOPER_SYSTEM

    assert "never policy_change" in SCOPER_SYSTEM


def test_clarification_emits_one_assistant_question():
    question = "Do you mean your current open AAOI campaign or all of them?"
    result = _run(
        Scope(intents=["trade_history"], needs_clarification=True, clarifying_question=question),
        text="how's AAOI",
    )
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == question


def test_clarification_sets_the_pending_flag():
    result = _run(
        Scope(intents=["trade_history"], needs_clarification=True, clarifying_question="Which ticker?"),
        text="how's it going",
    )
    assert result["pending_clarification"] is True


def test_clarification_without_question_falls_back_instead_of_silence():
    # A model that sets the flag but returns no question must not end the turn
    # with no reply at all.
    result = _run(Scope(intents=["trade_history"], needs_clarification=True))
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content.endswith("?")


def test_second_clarify_round_is_forced_to_proceed():
    # ADR-0006: one clarification round MAX is a code guarantee, not a prompt
    # hope. After a round was already asked, an insistent model is overridden.
    node = make_scope_node(
        AgentContext(
            chat_model=StubModel(
                Scope(intents=["trade_history"], needs_clarification=True,
                      clarifying_question="Still which ticker?")
            )
        )
    )
    result = node(
        {
            "messages": [
                HumanMessage(content="how's it going"),
                AIMessage(content="Which ticker?"),
                HumanMessage(content="you know, the usual"),
            ],
            "pending_clarification": True,
        }
    )
    assert result["scope"].needs_clarification is False
    assert result["pending_clarification"] is False
    assert any("assum" in a.lower() for a in result["scope"].assumptions)
    assert "messages" not in result  # proceeds into the graph, no second question


def test_capabilities_question_is_answered_from_the_live_roster():
    # Observed failure: "you are exposed to tavily web search tool" hit the
    # off_topic refusal, and the agent's earlier tool-list answer was a guess.
    # The reply must come from the ACTUAL roster, in code.
    @tool
    def search_web(query: str) -> str:
        """Search the live web."""
        return ""

    node = make_scope_node(
        AgentContext(
            chat_model=StubModel(Scope(intents=["capabilities"])),
            agent_tools=[search_web],
        )
    )
    result = node({"messages": [HumanMessage(content="do you have web search?")]})
    (reply,) = result["messages"]
    assert "search_web" in reply.content


def test_capabilities_without_tools_says_so_instead_of_listing_nothing():
    result = _run(Scope(intents=["capabilities"]), text="what tools do you have?")
    (reply,) = result["messages"]
    assert "No live tools" in reply.content


def test_capabilities_mixed_with_real_route_proceeds_without_short_circuit():
    result = _run(Scope(intents=["capabilities", "status_check"]))
    assert "messages" not in result


def test_scoper_sees_system_prompt_and_user_message():
    stub = StubModel(Scope(intents=["off_topic"]))
    node = make_scope_node(AgentContext(chat_model=stub))
    node({"messages": [HumanMessage(content="what's the weather")]})
    assert isinstance(stub.seen[0], SystemMessage)
    assert isinstance(stub.seen[-1], HumanMessage)
    assert "intake router" in stub.seen[0].content


def test_scoper_view_filters_the_tool_loops_traffic():
    # The answering agent's tool loop lives on the same thread; the scoper must
    # see only human/assistant text, never tool calls or tool dumps.
    stub = StubModel(Scope(intents=["status_check"]))
    node = make_scope_node(AgentContext(chat_model=stub))
    node({
        "messages": [
            HumanMessage(content="how's the market?"),
            AIMessage(content="", tool_calls=[{"name": "get_market_quote",
                                               "args": {"symbols": ["SPY"]},
                                               "id": "c1", "type": "tool_call"}]),
            ToolMessage(content="SPY: $601.23", name="get_market_quote", tool_call_id="c1"),
            AIMessage(content="SPY is trading at $601.23."),
            HumanMessage(content="and my book?"),
        ]
    })
    seen = stub.seen[1:]  # after the system prompt
    assert all(not isinstance(m, ToolMessage) for m in seen)
    assert all(not getattr(m, "tool_calls", None) for m in seen)
    assert [m.content for m in seen] == [
        "how's the market?", "SPY is trading at $601.23.", "and my book?",
    ]
