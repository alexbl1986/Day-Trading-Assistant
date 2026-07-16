"""daily_briefing: deterministic book pre-fetch + a mandatory desk search by the
answering agent, with intent-composed output headings. Offline with fakes.
"""

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from app.graphs.trading_assistant.answer import (
    BRIEFING_QUERY,
    make_answer_node,
)
from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.graph import build_graph
from app.graphs.trading_assistant.prefetch import make_prefetch_node
from app.graphs.trading_assistant.state import Scope
from app.graphs.trading_assistant.tools import make_desk_search_tool
from app.rag.retrieve import RetrievedDoc
from app.trading.domain import MissingData, Position


class _CaptureChat:
    """Same model bound or plain: records what it saw, answers with a canned text."""

    def __init__(self, text="Book looks fine."):
        self.text = text
        self.seen = None

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        raise AssertionError("not used in these tests")

    def invoke(self, messages):
        self.seen = messages
        return AIMessage(content=self.text)


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
        self.calls: list[str] = []

    def retrieve(self, query, *, user_id, k):
        self.calls.append(query)
        return list(self._docs)


def _position():
    return Position(
        symbol="NVDA", asset_class="STK", currency="USD",
        fx_rate_to_base=1.0, quantity=35, mark_price=194.83, position_value=6819.05,
        strike=None, expiry=None, right=None,
    )


def _doc(text):
    return RetrievedDoc(
        id="c1", text=text, score=0.03, doc_type="weekly", review_date="2026-07-06",
        source="review.pdf", section="High Beta unwind", pages=(4,),
    )


def _ctx(chat=None, tools=None):
    return AgentContext(
        chat_model=chat or _CaptureChat(),
        agent_tools=tools,
        load_positions=lambda u: [_position()],
        load_trades=lambda u: MissingData("ledger", "Upload a recent activity statement."),
        load_nav=lambda u: 100_000.0,
        default_user_id="alex",
    )


# --- pre-fetch: deterministic CSV work only --------------------------------


def test_briefing_prefetch_is_book_only():
    out = make_prefetch_node(_ctx())(
        {"scope": Scope(intents=["daily_briefing"]),
         "messages": [HumanMessage(content="morning briefing")], "user_id": "alex"}
    )
    tools = {item.tool for item in out["evidence"]}
    assert {"check_exposure", "scan_scaleout", "open_position_pnl", "list_positions",
            "performance_summary"} <= tools
    assert "search_desk_reviews" not in tools  # retrieval is the agent's job now


# --- prompt composition ------------------------------------------------------


def _system_for(intents, *, tools):
    chat = _CaptureChat()
    node = make_answer_node(_ctx(chat=chat, tools=tools))
    node({"messages": [HumanMessage(content="morning briefing")],
          "evidence": [], "missing": [],
          "scope": Scope(intents=intents)})
    return chat.seen[0].content


def _dummy_tool():
    from langchain_core.tools import tool

    @tool
    def search_desk_reviews(query: str) -> str:
        """Stub."""
        return "nothing"

    return search_desk_reviews


def test_briefing_prompt_mandates_desk_search_with_standing_query():
    system = _system_for(["daily_briefing"], tools=[_dummy_tool()])
    assert "MUST call search_desk_reviews" in system
    assert BRIEFING_QUERY in system


def test_briefing_prompt_composes_headings_including_book_vs_brief():
    system = _system_for(["daily_briefing"], tools=[_dummy_tool()])
    assert "Book status" in system
    assert "Your book vs the brief" in system
    assert "omit a heading" in system
    assert "Performance" not in system  # not asked for -> no heading


def test_performance_intent_adds_its_heading_to_the_briefing():
    # Regression: a briefing+performance question used to drop the performance
    # evidence because the briefing template had no slot for it.
    system = _system_for(["daily_briefing", "performance_review"], tools=[_dummy_tool()])
    assert "**Performance**" in system


def test_non_briefing_prompt_has_no_briefing_block():
    system = _system_for(["status_check"], tools=[_dummy_tool()])
    assert "MORNING BRIEFING" not in system
    assert "Your book vs the brief" not in system


# --- end-to-end: the mandatory search feeds untruncated text to the audit ---


class _Return:
    def __init__(self, value):
        self._value = value

    def invoke(self, messages):
        return self._value


class _BoundStub:
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages):
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class _BriefingStub:
    def __init__(self, scope, bound_responses):
        self._scope = scope
        self._bound = _BoundStub(bound_responses)

    def with_structured_output(self, schema):
        return _Return(self._scope if schema is Scope else None)

    def bind_tools(self, tools):
        return self._bound

    def invoke(self, messages):
        return AIMessage(content="unused")


def test_briefing_run_grounds_full_desk_text_and_held_names():
    # A long review section (well past the old 600-char cut) plus the held-name
    # footer must BOTH reach the grounding, so the desk's own figure ("25%")
    # and the held-name value back the answer.
    long_text = ("הדסק רואה סיכון של 25% לתיקון בשמות הצפופים NVDA QQQ SMH " * 30).strip()
    retriever = _FakeRetriever([_doc(long_text)])
    desk_tool = make_desk_search_tool(
        retriever, load_positions=lambda u: [_position()]
    )
    stub = _BriefingStub(
        Scope(intents=["daily_briefing"]),
        bound_responses=[
            AIMessage(content="", tool_calls=[{
                "name": "search_desk_reviews",
                "args": {"query": BRIEFING_QUERY},
                "id": "c1", "type": "tool_call",
            }]),
            AIMessage(content="Desk sees 25% correction risk; you hold NVDA ($6,819)."),
        ],
    )
    ctx = AgentContext(
        chat_model=stub,
        agent_tools=[desk_tool],
        load_positions=lambda u: [_position()],
        load_trades=lambda u: MissingData("ledger", "Upload a recent activity statement."),
        load_nav=lambda u: 100_000.0,
        default_user_id="alex",
    )
    result = build_graph(ctx).invoke(
        {"messages": [HumanMessage(content="morning briefing")], "user_id": "alex"}
    )
    assert retriever.calls == [BRIEFING_QUERY]
    assert long_text in result["synthesis"].grounding  # full text, no truncation
    assert "NVDA ($6,819 held)" in result["synthesis"].grounding  # the footer
    assert result["audit"].ok  # 25% and $6,819 are tool-backed
    assert result["messages"][-1].content.startswith("Desk sees 25%")
