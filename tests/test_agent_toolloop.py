"""The answering agent's tool loop: a bounded bind_tools loop on the
user-visible thread (quotes / desk retrieval / web), audit-gated drafts.

Offline: the chat model is a scripted stub whose bind_tools() returns scripted
tool-calling responses, and the tools run on fakes. The loop's contract: tool
results persist as ToolMessages in the thread (and thereby ground the audit),
the iteration cap always terminates the loop, and an audit bounce reruns the
UNBOUND model — it can rewrite, never fetch.
"""

from datetime import date, datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graphs.trading_assistant.answer import MAX_TOOL_ROUNDS
from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.graph import build_graph
from app.graphs.trading_assistant.state import Scope
from app.graphs.trading_assistant.tools import (
    NO_REVIEWS_SENTINEL,
    make_desk_search_tool,
    make_market_quote_tool,
)
from app.rag.retrieve import RetrievedDoc
from app.trading.domain import MissingData, Position, Quote
from app.trading.quotes import quote_yahoo_symbol


# --- bare-symbol quoting (the tool's engine) -------------------------------


def _raw_quote(price=601.23, currency="USD"):
    return {"price": price, "currency": currency, "epoch": 1_780_000_000,
            "market_state": "REGULAR"}


def test_quote_yahoo_symbol_quotes_bare_symbols():
    q = quote_yahoo_symbol("SPY", fetch=lambda s: _raw_quote())
    assert q.symbol == "SPY" and q.price == 601.23 and q.currency == "USD"
    assert q.market_state == "REGULAR"


def test_quote_yahoo_symbol_fails_loud_on_no_quote():
    import pytest

    with pytest.raises(ValueError, match="no quote available"):
        quote_yahoo_symbol("NOPE", fetch=lambda s: None)


# --- the market-quote tool --------------------------------------------------


def _fake_quote_fn(symbol):
    if symbol == "BAD":
        raise ValueError("no quote available for 'BAD'")
    return Quote(symbol=symbol, price=601.23, currency="USD",
                 as_of=datetime(2026, 7, 12, 15, 59, tzinfo=timezone.utc),
                 market_state="REGULAR")


def test_market_quote_tool_formats_dollars_and_reports_errors_per_symbol():
    tool = make_market_quote_tool(quote_fn=_fake_quote_fn)
    out = tool.invoke({"symbols": ["SPY", "BAD"]})
    assert "SPY: $601.23" in out  # $-format so the audit can back quoted prices
    assert "BAD: no quote" in out  # per-symbol error, not a raised exception


# --- the desk-search tool ----------------------------------------------------


def _doc(text, *, doc_type="weekly", review_date="2026-07-06", section="risks"):
    return RetrievedDoc(
        id="c1", text=text, score=0.03, doc_type=doc_type, review_date=review_date,
        source="review.pdf", section=section, pages=(1,),
    )


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(self, query, *, user_id, k):
        self.calls.append((query, user_id, k))
        return list(self._docs)


def _stock(symbol, value):
    return Position(
        symbol=symbol, asset_class="STK", currency="USD", fx_rate_to_base=1.0,
        quantity=1, mark_price=value, position_value=value,
        strike=None, expiry=None, right=None,
    )


# --- size_trade_signal (sizing-only trade_signal_eval) --------------------


def _sizing_tool(nav=100_000.0, positions=None):
    from app.graphs.trading_assistant.policy_model import DEFAULT_POLICY
    from app.graphs.trading_assistant.tools import make_size_signal_tool

    return make_size_signal_tool(
        load_nav=lambda u: nav,
        load_policy=lambda u: DEFAULT_POLICY,
        load_positions=(lambda u: positions) if positions is not None else None,
    )


def _size(tool, **args):
    """Invoke the sizing tool the way the tools node does: `user_id` injected
    alongside the model-parsed args (it is not in the model-facing schema)."""
    return tool.invoke({"user_id": "u1", **args})


def test_size_tool_sizes_an_option_signal_from_policy():
    tool = _sizing_tool(positions=[_stock("GOOGL", 10_797.0)])
    out = _size(tool, ticker="AAOI", kind="option", unit_price=3.10,
                detail="150 NEXT WEEK")
    assert "1.0% of NAV" in out                # the rule applied, from the record
    assert "$1,000" in out                     # the budget
    assert "3 contracts for $930" in out       # code-computed, audit-backed
    assert "AAOI 150 NEXT WEEK" in out         # parsed signal echoed — paper trail
    assert "not currently held" in out         # inventory cross-check
    assert "NOT CHECKED" in out                # chain/IV/exit caveats are mandatory


def test_size_tool_flags_a_name_already_in_the_book():
    tool = _sizing_tool(positions=[_stock("AAOI", 2_500.0)])
    out = _size(tool, ticker="AAOI", kind="option", unit_price=3.10)
    assert "ALREADY IN BOOK" in out
    assert "$2,500" in out


def test_size_tool_warns_when_the_entry_would_breach_the_options_cap():
    # $9,800 of calls already held: +$930 more = $10,730 > 10% of $100k NAV.
    fat_call = Position(
        symbol="NVDA", asset_class="OPT", currency="USD", fx_rate_to_base=1.0,
        quantity=10, mark_price=9.8, position_value=9_800.0,
        strike=100.0, expiry=date(2026, 8, 21), right="C",
    )
    tool = _sizing_tool(positions=[fat_call])
    out = _size(tool, ticker="AAOI", kind="option", unit_price=3.10)
    assert "BREACH" in out
    assert "$10,730" in out  # the after-entry exposure, computed in code


def test_size_tool_stock_kind_uses_the_stock_rule():
    out = _size(_sizing_tool(), ticker="PENG", kind="stock", unit_price=40.0)
    assert "3.0% of NAV" in out
    assert "75 shares for $3,000" in out


def test_size_tool_reports_an_over_budget_premium():
    out = _size(_sizing_tool(), ticker="SPY", kind="option", unit_price=15.0)
    assert "OVER BUDGET" in out  # $1,500/contract vs a $1,000 budget: zero bought


def test_size_tool_refuses_without_nav():
    out = _size(_sizing_tool(nav=None), ticker="AAOI", kind="option", unit_price=3.10)
    assert "NAV" in out and "statement" in out  # cold-start ask, not a zero-size
    assert "contracts" not in out


def test_size_tool_says_when_inventory_is_unchecked():
    from app.graphs.trading_assistant.policy_model import DEFAULT_POLICY
    from app.graphs.trading_assistant.tools import make_size_signal_tool

    tool = make_size_signal_tool(
        load_nav=lambda u: 100_000.0,
        load_policy=lambda u: DEFAULT_POLICY,
        load_positions=lambda u: MissingData(
            "positions snapshot", "Upload your tactical book export."
        ),
    )
    out = _size(tool, ticker="AAOI", kind="option", unit_price=3.10)
    assert "inventory not checked" in out  # sized anyway, blind spot named


def test_desk_tool_returns_every_chunk_in_full_with_source_headers():
    # Directive: NO truncation anywhere — a long chunk must arrive whole,
    # under a per-chunk [Source N: …] metadata header (course format).
    long_text = "סיכון QQQ ותנודתיות " * 120  # ~2,400 chars, far past any old cap
    tool = make_desk_search_tool(
        _FakeRetriever([_doc(long_text), _doc("שורה שנייה", section="hedges")]),
    )
    out = tool.invoke({"query": "risks this week", "user_id": "alex"})
    assert long_text in out
    assert (
        "[Source 1: review.pdf, doc_type=weekly, review_date=2026-07-06, "
        "section=risks, chunk_id=c1, pages=1, score=0.030]" in out
    )
    assert "[Source 2: review.pdf" in out and "section=hedges" in out


def test_desk_retriever_adapter_is_a_traceable_langchain_retriever():
    # Observability parity with the course stack: retrieval must flow through
    # a BaseRetriever so LangSmith auto-emits a run_type="retriever" child run
    # rendered as per-chunk Documents (no @traceable, same as Session 10).
    from langchain_core.retrievers import BaseRetriever

    from app.graphs.trading_assistant.tools import DeskReviewRetriever

    adapter = DeskReviewRetriever(hybrid=_FakeRetriever([_doc("chunk text")]), user_id="alex")
    assert isinstance(adapter, BaseRetriever)

    [doc] = adapter.invoke("q")
    assert doc.page_content == "chunk text"
    assert doc.metadata["chunk_id"] == "c1"
    assert doc.metadata["doc_type"] == "weekly"
    assert doc.metadata["review_date"] == "2026-07-06"
    assert doc.metadata["section"] == "risks"
    assert doc.metadata["source"] == "review.pdf"
    assert doc.metadata["pages"] == [1]
    assert doc.metadata["score"] == 0.03


def test_desk_tool_empty_corpus_returns_upload_ask():
    tool = make_desk_search_tool(_FakeRetriever([]))
    assert tool.invoke({"query": "anything", "user_id": "alex"}) == NO_REVIEWS_SENTINEL
    assert "upload" in NO_REVIEWS_SENTINEL.lower()


def test_desk_tool_footers_held_names_mentioned_in_results():
    doc = _doc("סיכון ל-QQQ, SMH, NVIDIA (NVDA) בגלל צפיפות פוזיציות")
    positions = [_stock("NVDA", 6819.05), _stock("GLW", 2558.27),
                 _stock("VD", 100.0)]  # "VD" is inside "NVDA": must NOT match
    tool = make_desk_search_tool(
        _FakeRetriever([doc]), load_positions=lambda u: positions
    )
    out = tool.invoke({"query": "risks", "user_id": "alex"})
    assert "HELD NAMES MENTIONED IN THESE REVIEWS: NVDA ($6,819 held)" in out
    assert "GLW" not in out  # held but not mentioned
    assert "VD ($" not in out  # substring of NVDA, not a word-boundary hit


# --- per-call user binding (deploy contract: one roster, every tenant) ------


def test_user_id_is_injected_never_model_visible():
    # The model's tool schema must not offer a tenant to pick; invocation
    # without an injected user_id must fail loud, never default silently.
    import pytest
    from pydantic import ValidationError

    for tool in (_sizing_tool(), make_desk_search_tool(_FakeRetriever([]))):
        assert "user_id" not in tool.args  # hidden from the bound model
        assert "user_id" in tool.args_schema.model_fields  # required at invoke
    with pytest.raises(ValidationError):
        make_desk_search_tool(_FakeRetriever([])).invoke({"query": "q"})


def test_tools_node_injects_the_callers_user_id():
    from app.graphs.trading_assistant.answer import make_tools_node

    retriever = _FakeRetriever([_doc("chunk")])
    ctx = AgentContext(
        chat_model=None,
        agent_tools=[make_desk_search_tool(retriever)],
        default_user_id="fallback",
    )
    node = make_tools_node(ctx)
    call = _tool_call_msg(name="search_desk_reviews", args={"query": "q"})

    node({"messages": [call], "user_id": "u42"})   # authenticated turn
    node({"messages": [call]})                     # no identity in state
    assert [uid for _, uid, _ in retriever.calls] == ["u42", "fallback"]


def test_desk_tool_footer_skips_missing_positions():
    tool = make_desk_search_tool(
        _FakeRetriever([_doc("NVDA looks crowded")]),
        load_positions=lambda u: MissingData("positions snapshot", "Upload it."),
    )
    assert "HELD NAMES" not in tool.invoke({"query": "risks", "user_id": "alex"})


# --- scripted stubs ---------------------------------------------------------


class _Return:
    def __init__(self, value):
        self._value = value

    def invoke(self, messages):
        return self._value


class _BoundStub:
    """The bind_tools() result: scripted responses, sticky on the last one.

    Sticky responses are COPIED with a fresh id — `add_messages` dedupes by
    message id, so returning the same object twice would replace, not append
    (real models always mint fresh messages)."""

    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages):
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0].model_copy(update={"id": None})


class _AgentStub:
    """Scope for the scoper, scripted turns for the BOUND model, and a plain
    answer for the UNBOUND model (the cap / audit-repair path)."""

    def __init__(self, scope, bound_responses, plain_answer):
        self._scope = scope
        self._bound = _BoundStub(bound_responses)
        self._plain_answer = plain_answer

    def with_structured_output(self, schema):
        # build_graph eagerly binds the policy extractor too; only the scoper
        # is ever invoked in these tests.
        return _Return(self._scope if schema is Scope else None)

    def bind_tools(self, tools):
        return self._bound

    def invoke(self, messages):
        return AIMessage(content=self._plain_answer)


def _tool_call_msg(name="get_market_quote", args=None, call_id="c1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {"symbols": ["SPY"]},
                     "id": call_id, "type": "tool_call"}],
    )


def _position():
    return Position(
        symbol="AAOI 16JAN26 40 C", asset_class="OPT", currency="USD",
        fx_rate_to_base=1.0, quantity=5, mark_price=4.0, position_value=7300.0,
        strike=40.0, expiry=date(2026, 1, 16), right="C",
    )


def _graph(stub, tools):
    ctx = AgentContext(
        chat_model=stub,
        load_positions=lambda u: [_position()],
        load_trades=lambda u: MissingData("ledger", "Upload a recent activity statement."),
        load_nav=lambda u: 100_000.0,
        agent_tools=tools,
        default_user_id="alex",
    )
    return build_graph(ctx)


# --- the loop end-to-end -----------------------------------------------------


def test_quote_persists_in_thread_and_backs_the_audit():
    stub = _AgentStub(
        Scope(intents=["market_regime"]),
        bound_responses=[_tool_call_msg(), AIMessage(content="SPY is trading at $601.23.")],
        plain_answer="unused",
    )
    result = _graph(stub, [make_market_quote_tool(quote_fn=_fake_quote_fn)]).invoke(
        {"messages": [HumanMessage(content="how's the market?")], "user_id": "alex"}
    )
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and "$601.23" in tool_msgs[0].content  # stays on the thread
    assert "$601.23" in result["synthesis"].grounding  # ...and grounds the audit
    assert result["audit"].ok
    assert result["messages"][-1].content == "SPY is trading at $601.23."
    assert result["tool_rounds"] == 1


def test_answer_with_no_tool_calls_goes_straight_to_audit():
    stub = _AgentStub(
        Scope(intents=["status_check"]),
        bound_responses=[AIMessage(content="You're at 7.3% of NAV.")],
        plain_answer="unused",
    )
    result = _graph(stub, [make_market_quote_tool(quote_fn=_fake_quote_fn)]).invoke(
        {"messages": [HumanMessage(content="am I within policy?")], "user_id": "alex"}
    )
    assert not [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert result["audit"].ok
    # The fixture's missing ledger appends the upload block after the answer.
    assert result["messages"][-1].content.startswith("You're at 7.3% of NAV.")


def test_iteration_cap_terminates_an_insistent_agent():
    # The bound stub ALWAYS wants another tool call; at the cap the UNBOUND
    # model runs and can only answer.
    stub = _AgentStub(
        Scope(intents=["market_regime"]),
        bound_responses=[_tool_call_msg()],  # sticky: tool calls forever
        plain_answer="SPY is trading at $601.23.",
    )
    result = _graph(stub, [make_market_quote_tool(quote_fn=_fake_quote_fn)]).invoke(
        {"messages": [HumanMessage(content="how's the market?")], "user_id": "alex"}
    )
    assert result["tool_rounds"] == MAX_TOOL_ROUNDS
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == MAX_TOOL_ROUNDS
    assert result["messages"][-1].content.startswith("SPY is trading")


def test_audit_bounce_reruns_without_tools():
    # Draft fabricates a figure -> audit bounces -> the repair runs on the
    # UNBOUND model (the bound stub would fabricate forever) and is delivered.
    stub = _AgentStub(
        Scope(intents=["status_check"]),
        bound_responses=[AIMessage(content="NAV is $999,999.")],  # sticky bad draft
        plain_answer="You're at 7.3% of NAV.",
    )
    result = _graph(stub, [make_market_quote_tool(quote_fn=_fake_quote_fn)]).invoke(
        {"messages": [HumanMessage(content="am I within policy?")], "user_id": "alex"}
    )
    assert result["synthesis_attempts"] == 2  # one bounce
    assert result["audit"].ok
    assert result["tool_rounds"] == 0  # the repair never re-entered the loop
    assert result["messages"][-1].content == "You're at 7.3% of NAV."
