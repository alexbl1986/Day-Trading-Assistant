"""Intake / scoper node.

The graph's routing brain: one structured-output call turns the user's message
(plus thread history) into a `Scope` — a multi-label intent, extracted tickers,
the `hypothetical` flag, and at most one clarifying question. Everything the
graph does downstream is deterministic and keys off this read, so the scoper is
the only place free-text becomes routing.
"""

from __future__ import annotations

from typing import Callable, Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.state import AgentState, Scope

SCOPER_SYSTEM = """You are the intake router for a retail options day-trader's assistant.
The user reconciles his IBKR book against his own exposure rules and his desk's Hebrew
daily/weekly market reviews. Classify his message into one or more of these routes and
extract the entities the downstream tools need. Do not answer the question yourself.

Routes (choose ALL that apply — a single message can need several):
- status_check: "am I within policy / exposure?" — book vs. exposure limits.
- rebalance_advice: "what should I change?" — full reconcile and concrete actions.
- desk_question: what the desk thinks about a name/theme/sector (from the reviews).
- trade_history: how a specific position/campaign is going (entry, scales, basis).
- performance_review: realized P/L, win rate, cost drag over a period.
- policy_change: a request to CHANGE a rule/threshold to a stated new value (e.g. "raise
  my options cap to 12%"). Asking what the rules ARE ("what's my current policy?") is a
  status_check read, never policy_change.
- market_regime: a live market read — macro, indices, or news/context on a name or
  theme, including explicit asks to search the web or check quotes (needs no uploaded data).
- trade_signal_eval: a pasted trade shorthand like "AAOI 150 NEXT WEEK 3.1".
- daily_briefing: "morning briefing" / "start my day" — the full composed rundown.
- capabilities: about the assistant itself — what it can and cannot do, which tools it
  has, or why it did / didn't do something on a previous turn.
- off_topic: anything else — not this book, the desk's reviews, these rules, live
  markets, or the assistant itself — refuse.

Extraction:
- tickers: uppercase symbols named or clearly implied.
- signal_text: the verbatim shorthand string, ONLY for trade_signal_eval, else null.
- hypothetical: true for "what if I raised my cap…" / "would I still be within policy if…".
  A hypothetical about a rule is ANALYSIS (status/rebalance), never a policy_change write.

Clarification is a LAST RESORT. Almost every message has a reasonable default — take it
and record it in `assumptions` (e.g. "assumed you mean the current open AAOI campaign")
instead of asking. Set needs_clarification true ONLY when the message is genuinely
undecidable AND no assumption could resolve it (e.g. a bare ticker with no action, or a
request that could mean two opposite trades). NEVER clarify a status / exposure / policy /
performance question, a named desk question, or a trade-history question about a named
ticker — just proceed. A missing detail a downstream tool can ask for (e.g. which position)
is not a reason to clarify here. Never ask more than one question."""

OFF_TOPIC_REFUSAL = (
    "I'm your trading-desk assistant — I can help with your book, your exposure and "
    "hedging rules, your desk's daily/weekly reviews, and trade-signal checks, but that "
    "one's outside what I do."
)

CAPABILITIES_INTRO = (
    "I'm your trading-desk assistant. I can reconcile your book against your exposure "
    "and hedging rules, search your desk's daily/weekly reviews, size a pasted trade "
    "signal, review realized performance, and compose your morning briefing."
)

# Emitted when the model wants to clarify but returned no question — a silent
# END with no reply is never acceptable.
DEFAULT_CLARIFY_QUESTION = (
    "Can you say a bit more about what you're after — which position, rule, or "
    "desk view do you mean?"
)


def _normalize_scope(scope: Scope, *, already_clarified: bool = False) -> Scope:
    """Deterministic safety net around the LLM's routing.

    Two guards the prompt alone cannot guarantee:
      * a hypothetical about a rule ("if I raised my cap to 12%…") is analysis,
        never a write — strip `policy_change` so the interrupt-gated path can
        never fire on it (fall back to `status_check` if that empties the routes);
      * ONE clarify round max — once a question was asked on this thread turn
        cycle, an insistent model is overridden into best-effort with a stated
        assumption, mirroring the `synthesis_attempts` cap.
    """
    if scope.hypothetical and "policy_change" in scope.intents:
        intents = [i for i in scope.intents if i != "policy_change"] or ["status_check"]
        scope = scope.model_copy(update={"intents": intents})
    if already_clarified and scope.needs_clarification:
        scope = scope.model_copy(
            update={
                "needs_clarification": False,
                "clarifying_question": None,
                "intents": scope.intents or ["status_check"],
                "assumptions": [
                    *scope.assumptions,
                    "Proceeding on best-effort assumptions — one clarifying "
                    "question was already asked.",
                ],
            }
        )
    return scope


def _scoper_view(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Human/assistant text only. The answering agent's tool loop runs on the
    same thread, but tool-call turns and tool results are noise to routing."""
    return [
        m
        for m in messages
        if isinstance(m, HumanMessage)
        or (isinstance(m, AIMessage) and not m.tool_calls)
    ]


def _capabilities_reply(tools: Sequence[BaseTool] | None) -> str:
    """Deterministic answer about the assistant itself, built from the ACTUAL
    roster — it can never claim (or deny) a tool this session doesn't have."""
    if not tools:
        return CAPABILITIES_INTRO + (
            "\n\nNo live tools are wired this session — I answer from uploaded data alone."
        )
    listed = "\n".join(
        f"- `{t.name}` — {(t.description or '').strip().splitlines()[0]}" for t in tools
    )
    return f"{CAPABILITIES_INTRO}\n\nLive tools this session:\n{listed}"


def make_scope_node(context: AgentContext) -> Callable[[AgentState], dict]:
    """Bind the scoper to its chat model; returns the LangGraph node callable."""
    model = context.chat_model.with_structured_output(Scope)

    def scope_node(state: AgentState) -> dict:
        conversation = [SystemMessage(content=SCOPER_SYSTEM), *_scoper_view(state["messages"])]
        scope: Scope = _normalize_scope(
            model.invoke(conversation),
            already_clarified=bool(state.get("pending_clarification")),
        )
        update: dict = {
            "scope": scope,
            # Per-turn resets: the thread's state survives across runs, so a
            # previous turn's repair budget, audit feedback, evidence, or
            # pending policy proposal must never leak into this one.
            "synthesis_attempts": 0,
            "audit_feedback": None,
            "evidence": [],
            "missing": [],
            "synthesis": None,
            "audit": None,
            "proposed_change": None,
            "policy_note": None,
            "pending_clarification": False,
            "tool_rounds": 0,
        }
        # One clarify round: surface the question as an assistant turn and stop;
        # the user's reply re-enters the graph as a fresh message, where the
        # pending flag turns any second ask into best-effort proceeding.
        if scope.needs_clarification:
            update["messages"] = [
                AIMessage(content=scope.clarifying_question or DEFAULT_CLARIFY_QUESTION)
            ]
            update["pending_clarification"] = True
        elif "capabilities" in scope.intents and set(scope.intents) <= {"capabilities", "off_topic"}:
            update["messages"] = [AIMessage(content=_capabilities_reply(context.agent_tools))]
        elif set(scope.intents) <= {"off_topic"}:
            update["messages"] = [AIMessage(content=OFF_TOPIC_REFUSAL)]
        return update

    return scope_node
