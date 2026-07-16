"""Assemble the trading-assistant graph from an injected `AgentContext`.

Topology: intake/scoper (per-turn state resets, one clarify round max) ->
deterministic CSV pre-fetch -> answering agent (a bounded tool loop on the
user-visible thread: desk retrieval / quotes / web) -> deterministic audit
(one bounce back to the answer node — tools off — then deliver with a warning
banner). A policy_change routes through the interrupt-gated prepare/confirm
pair first, then continues into pre-fetch when the message also asked a
question. A scoped-out message (awaiting a clarifying answer, or purely
off_topic) is answered by the scoper and skips the rest.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graphs.trading_assistant.answer import (
    make_answer_node,
    make_tools_node,
    route_after_answer,
)
from app.graphs.trading_assistant.audit import MAX_ATTEMPTS, make_audit_node
from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.policy import (
    make_policy_confirm_node,
    make_policy_prepare_node,
)
from app.graphs.trading_assistant.prefetch import make_prefetch_node
from app.graphs.trading_assistant.scope import make_scope_node
from app.graphs.trading_assistant.state import AgentState


def _route_after_scope(state: AgentState) -> str:
    scope = state["scope"]
    if scope.needs_clarification:
        return END  # the clarifying question was already emitted; wait for the reply
    if "capabilities" in scope.intents and set(scope.intents) <= {"capabilities", "off_topic"}:
        return END  # the scoper already answered from the live roster
    if set(scope.intents) <= {"off_topic"}:
        return END  # the scoper already emitted the refusal
    if "policy_change" in scope.intents:
        return "policy_prepare"  # a write takes precedence; gate it before anything else
    return "prefetch"


def _non_policy_intents(state: AgentState) -> set[str]:
    return set(state["scope"].intents) - {"policy_change", "off_topic"}


def _route_after_policy_prepare(state: AgentState) -> str:
    if state.get("proposed_change"):
        return "policy_confirm"
    # No concrete change parsed. If policy_change was the whole message, the
    # prepare node already asked which rule; but co-intents still owe an answer
    # (the ask travels as `policy_note`) — a misrouted read like "what's my
    # current policy?" must never dead-end on the write gate.
    return "prefetch" if _non_policy_intents(state) else END


def _route_after_policy_confirm(state: AgentState) -> str:
    # A mixed message ("raise my cap to 12% — am I within policy?") still owes
    # the non-policy half an answer, computed against the just-decided policy.
    return "prefetch" if _non_policy_intents(state) else END


def _route_after_audit(state: AgentState) -> str:
    if state["audit"].ok:
        return END
    if state.get("synthesis_attempts", 1) >= MAX_ATTEMPTS:
        return END  # delivered with a warning banner
    return "answer"  # one bounce to fix the unbacked figures (tools off)


def build_graph(
    context: AgentContext, *, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """Compile the graph against `context`. Tests inject fakes; dev injects real.

    `checkpointer` is left unset for the LangGraph platform (`langgraph dev` /
    deploy manage persistence); tests pass a `MemorySaver` so the policy
    `interrupt()` can pause and resume.
    """
    builder = StateGraph(AgentState)
    builder.add_node("scope", make_scope_node(context))
    builder.add_node("prefetch", make_prefetch_node(context))
    builder.add_node("answer", make_answer_node(context))
    builder.add_node("tools", make_tools_node(context))
    builder.add_node("audit", make_audit_node(context))
    builder.add_node("policy_prepare", make_policy_prepare_node(context))
    builder.add_node("policy_confirm", make_policy_confirm_node(context))

    builder.add_edge(START, "scope")
    builder.add_conditional_edges(
        "scope",
        _route_after_scope,
        {"prefetch": "prefetch", "policy_prepare": "policy_prepare", END: END},
    )
    builder.add_edge("prefetch", "answer")
    builder.add_conditional_edges(
        "answer", route_after_answer, {"tools": "tools", "audit": "audit"}
    )
    builder.add_edge("tools", "answer")
    builder.add_conditional_edges(
        "audit", _route_after_audit, {"answer": "answer", END: END}
    )
    builder.add_conditional_edges(
        "policy_prepare",
        _route_after_policy_prepare,
        {"policy_confirm": "policy_confirm", "prefetch": "prefetch", END: END},
    )
    builder.add_conditional_edges(
        "policy_confirm", _route_after_policy_confirm, {"prefetch": "prefetch", END: END}
    )
    return builder.compile(checkpointer=checkpointer)
