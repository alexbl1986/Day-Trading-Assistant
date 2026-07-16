"""Interrupt-gated policy writes.

A `policy_change` intent never writes on the model's say-so. Two nodes:
  * `policy_prepare` extracts the requested change (LLM) and validates it against
    the known fields — computation only, no write;
  * `policy_confirm` shows the exact current -> proposed change through a LangGraph
    `interrupt()` and only writes on explicit human approval, then bumps the version.

Splitting them means the LLM extraction runs once: `policy_prepare` commits its
result to state before the graph pauses, so the resume re-runs only `policy_confirm`.
The scoper's hypothetical guard already ensures a "what if I raised my cap" never
reaches here.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.policy_model import (
    DEFAULT_POLICY,
    FIELDS,
    PolicyChange,
    PolicyRecord,
    ProposedPolicyChange,
    apply_change,
    format_value,
    normalize_value,
)
from app.graphs.trading_assistant.state import AgentState
from app.graphs.trading_assistant.util import latest_user_text

_FIELD_LINES = "\n".join(f"- {key}: {spec.hint}" for key, spec in FIELDS.items())

POLICY_EXTRACT_SYSTEM = f"""The trader wants to change ONE exposure rule. Identify which
field and the new value. Fields (key: meaning):
{_FIELD_LINES}

Express the value in the field's native unit: percentages of NAV as decimals
(12% -> 0.12), gain triggers as ratios (+150% -> 1.5, +50% -> 0.5). If you cannot map
the request to exactly one field with confidence, set recognized=false."""


def make_policy_prepare_node(context: AgentContext) -> Callable[[AgentState], dict]:
    model = context.chat_model.with_structured_output(PolicyChange)

    def policy_prepare(state: AgentState) -> dict:
        user_id = state.get("user_id") or context.default_user_id
        policy = _load_policy(context, user_id)
        parsed: PolicyChange = model.invoke(
            [SystemMessage(content=POLICY_EXTRACT_SYSTEM), HumanMessage(content=latest_user_text(state["messages"]))]
        )
        if not parsed.recognized or parsed.field not in FIELDS:
            editable = ", ".join(spec.label for spec in FIELDS.values())
            remaining = set(state["scope"].intents) - {"policy_change", "off_topic"}
            if remaining:
                # Co-intents keep the run alive: the ask rides into the answer
                # as a note instead of a counter-question eating the question.
                return {
                    "policy": policy,
                    "policy_note": (
                        "The message was also read as a request to change a rule, but "
                        "no specific rule and new value could be identified. If the "
                        "user did want a change, ask which rule and to what value "
                        f"(editable: {editable}); otherwise just answer."
                    ),
                }
            return {
                "policy": policy,
                "messages": [
                    AIMessage(
                        content=(
                            "I couldn't tell which rule to change. I can adjust: "
                            f"{editable}. Which one, and to what value?"
                        )
                    )
                ],
            }

        value = normalize_value(parsed.field, parsed.new_value)
        spec = FIELDS[parsed.field]
        change = ProposedPolicyChange(
            field=parsed.field,
            label=spec.label,
            current=getattr(policy, parsed.field),
            proposed=value,
            summary=parsed.summary or f"Set {spec.label} to {format_value(parsed.field, value)}",
            next_version=policy.version + 1,
        )
        return {"policy": policy, "proposed_change": change}

    return policy_prepare


def make_policy_confirm_node(context: AgentContext) -> Callable[[AgentState], dict]:
    def policy_confirm(state: AgentState) -> dict:
        change: ProposedPolicyChange = state["proposed_change"]
        user_id = state.get("user_id") or context.default_user_id

        decision = interrupt(
            {
                "type": "policy_change",
                "field": change.field,
                "label": change.label,
                "current": format_value(change.field, change.current),
                "proposed": format_value(change.field, change.proposed),
                "summary": change.summary,
                "next_version": change.next_version,
            }
        )

        if not _approved(decision):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"No change made — {change.label} stays at "
                            f"{format_value(change.field, change.current)}."
                        )
                    )
                ]
            }

        updated = apply_change(_load_policy(context, user_id), change.field, change.proposed)
        _save_policy(context, user_id, updated)
        return {
            "policy": updated,
            "messages": [
                AIMessage(
                    content=(
                        f"✅ Updated {change.label}: {format_value(change.field, change.current)} → "
                        f"{format_value(change.field, change.proposed)}. Policy is now v{updated.version}."
                    )
                )
            ],
        }

    return policy_confirm


# -- helpers --------------------------------------------------------------


def _load_policy(context: AgentContext, user_id: str) -> PolicyRecord:
    if context.load_policy is None:
        return DEFAULT_POLICY
    return context.load_policy(user_id)


def _save_policy(context: AgentContext, user_id: str, policy: PolicyRecord) -> None:
    if context.save_policy is not None:
        context.save_policy(user_id, policy)


def _approved(decision) -> bool:
    """Approve only on an explicit yes; anything ambiguous is a safe reject."""
    if decision is True:
        return True
    if isinstance(decision, str):
        return decision.strip().lower() in {"approve", "approved", "yes", "y", "confirm", "ok"}
    if isinstance(decision, dict):
        if decision.get("approve") is True:
            return True
        return str(decision.get("decision", "")).strip().lower() in {"approve", "approved", "yes", "confirm"}
    return False
