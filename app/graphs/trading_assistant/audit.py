"""Deterministic audit node (ADR-0006, step 5) — code, never the LLM.

The one rule enforced here: every dollar amount and percentage in the answer must
trace to a number in the draft's grounding (the deterministic fact digest plus
every tool result on the thread). A figure that appears nowhere in the evidence is
a fabrication — the classic "confident wrong number" this app exists to prevent, and
the cold-start guarantee (no portfolio number when its store was never uploaded)
falls out for free, since a missing store contributes no numbers to the grounding.

Bare integers (years, strikes, option counts) are not audited — only $ and %,
the figures that carry compliance weight. One bounce back to the answer node
(tools off) with the offending figures, then deliver with a warning banner.

Rule-citation and quote-availability checks were dropped as redundant, not deferred:
the full rulebook rides in the digest and an unavailable quote contributes no figure
to the thread, so the numeric rule above covers both (ADR-0006, 2026-07-16 amendment).
"""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.messages import AIMessage

from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.state import AgentState, AuditResult

MAX_ATTEMPTS = 2  # initial draft + one bounce

# The sign may sit before or after the currency symbol ("-$1,200", "$-1,200").
_NUM = re.compile(
    r"(?P<pre>-)?\s*(?P<dollar>\$)?\s*(?P<post>-)?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<pct>%)?"
)


def extract_figures(text: str) -> list[tuple[str, float]]:
    """Every $ or % figure as (unit, value); bare numbers are ignored."""
    figures: list[tuple[str, float]] = []
    for m in _NUM.finditer(text):
        if not m.group("dollar") and not m.group("pct"):
            continue
        value = float(m.group("num").replace(",", ""))
        if m.group("pre") or m.group("post"):
            value = -value
        figures.append(("%" if m.group("pct") else "$", value))
    return figures


def _backed(unit: str, value: float, allowed: list[tuple[str, float]]) -> bool:
    # Magnitudes only: prose legitimately drops signs ("a loss of $X"), and a
    # wrong sign direction is a semantic error this audit cannot judge anyway.
    for a_unit, a_value in allowed:
        if a_unit != unit:
            continue
        if abs(abs(a_value) - abs(value)) <= max(0.1, 0.005 * abs(a_value)):
            return True
    return False


def audit_answer(answer: str, grounding: str) -> AuditResult:
    allowed = extract_figures(grounding)
    violations: list[str] = []
    for unit, value in extract_figures(answer):
        if not _backed(unit, value, allowed):
            sign = "-" if value < 0 else ""
            shown = f"{sign}${abs(value):,.2f}" if unit == "$" else f"{value:g}%"
            violations.append(shown)
    # dedupe, preserve order
    violations = list(dict.fromkeys(violations))
    return AuditResult(ok=not violations, violations=tuple(violations))


def make_audit_node(context: AgentContext) -> Callable[[AgentState], dict]:
    def audit_node(state: AgentState) -> dict:
        synthesis = state["synthesis"]
        result = audit_answer(synthesis.answer, synthesis.grounding)
        attempts = state.get("synthesis_attempts", 1)

        if result.ok:
            return {"audit": result, "messages": [AIMessage(content=synthesis.answer)]}

        if attempts >= MAX_ATTEMPTS:
            banner = (
                "\n\n> ⚠️ Some figures above could not be verified against the tool "
                "outputs and may be unreliable: " + ", ".join(result.violations) + "."
            )
            return {
                "audit": result,
                "messages": [AIMessage(content=synthesis.answer + banner)],
            }

        feedback = (
            "These figures in your answer are not present in the evidence — remove or "
            "correct them, using only numbers from the digest and tool results: "
            + ", ".join(result.violations)
        )
        return {"audit": result, "audit_feedback": feedback}

    return audit_node
