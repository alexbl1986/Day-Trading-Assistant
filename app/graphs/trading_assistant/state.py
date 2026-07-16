"""Graph state and the scoper's structured output.

`AgentState` is the single reasoning context that flows through the graph:
intake/scoper -> deterministic pre-fetch -> answering agent (tool loop on the
`messages` thread) -> audit. Slice 1 populates only `messages` and `scope`;
later slices add the evidence table, the audit-gated draft, and audit result.

`Scope` is the scoper node's structured output: a MULTI-LABEL intent (one message
can be both "am I within policy?" and "what does the desk think?"), the entities
we extract deterministically downstream, and the `hypothetical` flag that keeps
"if I raised my cap to 12%…" an analysis, never a policy write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.graphs.trading_assistant.policy_model import PolicyRecord, ProposedPolicyChange
from app.trading.domain import MissingData

# The 11 routes (the original eight, plus the two transcript routes daily_briefing
# and trade_signal_eval; capabilities added after meta-questions about the assistant
# itself hit the off_topic refusal). Multi-label: the scoper may return several.
Intent = Literal[
    "status_check",        # "am I within policy / exposure?" -> exposure + scaleout + pnl
    "rebalance_advice",    # "what should I change?" -> full reconcile
    "desk_question",       # "what does the desk think about X?" -> RAG
    "trade_history",       # "how's my AAOI campaign going?" -> ledger/campaigns
    "performance_review",  # realized P/L, win rate, cost drag
    "policy_change",       # "raise my options cap to 12%" -> interrupt-gated write
    "market_regime",       # live macro/index read -> quotes + Tavily (no uploads needed)
    "trade_signal_eval",   # pasted shorthand "AAOI 150 NEXT WEEK 3.1" -> TradePlan
    "daily_briefing",      # "morning briefing" -> composition of the fetches
    "capabilities",        # about the assistant itself (tools, "why didn't you…") -> roster reply
    "off_topic",           # not about this book/desk/rules -> refuse
]


class Scope(BaseModel):
    """Structured read of what the user is asking for (scoper node output)."""

    intents: list[Intent] = Field(
        description="All routes this message needs, not just the primary one."
    )
    tickers: list[str] = Field(
        default_factory=list,
        description="Uppercase tickers named or clearly implied (e.g. AAOI, TSMC).",
    )
    signal_text: str | None = Field(
        default=None,
        description="Verbatim pasted trade shorthand for trade_signal_eval, else null.",
    )
    hypothetical: bool = Field(
        default=False,
        description="True for 'what if I raised my cap…' — analyze, never write policy.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True ONLY when genuinely ambiguous; one clarify round, then proceed.",
    )
    clarifying_question: str | None = Field(
        default=None, description="The single question to ask when clarification is needed."
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Stated assumptions when proceeding best-effort instead of clarifying.",
    )


@dataclass(frozen=True)
class EvidenceItem:
    """One deterministic tool call's outcome, recorded for synthesis and audit.

    The evidence table is the contract between the graph's deterministic half and
    the LLM: synthesis copies numbers verbatim from `result`, and the audit node
    (Slice 3) checks that no number in the answer is absent here. `ok=False` with
    a `missing` value is the cold-start signal — the backing store was never
    uploaded, so this is "cannot answer yet", not "answered zero".
    """

    tool: str
    ok: bool
    result: object | None = None
    missing: MissingData | None = None
    note: str | None = None


@dataclass(frozen=True)
class SynthesisResult:
    """The answering agent's draft, gated by the audit before delivery.

    `answer` is the user-facing markdown. `grounding` is the deterministic fact
    digest plus every tool result on the thread — the ONLY source the audit
    node accepts numbers from, so a $/% figure absent here is unbacked.
    (The name is historical: the draft lives in `state["synthesis"]`.)
    """

    answer: str
    grounding: str
    upload_asks: tuple[str, ...]


@dataclass(frozen=True)
class AuditResult:
    """The deterministic audit's verdict; `violations` are unbacked figures."""

    ok: bool
    violations: tuple[str, ...]


class AgentState(TypedDict):
    """The single context threaded through every node."""

    messages: Annotated[list[AnyMessage], add_messages]
    # Scopes every store read; set from the request/login (config) on the server,
    # defaulted for local Studio runs.
    user_id: NotRequired[str]
    scope: NotRequired[Scope]
    # True while the turn just asked a clarifying question; the next run's
    # scoper reads it to enforce "one clarify round max" deterministically.
    pending_clarification: NotRequired[bool]
    # Everything below `scope` is PER-TURN state on a thread that persists
    # across runs: the scope node (first node of every run) resets it all, or
    # one turn's repair budget / audit feedback / pending proposal leaks into
    # the next (`| None` marks the keys whose reset value is None).
    evidence: NotRequired[list[EvidenceItem]]
    missing: NotRequired[list[MissingData]]
    # The answering agent's loop counter; the loop itself runs on `messages`,
    # so tool calls and results persist in the thread for follow-up questions.
    tool_rounds: NotRequired[int]
    synthesis: NotRequired[SynthesisResult | None]
    audit: NotRequired[AuditResult | None]
    synthesis_attempts: NotRequired[int]
    audit_feedback: NotRequired[str | None]
    policy: NotRequired[PolicyRecord]
    proposed_change: NotRequired[ProposedPolicyChange | None]
    # Set when a policy_change parse failed but co-intents keep the run alive:
    # the answering agent folds the "which rule?" ask into its answer instead
    # of the gate replacing the whole answer with a counter-question.
    policy_note: NotRequired[str | None]
