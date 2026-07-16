"""The trader's exposure rulebook as a typed, versioned record.

Pure data + pure functions only — no graph imports — so `state`, `deps`, and the
policy nodes can all depend on it without a cycle. The record is seeded from
`DEFAULT_POLICY` at first login and only ever changed through the interrupt-gated
`update_policy` flow (see `policy.py`). Every field here is a rule a route reads;
nothing enters the record that no code consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class PolicyRecord:
    """Versioned exposure rules. Fractions are decimals (0.10 == 10%)."""

    options_limit: float = 0.10          # options ≤ 10% of NAV
    option_sizing_pct: float = 0.01       # 1% of NAV per new option entry
    stock_sizing_pct_new: float = 0.03    # 3% of NAV per new stock entry
    existing_holding_cap: float = 0.06    # 6% cap on an existing holding
    hedge_ratio_low: float = 0.10         # hedge = put/call value, target 10–15%
    hedge_ratio_high: float = 0.15
    scale_out_first: float = 1.00         # +100% -> sell one contract (rung 1)
    scale_out_second: float = 2.00        # +200% -> sell another (rung 2)
    # Deliberately NOT here (backlogged): the moonshot endgame (+150% arms a +50% hard stop —
    # path-dependent, unenforceable from snapshots), the IV shield (returns
    # with full trade_signal_eval as a fail-loud manual reminder), and the
    # max-offensive-exposure cap ("offensive" was never defined). An
    # unenforced field that looks machine-managed is a liability.
    version: int = 1


DEFAULT_POLICY = PolicyRecord()


@dataclass(frozen=True)
class FieldSpec:
    label: str
    kind: str  # "nav_fraction" (shown as % of NAV) | "gain_ratio" (shown as +N%)
    hint: str  # guidance for the extraction model


# The editable rules (version is not user-editable). Keys match PolicyRecord fields.
FIELDS: dict[str, FieldSpec] = {
    "options_limit": FieldSpec("options exposure cap", "nav_fraction", "options as a share of NAV"),
    "option_sizing_pct": FieldSpec("per-option entry size", "nav_fraction", "NAV per new option entry"),
    "stock_sizing_pct_new": FieldSpec("per-stock entry size", "nav_fraction", "NAV per new stock entry"),
    "existing_holding_cap": FieldSpec("existing-holding cap", "nav_fraction", "cap on an existing holding"),
    "hedge_ratio_low": FieldSpec("hedge ratio floor", "nav_fraction", "low end of the put/call hedge band"),
    "hedge_ratio_high": FieldSpec("hedge ratio ceiling", "nav_fraction", "high end of the put/call hedge band"),
    "scale_out_first": FieldSpec("first scale-out trigger", "gain_ratio", "gain that sells the first contract"),
    "scale_out_second": FieldSpec("second scale-out trigger", "gain_ratio", "gain that sells the second contract"),
}


@dataclass(frozen=True)
class ProposedPolicyChange:
    """A validated, ready-to-confirm change carried in state across the interrupt."""

    field: str
    label: str
    current: float
    proposed: float
    summary: str
    next_version: int


class PolicyChange(BaseModel):
    """The scoper-adjacent extraction of a requested rule change (LLM output)."""

    recognized: bool = Field(description="False if the request can't map to exactly one field.")
    field: str = Field(default="", description="One policy field key, or empty if unrecognized.")
    new_value: float = Field(
        default=0.0,
        description="New value in the field's native unit: fractions as decimals (12% -> 0.12; +150% -> 1.5).",
    )
    summary: str = Field(default="", description="One-line human summary of the change.")


def normalize_value(field: str, value: float) -> float:
    """Coerce likely percent-as-integer mistakes into the field's native unit."""
    kind = FIELDS[field].kind
    if kind == "nav_fraction" and value > 1:      # "12" meant 12% -> 0.12
        return value / 100
    if kind == "gain_ratio" and value > 10:       # "150" meant +150% -> 1.5
        return value / 100
    return value


def format_value(field: str, value: float) -> str:
    if FIELDS[field].kind == "nav_fraction":
        return f"{value:.1%}"
    return f"+{value:.0%}"


def apply_change(policy: PolicyRecord, field: str, new_value: float) -> PolicyRecord:
    """Return a new record with `field` set and the version bumped."""
    if field not in FIELDS:
        raise ValueError(f"{field!r} is not an editable policy field")
    return replace(policy, **{field: new_value, "version": policy.version + 1})
