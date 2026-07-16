"""Scale-out scanning: the fills-based tranche ladder over open option positions.

Gain is measured as a ratio vs average entry price (current_mark / entry - 1),
which is multiplier- and currency-free. The ladder is STATEFUL: the campaign's
recorded scale-out sales are the rungs already taken — no sales yet means the
first tranche is due at `first_gain` (+100% default); one sale means the second
is due at `second_gain` (+200% default); two or more mean the remainder is a
moonshot runner, reported at ANY current gain as inventory, never as an action.
Thresholds arrive as scalars from the policy record (`scale_out_first` /
`scale_out_second`), keeping this layer decoupled from the graph's PolicyRecord
(same seam style as check_exposure). The runner's endgame (the trader's manual
"+150% arms a hard stop at +50%" rule) is deliberately NOT machine-checked:
it is path-dependent and snapshots cannot see the path.
"""

from __future__ import annotations

from app.trading.domain import (
    MissingData,
    Position,
    ScaleOutCandidate,
    ScaleOutSignal,
    Trade,
)
from app.trading.ledger import group_campaigns
from app.trading.symbols import (
    ContractKey,
    position_contract_key,
    statement_symbol_contract_key,
)

_FIRST_GAIN = 1.0   # +100% -> sell the first contract
_SECOND_GAIN = 2.0  # +200% -> sell the second


def classify_scaleout(
    gain: float,
    scales_taken: int,
    *,
    first_gain: float = _FIRST_GAIN,
    second_gain: float = _SECOND_GAIN,
) -> ScaleOutSignal:
    """Ladder state first, gain second: what you already sold decides which
    rung (if any) the current gain is measured against."""
    if scales_taken >= 2:
        return ScaleOutSignal.MOONSHOT_RUNNER
    if scales_taken == 1:
        return (
            ScaleOutSignal.SECOND_TRANCHE_DUE
            if gain >= second_gain
            else ScaleOutSignal.NONE
        )
    return ScaleOutSignal.FIRST_TRANCHE_DUE if gain >= first_gain else ScaleOutSignal.NONE


def scan_scaleout(
    positions: list[Position],
    trades: list[Trade],
    fallback_entry: dict[ContractKey, float] | None = None,
    *,
    first_gain: float = _FIRST_GAIN,
    second_gain: float = _SECOND_GAIN,
) -> list[ScaleOutCandidate] | MissingData:
    """Flag open OPTION positions by ladder state: tranche due, or runner.

    Options only: the tranche ladder (`first_gain`/`second_gain`, wired from
    the policy record's `scale_out_first`/`scale_out_second`) is a
    contract-selling rule ("sell one, let the rest ride"); stock positions are
    governed by the existing-holding cap in check_exposure, never by this scan.

    Cost basis and the sales count both come from the ledger (tactical
    CostBasisPrice is always 0), so with no ledger the scan refuses rather
    than implying "no candidates". Contracts with no opening fill this period
    (bought pre-window) fall back to the statement's Open-Positions Cost Price
    — same join as open_position_pnl — and honestly read as rung 0: sales
    taken before the window are as invisible as the entry was.
    """
    if not trades:
        return MissingData(store="ledger", remedy="Upload a recent activity statement.")

    state_by_key = {
        statement_symbol_contract_key(c.symbol): (c.avg_entry_price, c.scale_outs)
        for c in group_campaigns(trades)
        if c.is_open
    }
    fallback_entry = fallback_entry or {}

    candidates: list[ScaleOutCandidate] = []
    for pos in positions:
        if pos.asset_class != "OPT":
            continue  # the ladder is options-only; stocks answer to the holding cap
        key = position_contract_key(pos)
        entry, scales = state_by_key.get(key, (None, 0))
        if entry is None:
            entry = fallback_entry.get(key)
        if entry is None:
            continue  # neither source has a cost basis -> nothing to judge
        gain = pos.mark_price / entry - 1
        signal = classify_scaleout(
            gain, scales, first_gain=first_gain, second_gain=second_gain
        )
        if signal is not ScaleOutSignal.NONE:
            candidates.append(
                ScaleOutCandidate(
                    symbol=pos.symbol,
                    signal=signal,
                    gain=gain,
                    avg_entry_price=entry,
                    mark_price=pos.mark_price,
                    scales_taken=scales,
                    quantity=pos.quantity,
                )
            )
    return candidates
