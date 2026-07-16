"""Shared typed records for the trading assistant."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


@dataclass(frozen=True)
class MissingData:
    """A tool's cold-start refusal: the backing store was never uploaded.

    Distinct from an empty result — an empty snapshot must not read as
    "0% exposure, within policy". Carries the remedy to surface to the user.
    """

    store: str
    remedy: str


class ScaleOutSignal(Enum):
    NONE = "none"
    FIRST_TRANCHE_DUE = "first_tranche_due"    # no sales recorded, gain >= first rung
    SECOND_TRANCHE_DUE = "second_tranche_due"  # one sale recorded, gain >= second rung
    MOONSHOT_RUNNER = "moonshot_runner"        # ladder complete (2+ sales): report only


@dataclass(frozen=True)
class ScaleOutCandidate:
    """An open position the ladder scan flags (an action due, or a runner)."""

    symbol: str
    signal: ScaleOutSignal
    gain: float  # ratio vs avg entry: +100% == 1.0
    avg_entry_price: float
    mark_price: float
    scales_taken: int = 0  # closing-direction fills recorded in the open campaign
    quantity: float = 0.0  # contracts currently held, from the snapshot


@dataclass(frozen=True)
class TradeSizing:
    """What the per-entry sizing rule buys for a pasted trade signal."""

    kind: str          # "option" | "stock"
    pct_of_nav: float  # the sizing rule applied
    budget: float      # nav x pct
    unit_cost: float   # premium x 100 per contract, or the share price
    quantity: int      # whole contracts/shares the budget buys (floored)
    cost: float        # quantity x unit_cost


@dataclass(frozen=True)
class Position:
    """One line of the tactical book snapshot.

    `quantity` and `position_value` are signed (negative for shorts).
    `position_value` is in the position's native `currency`; multiply by
    `fx_rate_to_base` to compare across the book. Option fields are None for
    stocks. Cost basis is deliberately absent (it comes from the ledger).
    """

    symbol: str
    asset_class: str
    currency: str
    fx_rate_to_base: float
    quantity: float
    mark_price: float
    position_value: float
    strike: float | None = None
    expiry: date | None = None
    right: str | None = None


@dataclass(frozen=True)
class ExposureCheck:
    """One bucket of the book measured against a % -of-NAV policy limit.

    `value_base` is the bucket's exposure in base currency (USD); `pct_of_nav`
    is that over the account NAV; `limit` is the policy ceiling (0.10 == 10%).
    """

    label: str
    value_base: float
    pct_of_nav: float
    limit: float

    @property
    def within_policy(self) -> bool:
        return self.pct_of_nav <= self.limit


@dataclass(frozen=True)
class HedgeCheck:
    """The hedge ratio — put value / call value — against the policy band.

    His precise formula (not puts/NAV): base-currency absolute option values,
    shorts included by magnitude. Band-shaped, unlike `ExposureCheck`'s single
    ceiling: below `low` is under-hedged, above `high` over-hedged.
    """

    put_value_base: float
    call_value_base: float
    ratio: float
    low: float
    high: float

    @property
    def status(self) -> str:
        if self.ratio < self.low:
            return "under"
        if self.ratio > self.high:
            return "over"
        return "within"


@dataclass(frozen=True)
class ExposureReport:
    """check_exposure's result: the NAV used plus one check per policy bucket.

    `nav` is surfaced for provenance (the exact denominator every pct rides on).
    `hedge` is None when no band was requested or the book has no call value
    (no denominator — never a division blowup).
    """

    nav: float
    checks: tuple[ExposureCheck, ...]
    hedge: HedgeCheck | None = None


@dataclass(frozen=True)
class PositionPnL:
    """One open position's unrealized P/L vs its average entry (from the ledger).

    `unrealized_pl` is in base USD; `gain` is the ratio vs entry (+50% == 0.5).
    `avg_entry_price`/`mark_price` are per-unit, carried for transparency.
    """

    symbol: str
    avg_entry_price: float
    mark_price: float
    gain: float
    unrealized_pl: float
    cost_basis_source: str = "ledger"  # "ledger" (campaigns) | "statement" (Open Positions)


@dataclass(frozen=True)
class UnrealizedPnLReport:
    """open_position_pnl's result: per-position P/L plus the base-USD total."""

    total_unrealized_pl: float
    lines: tuple[PositionPnL, ...]


@dataclass(frozen=True)
class Quote:
    """A live (or last-known) market quote for one position.

    `as_of` (the price's own timestamp) and `market_state` travel with the
    price so a stale close can never pass for a live tick — the caller always
    sees WHEN it's from. `currency` is the venue's quote currency, checked
    against the position to catch a wrong-instrument mapping (cf. the SIVE
    currency bug). `symbol` is the Yahoo symbol actually quoted.
    """

    symbol: str
    price: float
    currency: str
    as_of: datetime
    market_state: str


@dataclass(frozen=True)
class Trade:
    """One fill from the Activity Statement's Trades section.

    `quantity`/`proceeds`/`realized_pl` etc. are signed as IBKR reports them.
    `symbol` is the raw contract string (option: `AAOI 16JAN26 40 C`);
    `root_ticker` is its underlying (`AAOI`), for per-symbol attribution.
    Strike/expiry/right decomposition is deferred until campaign logic needs it.
    """

    symbol: str
    root_ticker: str
    asset_category: str
    currency: str
    timestamp: datetime
    quantity: float
    price: float
    proceeds: float
    commission: float
    basis: float
    realized_pl: float
    mtm_pl: float
    code: str


@dataclass(frozen=True)
class Split:
    """One stock split from the statement's Corporate Actions section.

    `numerator`-for-`denominator` new shares per old (8-for-1 forward,
    1-for-8 reverse). `effective` is the action's Date/Time (after the
    close), the moment that separates pre-split fills from post-split ones.
    """

    symbol: str
    numerator: int
    denominator: int
    effective: datetime


@dataclass(frozen=True)
class TickerPerformance:
    """One underlying's summed realized P/L (IBKR's own signed column)."""

    root_ticker: str
    realized_pl: float


@dataclass(frozen=True)
class PerformanceReport:
    """performance_summary's result: Tier-1 realized attribution from the ledger.

    Every figure comes from IBKR's own signed Realized P/L / commission columns,
    never recomputed from fills. Win rate counts CLOSED campaigns only — an open
    runner isn't a win yet — and is None when nothing has closed.
    """

    total_realized_pl: float
    by_month: tuple[tuple[str, float], ...]  # ("2026-01", 600.0), chronological
    top_winners: tuple[TickerPerformance, ...]
    top_losers: tuple[TickerPerformance, ...]  # most negative first
    closed_campaigns: int
    winning_campaigns: int
    commission_total: float  # signed as IBKR reports it (negative = paid)

    @property
    def win_rate(self) -> float | None:
        if self.closed_campaigns == 0:
            return None
        return self.winning_campaigns / self.closed_campaigns


@dataclass(frozen=True)
class Campaign:
    """A continuously-open run of fills in one contract (same raw symbol).

    Starts at the first opening fill and ends when net quantity returns to
    zero; reopening the same contract is a new campaign (rolls don't chain).
    """

    symbol: str
    root_ticker: str
    fills: tuple[Trade, ...]
    net_quantity: float

    @property
    def is_open(self) -> bool:
        return self.net_quantity != 0

    @property
    def realized_pl(self) -> float:
        return sum(f.realized_pl for f in self.fills)

    @property
    def avg_entry_price(self) -> float:
        """Quantity-weighted price of the fills that opened the position.

        Only fills in the campaign's opening direction count, so a
        profit-taking scale-out doesn't move the entry price.
        """
        opened_long = self.fills[0].quantity > 0
        opens = [f for f in self.fills if (f.quantity > 0) == opened_long]
        qty = sum(abs(f.quantity) for f in opens)
        return sum(abs(f.quantity) * f.price for f in opens) / qty

    @property
    def scale_outs(self) -> int:
        """Closing-direction fills recorded so far — the ladder rungs taken.

        Counts fill events, not contracts ("sell one contract" is one rung,
        even if a single order prints as one multi-contract fill). Only fills
        inside the statement window are visible, so a pre-window scale-out
        does not count — same bound as the ledger itself.
        """
        opened_long = self.fills[0].quantity > 0
        return sum(1 for f in self.fills if (f.quantity > 0) != opened_long)

    @property
    def house_money(self) -> bool:
        """Open runner funded entirely by extracted cash.

        True when the position is still open and gross cash pulled from sales
        covers the cash paid on entries — i.e. signed proceeds net to >= 0.
        (Matches his transcript: extracted $600 vs $582 in → 100% house money.)
        """
        return self.is_open and sum(f.proceeds for f in self.fills) >= 0
