"""The answering agent's node mechanics: deterministic fact digest, grounding
(digest + thread tool results), the upload block, and the cold-start refusal.

Offline: a fake chat model returns a canned answer, so we test that the digest is
built verbatim from the typed evidence, that grounding picks up tool results from
the thread, and that a toolless route with no evidence refuses instead of inventing.
"""

from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graphs.trading_assistant.answer import build_facts, make_answer_node
from app.graphs.trading_assistant.deps import AgentContext
from app.graphs.trading_assistant.state import EvidenceItem
from app.trading.domain import (
    ExposureCheck,
    ExposureReport,
    HedgeCheck,
    MissingData,
    ScaleOutCandidate,
    ScaleOutSignal,
)


def _exposure_evidence():
    report = ExposureReport(
        nav=120_000.0,
        checks=(ExposureCheck(label="options", value_base=8760.0, pct_of_nav=0.073, limit=0.10),),
    )
    return EvidenceItem("check_exposure", ok=True, result=report)


# --- build_facts ---------------------------------------------------------


def test_digest_formats_exposure_numbers_verbatim():
    digest, asks = build_facts([_exposure_evidence()], [])
    assert "7.3% of NAV" in digest
    assert "limit 10%" in digest
    assert "$120,000" in digest
    assert "within policy" in digest
    assert asks == ()


def test_scaleout_ladder_state_is_rendered_in_digest():
    # The trader must be able to tell which rung each position is on: the due
    # flags carry the recorded-sales count and the action, and a runner is
    # reported as inventory (no action) even when it has melted below entry+.
    due = ScaleOutCandidate(
        "AAOI", ScaleOutSignal.FIRST_TRANCHE_DUE, 1.2, 2.0, 4.4,
        scales_taken=0, quantity=5,
    )
    runner = ScaleOutCandidate(
        "IGV", ScaleOutSignal.MOONSHOT_RUNNER, 0.25, 0.40, 0.50,
        scales_taken=2, quantity=1,
    )
    digest, _ = build_facts(
        [EvidenceItem("scan_scaleout", ok=True, result=[due, runner])], []
    )
    assert "AAOI: first tranche due" in digest
    assert "+120% vs entry" in digest
    assert "0 of 2 scale-out sales recorded" in digest
    assert "IGV: moonshot runner" in digest
    assert "2 sales recorded" in digest
    assert "+25%" in digest  # a melted runner shows its real gain
    assert "manual" in digest  # the endgame is explicitly the trader's job


def _hedged_exposure_evidence(puts: float, calls: float):
    report = ExposureReport(
        nav=120_000.0,
        checks=(ExposureCheck(label="options", value_base=8760.0, pct_of_nav=0.073, limit=0.10),),
        hedge=HedgeCheck(
            put_value_base=puts, call_value_base=calls,
            ratio=puts / calls, low=0.10, high=0.15,
        ),
    )
    return EvidenceItem("check_exposure", ok=True, result=report)


def test_under_hedged_digest_prices_the_gap_in_dollars():
    # All-calls book (his real failure mode): the band x call value, minus the
    # puts he holds, is the dollar range of puts to buy — computed in code so
    # the audit backs it. 10–15% of $4,730 = $473–$710.
    digest, _ = build_facts([_hedged_exposure_evidence(puts=0.0, calls=4730.0)], [])
    assert "UNDER-HEDGED" in digest
    assert "$473–$710" in digest
    assert "puts" in digest


def test_within_band_hedge_prints_no_gap():
    digest, _ = build_facts([_hedged_exposure_evidence(puts=567.6, calls=4730.0)], [])
    assert "within band" in digest
    assert "puts needed" not in digest


def _pnl_evidence(*lines):
    from app.trading.domain import PositionPnL, UnrealizedPnLReport

    pnl_lines = tuple(
        PositionPnL(symbol=s, avg_entry_price=e, mark_price=m, gain=g, unrealized_pl=pl)
        for s, e, m, g, pl in lines
    )
    report = UnrealizedPnLReport(
        total_unrealized_pl=sum(l.unrealized_pl for l in pnl_lines), lines=pnl_lines
    )
    return EvidenceItem("open_position_pnl", ok=True, result=report)


def test_pnl_outlier_is_decomposed_in_digest():
    # SMTOY carries -$19,274 of a -$20,186 total at -90.7% since entry (the real
    # ADR-ratio artifact): the digest must name it and flag the suspect basis,
    # so the total is never read as a book-wide bleed.
    digest, _ = build_facts(
        [_pnl_evidence(
            ("SMTOY", 88.53, 8.22, -0.907, -19274.0),
            ("AEHR", 2.0, 3.0, 0.5, -912.0),
        )],
        [],
    )
    assert "SMTOY" in digest
    assert "-$19,274" in digest
    assert "-90.7%" in digest
    assert "entry basis looks suspect" in digest


def test_outlier_share_is_of_the_net_total_not_gross_swings():
    # The real book: 60 positions with big offsetting winners and losers, yet
    # SMTOY explains 95% of the NET total the trader actually reads. Offsetting
    # pairs must not dilute the trigger (a gross-absolute denominator would).
    digest, _ = build_facts(
        [_pnl_evidence(
            ("SMTOY", 88.53, 8.22, -0.907, -19274.0),
            ("AEHR", 1.0, 3.0, 2.0, 12000.0),
            ("IGV", 4.0, 2.0, -0.5, -12000.0),
            ("GOOGL", 176.89, 359.91, 1.03, 5490.0),
            ("PENG", 4.0, 2.0, -0.5, -6402.0),
        )],
        [],
    )
    assert "SMTOY alone accounts for" in digest


def test_balanced_pnl_book_has_no_outlier_line():
    # Two names splitting the loss evenly: neither dominates, no decomposition.
    digest, _ = build_facts(
        [_pnl_evidence(
            ("AAOI", 2.0, 1.0, -0.5, -500.0),
            ("PENG", 4.0, 2.0, -0.5, -500.0),
        )],
        [],
    )
    assert "alone accounts for" not in digest


def test_digest_renders_the_full_policy_rulebook():
    # "What's my current policy?" is a day-one question (ADR-0006 cold-start),
    # but only check-embedded limits used to reach the model. The full record
    # must land in the digest so every rule value is citable and audit-backed.
    from app.graphs.trading_assistant.policy_model import DEFAULT_POLICY

    digest, _ = build_facts([EvidenceItem("policy_rules", ok=True, result=DEFAULT_POLICY)], [])
    assert "### Exposure rulebook (policy v1)" in digest
    assert "options exposure cap: 10.0%" in digest
    assert "per-option entry size: 1.0%" in digest
    assert "second scale-out trigger: +200%" in digest
    # Manual/undefined rules are deliberately NOT in the record (the record
    # carries only rules code enforces): moonshot endgame, IV shield, and the
    # never-defined offensive-exposure cap all live in the backlog instead.
    assert "moonshot" not in digest
    assert "IV-shield" not in digest
    assert "offensive" not in digest


def test_digest_stamps_statement_freshness():
    digest, _ = build_facts(
        [EvidenceItem("statement_as_of", ok=True, result=date(2026, 7, 3))], []
    )
    assert "2026-07-03" in digest


def test_missing_becomes_upload_asks():
    _, asks = build_facts([], [MissingData("ledger", "Upload a recent activity statement.")])
    assert asks == ("- ledger: Upload a recent activity statement.",)


# --- answer node ---------------------------------------------------------


class _FakeChat:
    def __init__(self, text):
        self.text = text
        self.seen = None

    def invoke(self, messages):
        self.seen = messages
        return AIMessage(content=self.text)


def _run(evidence, missing, answer="You're at 7.3% of NAV, within your 10% limit.",
         messages=None):
    node = make_answer_node(AgentContext(chat_model=_FakeChat(answer)))
    return node({
        "messages": messages or [HumanMessage(content="am I ok?")],
        "evidence": evidence,
        "missing": missing,
    })


def test_answer_grounds_draft_and_counts_attempt():
    out = _run([_exposure_evidence()], [])
    assert "7.3%" in out["synthesis"].answer
    assert "7.3% of NAV" in out["synthesis"].grounding
    assert out["synthesis_attempts"] == 1
    assert "messages" not in out  # the draft is audit-gated, never a thread message


def test_answer_appends_upload_block_deterministically():
    out = _run([_exposure_evidence()], [MissingData("ledger", "Upload a recent activity statement.")])
    answer = out["synthesis"].answer
    assert "To answer the rest, upload:" in answer
    assert "ledger: Upload a recent activity statement." in answer


def test_answer_refuses_when_toolless_with_no_evidence_or_missing():
    out = _run([], [])
    assert "don't have the tools" in out["synthesis"].answer
    assert out["synthesis"].grounding == ""


def test_trade_signal_prompt_mandates_the_sizing_tool():
    # A pasted signal must be sized by the deterministic tool, never freehand:
    # the system prompt names the tool and carries the verbatim shorthand.
    from app.graphs.trading_assistant.state import Scope

    fake = _FakeChat("sized.")
    node = make_answer_node(AgentContext(chat_model=fake))
    node({
        "messages": [HumanMessage(content="AAOI 150 NEXT WEEK 3.1")],
        "evidence": [_exposure_evidence()],
        "missing": [],
        "scope": Scope(
            intents=["trade_signal_eval"], signal_text="AAOI 150 NEXT WEEK 3.1"
        ),
    })
    system = fake.seen[0].content
    assert "size_trade_signal" in system
    assert "AAOI 150 NEXT WEEK 3.1" in system


def test_policy_note_rides_into_the_answer_prompt():
    # A failed policy-change parse with co-intents leaves a note in state; the
    # answering model must see it so the "which rule?" ask lands inside the
    # answer instead of replacing it.
    fake = _FakeChat("All within policy.")
    node = make_answer_node(AgentContext(chat_model=fake))
    node({
        "messages": [HumanMessage(content="what's my current policy?")],
        "evidence": [_exposure_evidence()],
        "missing": [],
        "policy_note": "no specific rule and new value could be identified",
    })
    system = fake.seen[0].content
    assert "no specific rule and new value could be identified" in system


def test_grounding_includes_tool_results_from_the_thread():
    # A number a tool fetched must be audit-backed even though it is not in the
    # pre-fetch digest — grounding = digest + every ToolMessage on the thread.
    thread = [
        HumanMessage(content="how's the market?"),
        AIMessage(content="", tool_calls=[{"name": "get_market_quote",
                                           "args": {"symbols": ["SPY"]},
                                           "id": "c1", "type": "tool_call"}]),
        ToolMessage(content="SPY: $601.23 (REGULAR)", name="get_market_quote",
                    tool_call_id="c1"),
    ]
    out = _run([_exposure_evidence()], [], answer="SPY is at $601.23.", messages=thread)
    assert "$601.23" in out["synthesis"].grounding
    assert "7.3% of NAV" in out["synthesis"].grounding  # digest still there


def test_prior_answers_do_not_ground():
    # A bannered (unbacked) figure delivered last turn must not launder itself
    # into this turn's grounding via the thread.
    thread = [
        HumanMessage(content="am I ok?"),
        AIMessage(content="NAV is $999,999."),  # last turn's delivered answer
        HumanMessage(content="and now?"),
    ]
    out = _run([_exposure_evidence()], [], messages=thread)
    assert "$999,999" not in out["synthesis"].grounding
