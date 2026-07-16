"""Build the frozen Task 5 evaluation dataset -> evals/eval_dataset.jsonl.

The four questions are curated from the real trader's Antigravity transcripts
(sessions 04/07-08/07/2026): one asked verbatim, and three instantiations of
his recurring question patterns (ticker-bias lookup, daily-boot briefing,
catalyst lookup) onto content the committed corpus actually covers. Queries
are English over the mixed English/Hebrew corpus — his real usage.

``reference_contexts`` are resolved at build time from the committed review
PDFs via the production chunker (``chunk_document``), selected by chunk_id —
so the frozen texts are exactly what the retriever can return from Qdrant.
Rerun after any extraction/chunking change; the build fails loudly if a
referenced chunk_id no longer exists.

Run from the repo root:  .venv/bin/python evals/build_eval_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.chunk import chunk_document  # noqa: E402

REVIEWS = ROOT / "data" / "reviews"
OUT = ROOT / "evals" / "eval_dataset.jsonl"

EXAMPLES = [
    {
        "user_input": (
            "What are the main tactical guidelines to follow for this week's "
            "trading sessions?"
        ),
        "question_type": "broad-multi-chunk",
        "question_source": (
            "verbatim trader question, 06/07 session: 'Study it, and summarize "
            "the main tactical guidelines to follow and keep in mind for this "
            "week's trading sessions'"
        ),
        "reference_context_ids": [
            "weekly-2026-07-06-s01-c00",
            "weekly-2026-07-06-s02-c00",
            "weekly-2026-07-06-s02-c01",
            "weekly-2026-07-06-s25-c00",
        ],
        "reference": (
            "The critical rule for the week: don't chase every headline — build "
            "the book around receivers with proven pricing power, funded "
            "carefully against spenders, crowded high-beta trades, and names "
            "that already got all their upside in advance. Core long exposure is "
            "the physical AI stack: memory/HBM/DRAM (Micron, SK Hynix, Samsung — "
            "getting shortage confirmation, LTAs, capex and price hikes), plus "
            "MLCC/capacitors and advanced packaging (Yageo, Samsung "
            "Electro-Mechanics, Murata, TDK, ASE, Amkor, TSMC) as clear "
            "receivers of BOM inflation, and quality semicap (ASML, AMAT, LRCX, "
            "KLAC) — buying only the names that are getting price, capacity, or "
            "contracts. Reduce beta in names that are capex buyers rather than "
            "bottleneck sellers: AI cloud spenders with expensive credit and "
            "unproven ROI (Oracle, Meta, CoreWeave, leveraged neoclouds). Treat "
            "the broadening rotation (regional banks, transports, consumer "
            "discretionary, Russell 2000 quality) as a tactical complement to "
            "crowded AI, not a replacement for the physical thesis. Keep hedges "
            "against a high-beta unwind: SPY/QQQ hedges, partial SMH trims, "
            "watching HYG/LQD credit spreads, and cash discipline; "
            "space/defense/drones can serve as a small policy-beta rotation "
            "valve. The PM decision for the open: start the week positive but "
            "not over-leveraged — overweight physical receivers, "
            "neutral-to-underweight leveraged spenders, tactical broadening "
            "exposure, hedged against a high-beta unwind; increase size only "
            "after price/contract confirmation or an orderly pullback, and "
            "never buy headlines after a gap-up."
        ),
    },
    {
        "user_input": "What is the desk's current view on Micron (MU)?",
        "question_type": "cross-document",
        "question_source": (
            "trader pattern, 04/07 session: 'Can you answer your third question "
            "regarding the bias on AAOI from the most recent desk review?' — "
            "instantiated on the desk's #1 weekly name (he bought MU calls on "
            "06/07)"
        ),
        "reference_context_ids": [
            "weekly-2026-07-06-s06-c00",
            "weekly-2026-07-06-s21-c00",
            "daily-2026-07-08-s02-c03",
            "daily-2026-07-08-s02-c04",
        ],
        "reference": (
            "The desk is core long Micron — it is the #1 name on the weekly "
            "list — but disciplined on entry and with a new legal risk flag. "
            "The weekly review frames MU inside a memory supercycle: Micron "
            "talks about DRAM shortage through 2028 and is building the "
            "Hiroshima expansion for 1-gamma DRAM and HBM, while LTAs and capex "
            "create a contract price floor (but also raise overcapacity risk "
            "after 2028); the cycle is no longer just spot DRAM — it is "
            "contractual, geopolitical and industrial. The desk action is core "
            "long, adding only after a pullback or fresh price confirmation, "
            "watching contract price hikes, margin guidance and NAND/DRAM split "
            "commentary. The daily review keeps the thesis intact but less "
            "clean: ADATA sees another 20-30% rise in Q3 DRAM contract prices "
            "(35-40% in NAND), positive for memory makers, but Samsung, SK "
            "hynix and Micron have been sued in the US over allegedly "
            "coordinated DRAM supply cuts and sharp price increases. The desk "
            "reads the lawsuit as not breaking the memory thesis but changing "
            "the quality of the story — shortage and high profitability "
            "attract regulation and lawsuits — something to watch, not "
            "necessarily an immediate sell trigger."
        ),
    },
    {
        "user_input": (
            "What is the desk's bottom line for today, and what actions does "
            "it recommend before the open?"
        ),
        "question_type": "doc-scoped-summary",
        "question_source": (
            "trader's standing daily-boot briefing, encoded in his AGENTS.md on "
            "04/07: 'a concise summary of the current daily and weekly desk "
            "overview... to prepare for the day's trading session'"
        ),
        "reference_context_ids": [
            "daily-2026-07-08-s01-c00",
            "daily-2026-07-08-s08-c00",
            "daily-2026-07-08-s09-c00",
        ],
        "reference": (
            "Bottom line: AI is not leaving the market — it is changing layers. "
            "Momentum names and crowded semis are absorbing pressure while "
            "money looks for cheaper, infrastructure-heavy exposure: memory, "
            "cloud, data centers, hyperscaler debt, China AI, photonics and "
            "SMR. In parallel, Iran/oil headlines bring the macro risk premium "
            "back, so today's trading must be selective, not momentum-chasing. "
            "Preference goes to names that show a connection between AI and "
            "revenue, capacity, low cost, or a long contract; be careful with "
            "names whose rise rested only on story, index flows, or momentum. "
            "Desk actions for today: (1) check oil, VIX and credit spreads "
            "before the open; (2) don't chase sharp jumps in NET/BABA/PENG "
            "without confirmation; (3) watch whether MU/SK hynix recover "
            "despite the DRAM lawsuit; (4) check whether the Amazon bond issue "
            "affects hyperscaler debt broadly; (5) separate the strong AI theme "
            "from crowded AI stocks. Base scenario: oil calms, the AI rotation "
            "continues, memory stays strong and QQQ stabilizes — hold selective "
            "exposure to quality infra, cloud, and memory without chasing."
        ),
    },
    {
        "user_input": (
            "What does the desk say about the Apple-Broadcom deal, and who "
            "benefits from it?"
        ),
        "question_type": "narrow-single-chunk",
        "question_source": (
            "trader pattern: catalyst/news lookup on a specific name (e.g. "
            "'Did you notice a new review in your folders?' -> summarize what's "
            "new on GLW/ESI), instantiated on a daily-review catalyst"
        ),
        "reference_context_ids": [
            "daily-2026-07-08-s02-c04",
        ],
        "reference": (
            "Apple is expanding its agreement with Broadcom into a deal "
            "expected to exceed $30 billion, producing more than 15 billion "
            "chips in the US. The desk reads it as more than an iPhone supplier "
            "deal: it is a lock-in of an American supply chain against "
            "geopolitical risk, tariffs and dependence on Asia. It is positive "
            "for Broadcom (AVGO) and for the reshoring theme. Names to watch: "
            "AVGO, AAPL, QCOM, SWKS, QRVO."
        ),
    },
]


def main() -> None:
    pdfs = sorted(REVIEWS.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs found under {REVIEWS}")

    texts_by_id: dict[str, str] = {}
    for pdf in pdfs:
        for chunk in chunk_document(str(pdf)):
            texts_by_id[chunk.chunk_id] = chunk.text
    print(f"chunked {len(pdfs)} PDFs -> {len(texts_by_id)} chunks")

    rows = []
    for example in EXAMPLES:
        missing = [
            cid for cid in example["reference_context_ids"] if cid not in texts_by_id
        ]
        if missing:
            raise SystemExit(f"unknown chunk_id(s) {missing} — corpus changed?")
        rows.append(
            {
                "user_input": example["user_input"],
                "reference": example["reference"],
                "reference_contexts": [
                    texts_by_id[cid] for cid in example["reference_context_ids"]
                ],
                "reference_context_ids": example["reference_context_ids"],
                "question_type": example["question_type"],
                "question_source": example["question_source"],
            }
        )

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} examples)")
    for i, row in enumerate(rows):
        n_ctx = len(row["reference_contexts"])
        print(f"  [{i}] {row['question_type']:<20} {n_ctx} ctx | {row['user_input']}")


if __name__ == "__main__":
    main()
