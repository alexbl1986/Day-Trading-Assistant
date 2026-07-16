"""Regenerate docs/chunk_preview/ from the committed review PDFs.

One markdown file per review, named after its doc_type and date
(``weekly_06_07_26.md``), showing every retrieval chunk in index order with
its full metadata — the human checkpoint for evaluating extraction/chunking
changes before they reach the index.

Run from Certification_Challenge:  .venv/bin/python scripts/preview_chunks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.chunk import Chunk, chunk_document  # noqa: E402

REVIEWS = ROOT / "data" / "reviews"
PREVIEW = ROOT / "docs" / "chunk_preview"


def _render(chunks: list[Chunk]) -> str:
    first = chunks[0]
    lengths = [len(c.text) for c in chunks]
    lines = [
        f"# Chunk preview — `{first.source}`",
        "",
        f"- **doc_type:** {first.doc_type} · **review_date:** {first.review_date}"
        f" · **source:** {first.source}  ",
        f"- **chunks:** {len(chunks)} · **char length:** min {min(lengths)}"
        f" / avg {sum(lengths) // len(lengths)} / max {max(lengths)}",
        "",
        "",
        "The section heading is the first line of each chunk's searchable text"
        " (also kept as `section`). Table rows are pipe-joined, cells in RTL"
        " reading order.",
        "",
        "---",
        "",
    ]
    for i, c in enumerate(chunks, start=1):
        lines += [
            "",
            f"### Chunk {i}/{len(chunks)} · `{c.chunk_id}`",
            f"`section:` {c.section!r} · `pages:` {list(c.pages)}"
            f" · `chars:` {len(c.text)}",
            "",
            "> " + c.text.replace("\n", "\n"),
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    pdfs = sorted(REVIEWS.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs found under {REVIEWS}")
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        chunks = chunk_document(str(pdf))
        yyyy, mm, dd = (chunks[0].review_date or "unknown-00-00").split("-")
        out = PREVIEW / f"{chunks[0].doc_type}_{dd}_{mm}_{yyyy[2:]}.md"
        out.write_text(_render(chunks), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)} ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
