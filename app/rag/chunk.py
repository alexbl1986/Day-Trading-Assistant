"""Adaptive, structure-aware chunking for the desk reviews.

The reviews carry no fixed template, so we chunk on the layout signal instead:
the per-doc char-weighted modal font size is the body, anything ~2pt larger is a
heading, and a chunk is the body between consecutive headings. Long sections are
packed to a character budget and repeated page furniture (headers/footers) is
dropped. Table regions arrive from extraction as pipe-joined row lines, so a
chunk keeps each name on the same row as its desk action.

The section heading is **prepended into the searchable text** of every chunk
(so BM25/dense can match on it) and is *also* kept as the ``section`` metadata
field for citation. Other metadata — source, chunk_id, review_date, doc_type,
pages — follows the course conventions in 01_Dense_Vector_Retrieval /
07_Advanced_Retrievers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.rag.extract import (
    Line,
    detect_doc_type,
    detect_review_date,
    extract_lines,
    extract_spans,
    modal_body_size,
)

# A line repeated on this many distinct pages is page furniture, not content.
_BOILERPLATE_PAGE_THRESHOLD = 3

# A chunk shorter than this is an orphaned fragment (stray heading punctuation,
# a lone char) with no retrieval value; it is merged into its neighbour.
_MIN_CHUNK_CHARS = 40


@dataclass(frozen=True)
class Chunk:
    """A retrieval chunk: packed sub-chunk of a section, heading-in-text."""

    text: str
    doc_type: str
    review_date: str | None
    source: str
    chunk_id: str
    section: str | None
    pages: tuple[int, ...]


@dataclass
class _Section:
    heading: str | None
    lines: list[Line]


def chunk_document(
    path: str,
    *,
    heading_gap: float = 2.0,
    min_chars: int = 200,
    max_chars: int = 1200,
) -> list[Chunk]:
    """The retrieval chunks (heading-injected, section-tagged) for a review."""
    spans = extract_spans(path)
    doc_type = detect_doc_type(spans)
    review_date = detect_review_date(spans)
    source = Path(path).name
    threshold = modal_body_size(spans) + heading_gap
    sections = _split_into_sections(_drop_boilerplate(extract_lines(path)), threshold)

    chunks: list[Chunk] = []
    for s_idx, section in enumerate(sections):
        packs = _pack(section.lines, min_chars, max_chars)
        if not packs:
            continue  # heading-only section -> nothing to index

        heading = section.heading
        # Reserve room so the injected heading keeps chunks within budget.
        body_budget = max(max_chars - (len(heading) + 1 if heading else 0), min_chars)
        c_idx = 0
        for piece in packs:
            pages = tuple(sorted({ln.page for ln in piece}))
            body = _join_lines(piece)
            for sub in _split_long(body, body_budget):
                chunks.append(
                    Chunk(
                        text=_with_heading(heading, sub),
                        doc_type=doc_type,
                        review_date=review_date,
                        source=source,
                        chunk_id=(
                            f"{doc_type}-{review_date or 'unknown'}"
                            f"-s{s_idx:02d}-c{c_idx:02d}"
                        ),
                        section=heading,
                        pages=pages,
                    )
                )
                c_idx += 1

    return _merge_fragments(chunks)


def _with_heading(heading: str | None, body: str) -> str:
    return f"{heading}\n{body}".strip() if heading else body


def _join_lines(lines: list[Line]) -> str:
    """Join a pack's lines with spaces, except around table rows (pipe-joined
    by extraction), which keep their own line so row boundaries survive."""
    body = ""
    prev_is_row = False
    for line in lines:
        is_row = " | " in line.text
        if body:
            body += "\n" if (is_row or prev_is_row) else " "
        body += line.text
        prev_is_row = is_row
    return body.strip()


def _is_heading(line: Line, threshold: float) -> bool:
    # Big enough to be a heading, and actually a title (has a letter, short,
    # not a lone page number or dash).
    if line.size < threshold:
        return False
    return bool(re.search(r"[A-Za-zא-ת]", line.text)) and len(line.text) <= 80


def _split_into_sections(lines: list[Line], threshold: float) -> list[_Section]:
    """Group body lines under their heading.

    A single visual heading is often laid out as several stacked heading-sized
    lines (Hebrew wrapping + mixed RTL/LTR tokens), so consecutive heading lines
    are joined into one heading rather than each starting an empty section —
    otherwise all but the last fragment would be dropped.
    """
    sections: list[_Section] = []
    current = _Section(heading=None, lines=[])
    for line in lines:
        if _is_heading(line, threshold):
            if current.lines:
                # Body already collected: this heading begins a new section.
                sections.append(current)
                current = _Section(heading=line.text, lines=[])
            elif current.heading is None:
                current.heading = line.text
            else:
                # Consecutive heading lines are one wrapped heading.
                current.heading = f"{current.heading} {line.text}".strip()
        else:
            current.lines.append(line)
    if current.heading is not None or current.lines:
        sections.append(current)
    return sections


def _drop_boilerplate(lines: list[Line]) -> list[Line]:
    pages_by_text: dict[str, set[int]] = {}
    for line in lines:
        pages_by_text.setdefault(line.text, set()).add(line.page)
    return [
        line
        for line in lines
        if len(pages_by_text[line.text]) < _BOILERPLATE_PAGE_THRESHOLD
    ]


def _pack(lines: list[Line], min_chars: int, max_chars: int) -> list[list[Line]]:
    """Group a section's lines into chunks within the character budget.

    Lines accumulate until the next would exceed ``max_chars``; a trailing chunk
    smaller than ``min_chars`` is merged back into the previous one so we don't
    emit fragments.
    """
    if not lines:
        return []
    packed: list[list[Line]] = []
    buf: list[Line] = []
    size = 0
    for line in lines:
        if buf and size + len(line.text) + 1 > max_chars:
            packed.append(buf)
            buf, size = [], 0
        buf.append(line)
        size += len(line.text) + 1
    if buf:
        packed.append(buf)

    if len(packed) > 1 and sum(len(l.text) for l in packed[-1]) < min_chars:
        packed[-2].extend(packed.pop())
    return packed


def _split_long(text: str, max_chars: int) -> list[str]:
    """Sub-split an over-budget chunk at whitespace, hard-cutting if it must."""
    pieces: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def _merge_fragments(chunks: list[Chunk]) -> list[Chunk]:
    """Fold sub-``_MIN_CHUNK_CHARS`` fragments into an adjacent chunk.

    A fragment merges into the previous chunk (keeping that chunk's section),
    or into the next one if it is the very first chunk.
    """
    merged: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) < _MIN_CHUNK_CHARS and merged:
            merged[-1] = _join(merged[-1], chunk)
        else:
            merged.append(chunk)
    # A leading fragment couldn't merge backward; merge it forward.
    if len(merged) > 1 and len(merged[0].text) < _MIN_CHUNK_CHARS:
        merged[1] = _join(merged[0], merged[1])
        merged.pop(0)
    return merged


def _join(a: Chunk, b: Chunk) -> Chunk:
    return Chunk(
        text=f"{a.text} {b.text}".strip(),
        doc_type=a.doc_type,
        review_date=a.review_date,
        source=a.source,
        chunk_id=a.chunk_id,
        section=a.section if a.section is not None else b.section,
        pages=tuple(sorted(set(a.pages) | set(b.pages))),
    )
