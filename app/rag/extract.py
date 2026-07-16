"""PyMuPDF structural extraction for the Hebrew desk reviews (ADR-0004).

We keep PyMuPDF's structural map — per-span font size + weight — because the
adaptive chunker needs it to tell headings from body. Every span's text is run
through the deterministic RTL-repair pass on the way out, so downstream code
never sees the extraction artifacts.

Layout tables get their own path: naive layout-order extraction shreds the
row structure that binds a name to its desk action, so ``extract_lines``
detects tables with ``page.find_tables()`` and emits one pipe-joined line per
row, cells in RTL reading order (rightmost column first). Cell text is
reconstructed from the page's text fragments by bbox — ``table.extract()``
itself returns character-reversed Hebrew and is never used.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import fitz  # pymupdf

from app.rag.rtl import repair_rtl

# pymupdf span "flags": bit 2**4 marks a synthetic/real bold face.
_BOLD_FLAG = 1 << 4

_DATE = re.compile(r"\b(\d{2})/(\d{2})/(\d{2})\b")  # DD/MM/YY in the review header


@dataclass(frozen=True)
class Span:
    """A single extracted text run with the layout signal the chunker uses."""

    text: str
    size: float
    font: str
    bold: bool
    page: int  # 1-indexed


@dataclass(frozen=True)
class Line:
    """A visual line: spans joined, with the size the chunker classifies on."""

    text: str
    size: float  # representative = largest span on the line (headings are uniform)
    bold: bool  # any bold span on the line
    page: int  # 1-indexed


_WS = re.compile(r"\s+")


def extract_spans(path: str) -> list[Span]:
    """Return every text span in reading order, RTL-repaired, across all pages."""
    spans: list[Span] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for s in line.get("spans", []):
                        font = s.get("font", "")
                        bold = bool(s.get("flags", 0) & _BOLD_FLAG) or "Bold" in font
                        spans.append(
                            Span(
                                text=repair_rtl(s["text"]),
                                size=round(s["size"], 1),
                                font=font,
                                bold=bold,
                                page=page_index,
                            )
                        )
    return spans


def extract_lines(path: str) -> list[Line]:
    """Return visual lines in reading order, spans joined and RTL-repaired.

    A line's representative ``size`` is the largest span size on it, so a
    heading line reads as a heading even if it carries a trailing small span.
    Blank lines are dropped. Lines inside a detected table region are replaced
    by one serialized line per table row (body-sized, so a big-font header row
    can never split a section), spliced into the flow at the table's position.
    """
    lines: list[Line] = []
    with fitz.open(path) as doc:
        body_size = _doc_body_size(doc)
        for page_index, page in enumerate(doc, start=1):
            lines.extend(_page_lines(page, page_index, body_size))
    return lines


@dataclass(frozen=True)
class _Fragment:
    """One dict-line's contribution to one table cell, kept with its position."""

    text: str
    x0: float
    y0: float


# Fragments whose y0 differ by no more than this sit on one visual row.
_ROW_TOLERANCE = 3.0


def _page_lines(page, page_index: int, body_size: float) -> list[Line]:
    """One page's lines: prose as-is, table regions as serialized rows."""
    tables = page.find_tables().tables
    table_rects = [fitz.Rect(t.bbox) for t in tables]
    entries: list[tuple[float, Line]] = []
    raw_by_table: list[list[dict]] = [[] for _ in tables]

    for block in page.get_text("dict")["blocks"]:
        for raw in block.get("lines", []):
            raw_spans = raw.get("spans", [])
            if not raw_spans:
                continue
            text = _WS.sub(" ", "".join(repair_rtl(s["text"]) for s in raw_spans)).strip()
            if not text:
                continue
            rect = fitz.Rect(raw["bbox"])
            center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
            table_index = next(
                (i for i, tr in enumerate(table_rects) if tr.contains(center)), None
            )
            if table_index is not None:
                raw_by_table[table_index].append(raw)
                continue
            size = round(max(s["size"] for s in raw_spans), 1)
            bold = any(
                (s.get("flags", 0) & _BOLD_FLAG) or "Bold" in s.get("font", "")
                for s in raw_spans
            )
            entries.append(
                (rect.y0, Line(text=text, size=size, bold=bold, page=page_index))
            )

    for table, raws in zip(tables, raw_by_table):
        entries.extend(_table_row_lines(table, raws, page_index, body_size))
    entries.sort(key=lambda entry: entry[0])
    return [line for _, line in entries]


def _table_row_lines(
    table, raw_lines: list[dict], page_index: int, body_size: float
) -> list[tuple[float, Line]]:
    """Serialize a table: one pipe-joined line per row, cells right-to-left.

    Cell content is gathered at *span* granularity (one dict-line can stretch
    across every column — e.g. a header row), but within one line's cell group
    the spans keep document order: that order is already logical, and x-sorting
    would reverse multi-span LTR runs.
    """
    grid = [[fitz.Rect(c) for c in row.cells if c is not None] for row in table.rows]
    cells = [(ri, ci, rect) for ri, row in enumerate(grid) for ci, rect in enumerate(row)]
    content: dict[tuple[int, int], list[_Fragment]] = {}
    for raw in raw_lines:
        groups: dict[tuple[int, int], list[str]] = {}
        left_edge: dict[tuple[int, int], float] = {}
        for span in raw.get("spans", []):
            rect = fitz.Rect(span["bbox"])
            point = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
            hit = next(
                ((ri, ci) for ri, ci, cell in cells if cell.contains(point)), None
            )
            if hit is None:
                if not cells:
                    continue
                ri, ci, _ = min(cells, key=lambda c: _distance(c[2], point))
                hit = (ri, ci)
            groups.setdefault(hit, []).append(repair_rtl(span["text"]))
            left_edge[hit] = min(left_edge.get(hit, rect.x0), rect.x0)
        for hit, texts in groups.items():
            text = _WS.sub(" ", "".join(texts)).strip()
            if text:
                content.setdefault(hit, []).append(
                    _Fragment(text=text, x0=left_edge[hit], y0=raw["bbox"][1])
                )

    rows: list[tuple[float, Line]] = []
    for ri, row in enumerate(table.rows):
        # Rightmost cell first: the logical first column of an RTL layout.
        order = sorted(range(len(grid[ri])), key=lambda ci: -grid[ri][ci].x0)
        text = " | ".join(_cell_text(content.get((ri, ci), [])) for ci in order)
        if not text.strip(" |"):
            continue
        rows.append(
            (
                fitz.Rect(row.bbox).y0,
                Line(text=text.strip(), size=body_size, bold=False, page=page_index),
            )
        )
    return rows


def _cell_text(fragments: list[_Fragment]) -> str:
    """A cell's text: fragments grouped into visual rows, right-to-left."""
    ordered: list[str] = []
    visual_row: list[_Fragment] = []
    for fragment in sorted(fragments, key=lambda f: f.y0):
        if visual_row and fragment.y0 - visual_row[0].y0 > _ROW_TOLERANCE:
            ordered.extend(f.text for f in sorted(visual_row, key=lambda f: -f.x0))
            visual_row = []
        visual_row.append(fragment)
    ordered.extend(f.text for f in sorted(visual_row, key=lambda f: -f.x0))
    return " ".join(ordered).strip()


def _distance(rect: "fitz.Rect", point: "fitz.Point") -> float:
    dx = max(rect.x0 - point.x, 0.0, point.x - rect.x1)
    dy = max(rect.y0 - point.y, 0.0, point.y - rect.y1)
    return dx * dx + dy * dy


def _doc_body_size(doc) -> float:
    """Char-weighted modal span size across the open document."""
    weights: Counter[float] = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for s in line.get("spans", []):
                    weights[round(s["size"], 1)] += len(s["text"].strip())
    return weights.most_common(1)[0][0] if weights else 0.0


def modal_body_size(spans: list[Span]) -> float:
    """Char-weighted most-common font size — the document's body text size."""
    weights: Counter[float] = Counter()
    for s in spans:
        weights[s.size] += len(s.text.strip())
    return weights.most_common(1)[0][0]


def page_count(path: str) -> int:
    with fitz.open(path) as doc:
        return doc.page_count


def detect_doc_type(spans: list[Span]) -> str:
    """Classify the review as ``"daily"`` or ``"weekly"`` from its title text."""
    text = " ".join(s.text for s in spans)
    if "יומית" in text:
        return "daily"
    if "שבועית" in text:
        return "weekly"
    return "unknown"


def detect_review_date(spans: list[Span]) -> str | None:
    """Return the review date as an ISO ``YYYY-MM-DD`` string, if present.

    Dates in the header are DD/MM/YY (two-digit year in the 2000s).
    """
    for s in spans:
        m = _DATE.search(s.text)
        if m:
            dd, mm, yy = m.groups()
            return f"20{yy}-{mm}-{dd}"
    return None
