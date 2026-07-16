# PyMuPDF + deterministic RTL-repair pass for desk-review extraction

The desk reviews are Hebrew (RTL) PDFs, and no extractor handles them cleanly out of the
box. A head-to-head on the real 08/07/26 daily review showed: **pypdf** extracts content
perfectly (correct punctuation and prefix order, zero artifacts) but flattens the document
to one word per line with no paragraph signal; **PyMuPDF** preserves the full structural
map (blocks + font sizes: 13pt section headers, 11pt sub-headers) that section-aware
chunking needs, but introduces ~173 systematic artifacts — leading punctuation that
belongs at the end of the same token (`,זיכרון` → `זיכרון,`) and a maqaf misplaced around
one-letter Hebrew prefixes on Latin acronyms (`-הAI` → `ה-AI`). Both artifact classes are
mechanical, so we chose **PyMuPDF plus a small deterministic repair pass**, unit-tested
against pypdf's output as ground truth. We rejected pypdf (structure loss is unrecoverable,
and structure is what chunking needs most) and vision-LLM page parsing (cleanest output,
but non-deterministic ingestion would muddy the Task 5/6 eval comparisons; also an API
dependency for a job that runs offline in seconds). PyMuPDF is AGPL: acceptable because
the cert repo is public; if commercialized later, the extraction step is one isolated
ingestion module and can be swapped without touching the agent.

## Amendment (2026-07-14) — table-aware serialization

A studio-run review showed the reviews' layout tables (weekly action map, daily stock
map) linearized into cell soup: PyMuPDF walks table cells in layout order, destroying
the row structure that binds a name to its desk action — the agent missed the
AMKR/PENG/RKLB calls precisely because of it. `extract_lines` now detects tables per
page with `page.find_tables()` and emits **one pipe-joined line per row, cells in RTL
reading order** (rightmost column = first logical column). Cell text is reconstructed
from the page's text-dict **spans** by bbox containment — span granularity is required
because a single dict-line can stretch across every column (header rows do) — keeping
document order *within* a cell so multi-span LTR runs are not reversed, and ordering
separate fragments right-to-left within a visual row. `table.extract()` itself was
rejected: it returns character-reversed Hebrew. Row lines carry the doc's modal body
size so a large-font header row can never split a section; raw fragments inside table
bboxes are excluded from the prose flow, and rows splice into it by y-position. The
chunker keeps each row on its own line, so name↔action bindings survive into chunks
intact (test-pinned on the real weekly action map).
