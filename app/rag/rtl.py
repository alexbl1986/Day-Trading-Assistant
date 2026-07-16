"""Deterministic RTL-repair pass for pymupdf-extracted Hebrew desk reviews.

pymupdf preserves the structural map we need for section-aware chunking but
introduces two systematic RTL artifacts (ADR-0004, verified on the real
08/07/26 daily): sentence-final punctuation is flipped to the START of a span,
and a maqaf that joins a one-letter Hebrew prefix to a Latin token lands before
the prefix. Both are mechanical, so we repair each span deterministically
before chunking, unit-tested against pypdf's output as ground truth.
"""

import re

# Sentence punctuation that gets flipped to the start of a span on extraction.
# A spaced dash (" - ") is a real dash, not an artifact, so "-" is excluded.
_LEADING_PUNCT = ".,:;!?"

# Maqaf artifact: a hyphen joining a one-letter Hebrew prefix (ה/ו/ב/כ/ל/מ/ש …)
# to a Latin token lands BEFORE the prefix ("-הAI") instead of after it
# ("ה-AI"). Match an unspaced hyphen + single Hebrew letter + Latin char and
# swap the first two. Hebrew-Hebrew and Latin-only hyphens don't match, so real
# hyphens ("GPT-4", "רב-לאומית") are left untouched.
_MAQAF = re.compile(r"-([א-ת])([A-Za-z])")

_HEBREW = re.compile(r"[א-ת]")


def repair_rtl(span_text: str) -> str:
    """Return ``span_text`` with both RTL extraction artifacts repaired.

    1. Leading sentence punctuation is moved to after the last non-whitespace
       character, preserving any trailing whitespace.
    2. A misplaced maqaf is moved from before a one-letter Hebrew prefix to
       between the prefix and its Latin token (applied to every occurrence).

    Clean spans are returned unchanged.
    """
    span_text = _repair_leading_punct(span_text)
    return _MAQAF.sub(r"\1-\2", span_text)


def _repair_leading_punct(span_text: str) -> str:
    # The flip only happens to RTL (Hebrew) runs; a leading "." on an LTR
    # fragment like ".5%" or ".NET" is real and must be left in place.
    if not span_text or span_text[0] not in _LEADING_PUNCT:
        return span_text
    if not _HEBREW.search(span_text):
        return span_text

    punct, rest = span_text[0], span_text[1:]
    body = rest.rstrip()
    trailing = rest[len(body):]
    return f"{body}{punct}{trailing}"
