"""Seam: repair_rtl(span_text) -> corrected span text.

pymupdf extracts the Hebrew desk reviews with two systematic RTL artifacts
(verified on the real 08/07/26 daily): sentence punctuation that belongs at the
END of a span is flipped to its START, and a maqaf (hyphen) that joins a
one-letter Hebrew prefix to a Latin token lands before the prefix instead of
after it. Both are deterministic and repaired per-span before chunking.
"""

import pytest

from app.rag.rtl import repair_rtl


@pytest.mark.parametrize(
    "raw, fixed",
    [
        (",זיכרון", "זיכרון,"),                       # leading comma -> trailing
        (":יותר", "יותר:"),                           # leading colon
        (".שכבה מניות סופגות", "שכבה מניות סופגות."),  # leading period, multi-word
        (",עיוור יותר ", "עיוור יותר, "),             # trailing whitespace preserved
    ],
)
def test_leading_punctuation_moves_to_end(raw, fixed):
    assert repair_rtl(raw) == fixed


@pytest.mark.parametrize(
    "raw, fixed",
    [
        ("-הAI", "ה-AI"),                                 # maqaf before 1-letter prefix
        ("-וSMR", "ו-SMR"),
        ("-בExcel", "ב-Excel"),
        ("-בOpenAI/Anthropic", "ב-OpenAI/Anthropic"),     # Latin token kept intact
        ("מהפכת -הAI נמשכת", "מהפכת ה-AI נמשכת"),          # mid-span, applied globally
    ],
)
def test_maqaf_prefix_moves_after_hebrew_letter(raw, fixed):
    assert repair_rtl(raw) == fixed


def test_clean_span_is_unchanged():
    # No leading punctuation, and a spaced dash is a real dash, not a maqaf.
    assert repair_rtl("לא נעלם מהשוק - הוא משנה ") == "לא נעלם מהשוק - הוא משנה "


def test_non_maqaf_hyphens_untouched():
    # A maqaf joins a Hebrew prefix to a Latin token; other hyphens are real.
    assert repair_rtl("GPT-4 turbo") == "GPT-4 turbo"      # Latin-digit hyphen
    assert repair_rtl("רב-לאומית") == "רב-לאומית"          # Hebrew-Hebrew hyphen


def test_leading_punct_on_ltr_fragment_untouched():
    # The flip is an RTL artifact; leading punctuation on a Latin/numeric run
    # is real and must not be moved to the end.
    assert repair_rtl(".5%") == ".5%"
    assert repair_rtl(".NET") == ".NET"
