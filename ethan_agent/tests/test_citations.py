from agent.citations import check_evidence, check_inline, quote_in_text
from agent.interfaces import Chunk
from agent.schemas import Evidence

CHUNK = Chunk(
    chunk_id="A1", ticker="EXMP", doc_type="10-K",
    text="Data Center segment revenue was $18.4 billion for fiscal 2025, an increase of 61%.",
)


def ev(source: str, quote: str) -> Evidence:
    return Evidence(source=source, quote=quote, stance="supports")


def test_exact_quote_passes():
    valid, issues = check_evidence([ev("S1", "an increase of 61%")], {"S1": CHUNK})
    assert len(valid) == 1 and not issues


def test_whitespace_case_and_smart_quotes_normalised():
    assert quote_in_text("DATA  CENTER segment\nrevenue", CHUNK.text)
    assert quote_in_text("it’s", "it's")


def test_bracketed_label_accepted():
    valid, _ = check_evidence([ev("[S1]", "an increase of 61%")], {"S1": CHUNK})
    assert valid[0].source == "S1"


def test_invented_label_dropped():
    valid, issues = check_evidence([ev("S9", "an increase of 61%")], {"S1": CHUNK})
    assert not valid and "not one of the sources" in issues[0]


def test_fabricated_quote_dropped():
    # A plausible but invented number must not pass.
    valid, issues = check_evidence([ev("S1", "an increase of 75% driven by gaming")], {"S1": CHUNK})
    assert not valid and "quote not found" in issues[0]


def test_tiny_copy_error_tolerated():
    assert quote_in_text("Data Center segment revenue was $18.4 billion for fiscal 2025, an increase of 61", CHUNK.text)


# ---------------------------------------------------------------- inline [S#] labels


def test_inline_valid_labels_untouched():
    text = "Revenue grew 61% [S1]. Margins rose [S1][S2]. Here's what the evidence shows."
    out, issues = check_inline(text, {"S1", "S2"})
    assert out == text and not issues


def test_inline_invented_label_removed():
    out, issues = check_inline("Revenue grew 61% [S1][S7].", {"S1"})
    assert out == "Revenue grew 61% [S1]." and "[S7]" in issues[0]


def test_inline_uncited_number_flagged_but_text_kept():
    out, issues = check_inline("Revenue grew 61% [S1]. Market share is 90%.", {"S1"})
    assert "Market share is 90%." in out
    assert len(issues) == 1 and "90" in issues[0]


def test_inline_ignores_thesis_numbers_years_and_ids():
    text = "You expect 50%+ growth. In fiscal 2025 the Q1 FY2026 report matters."
    _, issues = check_inline(text, set(), ignore_numbers_from="growing 50%+")
    assert not issues


def test_inline_sentence_only_label_removed_flags_number():
    _, issues = check_inline("Share is 90% [S9].", {"S1"})
    assert any("[S9]" in i for i in issues) and any("90" in i for i in issues)
