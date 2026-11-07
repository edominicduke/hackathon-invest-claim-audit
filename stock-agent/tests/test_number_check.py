"""Number check tests (Bogdan's part)."""

import pytest

from src.checks.number_check import check_numbers, find_numbers

SOURCE = [{
    "label": "10-Q p.8",
    "text": "Data center revenue was $4,213 million, up 18.4% from last year. "
            "Gross margin was 74.6%. The stock trades at 35.2x earnings. Total debt: $8.5B.",
}]


@pytest.mark.parametrize("claim", [
    "$4.2 billion",        # rounded from $4,213 million
    "$4,213 million",
    "$4.21B",
    "18%",                 # rounded from 18.4%
    "18.4 percent",
    "about 75%",           # 74.6 rounds to 75
    "35x earnings",
    "$8.5 billion",
    "$8,500 million",
])
def test_correct_numbers_are_found(claim):
    result = check_numbers(f"The report says {claim}.", SOURCE)
    assert result["total"] == 1, result
    assert result["all_found"], result
    assert result["numbers"][0]["source"] == "10-Q p.8"


@pytest.mark.parametrize("claim", [
    "$5.1 billion",        # made up
    "$4.3 billion",        # wrong after rounding
    "31%",
    "18.9%",               # too precise and wrong
    "4.2%",                # right digits, wrong kind (% not $)
    "50x earnings",
])
def test_made_up_numbers_are_flagged(claim):
    result = check_numbers(f"The report says {claim}.", SOURCE)
    assert result["total"] == 1, result
    assert result["not_found"] == [result["numbers"][0]["text"]]


def test_mixed_answer():
    answer = "Revenue was $4.2 billion (up 18%), and margins hit 81%. [10-Q, p.8]"
    result = check_numbers(answer, SOURCE)
    assert result["total"] == 3
    assert result["found"] == 2
    assert result["not_found"] == ["81%"]
    assert result["all_found"] is False


def test_ignores_citations_years_quarters_and_forms():
    text = "In Q3 2026 the 10-K and 10-Q [source 2, p.14] were filed."
    assert find_numbers(text) == []


def test_ignores_time_spans_but_checks_percentage_points():
    nums = find_numbers("Over 12 months and 3 years, margin rose 1.5 percentage points.")
    assert [(n["text"], n["kind"]) for n in nums] == [("1.5 percentage points", "percent")]


def test_no_sources():
    result = check_numbers("Revenue was $4.2 billion.", [])
    assert result["found"] == 0
    assert result["note"]


def test_plain_string_sources_work():
    result = check_numbers("Margin was 74.6%.", ["Gross margin was 74.6%."])
    assert result["all_found"]
    assert result["numbers"][0]["source"] == "source 1"


def test_no_numbers_is_all_found():
    result = check_numbers("Management sounded confident.", SOURCE)
    assert result["total"] == 0 and result["all_found"]
