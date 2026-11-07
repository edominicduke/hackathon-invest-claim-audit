"""Storage tests (Abdullah). Run: python -m pytest -q"""

import pytest

from src.app.store import NotFoundError, Storage, ValidationError


def rec_data(**over):
    data = {
        "company": "nvda",
        "user_reason": "AI chip demand is growing",
        "verdict": "yellow",
        "summary": "Data center revenue was $4.2 billion, up 18%.",
        "sources": [{"label": "10-Q p.8", "text": "Data center revenue was $4,213 million, up 18.4%."}],
        "as_of_date": "2026-01-05",
    }
    data.update(over)
    return data


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path)


def test_create_and_get_recommendation(storage):
    rec = storage.recommendations.create(rec_data())
    assert rec["company"] == "NVDA"          # ticker is upper-cased
    assert rec["results"] == {}
    assert storage.recommendations.get(rec["id"]) == rec


def test_sources_can_be_plain_strings(storage):
    rec = storage.recommendations.create(rec_data(sources=["Q3 earnings call"]))
    assert rec["sources"] == [{"label": "Q3 earnings call"}]


def test_list_filters(storage):
    storage.recommendations.create(rec_data(company="NVDA", verdict="green"))
    storage.recommendations.create(rec_data(company="AAPL", verdict="red"))
    assert len(storage.recommendations.list()) == 2
    assert [r["company"] for r in storage.recommendations.list(company="aapl")] == ["AAPL"]
    assert [r["verdict"] for r in storage.recommendations.list(verdict="green")] == ["green"]


def test_list_keeps_creation_order(storage):
    ids = [storage.recommendations.create(rec_data()) ["id"] for _ in range(5)]
    assert [r["id"] for r in storage.recommendations.list()] == ids


@pytest.mark.parametrize("bad", [
    {"company": ""},
    {"company": "NOT A TICKER"},
    {"company": "WAYTOOLONGTICKER"},
    {"user_reason": ""},
    {"user_reason": "x" * 4001},
    {"verdict": "buy"},
    {"sources": "not a list"},
    {"sources": [{"label": ""}]},
    {"sources": ["s"] * 51},
    {"as_of_date": "07/11/2026"},
    {"as_of_date": "2999-01-01"},
    {"price_on_date": -5},
])
def test_validation_errors(storage, bad):
    with pytest.raises(ValidationError):
        storage.recommendations.create(rec_data(**bad))


def test_unknown_id_is_not_found(storage):
    with pytest.raises(NotFoundError):
        storage.recommendations.get("20260101-000000-abcdef")
    with pytest.raises(NotFoundError):
        storage.recommendations.get("../../etc/passwd")


def test_recommendation_cannot_be_edited_only_results(storage):
    rec = storage.recommendations.create(rec_data())
    assert not hasattr(storage.recommendations, "update")
    updated = storage.recommendations.update_results(rec["id"], {"1m": {"beat_market": True}}, price_on_date=140.0)
    assert updated["results"]["1m"]["beat_market"] is True
    assert updated["price_on_date"] == 140.0
    assert updated["verdict"] == rec["verdict"]


def test_challenge_create_update_keeps_fields(storage):
    rec = storage.recommendations.create(rec_data())
    ch = storage.challenges.create(rec["id"], {
        "bear_case": ["AI revenue is a small share of sales", "Margins fell twice"],
        "answers": {"Why now?": "AI is growing"},
    })
    assert ch["company"] == "NVDA"
    assert ch["user_reason"] == rec["user_reason"]

    # Send only one new answer: old answer and bear case stay.
    ch2 = storage.challenges.update(ch["id"], {"answers": {"What would make you sell?": "Sales drop 2 quarters"}})
    assert ch2["answers"] == {"Why now?": "AI is growing", "What would make you sell?": "Sales drop 2 quarters"}
    assert ch2["bear_case"] == ch["bear_case"]

    ch3 = storage.challenges.update(ch["id"], {"verdict": "yellow", "verdict_reason": "Timing risk before earnings"})
    assert ch3["verdict"] == "yellow"
    assert ch3["answers"] == ch2["answers"]


def test_challenge_rules(storage):
    with pytest.raises(NotFoundError):
        storage.challenges.create("20260101-000000-abcdef", {})
    rec = storage.recommendations.create(rec_data())
    ch = storage.challenges.create(rec["id"], {})
    with pytest.raises(ValidationError):
        storage.challenges.update(ch["id"], {"company": "AAPL"})   # not editable
    with pytest.raises(ValidationError):
        storage.challenges.update(ch["id"], {"verdict": "maybe"})
    with pytest.raises(ValidationError):
        storage.challenges.update(ch["id"], {"answers": {f"q{i}": "a" for i in range(21)}})


def test_delete_recommendation_deletes_its_challenges(storage):
    rec = storage.recommendations.create(rec_data())
    other = storage.recommendations.create(rec_data(company="AAPL"))
    storage.challenges.create(rec["id"], {})
    storage.challenges.create(rec["id"], {})
    keep = storage.challenges.create(other["id"], {})

    assert len(storage.recommendation_with_challenges(rec["id"])["challenges"]) == 2
    storage.delete_recommendation(rec["id"])
    assert [c["id"] for c in storage.challenges.list()] == [keep["id"]]
    with pytest.raises(NotFoundError):
        storage.recommendations.get(rec["id"])


def test_data_survives_restart(tmp_path):
    rec = Storage(tmp_path).recommendations.create(rec_data())
    again = Storage(tmp_path)  # like restarting the app
    assert again.recommendations.get(rec["id"])["company"] == "NVDA"
