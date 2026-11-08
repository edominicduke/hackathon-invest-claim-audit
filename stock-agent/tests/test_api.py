"""Route tests. Needs fastapi + httpx (pip install -r requirements.txt)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.app.server import create_app  # noqa: E402

REC = {
    "company": "NVDA",
    "user_reason": "AI chip demand is growing",
    "verdict": "green",
    "summary": "Revenue was $4.2 billion, up 18%. Margin hit 81%.",
    "sources": [{"label": "10-Q p.8", "text": "Revenue was $4,213 million, up 18.4%. Margin was 74.6%."}],
    "as_of_date": "2026-01-05",
}


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path)))


def test_full_flow(client):
    r = client.post("/api/recommendations", json=REC)
    assert r.status_code == 201
    rec_id = r.json()["id"]

    r = client.post(f"/api/recommendations/{rec_id}/challenges", json={"answers": {"Why now?": "Earnings soon"}})
    assert r.status_code == 201
    ch_id = r.json()["id"]

    r = client.put(f"/api/challenges/{ch_id}", json={"verdict": "yellow"})
    assert r.json()["answers"] == {"Why now?": "Earnings soon"}

    r = client.get(f"/api/recommendations/{rec_id}")
    assert [c["id"] for c in r.json()["challenges"]] == [ch_id]

    assert len(client.get("/api/recommendations", params={"company": "nvda"}).json()) == 1
    assert client.get("/api/track-record").json()["total_recommendations"] == 1

    nc = client.get(f"/api/recommendations/{rec_id}/number-check").json()
    assert nc["not_found"] == ["81%"]

    assert client.delete(f"/api/recommendations/{rec_id}").status_code == 204
    assert client.get(f"/api/challenges/{ch_id}").status_code == 404


def test_errors(client):
    assert client.post("/api/recommendations", json={**REC, "verdict": "buy"}).status_code == 400
    assert client.get("/api/recommendations/20260101-000000-abcdef").status_code == 404
    assert client.put("/api/recommendations/x", json={}).status_code == 405  # no editing
    assert client.post("/api/number-check", json={"answer": ""}).status_code == 400


def test_number_check_route(client):
    r = client.post("/api/number-check", json={"answer": "Margin was 75%.", "sources": REC["sources"]})
    assert r.json()["all_found"] is True


def test_number_check_with_sec_route(tmp_path):
    from tests.test_sec_facts import fake_sec

    sec, _ = fake_sec()
    client = TestClient(create_app(str(tmp_path), sec=sec))
    r = client.post("/api/number-check", json={"answer": "EPS was $1.08, not $2.50.", "company": "NVDA"})
    body = r.json()
    assert body["sec_error"] is None
    assert [n["official_sec"] for n in body["numbers"]] == [True, False]
