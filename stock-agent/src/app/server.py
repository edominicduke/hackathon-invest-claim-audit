"""FastAPI routes for recommendations, challenges and the track record.

Run:   python -m src.app            (then open http://127.0.0.1:8000/docs)
Teammates' screens (Ethan) and the results check (Bogdan's part) call these routes.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException

from src.app.store import NotFoundError, Storage, ValidationError
from src.checks.number_check import check_numbers, check_numbers_with_sec
from src.checks.results_check import track_record
from src.checks.sec_facts import SecFacts


def create_app(data_dir: str = "data", sec: SecFacts | None = None) -> FastAPI:
    app = FastAPI(title="Stock Agent - Storage & Checks")
    storage = Storage(data_dir)
    sec = sec or SecFacts()  # keeps SEC downloads cached while the app runs
    app.state.storage = storage

    def run(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotFoundError:
            raise HTTPException(status_code=404, detail="not found")
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------ recommendations (#4)
    @app.post("/api/recommendations", status_code=201)
    def create_recommendation(body: dict[str, Any] = Body(...)):
        return run(storage.recommendations.create, body)

    @app.get("/api/recommendations")
    def list_recommendations(company: Optional[str] = None, verdict: Optional[str] = None):
        return run(storage.recommendations.list, company=company, verdict=verdict)

    @app.get("/api/recommendations/{rec_id}")
    def get_recommendation(rec_id: str):
        return run(storage.recommendation_with_challenges, rec_id)

    @app.delete("/api/recommendations/{rec_id}", status_code=204)
    def delete_recommendation(rec_id: str):
        run(storage.delete_recommendation, rec_id)

    # No PUT on recommendations on purpose: the track record must stay honest.

    # ------------------------------------------------ challenges (#9)
    @app.post("/api/recommendations/{rec_id}/challenges", status_code=201)
    def create_challenge(rec_id: str, body: dict[str, Any] = Body(default={})):
        return run(storage.challenges.create, rec_id, body)

    @app.get("/api/recommendations/{rec_id}/challenges")
    def list_challenges_for(rec_id: str):
        run(storage.recommendations.get, rec_id)
        return storage.challenges.list(recommendation_id=rec_id)

    @app.get("/api/challenges/{ch_id}")
    def get_challenge(ch_id: str):
        return run(storage.challenges.get, ch_id)

    @app.put("/api/challenges/{ch_id}")
    def update_challenge(ch_id: str, body: dict[str, Any] = Body(...)):
        return run(storage.challenges.update, ch_id, body)

    @app.delete("/api/challenges/{ch_id}", status_code=204)
    def delete_challenge(ch_id: str):
        run(storage.challenges.delete, ch_id)

    # ------------------------------------------------ track record (#4)
    @app.get("/api/track-record")
    def get_track_record():
        return track_record(storage.recommendations.all())

    # ------------------------------------------------ number check (#3)
    @app.post("/api/number-check")
    def number_check(body: dict[str, Any] = Body(...)):
        """Body: {"answer": "...", "sources": [{"label": "...", "text": "..."}], "company": "NVDA"}

        Add "company" to also compare against official SEC numbers.
        """
        answer = body.get("answer")
        sources = body.get("sources") or []
        company = body.get("company")
        if not isinstance(answer, str) or not answer.strip():
            raise HTTPException(status_code=400, detail="'answer' is required")
        if not isinstance(sources, list):
            raise HTTPException(status_code=400, detail="'sources' must be a list")
        if company:
            return check_numbers_with_sec(answer, sources, str(company), sec)
        return check_numbers(answer, sources)

    @app.get("/api/recommendations/{rec_id}/number-check")
    def number_check_saved(rec_id: str, use_sec: bool = False):
        """Checks a saved recommendation's summary against its saved sources
        (and official SEC numbers when use_sec=true)."""
        rec = run(storage.recommendations.get, rec_id)
        if not rec.get("summary"):
            raise HTTPException(status_code=400, detail="this recommendation has no summary to check")
        if use_sec:
            return check_numbers_with_sec(rec["summary"], rec["sources"], rec["company"], sec)
        return check_numbers(rec["summary"], rec["sources"])

    return app
