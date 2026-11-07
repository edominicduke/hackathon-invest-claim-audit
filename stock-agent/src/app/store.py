"""Storage for the stock agent: recommendations and Challenge My Trade sessions.

Same pattern as the KPMG project: one JSON file per item (JsonStore).
Saving, opening, listing and deleting cost no LLM calls.

Links:  recommendation -> challenges
Rules:
  - A recommendation cannot be edited after it is saved (honest track record).
    Only its "results" field can be filled in later by the results check.
  - Deleting a recommendation also deletes its challenges.
  - PUT on a challenge keeps the fields you leave out.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

VERDICTS = ("green", "yellow", "red", "not_enough_evidence")

MAX_SHORT = 120        # author, labels
MAX_TEXT = 4000        # reasons, answers, summaries
MAX_LIST = 50          # sources, list items
MAX_ANSWERS = 20       # questions in a challenge
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class NotFoundError(Exception):
    """Unknown id -> 404."""


class ValidationError(Exception):
    """Bad input -> 400."""


def _now() -> str:
    # Microseconds keep items in the order they were created (KPMG comment bug fix).
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    # Same style as KPMG ids, e.g. 20261107-163012-a59ac9
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


# ---------------------------------------------------------------- validation helpers
def _text(value: Any, field: str, max_len: int = MAX_TEXT, required: bool = True) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ValidationError(f"'{field}' is required")
        return None
    if not isinstance(value, str):
        raise ValidationError(f"'{field}' must be text")
    value = value.strip()
    if len(value) > max_len:
        raise ValidationError(f"'{field}' must be at most {max_len} characters")
    return value


def _ticker(value: Any) -> str:
    value = _text(value, "company", 10)
    value = value.upper()
    if not TICKER_RE.match(value):
        raise ValidationError("'company' must be a stock ticker like NVDA or BRK.B")
    return value


def _verdict(value: Any, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if value not in VERDICTS:
        raise ValidationError(f"'verdict' must be one of {', '.join(VERDICTS)}")
    return value


def _date(value: Any, field: str) -> str:
    if value is None:
        return date.today().isoformat()
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValidationError(f"'{field}' must be a date like 2026-11-07") from exc
    if parsed > date.today():
        raise ValidationError(f"'{field}' cannot be in the future")
    return parsed.isoformat()


def _sources(value: Any) -> list[dict]:
    """Each source: {"label": "10-Q p.8", "document": ..., "date": ..., "text": ...}.

    Only "label" is required. "text" is the passage the AI used; the number
    check reads it.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValidationError("'sources' must be a list")
    if len(value) > MAX_LIST:
        raise ValidationError(f"'sources' can have at most {MAX_LIST} items")
    clean = []
    for i, src in enumerate(value):
        if isinstance(src, str):
            src = {"label": src}
        if not isinstance(src, dict):
            raise ValidationError(f"sources[{i}] must be text or an object")
        item = {"label": _text(src.get("label"), f"sources[{i}].label", MAX_SHORT)}
        for key, limit in (("document", MAX_SHORT), ("date", 20), ("page", 20), ("url", 500), ("text", MAX_TEXT)):
            if src.get(key) is not None:
                item[key] = _text(str(src[key]), f"sources[{i}].{key}", limit, required=False)
        clean.append(item)
    return clean


def _text_or_list(value: Any, field: str) -> str | list[str] | None:
    """Bear case can be one paragraph or a list of reasons."""
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) > MAX_LIST:
            raise ValidationError(f"'{field}' can have at most {MAX_LIST} items")
        return [_text(v, f"{field}[{i}]") for i, v in enumerate(value)]
    return _text(value, field, required=False)


def _answers(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValidationError("'answers' must be an object of question -> answer")
    if len(value) > MAX_ANSWERS:
        raise ValidationError(f"'answers' can have at most {MAX_ANSWERS} questions")
    return {
        _text(q, "answers question", 300): _text(a, f"answer to '{q}'", required=False) or ""
        for q, a in value.items()
    }


# ---------------------------------------------------------------- JsonStore (from KPMG)
class JsonStore:
    """One JSON file per item inside a folder."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def _path(self, item_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9\-]+", item_id or ""):
            raise NotFoundError(item_id)
        return self.folder / f"{item_id}.json"

    def save(self, item: dict) -> dict:
        path = self._path(item["id"])
        # Write to a temp file then rename, so a crash never leaves half a file.
        fd, tmp = tempfile.mkstemp(dir=self.folder, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return item

    def get(self, item_id: str) -> dict:
        path = self._path(item_id)
        if not path.exists():
            raise NotFoundError(item_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def all(self) -> list[dict]:
        items = [json.loads(p.read_text(encoding="utf-8")) for p in self.folder.glob("*.json")]
        return sorted(items, key=lambda x: x.get("created_at", ""))

    def delete(self, item_id: str) -> None:
        path = self._path(item_id)
        if not path.exists():
            raise NotFoundError(item_id)
        path.unlink()


# ---------------------------------------------------------------- recommendations
class RecommendationStore(JsonStore):
    """data/recommendations/ - one file per recommendation the agent makes."""

    def create(self, data: dict) -> dict:
        now = _now()
        item = {
            "id": _new_id(),
            "company": _ticker(data.get("company")),
            "user_reason": _text(data.get("user_reason"), "user_reason"),
            "verdict": _verdict(data.get("verdict")),
            "summary": _text(data.get("summary"), "summary", required=False),
            "confidence": data.get("confidence") if data.get("confidence") in ("low", "medium", "high") else None,
            "sources": _sources(data.get("sources")),
            # as_of_date lets you save past-dated test recommendations, so the
            # results check can be tried right away.
            "as_of_date": _date(data.get("as_of_date"), "as_of_date"),
            "price_on_date": None,
            "created_by": _text(data.get("created_by"), "created_by", MAX_SHORT, required=False) or "agent",
            "created_at": now,
            "results": {},
            "results_updated_at": None,
        }
        price = data.get("price_on_date")
        if price is not None:
            if not isinstance(price, (int, float)) or isinstance(price, bool) or price <= 0:
                raise ValidationError("'price_on_date' must be a positive number")
            item["price_on_date"] = float(price)
        return self.save(item)

    def list(self, company: str | None = None, verdict: str | None = None) -> list[dict]:
        items = self.all()
        if company:
            items = [r for r in items if r["company"] == company.upper()]
        if verdict:
            items = [r for r in items if r["verdict"] == verdict]
        return items

    def update_results(self, rec_id: str, results: dict, price_on_date: float | None = None) -> dict:
        """The ONLY change allowed after saving. Called by the results check."""
        if not isinstance(results, dict):
            raise ValidationError("'results' must be an object")
        item = self.get(rec_id)
        item["results"].update(results)
        if price_on_date is not None and item.get("price_on_date") is None:
            item["price_on_date"] = float(price_on_date)
        item["results_updated_at"] = _now()
        return self.save(item)


# ---------------------------------------------------------------- challenges
class ChallengeStore(JsonStore):
    """data/challenges/ - one file per Challenge My Trade session."""

    EDITABLE = ("bear_case", "drop_reason", "answers", "verdict", "verdict_reason")

    def __init__(self, folder: str | Path, recommendations: RecommendationStore):
        super().__init__(folder)
        self.recommendations = recommendations

    def create(self, rec_id: str, data: dict) -> dict:
        rec = self.recommendations.get(rec_id)  # unknown -> 404
        now = _now()
        item = {
            "id": _new_id(),
            "recommendation_id": rec_id,
            "company": rec["company"],
            "user_reason": _text(data.get("user_reason"), "user_reason", required=False) or rec["user_reason"],
            "bear_case": _text_or_list(data.get("bear_case"), "bear_case"),
            "drop_reason": _text(data.get("drop_reason"), "drop_reason", required=False),
            "answers": _answers(data.get("answers")),
            "verdict": _verdict(data.get("verdict"), required=False),
            "verdict_reason": _text(data.get("verdict_reason"), "verdict_reason", required=False),
            "created_by": _text(data.get("created_by"), "created_by", MAX_SHORT, required=False) or "user",
            "created_at": now,
            "updated_at": now,
        }
        return self.save(item)

    def update(self, ch_id: str, data: dict) -> dict:
        """PUT keeps fields you leave out. Answers are merged, not replaced."""
        item = self.get(ch_id)
        unknown = set(data) - set(self.EDITABLE)
        if unknown:
            raise ValidationError(f"cannot change: {', '.join(sorted(unknown))}")
        if "bear_case" in data:
            item["bear_case"] = _text_or_list(data["bear_case"], "bear_case")
        if "drop_reason" in data:
            item["drop_reason"] = _text(data["drop_reason"], "drop_reason", required=False)
        if "answers" in data:
            merged = {**item["answers"], **_answers(data["answers"])}
            item["answers"] = _answers(merged)
        if "verdict" in data:
            item["verdict"] = _verdict(data["verdict"], required=False)
        if "verdict_reason" in data:
            item["verdict_reason"] = _text(data["verdict_reason"], "verdict_reason", required=False)
        item["updated_at"] = _now()
        return self.save(item)

    def list(self, recommendation_id: str | None = None) -> list[dict]:
        items = self.all()
        if recommendation_id:
            items = [c for c in items if c["recommendation_id"] == recommendation_id]
        return items


# ---------------------------------------------------------------- one object for the app
class Storage:
    def __init__(self, data_dir: str | Path = "data"):
        data_dir = Path(data_dir)
        self.recommendations = RecommendationStore(data_dir / "recommendations")
        self.challenges = ChallengeStore(data_dir / "challenges", self.recommendations)

    def recommendation_with_challenges(self, rec_id: str) -> dict:
        rec = self.recommendations.get(rec_id)
        return {**rec, "challenges": self.challenges.list(recommendation_id=rec_id)}

    def delete_recommendation(self, rec_id: str) -> None:
        self.recommendations.get(rec_id)  # 404 first
        for ch in self.challenges.list(recommendation_id=rec_id):
            self.challenges.delete(ch["id"])
        self.recommendations.delete(rec_id)
