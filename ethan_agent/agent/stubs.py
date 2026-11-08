"""Stand-ins for teammates' components, so the agent runs end to end today.

Swap them out by passing real implementations to ``ThesisAuditor``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .interfaces import Chunk
from .schemas import ThesisAudit, UserAnswer

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = set(
    "a an and are as at be by for from has have in is it its of on or our that the "
    "their this to was were will with we which".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class FixtureRetriever:
    """BM25 over JSON chunk files. Stand-in for Guru's search."""

    def __init__(self, fixture_dir: str | Path):
        self.chunks: list[Chunk] = []
        for path in sorted(Path(fixture_dir).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            self.chunks += [Chunk(**c) for c in data["chunks"]]
        self._docs = [_tokens(f"{c.section or ''} {c.text}") for c in self.chunks]
        self._avg_len = sum(map(len, self._docs)) / max(len(self._docs), 1)
        df = Counter(t for d in self._docs for t in set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, *, ticker: str, k: int = 4) -> list[Chunk]:
        q = _tokens(query)
        target = ticker.upper()
        scored = []
        for chunk, doc in zip(self.chunks, self._docs):
            if chunk.ticker.upper() != target and target not in {r.upper() for r in chunk.related_tickers}:
                continue
            tf = Counter(doc)
            score = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    score += self._idf[t] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(doc) / self._avg_len))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [c for _, c in scored[:k]]


class JsonFileStore:
    """Writes audits and answers to disk. Stand-in for Abdullah's storage."""

    def __init__(self, root: str | Path = "data/store"):
        self.root = Path(root)

    def save_audit(self, audit: ThesisAudit) -> None:
        path = self.root / "audits" / f"{audit.audit_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(audit.model_dump_json(indent=2), encoding="utf-8")

    def save_answers(self, audit_id: str, answers: list[UserAnswer]) -> None:
        path = self.root / "answers" / f"{audit_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([a.model_dump() for a in answers], indent=2), encoding="utf-8")


class NoopNumberChecker:
    """Placeholder for Bogdan's number check (uses official SEC XBRL numbers)."""

    def check(self, audit, chunks) -> list[str]:  # noqa: ARG002
        return []
