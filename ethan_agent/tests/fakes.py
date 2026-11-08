"""A scripted LLM so the pipeline's code paths are tested without API calls."""

from __future__ import annotations

import re
from typing import Any, Callable

from agent.schemas import Evidence

_SOURCE = re.compile(r'<source label="(S\d+)"[^>]*>\n(.*?)\n</source>', re.S)


class FakeLLM:
    model = "fake-llm"

    def __init__(self, handlers: dict[type, Any]):
        # handler: an instance to return, or a callable(system, user) -> instance
        self.handlers = handlers
        self.calls: list[tuple[str, str]] = []

    def structured(self, *, system: str, user: str, output: type, effort: str = "medium"):
        self.calls.append((output.__name__, user))
        handler = self.handlers[output]
        result = handler(system, user) if callable(handler) else handler
        return result.model_copy(deep=True)

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)


def label_for(user: str, snippet: str) -> str:
    """Find the [S#] label the pipeline gave the source containing ``snippet``,
    the way a model reading the prompt would."""
    for label, body in _SOURCE.findall(user):
        if snippet in body:
            return label
    raise KeyError(f"no source containing {snippet!r}")


def cite(user: str, quote: str, stance: str = "supports") -> Evidence:
    return Evidence(source=label_for(user, quote), quote=quote, stance=stance)


def by_claim(mapping: dict[str, Any]) -> Callable[[str, str], Any]:
    """Route a judge call to a canned answer by a substring of the claim line.
    Values may be instances or callables(user) -> instance."""

    def handler(_system: str, user: str):
        claim_line = next(line for line in user.splitlines() if line.startswith("Claim ("))
        for key, value in mapping.items():
            if key in claim_line:
                return value(user) if callable(value) else value
        raise KeyError(claim_line)

    return handler
