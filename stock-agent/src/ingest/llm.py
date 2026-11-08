"""Small shared helper for JSON calls to Gemini on Vertex (used by verify + eval)."""

from __future__ import annotations

import json
import os
import random
import time

MODEL = os.environ.get("EVAL_MODEL", "gemini-3.8-flash")


def ask_json(prompt: str, temperature: float = 0.2, retries: int = 5, model: str | None = None) -> dict:
    from google.genai import errors, types
    from .embed import client
    for attempt in range(retries):
        try:
            resp = client().models.generate_content(
                model=model or MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature, response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                ),
            )
            return json.loads(resp.text)
        except errors.APIError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            raise
        except (json.JSONDecodeError, TypeError):
            continue
    raise RuntimeError("Gemini call failed after retries")
