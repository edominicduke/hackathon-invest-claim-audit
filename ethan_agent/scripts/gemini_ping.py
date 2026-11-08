"""Time one 'Read my thesis' call against Gemini with retries OFF, so a rate limit
or other error shows immediately instead of as a silent wait.

    .venv\\Scripts\\python scripts\\gemini_ping.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from agent import prompts as P  # noqa: E402
from agent.llm_gemini import GeminiLLM  # noqa: E402
from agent.schemas import ParsedThesis  # noqa: E402

if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    sys.exit('Set your key first:  $env:GEMINI_API_KEY = "..."')

TIMEOUT_S = 60
client = genai.Client(
    http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1))
)
llm = GeminiLLM(client=client, attempts=1)
print(f"Model: {llm.model}  (retries off, {TIMEOUT_S}s timeout) - waiting for Gemini...", flush=True)

thesis = "I'm buying EXMP because AI demand will keep data center revenue growing 50%+."
start = time.perf_counter()
try:
    parsed = llm.structured(
        system=P.EXTRACT, user=f"<thesis>\n{thesis}\n</thesis>", output=ParsedThesis, effort="low"
    )
    print(f"OK in {time.perf_counter() - start:.1f}s -> ticker={parsed.ticker}, direction={parsed.direction.value}")
except Exception as exc:  # show the real error, whatever it is
    print(f"FAILED after {time.perf_counter() - start:.1f}s: {type(exc).__name__}: {exc}")
