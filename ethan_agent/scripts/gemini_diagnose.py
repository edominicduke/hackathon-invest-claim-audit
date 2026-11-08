"""Find out WHY a Gemini call is slow: add one ingredient of our real request at a time.

    .venv\\Scripts\\python scripts\\gemini_diagnose.py                  # uses $env:GEMINI_MODEL or default
    .venv\\Scripts\\python scripts\\gemini_diagnose.py other-model-name  # compare a different model

Each step has its own timeout and no retries, so one slow step can't hide the others.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from agent import prompts as P  # noqa: E402
from agent.llm import strict_schema  # noqa: E402
from agent.llm_gemini import DEFAULT_GEMINI_MODEL  # noqa: E402
from agent.schemas import Decomposition, ParsedThesis  # noqa: E402

if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    sys.exit('Set your key first:  $env:GEMINI_API_KEY = "..."')

args = [a for a in sys.argv[1:] if not a.startswith("--")]
DECOMPOSE = "--decompose" in sys.argv
MODEL = args[0] if args else os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
TIMEOUT_S = 90 if DECOMPOSE else 45
client = genai.Client(
    http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1))
)

THESIS = "<thesis>\nI'm buying EXMP because AI demand will keep data center revenue growing 50%+.\n</thesis>"
SCHEMA = strict_schema(ParsedThesis)
low = types.ThinkingLevel("LOW")
minimal = types.ThinkingLevel("MINIMAL")

STEPS = [
    ("1. plain text, no config", "Reply with the single word: hi", {}),
    ("2. our prompt, plain text", THESIS, {"system_instruction": P.EXTRACT}),
    ("3. + JSON mode, no schema", THESIS, {"system_instruction": P.EXTRACT, "response_mime_type": "application/json"}),
    ("4. + our schema", THESIS, {"system_instruction": P.EXTRACT, "response_mime_type": "application/json",
                                 "response_json_schema": SCHEMA}),
    ("5. + thinking LOW (= the app)", THESIS, {"system_instruction": P.EXTRACT, "response_mime_type": "application/json",
                                               "response_json_schema": SCHEMA,
                                               "thinking_config": types.ThinkingConfig(thinking_level=low)}),
    ("6. schema + thinking MINIMAL", THESIS, {"system_instruction": P.EXTRACT, "response_mime_type": "application/json",
                                              "response_json_schema": SCHEMA,
                                              "thinking_config": types.ThinkingConfig(thinking_level=minimal)}),
]

if DECOMPOSE:
    # Same input the app sends for "Breaking the thesis into checkable claims".
    D_USER = (
        "Company: EXMP (Exemplar Semiconductor)\nDirection: buy\nHorizon: not stated\n\n<thesis>\n"
        "I'm buying EXMP because AI demand will keep data center revenue growing 50%+ and they have "
        "no real competition.\n</thesis>"
    )
    D_SCHEMA = strict_schema(Decomposition)
    base = {"system_instruction": P.DECOMPOSE, "response_mime_type": "application/json"}
    STEPS = [
        ("A. JSON mode, no schema, LOW", D_USER, {**base, "thinking_config": types.ThinkingConfig(thinking_level=low)}),
        ("B. our schema, thinking LOW", D_USER, {**base, "response_json_schema": D_SCHEMA,
                                                 "thinking_config": types.ThinkingConfig(thinking_level=low)}),
        ("C. our schema, thinking MEDIUM (app)", D_USER, {**base, "response_json_schema": D_SCHEMA,
                                                          "thinking_config": types.ThinkingConfig(
                                                              thinking_level=types.ThinkingLevel("MEDIUM"))}),
    ]

print(f"Model: {MODEL}   (each step: no retries, {TIMEOUT_S}s timeout)\n", flush=True)
for name, contents, cfg in STEPS:
    print(f"{name:<34}", end=" ", flush=True)
    start = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL, contents=contents, config=types.GenerateContentConfig(**cfg) if cfg else None
        )
        usage = resp.usage_metadata
        thoughts = getattr(usage, "thoughts_token_count", None) if usage else None
        out = getattr(usage, "candidates_token_count", None) if usage else None
        print(f"OK   {time.perf_counter() - start:5.1f}s   thinking tokens={thoughts}  output tokens={out}")
    except Exception as exc:
        msg = str(exc).splitlines()[0][:90]
        print(f"FAIL {time.perf_counter() - start:5.1f}s   {type(exc).__name__}: {msg}")
