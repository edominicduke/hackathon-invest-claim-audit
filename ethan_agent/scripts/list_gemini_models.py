"""List the Gemini models your key can use for text generation.

    $env:GEMINI_API_KEY = "..."
    .venv\\Scripts\\python scripts\\list_gemini_models.py          # text models only
    .venv\\Scripts\\python scripts\\list_gemini_models.py --all    # everything

Then pick one with:  $env:GEMINI_MODEL = "<name from the list>"
"""

import os
import sys

from google import genai

if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    sys.exit('Set your key first:  $env:GEMINI_API_KEY = "..."')

show_all = "--all" in sys.argv
rows = []
# Keep a reference: the SDK closes the connection when the Client is garbage-collected,
# and models.list() is a lazy pager that only fetches once the loop starts.
client = genai.Client()
for m in client.models.list():
    actions = m.supported_actions or []
    if not show_all and "generateContent" not in actions:
        continue  # skip embedding / image / audio-only models
    name = (m.name or "").removeprefix("models/")
    rows.append((name, "yes" if m.thinking else "-", m.input_token_limit or 0, m.output_token_limit or 0, m.display_name or ""))

rows.sort()
print(f"{'MODEL (use as GEMINI_MODEL)':<42} {'THINKING':<9} {'INPUT TOK':>10} {'OUTPUT TOK':>11}  DISPLAY NAME")
for name, thinking, inp, out, display in rows:
    print(f"{name:<42} {thinking:<9} {inp:>10,} {out:>11,}  {display}")
print(f"\n{len(rows)} models. Current default in the app: gemini-3.8-flash")
