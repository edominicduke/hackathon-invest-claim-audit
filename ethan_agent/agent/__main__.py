"""CLI for teammates:  python -m agent "I'm buying EXMP because ..."  -> audit JSON on stdout."""

import sys

from . import build_default_auditor
from .llm import enable_call_logging

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python -m agent "<your thesis>"')
    enable_call_logging()  # progress + per-call timing go to stderr; JSON goes to stdout
    audit = build_default_auditor().audit(" ".join(sys.argv[1:]))
    print(audit.model_dump_json(indent=2))
