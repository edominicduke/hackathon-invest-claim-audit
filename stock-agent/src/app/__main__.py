"""Start the app:  python -m src.app   (add --data-dir to choose a folder)."""

import argparse

import uvicorn

from src.app.server import create_app

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

uvicorn.run(create_app(args.data_dir), host="127.0.0.1", port=args.port)
