"""Embeddings with Vertex AI's gemini-embedding-001.

One-time setup (each teammate, on their own laptop):
  pip install google-genai numpy
  gcloud auth application-default login          # sign in with YOUR Google account
  gcloud config set project <PROJECT_ID>
  gcloud services enable aiplatform.googleapis.com   # once per project

  export GOOGLE_CLOUD_PROJECT=<PROJECT_ID>
  export GOOGLE_CLOUD_LOCATION=global
  export GOOGLE_GENAI_USE_VERTEXAI=True

Check it works:
  python -m ingest.embed

Things to know about this model:
  - Vertex accepts ONE text per request for gemini-embedding-001, so this
    module sends requests in parallel threads instead of batching.
  - Each text is capped at 2,048 tokens; longer text is silently cut off.
    Keep chunks under ~1,500 tokens.
  - Chunks and queries use different task types (RETRIEVAL_DOCUMENT vs
    RETRIEVAL_QUERY). Always use embed_query() for user questions.
  - Vectors are L2-normalized, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors, types

from .common import DATA_ROOT

MODEL = "gemini-embedding-001"
DIM = 768            # 3072 is the full size; 768 keeps the index 4x smaller with little quality loss
WORKERS = 8          # parallel requests; lower this if you keep hitting quota errors
INDEX_DIR = DATA_ROOT / "index"


_client: genai.Client | None = None
_client_lock = threading.Lock()


def client() -> genai.Client:
    """One shared client for all threads.

    Created under a lock: if several threads each built their own, the extras
    get garbage-collected, and the SDK closes a client when it's collected,
    which kills requests still in flight ("client has been closed").
    Reads GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION / GOOGLE_GENAI_USE_VERTEXAI.
    """
    global _client
    with _client_lock:
        if _client is None:
            _client = genai.Client()
        return _client


def _embed_one(text: str, task_type: str, max_retries: int = 6) -> np.ndarray:
    c = client()  # hold a reference for the whole request
    for attempt in range(max_retries):
        try:
            resp = c.models.embed_content(
                model=MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=DIM),
            )
            v = np.asarray(resp.embeddings[0].values, dtype=np.float32)
            return v / (np.linalg.norm(v) + 1e-12)
        except errors.APIError as e:
            # 429 = quota/rate limit, 5xx = transient. Back off and retry those.
            if e.code == 429 or e.code >= 500:
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            raise
    raise RuntimeError(f"Gave up after {max_retries} retries (rate limited?)")


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed chunks for the index. Returns an (n, DIM) array, same order as texts."""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        vecs = list(pool.map(lambda t: _embed_one(t, "RETRIEVAL_DOCUMENT"), texts))
    return np.vstack(vecs) if vecs else np.zeros((0, DIM), dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed one user question. Shape (DIM,)."""
    return _embed_one(query, "RETRIEVAL_QUERY")


def embed_chunks_cached(chunk_ids: list[str], texts: list[str]) -> tuple[list[str], np.ndarray]:
    """Embed only chunks not already on disk, then save the updated index.

    Re-running after fetching one more ticker only pays for the new chunks.
    Returns (ids, vectors) for the whole index.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    ids_path, vec_path = INDEX_DIR / "ids.json", INDEX_DIR / "embeddings.npy"

    old_ids = json.loads(ids_path.read_text()) if ids_path.exists() else []
    old_vecs = np.load(vec_path) if vec_path.exists() else np.zeros((0, DIM), dtype=np.float32)
    have = set(old_ids)

    new = [(i, t) for i, t in zip(chunk_ids, texts) if i not in have]
    print(f"{len(have)} chunks already embedded, {len(new)} new")
    if new:
        new_vecs = embed_documents([t for _, t in new])
        old_ids += [i for i, _ in new]
        old_vecs = np.vstack([old_vecs, new_vecs])
        np.save(vec_path, old_vecs)
        ids_path.write_text(json.dumps(old_ids))
    return old_ids, old_vecs


if __name__ == "__main__":
    # Smoke test: the query should score higher against the matching passage.
    docs = [
        "Data Center revenue was $30.8 billion, up 112% from a year ago.",
        "The company leases its headquarters in Santa Clara, California.",
    ]
    d = embed_documents(docs)
    q = embed_query("How fast is data center revenue growing?")
    print(f"dim={d.shape[1]}  scores: revenue={d[0] @ q:.3f}  headquarters={d[1] @ q:.3f}")
    print("OK" if d[0] @ q > d[1] @ q else "Unexpected ranking, check setup")
