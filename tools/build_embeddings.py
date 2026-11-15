#!/usr/bin/env python3
"""Embed the knowledge-base chunks once and save them to data/kb_embeddings.npz.

    OPENAI_API_KEY=sk-... .venv/bin/python tools/build_embeddings.py

Run it again whenever you change kb/*.md, then commit data/kb_embeddings.npz.
(If the file is stale the app recomputes embeddings at startup, so it only saves time and a few cents.)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import OpenAI  # noqa: E402

import rag  # noqa: E402

key = os.environ.get("OPENAI_API_KEY")
if not key:
    sys.exit("Set OPENAI_API_KEY in your environment first.")
vecs = rag.build_embeddings_file(OpenAI(api_key=key))
print(f"saved {vecs.shape[0]} embeddings ({vecs.shape[1]} dims) to {rag.EMB_FILE}")
