#!/usr/bin/env python3
"""Offline tests for rag.py: no network, no API key.   .venv/bin/python tools/test_rag.py"""
import hashlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rag  # noqa: E402

# the same 30 question -> expected-note pairs the n8n tests use
mjs = (ROOT / "tools" / "test_workflows.mjs").read_text()
pairs = re.findall(r"^\s*\['((?:[^'\\]|\\.)*)', '((?:[^'\\]|\\.)*)'\],?\s*$", mjs, re.M)
assert len(pairs) >= 25, f"expected the question list, found {len(pairs)}"

chunks = rag.load_chunks()
kb = rag.KnowledgeBase(chunks, None)
hits = 0
for q, want in pairs:
    got = rag.hybrid_search(kb, None, [q])["retrieved"]
    titles = [r["title"] for r in got]
    assert want in titles, f'"{q}": expected "{want}" in top 5, got {titles}'
    hits += 1
print(f"ok  BM25 + fusion finds the right note in the top 5 for {hits}/{len(pairs)} questions")


# ---- fake OpenAI client: deterministic bag-of-words embeddings + scripted chat answers
def fake_vec(text, dim=256):
    v = np.zeros(dim, dtype=np.float32)
    for t in rag.tokens(text):
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dim] += 1
    return v

class FakeClient:
    def __init__(self, fail_chat=False, fail_embed=False):
        self.fail_chat, self.fail_embed, self.chat_calls = fail_chat, fail_embed, []
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
    def _embed(self, model, input):
        if self.fail_embed: raise RuntimeError("embed down")
        return SimpleNamespace(data=[SimpleNamespace(embedding=fake_vec(t).tolist()) for t in input])
    def _chat(self, model, messages, temperature, max_tokens):
        if self.fail_chat: raise RuntimeError("chat down")
        self.chat_calls.append(messages)
        if "search" in messages[0]["content"][:60].lower() or "3 short search queries" in messages[0]["content"]:
            text = "1. safety car\n2) virtual safety car\n- 'yellow flag slow down'\n- Why did everyone slow down?"
        else:
            text = "The Safety Car is out, so everyone bunches up behind it [1][2]."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

kb2 = rag.KnowledgeBase(chunks, np.array([fake_vec(c["text"]) / (np.linalg.norm(fake_vec(c["text"])) or 1) for c in chunks]))

# prepare(): context line, history, limits
p = rag.prepare("  Why slow?  ", {"status": "Safety Car", "lap": "23", "session": "Race"},
                [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
assert p["live"] == "Session: Race | Lap: 23 | Track status shown on screen: Safety Car", p["live"]
assert p["live_hint"] == "Safety Car" and p["history_text"] == "Viewer: hi\nAssistant: hello"
assert rag.prepare("x" * 900, {}, [])["question"] == "x" * 600
assert rag.prepare("q", None, None)["live"] == "not provided"
try: rag.prepare("   ", {}, []); raise SystemExit("empty question should fail")
except ValueError: pass
print("ok  prepare(): context line, history, 600-char limit, empty question rejected")

# parse_subqueries(): numbering/quotes stripped, duplicates and the original question dropped, max 4
qs = rag.parse_subqueries("1. safety car\n2) virtual safety car\n- 'yellow flag slow down'\n- Why did everyone slow down?\n- extra",
                          "Why did everyone slow down?", "Safety Car")
assert qs == ["Why did everyone slow down?", "Why did everyone slow down? Safety Car", "safety car", "virtual safety car"], qs
print("ok  parse_subqueries(): cleaned, de-duplicated, capped at 4 queries")

# full pipeline with context steering retrieval
fc = FakeClient()
out = rag.answer_question(kb2, fc, "Why did everyone slow down?", {"status": "Safety Car"}, [])
assert out["ok"] and out["mode"] == "rag" and "[1]" in out["answer"]
assert [s["cited"] for s in out["sources"][:2]] == [True, True] and len(out["sources"]) == 5
assert any("afety" in s["title"] for s in out["sources"]), [s["title"] for s in out["sources"]]
assert out["subqueries"][0] == "Why did everyone slow down?" and len(out["subqueries"]) == 4
assert any(s["dense"] is not None for s in out["sources"]), "dense scores should be present"
answer_prompt = fc.chat_calls[-1]
assert answer_prompt[0]["content"] == rag.RAG_SYSTEM and "REFERENCE NOTES:\n[1] " in answer_prompt[1]["content"]
assert "Track status shown on screen: Safety Car" in answer_prompt[1]["content"]
print("ok  full pipeline: decompose -> hybrid search -> grounded answer with citations")

# failure modes degrade gracefully
out = rag.answer_question(kb2, FakeClient(fail_chat=True), "What is parc fermé?", {}, [])
assert out["ok"] is False and out["answer"] == rag.FALLBACK_ANSWER and out["sources"], out
out = rag.answer_question(kb2, FakeClient(fail_embed=True), "What is parc fermé?", {}, [])
assert out["ok"] and all(s["dense"] is None for s in out["sources"]) and out["sources"][0]["title"] == "Parc fermé"
print("ok  LLM failure -> friendly fallback; embedding failure -> keyword-only search still answers")

# embeddings cache: stale file is ignored
assert not np.array_equal(rag._text_hashes(chunks), rag._text_hashes(chunks[:-1] + [{**chunks[-1], "text": "changed"}]))
print("ok  embeddings cache is keyed by note text, so edited notes are re-embedded")
print("\nall checks passed")
