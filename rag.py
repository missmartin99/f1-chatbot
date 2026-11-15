"""Pit Wall Pal RAG pipeline in plain Python (a port of n8n workflow 03).

    question -> decompose (LLM) -> dense search (embeddings) + BM25 -> Reciprocal Rank Fusion
             -> answer ONLY from the top notes, with [n] citations

It uses the same prompts, knowledge base, models and ranking logic as the n8n workflow, so
answers should match. Prompts and chunking are imported from tools/build_workflows.py so there
is one source of truth. Chunk embeddings are cached in data/kb_embeddings.npz (rebuild with
`python tools/build_embeddings.py`); if that file is missing or stale they are recomputed.

The OpenAI key is read by the caller and passed in; this module never logs or returns it.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from tools.build_workflows import (  # same prompts / chunks / models as the n8n workflows
    CHAT_MODEL, DECOMPOSE_SYSTEM, EMBED_MODEL, RAG_SYSTEM, load_chunks,
)

ROOT = Path(__file__).resolve().parent
EMB_FILE = ROOT / "data" / "kb_embeddings.npz"

FALLBACK_ANSWER = "Sorry, I couldn't get an answer from the language model just now. Please try again."
TOP_K_DENSE = 4          # per query, same as the n8n vector store node
FINAL_NOTES = 5
RRF_K = 60

# ---------------------------------------------------------------- text normalising (same as the JS)
SYN = {"tire": "tyre", "tires": "tyres", "curb": "kerb", "curbs": "kerbs", "color": "colour",
       "colors": "colours", "quali": "qualifying"}
STOP = set((
    "a an the is are was were be been being of to in on at for from by with and or but if then than that this these those "
    "it its as do does did what whats why how when who which can could should would will may might i you he she we they me my your our their "
    "there here so not no yes about into over under up down out off just also very more most some any tell mean means "
    "work works happen happens get gets"
).split())


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    if len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


def tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))        # fermé -> ferme
    toks = [SYN.get(t, t) for t in re.findall(r"[a-z0-9]+", s)]
    return [_stem(t) for t in toks if t not in STOP]


# ---------------------------------------------------------------- knowledge base index
class KnowledgeBase:
    """Chunks + BM25 index + embedding matrix. Build once, reuse for every question."""

    K1, B = 1.5, 0.75

    def __init__(self, chunks: list[dict], vectors: np.ndarray | None):
        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        self.vectors = vectors
        docs = []
        for c in chunks:
            toks = tokens((c["title"] + " ") * 3 + c["text"])      # title counted 3x so exact terms win
            docs.append((c["id"], Counter(toks), len(toks)))
        self.docs = docs
        self.n = len(docs)
        self.avgdl = sum(d[2] for d in docs) / self.n
        self.df: Counter = Counter()
        for _, tf, _ in docs:
            self.df.update(tf.keys())

    def bm25(self, query_tokens: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        uniq = set(query_tokens)
        for cid, tf, dl in self.docs:
            s = 0.0
            for t in uniq:
                f = tf.get(t)
                if not f:
                    continue
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * (f * (self.K1 + 1)) / (f + self.K1 * (1 - self.B + self.B * dl / self.avgdl))
            if s > 0:
                out[cid] = s
        return out


def _text_hashes(chunks: list[dict]) -> np.ndarray:
    return np.array([hashlib.sha1(c["text"].encode("utf-8")).hexdigest() for c in chunks])


def embed_texts(client, texts: list[str]) -> np.ndarray:
    vecs: list[list[float]] = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 100])
        vecs.extend(d.embedding for d in resp.data)
    m = np.array(vecs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def build_embeddings_file(client, chunks: list[dict] | None = None) -> np.ndarray:
    chunks = chunks or load_chunks()
    vectors = embed_texts(client, [c["text"] for c in chunks])
    EMB_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(EMB_FILE, hashes=_text_hashes(chunks), vectors=vectors)
    return vectors


def load_knowledge_base(client=None) -> KnowledgeBase:
    """Use the cached embeddings when they match the current notes, otherwise recompute them."""
    chunks = load_chunks()
    vectors = None
    if EMB_FILE.exists():
        try:
            f = np.load(EMB_FILE, allow_pickle=False)
            if np.array_equal(f["hashes"], _text_hashes(chunks)):
                vectors = f["vectors"].astype(np.float32)
        except Exception:
            vectors = None
    if vectors is None and client is not None:
        try:
            vectors = build_embeddings_file(client, chunks)
        except Exception:
            try:  # could not write the cache (read-only disk); still use fresh vectors in memory
                vectors = embed_texts(client, [c["text"] for c in chunks])
            except Exception:
                vectors = None            # keyword-only search still works
    return KnowledgeBase(chunks, vectors)


# ---------------------------------------------------------------- pipeline steps
def prepare(question, context, history) -> dict:
    q = str(question or "").strip()[:600]
    if not q:
        raise ValueError("The request needs a question.")
    ctx = context if isinstance(context, dict) else {}
    parts = []
    if ctx.get("event"):   parts.append("Event: " + str(ctx["event"])[:80])
    if ctx.get("session"): parts.append("Session: " + str(ctx["session"])[:40])
    if ctx.get("lap"):     parts.append("Lap: " + str(ctx["lap"])[:20])
    if ctx.get("status"):  parts.append("Track status shown on screen: " + str(ctx["status"])[:60])
    if ctx.get("note"):    parts.append("Viewer note: " + str(ctx["note"])[:200])
    live = " | ".join(parts) if parts else "not provided"
    live_hint = " ".join(str(x)[:80] for x in (ctx.get("status"), ctx.get("note")) if x)
    hist = history[-6:] if isinstance(history, list) else []
    history_text = "\n".join(
        ("Viewer: " if isinstance(m, dict) and m.get("role") == "user" else "Assistant: ")
        + str((m or {}).get("content", "") if isinstance(m, dict) else "")[:500]
        for m in hist
    ) or "(no earlier messages)"
    return {"question": q, "live": live, "live_hint": live_hint, "history_text": history_text}


def parse_subqueries(raw: str, question: str, live_hint: str) -> list[str]:
    subs = []
    for line in str(raw).split("\n"):
        s = re.sub(r"^[\s\-\*•\d\.\)]+", "", line)
        s = re.sub(r"^[\"']+|[\"']+$", "", s).strip()
        if 2 < len(s) < 160:
            subs.append(s)
    subs = [s for s in dict.fromkeys(subs) if s.lower() != question.lower()]
    queries = [question]
    if live_hint:
        queries.append(f"{question} {live_hint}")
    for s in subs:
        if len(queries) < 4:
            queries.append(s)
    return queries


def hybrid_search(kb: KnowledgeBase, client, queries: list[str]) -> dict:
    # BM25 for every query, best score per chunk
    lexical: dict[str, float] = {}
    for q in queries:
        for cid, s in kb.bm25(tokens(q)).items():
            lexical[cid] = max(lexical.get(cid, 0.0), s)
    lexical_ranked = [cid for cid, _ in sorted(lexical.items(), key=lambda e: -e[1])[:8]]

    # dense: top 4 per query by cosine similarity, pooled, best score per chunk
    dense: dict[str, dict] = {}
    if kb.vectors is not None and client is not None:
        try:
            qv = embed_texts(client, queries)
            order = 0
            for row in qv:
                sims = kb.vectors @ row
                for idx in np.argsort(-sims, kind="stable")[:TOP_K_DENSE]:
                    cid, score = kb.chunks[int(idx)]["id"], float(sims[idx])
                    prev = dense.get(cid)
                    if prev is None:
                        dense[cid] = {"score": score, "order": order}
                        order += 1
                    elif score > prev["score"]:
                        prev["score"] = score
        except Exception:
            dense = {}                     # fall back to keyword-only for this question
    dense_ranked = [cid for cid, _ in sorted(dense.items(), key=lambda e: (-e[1]["score"], e[1]["order"]))]

    # Reciprocal Rank Fusion
    fused: dict[str, float] = {}
    for ranked in (dense_ranked, lexical_ranked):
        for i, cid in enumerate(ranked):
            fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + i + 1)
    top = sorted(fused.items(), key=lambda e: -e[1])[:FINAL_NOTES]

    retrieved = []
    for cid, rrf in top:
        c = kb.by_id[cid]
        retrieved.append({
            "id": cid, "title": c["title"], "category": c["category"], "text": c["text"],
            "rrf": round(rrf, 4),
            "dense": round(dense[cid]["score"], 3) if cid in dense else None,
            "bm25": round(lexical[cid], 2) if cid in lexical else None,
        })
    context_text = "\n\n".join(
        f"[{i + 1}] {r['title']} ({r['category']})\n{r['text']}" for i, r in enumerate(retrieved)
    ) or "(no matching notes were found)"
    return {"retrieved": retrieved, "context_text": context_text,
            "debug": {"denseCandidates": len(dense_ranked), "lexicalCandidates": len(lexical_ranked)}}


def _chat(client, system: str, user: str, temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def answer_question(kb: KnowledgeBase, client, question, context=None, history=None) -> dict:
    """Same response shape as the n8n RAG webhook: mode, answer, ok, sources, subqueries, debug."""
    p = prepare(question, context, history)

    try:
        raw = _chat(client, DECOMPOSE_SYSTEM,
                    f"Viewer question: {p['question']}\nLive context: {p['live']}", 0.3, 100)
    except Exception:
        raw = ""                            # search the original question only
    queries = parse_subqueries(raw, p["question"], p["live_hint"])

    found = hybrid_search(kb, client, queries)

    try:
        text = _chat(client, RAG_SYSTEM,
                     f"LIVE STREAM CONTEXT: {p['live']}\n\nRECENT CHAT:\n{p['history_text']}\n\n"
                     f"REFERENCE NOTES:\n{found['context_text']}\n\nVIEWER QUESTION: {p['question']}", 0.2, 300)
    except Exception:
        text = ""

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    sources = [{"n": i + 1, "id": r["id"], "title": r["title"], "category": r["category"],
                "cited": (i + 1) in cited, "rrf": r["rrf"], "dense": r["dense"], "bm25": r["bm25"]}
               for i, r in enumerate(found["retrieved"])]
    return {"mode": "rag", "answer": text or FALLBACK_ANSWER, "ok": bool(text),
            "sources": sources, "subqueries": queries, "debug": found["debug"]}
