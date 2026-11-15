#!/usr/bin/env python3
"""Build the three importable n8n workflows from the markdown knowledge base.

    python3 tools/build_workflows.py

Reads  kb/*.md          (each "## Heading" becomes one retrievable chunk)
Writes n8n/01_ingest_f1_knowledge_base.json
       n8n/02_f1_llm_chatbot.json
       n8n/03_f1_rag_chatbot.json
       n8n/kb_chunks.json   (the chunks, for inspection)

Edit the markdown, re-run this script, re-import the JSON. Standard library only.
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "kb"
OUT_DIR = ROOT / "n8n"

MAX_CHUNK_CHARS = 900          # the n8n Default Data Loader splits above ~1000
MEMORY_KEY = "f1_knowledge_base"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


# --------------------------------------------------------------------------
# 1. Knowledge base -> chunks
# --------------------------------------------------------------------------
def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:60]


def split_long(text: str, limit: int) -> list[str]:
    """Split on sentence boundaries so no piece is longer than `limit`."""
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, cur = [], ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > limit:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    seen: set[str] = set()
    for path in sorted(KB_DIR.glob("*.md")):
        category = path.stem
        title_line = None
        sections: list[tuple[str, list[str]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# ") and title_line is None:
                title_line = line[2:].strip()
            elif line.startswith("## "):
                sections.append((line[3:].strip(), []))
            elif sections:
                sections[-1][1].append(line)
        category_name = title_line or category
        file_key = re.sub(r"^\d+_", "", category)
        for title, body_lines in sections:
            body = " ".join(l.strip() for l in body_lines if l.strip())
            if not body:
                continue
            for part_no, part in enumerate(split_long(body, MAX_CHUNK_CHARS - len(title) - 5), 1):
                cid = f"{file_key}-{slug(title)}" + (f"-{part_no}" if part_no > 1 else "")
                if cid in seen:
                    raise SystemExit(f"duplicate chunk id: {cid}")
                seen.add(cid)
                chunks.append({
                    "id": cid,
                    "title": title,
                    "category": category_name,
                    "text": f"{title}. {part}",
                })
    return chunks


# --------------------------------------------------------------------------
# 2. Prompts (no curly braces: n8n's chain node treats {x} as a template var)
# --------------------------------------------------------------------------
LLM_SYSTEM = """You are Pit Wall Pal, a friendly Formula 1 explainer built into a live-stream viewing page. The viewer is a beginner who is watching a session right now and got confused by something.

How to answer:
- Use plain words. Explain any F1 jargon you use. Be warm, never condescending.
- Keep it short: 2 to 4 sentences, about 70 words at most, unless the viewer asks for more detail ("tell me more").
- No bullet lists, no headings. Write like you are talking to a friend on the sofa.
- If a LIVE STREAM CONTEXT line is provided, use it to connect your answer to what the viewer is seeing.

Honesty rules:
- You do NOT have the live timing feed. Never claim to know current positions, gaps, lap times, who is winning, or what just happened in the race. If asked, say you cannot see the live feed, point them to the on-screen graphics, and offer to explain the terms they are seeing.
- Formula 1 rules change every year, and 2026 brought the biggest changes in years (new engines, a much bigger electric part, active aerodynamics, a new team). If an answer depends on the current rules and you are not sure, say so plainly instead of guessing. Never invent rule numbers, penalty sizes or point values.

Example 1
Viewer: what is an undercut?
Assistant: An undercut is a pit stop trick. A driver pits before the car ahead, uses the grip of fresh tyres to set a quick lap, and if it works they come out in front once the rival has also stopped.

Example 2
Viewer: who is leading right now?
Assistant: I cannot see the live timing, so I do not know who is leading at the moment. Check the timing tower on your screen, and if there is a term in it you do not recognise, ask me and I will explain it."""

DECOMPOSE_SYSTEM = """You help a search engine that looks through a Formula 1 beginner knowledge base (glossary, flags, race weekend format, points, penalties and stewards, Safety Car and red flags, tyres and strategy, the 2026 cars, and how to read the TV graphics).

Rewrite the viewer's question into exactly 3 short search queries, one per line, each looking at a different angle: for example the exact term, the underlying rule, and the situation that is happening. Use proper F1 terms. For instance, if a viewer asks why everybody slowed down, think of Safety Car, Virtual Safety Car and yellow flag.

Output only the 3 queries, one per line. No numbering, no quotes, no extra text."""

RAG_SYSTEM = """You are Pit Wall Pal, a friendly Formula 1 explainer built into a live-stream viewing page. The viewer is a beginner who is watching a session right now and got confused by something.

You are given numbered REFERENCE NOTES from an F1 knowledge base with each question. Follow these rules:
- Answer using ONLY the reference notes. If they do not cover the question, say you do not have that in your F1 notes, and suggest a related thing you can explain. Do not fill gaps from memory.
- If the notes conflict with what you remember, trust the notes. The sport changed a lot in 2026, for example DRS was removed.
- After each fact you take from a note, add its number in square brackets, like [1] or [2]. Only use numbers that exist. Do not add a separate source list.
- You do NOT have the live timing feed. Never claim to know current positions, gaps, lap times, who is winning, or what just happened. If asked, say you cannot see the live feed and point to the on-screen graphics. You may still explain what a term or situation means.
- If a LIVE STREAM CONTEXT line is provided, use it to connect the answer to what the viewer is seeing.
- Use plain words and explain jargon. Keep it short: 2 to 4 sentences, about 70 words at most, unless the viewer asks for more detail. No bullet lists, no headings.
- Never invent rules, numbers or names."""


# --------------------------------------------------------------------------
# 3. Code-node sources (plain JavaScript run inside n8n)
# --------------------------------------------------------------------------
PREPARE_JS = r"""// Normalise the request sent by the chat page.
const body = $input.first().json.body || {};
const question = String(body.question || '').trim().slice(0, 600);
if (!question) {
  throw new Error('The request body needs a "question" field.');
}

// What the viewer is currently watching (optional, sent by the page).
const ctx = body.context || {};
const parts = [];
if (ctx.event)   parts.push('Event: ' + String(ctx.event).slice(0, 80));
if (ctx.session) parts.push('Session: ' + String(ctx.session).slice(0, 40));
if (ctx.lap)     parts.push('Lap: ' + String(ctx.lap).slice(0, 20));
if (ctx.status)  parts.push('Track status shown on screen: ' + String(ctx.status).slice(0, 60));
if (ctx.note)    parts.push('Viewer note: ' + String(ctx.note).slice(0, 200));
const live = parts.length ? parts.join(' | ') : 'not provided';
const liveHint = [ctx.status, ctx.note].filter(Boolean).map(x => String(x).slice(0, 80)).join(' ');

// Short chat history (the page is stateless, so it sends the last turns).
const history = Array.isArray(body.history) ? body.history.slice(-6) : [];
const historyText = history
  .map(m => ((m && m.role === 'user') ? 'Viewer: ' : 'Assistant: ') + String((m && m.content) || '').slice(0, 500))
  .join('\n') || '(no earlier messages)';

return [{ json: { question, live, liveHint, historyText } }];
"""

PARSE_SUBQUERIES_JS = r"""// Turn the LLM's 3 lines into one n8n item per search query.
const prep = $('Prepare').first().json;
const first = $input.first().json || {};
const raw = String(first.text || first.output || '');

let subs = raw.split('\n')
  .map(s => s.replace(/^[\s\-\*•\d\.\)]+/, '').replace(/^["']+|["']+$/g, '').trim())
  .filter(s => s.length > 2 && s.length < 160);
subs = [...new Set(subs)].filter(s => s.toLowerCase() !== prep.question.toLowerCase());

// Always search the original question. If the viewer set a track status or note,
// also search "question + what is on screen" so e.g. "why did they slow down?"
// under a Safety Car banner finds the Safety Car notes.
const queries = [prep.question];
if (prep.liveHint) queries.push(prep.question + ' ' + prep.liveHint);
for (const s of subs) { if (queries.length < 4) queries.push(s); }

return queries.map(q => ({ json: { q } }));
"""

HYBRID_MERGE_JS = r"""// ---------------------------------------------------------------------------
// Hybrid retrieval, same idea as the class project (dense + BM25), in plain JS:
//   dense hits  : already retrieved by the vector store node above (semantic)
//   BM25 hits   : computed here over the same knowledge base (exact words)
//   fusion      : Reciprocal Rank Fusion merges both rankings into the final top 5
// ---------------------------------------------------------------------------
const KB = __KB_JSON__;

const prep = $('Prepare').first().json;
const queries = $('Parse sub-queries').all().map(i => i.json.q);

// ---- tiny text normaliser: lower-case, strip accents (fermé -> ferme), UK/US spelling, light stemming
const SYN = { tire: 'tyre', tires: 'tyres', curb: 'kerb', curbs: 'kerbs', color: 'colour', colors: 'colours', quali: 'qualifying' };
const STOP = new Set(('a an the is are was were be been being of to in on at for from by with and or but if then than that this these those ' +
  'it its as do does did what whats why how when who which can could should would will may might i you he she we they me my your our their ' +
  'there here so not no yes about into over under up down out off just also very more most some any tell mean means ' +
  'work works happen happens get gets').split(' '));
function stem(t) {
  if (t.length > 4 && t.endsWith('ies')) t = t.slice(0, -3) + 'y';
  else if (t.length > 3 && t.endsWith('s') && !t.endsWith('ss')) t = t.slice(0, -1);
  if (t.length > 5 && t.endsWith('ing')) t = t.slice(0, -3);
  if (t.length > 4 && t.endsWith('e')) t = t.slice(0, -1);
  return t;
}
function tokens(s) {
  return (String(s).toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').match(/[a-z0-9]+/g) || [])
    .map(t => SYN[t] || t).filter(t => !STOP.has(t)).map(stem);
}

// ---- BM25 index over the knowledge base (title counted 3x so exact terms win)
const docs = KB.map(d => {
  const toks = tokens((d.title + ' ').repeat(3) + d.text);
  const tf = new Map();
  toks.forEach(t => tf.set(t, (tf.get(t) || 0) + 1));
  return { id: d.id, tf, len: toks.length };
});
const N = docs.length;
const avgdl = docs.reduce((a, d) => a + d.len, 0) / N;
const df = new Map();
docs.forEach(d => d.tf.forEach((_, t) => df.set(t, (df.get(t) || 0) + 1)));
const K1 = 1.5, B = 0.75;
function bm25(queryTokens) {
  const out = new Map();
  for (const d of docs) {
    let s = 0;
    for (const t of new Set(queryTokens)) {
      const f = d.tf.get(t);
      if (!f) continue;
      const idf = Math.log(1 + (N - df.get(t) + 0.5) / (df.get(t) + 0.5));
      s += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * d.len / avgdl));
    }
    if (s > 0) out.set(d.id, s);
  }
  return out;
}

// ---- 1) BM25 for every query, keep the best score per chunk
const lexical = new Map();
for (const q of queries) {
  for (const [id, s] of bm25(tokens(q))) lexical.set(id, Math.max(lexical.get(id) || 0, s));
}
const lexicalRanked = [...lexical.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8).map(e => e[0]);

// ---- 2) dense hits from the vector store node (one item per hit, all sub-queries pooled)
const byId = new Map(KB.map(d => [d.id, d]));
const dense = new Map();
let order = 0;
for (const it of $input.all()) {
  const j = it.json || {};
  const doc = j.document || j;
  const id = doc.metadata && doc.metadata.id;
  if (!id || !byId.has(id)) continue;                       // empty item or unknown chunk
  const score = typeof j.score === 'number' ? j.score : (typeof doc.score === 'number' ? doc.score : 0);
  const prev = dense.get(id);
  if (!prev || score > prev.score) dense.set(id, { score, order: prev ? prev.order : order++ });
}
const denseRanked = [...dense.entries()]
  .sort((a, b) => (b[1].score - a[1].score) || (a[1].order - b[1].order))
  .map(e => e[0]);

// ---- 3) Reciprocal Rank Fusion: 1/(60 + rank) from each list, summed
const RRF_K = 60;
const fused = new Map();
const addRanks = list => list.forEach((id, i) => fused.set(id, (fused.get(id) || 0) + 1 / (RRF_K + i + 1)));
addRanks(denseRanked);
addRanks(lexicalRanked);
const top = [...fused.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);

const retrieved = top.map(([id, rrf]) => {
  const d = byId.get(id);
  return {
    id, title: d.title, category: d.category, text: d.text,
    rrf: Number(rrf.toFixed(4)),
    dense: dense.has(id) ? Number(dense.get(id).score.toFixed(3)) : null,
    bm25: lexical.has(id) ? Number(lexical.get(id).toFixed(2)) : null,
  };
});
const contextText = retrieved.length
  ? retrieved.map((r, i) => '[' + (i + 1) + '] ' + r.title + ' (' + r.category + ')\n' + r.text).join('\n\n')
  : '(no matching notes were found)';

return [{ json: {
  queries,
  contextText,
  retrieved,
  debug: { denseCandidates: denseRanked.length, lexicalCandidates: lexicalRanked.length },
} }];
"""

FORMAT_LLM_JS = r"""const first = $input.first().json || {};
const answer = String(first.text || first.output || '').trim();
return [{ json: {
  mode: 'llm',
  answer: answer || "Sorry, I couldn't get an answer from the language model just now. Please try again.",
  ok: Boolean(answer),
} }];
"""

FORMAT_RAG_JS = r"""const first = $input.first().json || {};
const answer = String(first.text || first.output || '').trim();
const m = $('Hybrid merge').first().json;

// Which numbered notes did the model actually cite, e.g. [2]?
const cited = new Set((answer.match(/\[(\d+)\]/g) || []).map(x => Number(x.slice(1, -1))));
const sources = m.retrieved.map((r, i) => ({
  n: i + 1, id: r.id, title: r.title, category: r.category, cited: cited.has(i + 1),
  rrf: r.rrf, dense: r.dense, bm25: r.bm25,
}));

return [{ json: {
  mode: 'rag',
  answer: answer || "Sorry, I couldn't get an answer from the language model just now. Please try again.",
  ok: Boolean(answer),
  sources,
  subqueries: m.queries,
  debug: m.debug,
} }];
"""

LOAD_CHUNKS_JS = r"""// The knowledge base chunks are embedded in this node (generated by tools/build_workflows.py).
// One n8n item per chunk. The vector store embeds `text` and keeps id/title/category as metadata.
const KB = __KB_JSON__;
return KB.map(d => ({ json: d }));
"""


# --------------------------------------------------------------------------
# 4. Node builders
# --------------------------------------------------------------------------
def nid() -> str:
    return str(uuid.uuid4())


def node(name: str, ntype: str, version, params: dict, pos: list[int], **extra) -> dict:
    n = {
        "parameters": params,
        "id": nid(),
        "name": name,
        "type": ntype,
        "typeVersion": version,
        "position": pos,
    }
    n.update(extra)
    return n


def webhook(path: str, pos) -> dict:
    return node(
        "Chat Webhook", "n8n-nodes-base.webhook", 2,
        {"httpMethod": "POST", "path": path, "responseMode": "responseNode",
         "options": {"allowedOrigins": "*"}},
        pos, webhookId=nid(),
    )


def respond(pos) -> dict:
    return node(
        "Respond to Webhook", "n8n-nodes-base.respondToWebhook", 1.1,
        {"respondWith": "firstIncomingItem",
         "options": {"responseHeaders": {"entries": [
             {"name": "Access-Control-Allow-Origin", "value": "*"}]}}},
        pos,
    )


def code(name: str, js: str, pos) -> dict:
    return node(name, "n8n-nodes-base.code", 2, {"jsCode": js}, pos)


def chat_model(name: str, pos, temperature: float, max_tokens: int) -> dict:
    return node(
        name, "@n8n/n8n-nodes-langchain.lmChatOpenAi", 1.2,
        {"model": {"__rl": True, "value": CHAT_MODEL, "mode": "list", "cachedResultName": CHAT_MODEL},
         "options": {"temperature": temperature, "maxTokens": max_tokens}},
        pos,
    )


def chain(name: str, system: str, text_expr: str, pos) -> dict:
    return node(
        name, "@n8n/n8n-nodes-langchain.chainLlm", 1.5,
        {"promptType": "define", "text": text_expr,
         "messages": {"messageValues": [{"type": "SystemMessagePromptTemplate", "message": system}]}},
        pos, onError="continueRegularOutput",
    )


def embeddings(name: str, pos) -> dict:
    return node(name, "@n8n/n8n-nodes-langchain.embeddingsOpenAi", 1.2,
                {"model": EMBED_MODEL, "options": {}}, pos)


def memory_key() -> dict:
    return {"__rl": True, "value": MEMORY_KEY, "mode": "list", "cachedResultName": MEMORY_KEY}


def sticky(text: str, pos, w: int, h: int, color: int = 7) -> dict:
    return node("Note " + nid()[:4], "n8n-nodes-base.stickyNote", 1,
                {"content": text, "height": h, "width": w, "color": color}, pos)


def link(conns: dict, src: str, dst: str, kind: str = "main", idx: int = 0) -> None:
    conns.setdefault(src, {}).setdefault(kind, [[]])
    while len(conns[src][kind]) <= idx:
        conns[src][kind].append([])
    conns[src][kind][idx].append({"node": dst, "type": kind, "index": 0})


def workflow(name: str, nodes: list[dict], conns: dict) -> dict:
    return {
        "name": name,
        "nodes": nodes,
        "connections": conns,
        "pinData": {},
        "active": False,
        "settings": {"executionOrder": "v1"},
        "meta": {"templateCredsSetupCompleted": False},
        "tags": [],
    }


# --------------------------------------------------------------------------
# 5. The three workflows
# --------------------------------------------------------------------------
def build_ingest(kb_json: str) -> dict:
    nodes = [
        sticky("## 1. Ingest the F1 knowledge base\n"
               "Run this **once** (and again after every n8n restart, because the Simple Vector Store "
               "lives in memory).\n\n"
               "**Manual Trigger → Load KB chunks → Store F1 notes**\n\n"
               "The chunks come from the `kb/` markdown files. Embeddings: OpenAI "
               f"`{EMBED_MODEL}`. Memory key: `{MEMORY_KEY}`. The RAG chatbot must use the same "
               "key and the same embedding model.",
               [-40, -60], 560, 230),
        node("Run once", "n8n-nodes-base.manualTrigger", 1, {}, [0, 260]),
        code("Load KB chunks", LOAD_CHUNKS_JS.replace("__KB_JSON__", kb_json), [240, 260]),
        node("Store F1 notes", "@n8n/n8n-nodes-langchain.vectorStoreInMemory", 1.3,
             {"mode": "insert", "memoryKey": memory_key(), "clearStore": True}, [480, 260]),
        embeddings("Embeddings OpenAI", [400, 500]),
        node("Default Data Loader", "@n8n/n8n-nodes-langchain.documentDefaultDataLoader", 1,
             {"jsonMode": "expressionData", "jsonData": "={{ $json.text }}",
              "options": {"metadata": {"metadataValues": [
                  {"name": "id", "value": "={{ $json.id }}"},
                  {"name": "title", "value": "={{ $json.title }}"},
                  {"name": "category", "value": "={{ $json.category }}"}]}}},
             [620, 500]),
    ]
    c: dict = {}
    link(c, "Run once", "Load KB chunks")
    link(c, "Load KB chunks", "Store F1 notes")
    link(c, "Embeddings OpenAI", "Store F1 notes", "ai_embedding")
    link(c, "Default Data Loader", "Store F1 notes", "ai_document")
    return workflow("F1 chatbot - 1. Ingest knowledge base", nodes, c)


def build_llm() -> dict:
    nodes = [
        sticky("## LLM-only chatbot (prompting, no retrieval)\n"
               "**Webhook → Prepare → Pit Wall Pal (LLM) → Format → Respond**\n\n"
               "Everything the model knows comes from its training plus the system prompt "
               "(persona, rules, two examples). Great for explaining stable ideas like an undercut. "
               "Weak for anything that changed recently, which is the point of comparing it with the RAG bot.\n\n"
               "POST body: question, context (event, session, lap, status, note), history.",
               [-40, -80], 620, 250),
        webhook("f1-llm-chat", [0, 300]),
        code("Prepare", PREPARE_JS, [240, 300]),
        chain("Pit Wall Pal (LLM only)", LLM_SYSTEM,
              "=LIVE STREAM CONTEXT: {{ $json.live }}\n\nRECENT CHAT:\n{{ $json.historyText }}\n\nVIEWER QUESTION: {{ $json.question }}",
              [480, 300]),
        chat_model("OpenAI Chat Model", [500, 540], 0.3, 300),
        code("Format response", FORMAT_LLM_JS, [780, 300]),
        respond([1020, 300]),
    ]
    c: dict = {}
    link(c, "Chat Webhook", "Prepare")
    link(c, "Prepare", "Pit Wall Pal (LLM only)")
    link(c, "OpenAI Chat Model", "Pit Wall Pal (LLM only)", "ai_languageModel")
    link(c, "Pit Wall Pal (LLM only)", "Format response")
    link(c, "Format response", "Respond to Webhook")
    return workflow("F1 chatbot - 2. LLM-only chatbot", nodes, c)


def build_rag(kb_json: str) -> dict:
    nodes = [
        sticky("## RAG chatbot (mirrors the class Myntra M2 + M3 pipeline)\n"
               "**Webhook → Prepare → Decompose (LLM) → Parse sub-queries → Find notes "
               "(vector store, dense) → Hybrid merge (BM25 + RRF) → Answer (LLM) → Format → Respond**\n\n"
               "1. The LLM rewrites the question into 3 search angles (query decomposition).\n"
               "2. Each query runs through dense search (vector store) AND BM25 keyword search.\n"
               "3. Reciprocal Rank Fusion merges both rankings; top 5 notes go to the LLM.\n"
               "4. The LLM answers ONLY from those notes and cites them as [1], [2].\n\n"
               "Run workflow 1 first so the vector store is filled.",
               [-40, -150], 900, 280),
        webhook("f1-rag-chat", [0, 300]),
        code("Prepare", PREPARE_JS, [240, 300]),
        chain("Decompose", DECOMPOSE_SYSTEM,
              "=Viewer question: {{ $json.question }}\nLive context: {{ $json.live }}",
              [480, 300]),
        chat_model("Decompose model", [500, 540], 0.3, 100),
        code("Parse sub-queries", PARSE_SUBQUERIES_JS, [760, 300]),
        node("Find notes (dense)", "@n8n/n8n-nodes-langchain.vectorStoreInMemory", 1.3,
             {"mode": "load", "memoryKey": memory_key(), "prompt": "={{ $json.q }}",
              "topK": 4, "includeDocumentMetadata": True},
             [1000, 300], alwaysOutputData=True, onError="continueRegularOutput"),
        embeddings("Embeddings OpenAI", [1020, 540]),
        code("Hybrid merge", HYBRID_MERGE_JS.replace("__KB_JSON__", kb_json), [1260, 300]),
        chain("Answer (grounded)", RAG_SYSTEM,
              "=LIVE STREAM CONTEXT: {{ $('Prepare').first().json.live }}\n\nRECENT CHAT:\n{{ $('Prepare').first().json.historyText }}\n\nREFERENCE NOTES:\n{{ $json.contextText }}\n\nVIEWER QUESTION: {{ $('Prepare').first().json.question }}",
              [1500, 300]),
        chat_model("Answer model", [1520, 540], 0.2, 300),
        code("Format response", FORMAT_RAG_JS, [1780, 300]),
        respond([2020, 300]),
    ]
    c: dict = {}
    link(c, "Chat Webhook", "Prepare")
    link(c, "Prepare", "Decompose")
    link(c, "Decompose model", "Decompose", "ai_languageModel")
    link(c, "Decompose", "Parse sub-queries")
    link(c, "Parse sub-queries", "Find notes (dense)")
    link(c, "Embeddings OpenAI", "Find notes (dense)", "ai_embedding")
    link(c, "Find notes (dense)", "Hybrid merge")
    link(c, "Hybrid merge", "Answer (grounded)")
    link(c, "Answer model", "Answer (grounded)", "ai_languageModel")
    link(c, "Answer (grounded)", "Format response")
    link(c, "Format response", "Respond to Webhook")
    return workflow("F1 chatbot - 3. RAG chatbot", nodes, c)


# --------------------------------------------------------------------------
def main() -> None:
    chunks = load_chunks()
    if not chunks:
        sys.exit("no chunks found in kb/")
    OUT_DIR.mkdir(exist_ok=True)
    kb_json = json.dumps(chunks, ensure_ascii=False, separators=(",", ":"))

    (OUT_DIR / "kb_chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    for fname, wf in [
        ("01_ingest_f1_knowledge_base.json", build_ingest(kb_json)),
        ("02_f1_llm_chatbot.json", build_llm()),
        ("03_f1_rag_chatbot.json", build_rag(kb_json)),
    ]:
        (OUT_DIR / fname).write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote n8n/{fname}  ({len(wf['nodes'])} nodes)")

    longest = max(len(c["text"]) for c in chunks)
    print(f"{len(chunks)} chunks from {len(list(KB_DIR.glob('*.md')))} files; longest chunk {longest} chars")


if __name__ == "__main__":
    main()
