# Pit Wall Pal: an F1 companion chatbot (LLM-only + RAG) built in n8n

A chat sidebar for a live F1 stream (F1 TV, FanCode, anything). A newcomer who gets lost mid-race
asks "why did everyone slow down?" or "what's parc fermé?" and gets a short plain-English answer.
You get **two chatbots** behind the same page so you can compare them:

| Bot | What it is | n8n workflow |
|---|---|---|
| **LLM-only** | Prompt engineering only: persona, rules, two few-shot examples, live-stream context. No retrieval. | `n8n/02_f1_llm_chatbot.json` |
| **RAG** | Question rewriting, hybrid retrieval over an F1 knowledge base, answer grounded in the retrieved notes with citations. | `n8n/01_ingest_f1_knowledge_base.json` + `n8n/03_f1_rag_chatbot.json` |

## How it maps to the class Myntra project

| Class project (Python) | This project (n8n) |
|---|---|
| `myntra_nodes()` loads the CSV into text nodes | `kb/*.md` → one chunk per `##` heading (208 chunks), embedded in the **Load KB chunks** node |
| `VectorStoreIndex` + bge-small embeddings | **Simple Vector Store** + OpenAI `text-embedding-3-small` |
| `BM25Retriever` | BM25 written in the **Hybrid merge** Code node (same corpus) |
| merge both top-10s, cross-encoder rerank | merge with **Reciprocal Rank Fusion** (see limitations: no cross-encoder in n8n) |
| M3 `decompose()`: LLM rewrites the query into 3 sub-queries | **Decompose** chain + **Parse sub-queries** node (3 rewrites + the original) |
| `answer_query()` "use ONLY the products listed" | **Answer (grounded)** chain: "use ONLY the numbered notes", cites `[1]`, `[2]` |
| Streamlit UI | `web/index.html` |
| `goldset.json` + RAGAS | `eval/f1_goldset.json` + `tools/compare_bots.py` |

What is new for the live-stream use case: the page sends **what is on screen** (session, lap, track status,
a free-text note) with each question, the bot is told it has **no live timing** and must say so instead of
guessing, answers are capped at about 70 words, and chat history is sent by the page (the webhooks are stateless).

## Setup (about 15 minutes)

1. **Run n8n.** `npx n8n` (needs a recent Node.js), or Docker: `docker run -p 5678:5678 -v n8n_data:/home/node/.n8n n8nio/n8n`.
   Open http://localhost:5678.
2. **Create an OpenAI credential** in n8n (Credentials → New → OpenAI API → paste your key).
3. **Import the three workflows** (Workflows → ⋯ → Import from file), in this order:
   `01_ingest…`, `02_f1_llm…`, `03_f1_rag…`.
   Open each OpenAI node (Chat Model, Embeddings) and select your credential.
4. **Fill the vector store:** open workflow 1, click **Execute workflow**. It embeds 208 short chunks, which costs a fraction of a cent.
   Re-run it after every n8n restart, because the Simple Vector Store is in-memory.
5. **Activate** workflows 2 and 3 (toggle top right). The page uses the production URLs `/webhook/...`.
6. **Open `web/index.html`** in a browser. Under *Connection to n8n* the base URL defaults to
   `http://localhost:5678/webhook`. Ask something.

Quick test without the page:
```bash
curl -s -X POST http://localhost:5678/webhook/f1-rag-chat -H 'Content-Type: application/json' \
  -d '{"question":"Is DRS still used?","context":{"session":"Race"},"history":[]}'
```

## The demo that makes the point (LLM-only vs RAG)

Switch the page to **Compare** and ask:

1. *"Is DRS still used?"* and *"What is Overtake Mode?"*: F1 changed its overtaking aid for 2026. A model trained earlier tends to describe DRS from memory or hedge; RAG answers from the notes.
2. *"Why did everyone slow down?"* with Track status set to **Safety Car**: the context line steers retrieval.
3. *"Who is leading right now?"*: both bots should say they cannot see live timing.
4. *"What's an undercut?"*: a stable concept; expect both bots to do fine. Not every question needs RAG.

For numbers rather than anecdotes: `python3 tools/compare_bots.py` runs the 23-question set through both webhooks and writes
`eval/results.md` and `eval/results.csv`. The pass column is a regex smoke test, so read the answers too.
The goldset has the same shape as your class one (exact match, cross-section/concept, multi-hop) plus **fresh_2026**,
**live_data** and **out_of_scope** categories. To use RAGAS as in class, export the retrieved notes (`sources`) and answers from the webhook response.

## Editing the knowledge base

Edit or add `##` sections in `kb/*.md` (keep each under about 800 characters), then:
```bash
python3 tools/build_workflows.py      # regenerates n8n/*.json
node tools/test_workflows.mjs         # offline checks (no n8n, no API key)
```
Re-import workflows 1 and 3 and re-run the ingest. The KB was written from FIA and F1 sources in November 2026
(sources below). **Rules change every season, so re-check it before you rely on it.**

## Files

```
kb/                      8 markdown files: glossary, flags, weekend + points, penalties, race control, tyres, 2026 cars, TV graphics
n8n/                     generated, importable workflows (+ kb_chunks.json to inspect the chunks)
web/index.html           the chat page (no build step, no dependencies)
tools/build_workflows.py KB + prompts + Code-node JS  →  n8n JSON
tools/test_workflows.mjs offline tests: graph structure + the Code-node logic
tools/compare_bots.py    runs eval/f1_goldset.json against both bots
eval/f1_goldset.json     23 questions with regex checks
```

## Honest limitations

- **The n8n workflows have not been run inside a live n8n instance.** I built them without access to one, validated the JSON structure,
  and ran the actual Code-node JavaScript offline (request parsing, sub-query parsing, BM25 + fusion, response formatting; the keyword half alone
  put the right note in the top 5 for all 24 test questions). I checked parameter names against n8n's docs, but node versions drift. If a node shows a
  warning after import, open it and re-select the credential, model, or **Memory Key** (`f1_knowledge_base`, identical in workflows 1 and 3).
  The dense (embedding) half and the LLM prompts are untested against real model output.
- **No cross-encoder reranker.** The class project used `bge-reranker-base` locally. n8n cannot run that, so results are fused with
  Reciprocal Rank Fusion instead. Upgrade path: n8n's Cohere Reranker node plugs into the vector store's *Rerank Results* option.
- **No live data.** The bot cannot know positions, gaps or incidents. A real product would add a live-timing API (for example OpenF1)
  as another retrieval source. Until then the bot says so.
- **In-memory vector store.** Fine for development (n8n's own advice), lost on restart, and memory keys are global to the n8n instance.
  For anything lasting, swap **Simple Vector Store** for Qdrant, Supabase or Pinecone in both workflows and re-embed.
- **Public webhooks.** Anyone who can reach them spends your OpenAI credit. Before exposing them, add Header Auth to the Webhook nodes
  and lock the CORS origin to your site. Answers are capped at about 300 tokens, and questions at 600 characters.
- **Latency.** RAG makes two LLM calls (decompose, answer) plus embedding calls, so it will be noticeably slower than LLM-only. To trade quality for speed, delete **Decompose** and its model node and connect **Prepare** straight to **Parse sub-queries**; it then searches just the original question (plus the on-screen context).
- **Swapping providers.** Class used OpenRouter or OpenAI. For OpenRouter replace the two chat-model nodes with n8n's OpenRouter Chat Model. Keep OpenAI (or another embeddings node) for embeddings, and keep the same embedding model in workflows 1 and 3.
- **Not affiliated with Formula 1, F1 TV or FanCode.** "Pit Wall Pal" is a placeholder name; the page uses no official logos.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Page says "Could not reach …" | n8n not running, wrong base URL, or workflow not Active. For `/webhook-test/…` click *Listen for test event* in the editor first. |
| HTTP 404 from n8n | Workflow is not Active (production URL) or the path differs (`f1-llm-chat`, `f1-rag-chat`). |
| RAG answers "not in my notes" for everything | Ingest not run since the last n8n restart, or the Memory Key differs between workflows 1 and 3. |
| Browser CORS error | Keep `allowedOrigins` on the Webhook node and the `Access-Control-Allow-Origin` header on Respond to Webhook. |
| Workflow hangs with no response | A node failed before Respond to Webhook; open the execution in n8n to see which. The chain nodes continue on error so you get a friendly fallback answer. |

## Sources used for the knowledge base

- Formula 1: [Key terms for the 2026 rules](https://www.formula1.com/en/latest/article/f1-2026-regulations-explained-the-key-terms-for-formula-1s-new-for-2026-rules.3T5BU6TC9quGcIpGzoWkY0) (Boost, Overtake Mode, Recharge, Active Aero)
- McLaren: [Explaining F1's new 2026 regulations](https://www.mclaren.com/racing/formula-1/2026/explaining-f1s-new-2026-regulations/) and [guide to the 2026 Sprint](https://www.mclaren.com/racing/formula-1/2026/guide-to-the-f1-sprint/)
- Crash.net: [2026 regulations: aero, chassis and engine](https://www.crash.net/f1/news/1050183/1/f1-2026-regulations-changes-aero-chassis-engine)
- Honda: [2026 F1 Explained: race day format](https://global.honda/en/F1/features/f1-explained/03/) and [qualifying](https://global.honda/en/F1/features/f1-explained/09/)
- PlanetF1: [2026 qualifying format (Q3 extra minute)](https://www.planetf1.com/news/f1-2026-qualifying-format-change-fia-extra-minute-q3)
- Sky Sports: [2026 Sprint format and points](https://www.skysports.com/f1/news/12040/13518235/f1-sprint-schedule-points-results-format-explained-qualifying-race-and-venues-for-2026-season)
- The Race: [2026 penalty and driving guideline changes](https://www.the-race.com/formula-1/f1-2026-penalty-driving-guidelines-changes/)
- FIA: [2026 F1 Sporting Regulations (Section B)](https://api.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_07_-_2026-06-25.pdf)
- Motor Sport Magazine: [F1 flags guide](https://motorsportmagazine.com/articles/single-seaters/f1/what-does-each-f1-flag-mean)
- Silverstone: [F1 penalties explained](https://www.silverstone.co.uk/index.php/news/f1-penalties-explained-f101-beginners-guide-penalty-system)
- Williams, Autosport and Planet F1 glossaries for jargon definitions
- n8n docs: [Simple Vector Store](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstoreinmemory/)
