#!/usr/bin/env python3
"""Ask the LLM-only bot and the RAG bot the same questions and compare.

    python3 tools/compare_bots.py                      # production URLs (workflows must be Active)
    python3 tools/compare_bots.py --base http://localhost:5678/webhook-test   # test URLs
    python3 tools/compare_bots.py --only fresh_2026 live_data                 # some categories

Writes eval/results.md (answers side by side) and eval/results.csv.
The pass/fail column is a crude regex smoke check from eval/f1_goldset.json,
so READ the answers too. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval" / "f1_goldset.json"


def ask(url: str, question: str, timeout: int = 90) -> dict:
    body = json.dumps({"question": question, "context": {}, "history": []}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        if isinstance(data, list):
            data = data[0] if data else {}
        data["_seconds"] = round(time.time() - t0, 1)
        return data
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return {"answer": f"[request failed: {e}]", "ok": False, "_seconds": round(time.time() - t0, 1)}


def passes(answer: str, patterns: list[str]) -> bool:
    return all(re.search(p, answer, re.I) for p in patterns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:5678/webhook", help="n8n webhook base URL")
    ap.add_argument("--only", nargs="*", help="only these categories")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    questions = json.loads(GOLD.read_text())["questions"]
    if args.only:
        questions = [q for q in questions if q["category"] in args.only]

    rows = []
    score = defaultdict(lambda: {"llm": [0, 0], "rag": [0, 0]})
    for q in questions:
        llm = ask(f"{base}/f1-llm-chat", q["question"])
        rag = ask(f"{base}/f1-rag-chat", q["question"])
        l_ok = passes(llm.get("answer", ""), q["must_include"])
        r_ok = passes(rag.get("answer", ""), q["must_include"])
        score[q["category"]]["llm"][0] += l_ok
        score[q["category"]]["llm"][1] += 1
        score[q["category"]]["rag"][0] += r_ok
        score[q["category"]]["rag"][1] += 1
        cited = [s["title"] for s in rag.get("sources", []) if s.get("cited")]
        rows.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "llm_pass": l_ok, "llm_answer": llm.get("answer", ""), "llm_s": llm["_seconds"],
            "rag_pass": r_ok, "rag_answer": rag.get("answer", ""), "rag_s": rag["_seconds"],
            "rag_cited": "; ".join(cited),
        })
        print(f"{q['id']} [{q['category']}]  LLM {'PASS' if l_ok else 'fail'} ({llm['_seconds']}s)   "
              f"RAG {'PASS' if r_ok else 'fail'} ({rag['_seconds']}s)")

    out_csv = ROOT / "eval" / "results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# LLM-only vs RAG", "", "| category | LLM-only | RAG |", "|---|---|---|"]
    tot = {"llm": [0, 0], "rag": [0, 0]}
    for cat, s in score.items():
        lines.append(f"| {cat} | {s['llm'][0]}/{s['llm'][1]} | {s['rag'][0]}/{s['rag'][1]} |")
        for k in tot:
            tot[k][0] += s[k][0]
            tot[k][1] += s[k][1]
    lines.append(f"| **total** | **{tot['llm'][0]}/{tot['llm'][1]}** | **{tot['rag'][0]}/{tot['rag'][1]}** |")
    lines += ["", "## Answers", ""]
    for r in rows:
        lines += [
            f"### {r['id']} ({r['category']}): {r['question']}", "",
            f"**LLM-only** ({'pass' if r['llm_pass'] else 'fail'}, {r['llm_s']}s): {r['llm_answer']}", "",
            f"**RAG** ({'pass' if r['rag_pass'] else 'fail'}, {r['rag_s']}s): {r['rag_answer']}", "",
            f"*Notes cited by RAG:* {r['rag_cited'] or '(none)'}", "",
        ]
    (ROOT / "eval" / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[:4 + len(score) + 1]))
    print("\nWrote eval/results.md and eval/results.csv")


if __name__ == "__main__":
    main()
