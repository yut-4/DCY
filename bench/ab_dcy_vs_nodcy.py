#!/usr/bin/env python3
"""A/B: same model, same tasks, with DCY context vs without.

Measures the three quantities the validation protocol requires per item,
never a single accuracy number:

  gold_in_page : was the gold symbol even present in the distilled page?
                 (retrieval ceiling — bounds what the model could possibly do)
  consumed     : does the model's answer token appear in the page it was shown?
                 (consumption — did it read the injected context at all)
  correct      : does the answer contain the gold symbol? (correctness)

The no-context arm on the same items is the hallucination baseline: whatever
the model emits there comes from pretraining alone.

Latency is reported per arm separately from DCY's own retrieval time, so
"DCY is slower" is attributable to the retriever or to the longer prompt
rather than being one blended number.

Matching is done on the FULL answer; truncation is display-only.

Run:
  python3 bench/ab_dcy_vs_nodcy.py --model qwen3:0.6b --limit 8
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import httpx

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


def ollama_generate(base: str, model: str, prompt: str, timeout: float) -> tuple[str, float, dict]:
    """Returns (text, wall_seconds, raw_timing_fields)."""
    t0 = time.perf_counter()
    r = httpx.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        },
        timeout=timeout,
    )
    wall = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    text = data.get("response", "") or ""
    # qwen3 thinking: content may land in `thinking`, leaving response thin.
    if not text.strip():
        text = data.get("thinking", "") or ""
    raw_len = len(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    timing = {
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "total_duration_ms": round((data.get("total_duration") or 0) / 1e6, 1),
        "thinking_chars_stripped": raw_len - len(text),
    }
    return text, wall, timing


def dcy_query(dcy: Path, db: Path, goal: str, budget: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(dcy), "query", str(db), goal, str(budget)],
        capture_output=True, text=True, timeout=120,
    )
    return proc.stdout, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description="DCY vs no-DCY A/B on one model")
    ap.add_argument("--dcy", type=Path, default=Path("build/dcy"))
    ap.add_argument("--db", type=Path, default=Path("build/tree-sitter-lib.sqlite"))
    ap.add_argument("--tasks", type=Path, default=Path("bench/tree-sitter-lib-tasks.jsonl"))
    ap.add_argument("--model", default="qwen3:0.6b")
    ap.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=Path("build/ab-dcy-vs-nodcy.json"))
    args = ap.parse_args()

    for p in (args.dcy, args.db, args.tasks):
        if not p.exists():
            ap.error(f"missing: {p}")

    tasks = []
    for line in args.tasks.read_text().splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    tasks = tasks[: args.limit]

    records = []
    print(f"model={args.model} budget={args.budget}B db={args.db} tasks={len(tasks)}")
    print("-" * 118)
    print(f"{'TASK':<16} {'GOLD':<32} {'NO_CTX':<22} {'WITH_DCY':<24} FLAGS")
    print("-" * 118)

    for t in tasks:
        goal = t["goal"]
        gold = t["gold"][0].rsplit(":", 1)[1]

        # Arm A: no context. Pure pretraining recall.
        p_none = (f"Question: {goal}\n"
                  "Answer with ONLY the C function name, nothing else.")
        ans_none, wall_none, tim_none = ollama_generate(
            args.ollama_base, args.model, p_none, args.timeout)

        # Arm B: DCY page injected.
        page, dcy_secs = dcy_query(args.dcy, args.db, goal, args.budget)
        p_dcy = (f"Repository context:\n{page}\n\n"
                 f"Question: {goal}\n"
                 "Answer with ONLY the C function name from the context above, "
                 "nothing else.")
        ans_dcy, wall_dcy, tim_dcy = ollama_generate(
            args.ollama_base, args.model, p_dcy, args.timeout)

        gold_in_page = bool(re.search(rf"\b{re.escape(gold)}\b", page))
        tok = WORD_RE.search(ans_dcy)
        consumed = bool(tok and re.search(rf"\b{re.escape(tok.group(0))}\b", page))
        correct_dcy = bool(re.search(rf"\b{re.escape(gold)}\b", ans_dcy))
        correct_none = bool(re.search(rf"\b{re.escape(gold)}\b", ans_none))

        flags = ("gold_in_page" if gold_in_page else "gold_MISSING")
        flags += " CONSUMED" if consumed else " off-page"
        if correct_dcy:
            flags += " CORRECT"
        if correct_none:
            flags += " (no-ctx also correct)"

        rec = {
            "id": t["id"], "goal": goal, "gold": gold,
            "answer_no_context": ans_none, "answer_with_dcy": ans_dcy,
            "gold_in_page": gold_in_page, "consumed": consumed,
            "correct_no_context": correct_none, "correct_with_dcy": correct_dcy,
            "secs_no_context": round(wall_none, 2),
            "secs_dcy_retrieval": round(dcy_secs, 3),
            "secs_llm_with_dcy": round(wall_dcy, 2),
            "secs_total_with_dcy": round(dcy_secs + wall_dcy, 2),
            "page_chars": len(page),
            "timing_no_context": tim_none, "timing_with_dcy": tim_dcy,
        }
        records.append(rec)
        disp_none = " ".join(ans_none.split())[:22]
        disp_dcy = " ".join(ans_dcy.split())[:24]
        print(f"{t['id']:<16} {gold:<32} {disp_none:<22} {disp_dcy:<24} {flags}")

    n = len(records)
    def s(key):
        return sum(1 for r in records if r[key])
    def med(key):
        return round(statistics.median(r[key] for r in records), 2)

    summary = {
        "model": args.model, "db": str(args.db), "tasks": n,
        "budget_bytes": args.budget,
        "gold_in_page": s("gold_in_page"),
        "consumed": s("consumed"),
        "correct_no_context": s("correct_no_context"),
        "correct_with_dcy": s("correct_with_dcy"),
        "median_secs_no_context": med("secs_no_context"),
        "median_secs_dcy_retrieval": med("secs_dcy_retrieval"),
        "median_secs_llm_with_dcy": med("secs_llm_with_dcy"),
        "median_secs_total_with_dcy": med("secs_total_with_dcy"),
        "median_page_chars": med("page_chars"),
        "median_prompt_tokens_no_context": statistics.median(
            r["timing_no_context"]["prompt_eval_count"] or 0 for r in records),
        "median_prompt_tokens_with_dcy": statistics.median(
            r["timing_with_dcy"]["prompt_eval_count"] or 0 for r in records),
        "thinking_chars_stripped_total": sum(
            (r["timing_no_context"]["thinking_chars_stripped"] or 0)
            + (r["timing_with_dcy"]["thinking_chars_stripped"] or 0)
            for r in records),
    }

    print()
    print(f"gold symbol present in DCY page : {summary['gold_in_page']}/{n}   (retrieval ceiling)")
    print(f"answer copied from injected page: {summary['consumed']}/{n}   (consumption)")
    print(f"correct  no-context: {summary['correct_no_context']}/{n}   with-DCY: {summary['correct_with_dcy']}/{n}")
    print()
    print(f"median latency  no-context : {summary['median_secs_no_context']}s"
          f"  ({summary['median_prompt_tokens_no_context']} prompt tok)")
    print(f"median latency  DCY retrieval: {summary['median_secs_dcy_retrieval']}s"
          f"  (page {summary['median_page_chars']} chars)")
    print(f"median latency  LLM w/ DCY : {summary['median_secs_llm_with_dcy']}s"
          f"  ({summary['median_prompt_tokens_with_dcy']} prompt tok)")
    print(f"median latency  total w/DCY: {summary['median_secs_total_with_dcy']}s")

    args.out.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(f"\nraw artifact: {args.out}")


if __name__ == "__main__":
    main()
