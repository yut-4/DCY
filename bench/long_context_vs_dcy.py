#!/usr/bin/env python3
"""Long-context dump vs DCY distilled page — same model, same question.

This is the comparison that actually tests DCY's claim. The baseline is NOT
"no context" (that only measures pretraining recall); it is "stuff the raw
source into the prompt", which is what a normal RAG-less pipeline does.

Arms per task, all on the SAME model:
  dump-NkB : concatenated raw source files, guaranteed to CONTAIN the gold
             symbol, truncated to N KB and injected whole.
  dcy-query-NNN : ONE `dcy query` + ONE LLM call, distilled page under a byte
             budget. This is flat retrieval, NOT the v2 paging runtime — it
             makes a single inference. Naming it plain "dcy" previously caused
             its latency to be read as the paging runtime's; the paging runtime
             costs ~5 inferences and is measured by bench/arch_comparison.py
             under the separate arm name `dcy-v2`.

Reported per arm: wall latency split into retrieval vs LLM, prompt tokens as
counted by the SERVER (not our guess), whether gold was present in what the
model was shown (ceiling), whether the answer token came from what it was
shown (consumption), and whether it matches gold (correctness).

num_predict is deliberately generous: a thinking model truncated mid-reasoning
emits its scratchpad instead of an answer, which scores as a content failure
when it is really a token-budget failure. Set --num-predict low only to
reproduce that artifact on purpose.

Run:
  python3 bench/long_context_vs_dcy.py --model qwen3:0.6b --limit 4 \
      --dump-sizes 4096 16384 65536
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import subprocess
import time
from pathlib import Path

import httpx

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


def ollama_generate(base: str, model: str, prompt: str, timeout: float,
                    num_predict: int, num_ctx: int) -> tuple[str, float, dict]:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": num_predict,
                        "num_ctx": num_ctx},
        },
        timeout=timeout,
    )
    wall = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    text = d.get("response", "") or d.get("thinking", "") or ""
    raw_len = len(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    ptok = d.get("prompt_eval_count")
    timing = {
        "prompt_eval_count": ptok,
        "eval_count": d.get("eval_count"),
        "prompt_eval_ms": round((d.get("prompt_eval_duration") or 0) / 1e6, 1),
        "eval_ms": round((d.get("eval_duration") or 0) / 1e6, 1),
        "hit_predict_cap": d.get("eval_count") == num_predict,
        # Ollama silently drops tokens past num_ctx. If the server-counted
        # prompt lands at the window edge, the model did NOT see the whole
        # context we built, and every gold/consumption flag computed on our
        # local string is unreliable for that row.
        "ctx_truncated": bool(ptok and ptok >= num_ctx - 8),
        "think_chars_stripped": raw_len - len(text),
    }
    return text, wall, timing


def build_dump(conn: sqlite3.Connection, gold: str, size: int) -> str:
    """Raw source containing gold, padded with other files, cut to `size` bytes.

    The gold-bearing file goes FIRST so truncation never silently removes the
    answer — otherwise a long-context loss would be indistinguishable from the
    answer simply not being in the prompt.
    """
    row = conn.execute(
        "SELECT f.path, f.content FROM files f JOIN entities e ON e.file_id=f.id "
        "WHERE e.name=? LIMIT 1", (gold,)).fetchone()
    parts, seen = [], set()
    if row:
        parts.append(f"===== {row[0]} =====\n{row[1]}")
        seen.add(row[0])
    for path, content in conn.execute(
            "SELECT path, content FROM files ORDER BY length(content) DESC"):
        if path in seen:
            continue
        parts.append(f"===== {path} =====\n{content}")
        if sum(len(p) for p in parts) > size:
            break
    return "\n\n".join(parts)[:size]


def dcy_query(dcy: Path, db: Path, goal: str, budget: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    p = subprocess.run([str(dcy), "query", str(db), goal, str(budget)],
                       capture_output=True, text=True, timeout=120)
    return p.stdout, time.perf_counter() - t0


def score(ctx: str, answer: str, gold: str) -> dict:
    tok = WORD_RE.search(answer)
    return {
        "gold_in_context": bool(re.search(rf"\b{re.escape(gold)}\b", ctx)),
        "consumed": bool(tok and re.search(rf"\b{re.escape(tok.group(0))}\b", ctx)),
        "correct": bool(re.search(rf"\b{re.escape(gold)}\b", answer)),
    }


PROMPT = ("Repository context:\n{ctx}\n\n"
          "Question: {goal}\n"
          "Answer with ONLY the C function name from the context above, "
          "nothing else.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcy", type=Path, default=Path("build/dcy"))
    ap.add_argument("--db", type=Path, default=Path("build/tree-sitter-lib.sqlite"))
    ap.add_argument("--tasks", type=Path, default=Path("bench/tree-sitter-lib-tasks.jsonl"))
    ap.add_argument("--model", default="qwen3:0.6b")
    ap.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    ap.add_argument("--budget", type=int, default=512, help="DCY byte budget")
    ap.add_argument("--dump-sizes", type=int, nargs="+", default=[4096, 16384, 65536])
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--num-ctx", type=int, default=16384,
                    help="Ollama context window. MUST exceed the largest dump "
                         "arm's token count or Ollama silently truncates the "
                         "prompt and the long-context arm measures nothing.")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", type=Path, default=Path("build/long-context-vs-dcy.json"))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    tasks = tasks[: args.limit]

    dcy_arm = f"dcy-query-{args.budget}"
    arms = [f"dump-{s // 1024}kB" for s in args.dump_sizes] + [dcy_arm]
    print(f"model={args.model}  tasks={len(tasks)}  num_predict={args.num_predict}")
    print(f"arms: {', '.join(arms)}   (dcy budget {args.budget}B)\n")

    records = []
    for t in tasks:
        goal, gold = t["goal"], t["gold"][0].rsplit(":", 1)[1]
        print(f"[{t['id']}] gold={gold}")
        for size in args.dump_sizes:
            ctx = build_dump(conn, gold, size)
            ans, wall, tim = ollama_generate(
                args.ollama_base, args.model,
                PROMPT.format(ctx=ctx, goal=goal), args.timeout,
                args.num_predict, args.num_ctx)
            sc = score(ctx, ans, gold)
            rec = {"id": t["id"], "gold": gold, "arm": f"dump-{size // 1024}kB",
                   "context_chars": len(ctx), "retrieval_secs": 0.0,
                   "llm_secs": round(wall, 2), "total_secs": round(wall, 2),
                   "answer": ans, **sc, **tim}
            records.append(rec)
            print(f"   {rec['arm']:<12} {wall:7.1f}s  "
                  f"{tim['prompt_eval_count'] or 0:>6} ptok  "
                  f"gold_in_ctx={sc['gold_in_context']!s:<5} "
                  f"correct={sc['correct']!s:<5} cut={tim['ctx_truncated']!s:<5} "
                  f"| {' '.join(ans.split())[:40]}")

        page, r_secs = dcy_query(args.dcy, args.db, goal, args.budget)
        ans, wall, tim = ollama_generate(
            args.ollama_base, args.model,
            PROMPT.format(ctx=page, goal=goal), args.timeout,
            args.num_predict, args.num_ctx)
        sc = score(page, ans, gold)
        rec = {"id": t["id"], "gold": gold, "arm": dcy_arm,
               "context_chars": len(page), "retrieval_secs": round(r_secs, 3),
               "llm_secs": round(wall, 2), "total_secs": round(r_secs + wall, 2),
               "answer": ans, **sc, **tim}
        records.append(rec)
        print(f"   {dcy_arm:<14} {r_secs + wall:7.1f}s  "
              f"{tim['prompt_eval_count'] or 0:>6} ptok  "
              f"gold_in_ctx={sc['gold_in_context']!s:<5} "
              f"correct={sc['correct']!s:<5} cut={tim['ctx_truncated']!s:<5} "
              f"| {' '.join(ans.split())[:40]}")
        print()

    n = len(tasks)
    print("=" * 78)
    print(f"{'ARM':<12} {'med_secs':>9} {'med_ptok':>9} {'gold_in_ctx':>12} "
          f"{'consumed':>9} {'correct':>8} {'ctx_cut':>8}")
    print("-" * 78)
    summary = {}
    for arm in arms:
        rs = [r for r in records if r["arm"] == arm]
        if not rs:
            continue
        row = {
            "median_total_secs": round(statistics.median(r["total_secs"] for r in rs), 2),
            "median_prompt_tokens": statistics.median(r["prompt_eval_count"] or 0 for r in rs),
            "median_context_chars": statistics.median(r["context_chars"] for r in rs),
            "gold_in_context": sum(r["gold_in_context"] for r in rs),
            "consumed": sum(r["consumed"] for r in rs),
            "correct": sum(r["correct"] for r in rs),
            "hit_predict_cap": sum(r["hit_predict_cap"] for r in rs),
            "ctx_truncated": sum(r["ctx_truncated"] for r in rs),
            "n": len(rs),
        }
        summary[arm] = row
        print(f"{arm:<12} {row['median_total_secs']:>9} {row['median_prompt_tokens']:>9} "
              f"{str(row['gold_in_context']) + '/' + str(n):>12} "
              f"{str(row['consumed']) + '/' + str(n):>9} "
              f"{str(row['correct']) + '/' + str(n):>8} "
              f"{str(row['ctx_truncated']) + '/' + str(n):>8}")

    cut = [a for a, r in summary.items() if r["ctx_truncated"]]
    if cut:
        print(f"\nWARNING: arms {cut} hit the num_ctx={args.num_ctx} window; "
              "Ollama dropped prompt tokens, so their gold/consumption flags "
              "describe text the model never saw. Raise --num-ctx to compare.")

    args.out.write_text(json.dumps(
        {"model": args.model, "num_predict": args.num_predict,
         "num_ctx": args.num_ctx,
         "dcy_budget_bytes": args.budget,
         "summary": summary, "records": records}, indent=2))
    print(f"\nraw artifact: {args.out}")


if __name__ == "__main__":
    main()
