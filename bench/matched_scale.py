#!/usr/bin/env python3
"""Matched-scale experiment: same task IDs, same budget, across corpus scales.

This isolates the true ∂PT/∂VT by holding tasks constant. Only tasks whose
gold symbols exist in the smallest index are scored at every scale.
"""

import argparse
import json
import re
import sqlite3
import statistics
import subprocess
import tiktoken
from pathlib import Path


def view(dcy, db, mode, goal, budget):
    process = subprocess.run([str(dcy), mode, str(db), goal, str(budget)],
                             capture_output=True, text=True, check=False)
    if process.returncode:
        raise RuntimeError(f"{goal!r}: {process.stderr.strip()}")
    return process.stdout


def emitted(rendered, names):
    return {names[int(i)] for i in re.findall(r"\[ref=E(\d+)\]", rendered) if int(i) in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcy", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", required=True,
                        help="VT_TOKENS=path/to/index.sqlite pairs")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--mode", default="query")
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--split", choices=["design", "held-out", "all"], default="all")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    encoder = tiktoken.get_encoding(args.encoding)
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if args.split != "all":
        tasks = [t for t in tasks if t.get("split", "design") == args.split]

    # Find the intersection of tasks that are scorable at the SMALLEST scale.
    # That is the set we hold constant across all scales.
    scales = []
    for pair in args.scales:
        vt_text, db_path = pair.split("=", 1)
        vt, db = int(vt_text), Path(db_path)
        connection = sqlite3.connect(db)
        names = {i: f"{p}:{n}" for i, p, n in connection.execute(
            "SELECT e.id,f.path,e.name FROM entities e JOIN files f ON f.id=e.file_id")}
        connection.close()
        scales.append((vt, db, names))

    smallest = scales[0][2]
    common_tasks = [t for t in tasks if set(t["gold"]) <= set(smallest.values())]
    print(f"Total tasks in split: {len(tasks)}")
    print(f"Tasks scorable at smallest scale ({scales[0][0]} VT): {len(common_tasks)}")
    if not common_tasks:
        raise SystemExit("No tasks common to all scales")

    records = []
    for vt, db, names in scales:
        for budget in args.budgets:
            recalls, prompt_tokens, latencies = [], [], []
            for task in common_tasks:
                rendered = view(args.dcy, db, args.mode, task["goal"], budget)
                found = emitted(rendered, names)
                recalls.append(len(set(task["gold"]) & found) / len(task["gold"]))
                prompt_tokens.append(len(encoder.encode(rendered, disallowed_special=())))
            records.append({
                "vt_tokens": vt, "budget_bytes": budget, "tasks": len(common_tasks),
                "injected_recall": round(sum(recalls) / len(recalls), 4),
                "full_gold_rate": round(sum(1 for r in recalls if r == 1) / len(recalls), 4),
                "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 1),
                "prompt_tokens_p95": round(sorted(prompt_tokens)[max(0, int(0.95 * len(prompt_tokens)) - 1)], 1),
                "prompt_tokens_max": max(prompt_tokens),
                "vcr": round(vt / statistics.mean(prompt_tokens), 1),
            })

    slopes = {}
    for budget in args.budgets:
        points = [(r["vt_tokens"], r["prompt_tokens_mean"]) for r in records
                  if r["budget_bytes"] == budget]
        if len(points) >= 2:
            (x0, y0), (x1, y1) = points[0], points[-1]
            slopes[str(budget)] = {
                "d_prompt_tokens_per_1k_vt": round((y1 - y0) / ((x1 - x0) / 1000), 5),
                "prompt_tokens_first": y0, "prompt_tokens_last": y1,
                "vt_first": x0, "vt_last": x1,
            }

    report = {
        "kind": "matched_scale_pilot", "encoding": args.encoding, "mode": args.mode,
        "split": args.split, "common_tasks": len(common_tasks),
        "scales": [s[0] for s in scales],
        "records": records, "prompt_growth": slopes,
        "limitations": [
            "Only tasks present in the smallest index are measured; this is a subset",
            "VT is corpus size under a fixed tokenizer, not model comprehension",
            "VCR is an addressability ratio and says nothing about comprehension",
            "injected recall is not task success",
        ],
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(f"{'VT':>10s} {'budget':>7s} {'n':>4s} {'recall':>7s} {'prompt_tok':>11s} {'p95':>7s} {'VCR':>8s}")
    for r in records:
        print(f"{r['vt_tokens']:10,} {r['budget_bytes']:7d} {r['tasks']:4d} "
              f"{r['injected_recall']:7.3f} {r['prompt_tokens_mean']:11.1f} "
              f"{r['prompt_tokens_p95']:7.0f} {r['vcr']:8.1f}")
    print("\nprompt token growth per 1,000 additional VT tokens:")
    for budget, slope in slopes.items():
        print(f"  budget {budget:>5s} B: {slope['d_prompt_tokens_per_1k_vt']:+.5f} tokens/1k VT "
              f"({slope['prompt_tokens_first']:.1f} -> {slope['prompt_tokens_last']:.1f} "
              f"over {slope['vt_first']:,} -> {slope['vt_last']:,} VT)")


if __name__ == "__main__":
    main()