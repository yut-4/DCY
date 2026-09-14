#!/usr/bin/env python3
"""Reproducible retrieval-only pilot; does not measure LLM task success."""

import argparse
import json
import re
import sqlite3
import statistics
import subprocess
import time
from pathlib import Path


def run(command):
    start = time.perf_counter_ns()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    if process.returncode:
        raise RuntimeError(f"{command}: {process.stderr.strip()}")
    return process.stdout, elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcy", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--modes", nargs="+", default=["query-fts", "query"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or any(b < 1 for b in args.budgets):
        parser.error("repeats and budgets must be positive")

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if not tasks:
        parser.error("task manifest is empty")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    index_output, index_ms = run([str(args.dcy), "index", str(args.repo), str(args.db)])
    generation = re.search(r"generation ([0-9a-f]{64})", index_output)
    if not generation:
        raise RuntimeError("index did not report a generation")
    connection = sqlite3.connect(args.db)
    names = {
        entity_id: f"{path}:{name}"
        for entity_id, path, name in connection.execute(
            "SELECT e.id,f.path,e.name FROM entities e JOIN files f ON f.id=e.file_id"
        )
    }
    connection.close()

    records = []
    for task in tasks:
        goal = task["goal"]
        gold = set(task["gold"])
        for budget in args.budgets:
            for mode in args.modes:
                if mode not in {"query-fts", "query", "query-mlpack"}:
                    parser.error(f"unsupported mode: {mode}")
                for repeat in range(args.repeats):
                    rendered, latency_ms = run(
                        [str(args.dcy), mode, str(args.db), goal, str(budget)]
                    )
                    emitted = {names[int(i)] for i in re.findall(r"^E(\d+)\b", rendered, re.M)}
                    hits = gold & emitted
                    records.append({
                        "task": task["id"], "mode": mode, "budget_bytes": budget,
                        "repeat": repeat, "latency_ms": latency_ms,
                        "context_bytes": len(rendered.encode()), "gold_count": len(gold),
                        "retrieved_count": len(emitted), "hit_count": len(hits),
                        "recall": len(hits) / len(gold) if gold else None,
                        "all_gold_present": gold <= emitted,
                    })

    report = {
        "kind": "retrieval_only_pilot", "generation": generation.group(1),
        "index_ms": index_ms, "index_stdout": index_output.strip(),
        "tasks": len(tasks), "records": records,
        "limitations": [
            "byte budgets are not model token budgets",
            "CLI process startup is included in latency",
            "gold symbol recall is not patch success or evidence precision",
        ],
    }
    if args.summary_only:
        categories = {task["id"]: task.get("category", "uncategorized") for task in tasks}
        groups = {}
        for record in records:
            key = (record["mode"], record["budget_bytes"])
            groups.setdefault(key, []).append(record)
        summary = []
        for (mode, budget), rows in sorted(groups.items()):
            times = sorted(row["latency_ms"] for row in rows)
            by_category = {}
            for category in sorted(set(categories.values())):
                subset = [row for row in rows if categories[row["task"]] == category]
                by_category[category] = {
                    "mean_recall": sum(row["recall"] for row in subset) / len(subset),
                    "all_gold_rate": sum(row["all_gold_present"] for row in subset) / len(subset),
                }
            summary.append({
                "mode": mode, "budget_bytes": budget, "n": len(rows),
                "mean_recall": sum(row["recall"] for row in rows) / len(rows),
                "all_gold_rate": sum(row["all_gold_present"] for row in rows) / len(rows),
                "latency_ms_p50": statistics.median(times),
                "latency_ms_p95": times[max(0, int(0.95 * len(times)) - 1)],
                "context_bytes_mean": statistics.mean(row["context_bytes"] for row in rows),
                "context_bytes_max": max(row["context_bytes"] for row in rows),
                "categories": by_category,
                "missed_tasks": sorted({row["task"] for row in rows if not row["all_gold_present"]}),
            })
        report["records"] = len(records)
        report["summary"] = summary
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
