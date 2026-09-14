#!/usr/bin/env python3
"""Offline E2 validation: does J_DCY rank goal trajectories usefully?

This is deliberately not a planner. A manifest supplies candidate trajectories
and externally verified success labels. The runner executes each step through
the real DCY CLI, derives observable C_goal and DB from gold evidence, computes
E_DCY per step, E_DCY^Sigma and J_DCY, then compares J's ranking with observed
success per cost using Spearman's rho.

Observable proxies used here:
  C_goal = relevant selected entities / all selected entities (goal precision)
  DB     = relevant selected entities / required entities (database recall)
  VT     = fixed corpus token count supplied by --virtual-tokens
  G      = fixed effective partition count supplied by --partitions

These proxies are benchmark measurements, not universal definitions. Success
labels must come from tests/review in a real task benchmark; this fixture keeps
them explicit in JSON so the metric cannot silently invent success.
"""

import argparse
import json
import math
import re
import sqlite3
import statistics
import subprocess
import time
from pathlib import Path


def query(dcy, db, goal, budget):
    start = time.perf_counter_ns()
    p = subprocess.run([str(dcy), "query", str(db), goal, str(budget)],
                       capture_output=True, text=True, check=False)
    latency = (time.perf_counter_ns() - start) / 1_000_000
    if p.returncode:
        raise RuntimeError(f"{goal!r}: {p.stderr.strip()}")
    return p.stdout, latency


def source(dcy, db, entity_id, generation, max_bytes=4096):
    p = subprocess.run([str(dcy), "source", str(db), str(entity_id), generation,
                        str(max_bytes)], capture_output=True, text=True, check=False)
    if p.returncode:
        raise RuntimeError(f"source E{entity_id}: {p.stderr.strip()}")
    return p.stdout


def refs(text, names):
    return {names[int(i)] for i in re.findall(r"\[ref=E(\d+)\]", text) if int(i) in names}


def spearman(xs, ys):
    def ranks(values):
        indexed = sorted(enumerate(values), key=lambda item: item[1])
        result = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i + 1
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                result[indexed[k][0]] = rank
            i = j
        return result

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcy", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--budget-bytes", type=int, default=512)
    ap.add_argument("--virtual-tokens", type=float, required=True)
    ap.add_argument("--partitions", type=float, required=True)
    ap.add_argument("--external", action="store_true",
                    help="verify assertions through dcy source instead of route success labels")
    ap.add_argument("--lambda-cost", type=float, default=0.001)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.virtual_tokens <= 0 or args.partitions <= 0 or args.lambda_cost < 0:
        ap.error("virtual-tokens and partitions must be positive; lambda must be non-negative")

    connection = sqlite3.connect(args.db)
    names = {i: f"{p}:{n}" for i, p, n in connection.execute(
        "SELECT e.id,f.path,e.name FROM entities e JOIN files f ON f.id=e.file_id")}
    ids = {value: key for key, value in names.items()}
    generation = connection.execute("SELECT value FROM meta WHERE key='generation'").fetchone()[0]
    connection.close()
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    records = []
    for task in tasks:
        required = set(task["required"])
        for route in task["routes"]:
            measurements, total_cost = [], 0.0
            previous = set()
            union = set()
            for step in route["steps"]:
                rendered, latency = query(args.dcy, args.db, step["goal"], args.budget_bytes)
                selected = refs(rendered, names)
                step_required = set(step.get("required", required))
                relevant = selected & step_required
                c_goal = len(relevant) / len(selected) if selected else 0.0
                db_utility = len(relevant) / len(step_required) if step_required else 0.0
                # Continuity is reported diagnostically; local C_goal remains
                # observable precision so it does not double-count prior state.
                continuity = len(relevant & previous) / len(relevant) if relevant else 0.0
                previous |= relevant
                union |= relevant
                working_set = args.virtual_tokens / args.partitions
                local = args.virtual_tokens * c_goal * db_utility / (working_set * working_set)
                prompt_tokens = len(re.findall(r"\S+", rendered))
                step_cost = prompt_tokens + latency / 10.0 + 1.0
                total_cost += step_cost
                measurements.append({
                    "goal": step["goal"], "selected": sorted(selected),
                    "required": sorted(step_required), "relevant": sorted(relevant),
                    "C_goal": round(c_goal, 6), "DB": round(db_utility, 6),
                    "continuity": round(continuity, 6), "E_DCY": local,
                    "prompt_tokens_proxy": prompt_tokens, "latency_ms": round(latency, 3),
                    "cost": step_cost,
                })
            sigma = sum(m["E_DCY"] for m in measurements)
            objective = sigma - args.lambda_cost * total_cost
            if args.external:
                assertions = task.get("assertions", {})
                if set(assertions) != required:
                    raise SystemExit(f"{task['id']}: assertions must cover required symbols")
                success = union >= required
                for symbol, pattern in assertions.items():
                    entity_id = ids.get(symbol)
                    if entity_id is None or not re.search(pattern, source(args.dcy, args.db, entity_id, generation)):
                        success = False
                        break
            else:
                success = union >= required
            observed_utility = (1.0 if success else 0.0) / total_cost if total_cost else 0.0
            records.append({
                "task": task["id"], "route": route["id"], "success": bool(success),
                "steps": len(measurements), "E_DCY_sigma": sigma,
                "mean_E_DCY": sigma / len(measurements), "cost": total_cost,
                "J_DCY": objective, "observed_utility": observed_utility,
                "measurements": measurements,
            })

    by_task = {}
    for record in records:
        by_task.setdefault(record["task"], []).append(record)
    comparisons = []
    for task_id, rows in by_task.items():
        best_j = max(rows, key=lambda r: r["J_DCY"])
        best_observed = max(rows, key=lambda r: r["observed_utility"])
        comparisons.append({
            "task": task_id, "best_by_J": best_j["route"],
            "best_observed": best_observed["route"],
            "same_best": best_j["route"] == best_observed["route"],
        })
    report = {
        "kind": "goal_trajectory_validation",
        "budget_bytes": args.budget_bytes, "virtual_tokens": args.virtual_tokens,
        "partitions": args.partitions, "lambda_cost": args.lambda_cost,
        "tasks": len(tasks), "trajectories": len(records), "records": records,
        "spearman_J_vs_observed_utility": spearman(
            [r["J_DCY"] for r in records], [r["observed_utility"] for r in records]),
        "best_route_matches": sum(c["same_best"] for c in comparisons),
        "best_route_comparisons": comparisons,
        "limitations": [
            "offline candidate trajectories; no planner is implemented",
            "C_goal and DB are observable gold-based proxies for this fixture",
            "success labels are manifest inputs and must come from tests/review in real benchmarks",
            "cost components use prompt word count and latency normalization, not billing units",
        ],
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(f"tasks={len(tasks)} trajectories={len(records)}")
    print(f"Spearman rho(J, observed utility)={report['spearman_J_vs_observed_utility']:.4f}")
    print(f"best route agreement={report['best_route_matches']}/{len(comparisons)}")
    for row in records:
        print(f"  {row['task']}/{row['route']}: success={row['success']} "
              f"EΣ={row['E_DCY_sigma']:.6f} J={row['J_DCY']:.6f} "
              f"observed={row['observed_utility']:.6f}")


if __name__ == "__main__":
    main()
