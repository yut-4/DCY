#!/usr/bin/env python3
"""E3: calibrate and test objective functions on held-out trajectories.

Routes are generated deterministically from the 100-task manifest. Each route is
executed through the real DCY CLI. A route succeeds when the union of relevant
gold entities retrieved by its steps covers the task's required entities. This
is an evidence-success proxy, not hidden-test success.

Lambda is selected only on the design split by route-choice agreement, then
frozen on held-out. Scores compared:
  sigma, average sigma, cheapest (-cost), J = sigma - lambda*cost,
  normalized J using per-split min-max terms, and a deterministic random control.
"""

import argparse
import json
import math
import random
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
    if p.returncode:
        raise RuntimeError(f"{goal!r}: {p.stderr.strip()}")
    return p.stdout, (time.perf_counter_ns() - start) / 1_000_000


def source(dcy, db, entity_id, generation, cache):
    if entity_id not in cache:
        p = subprocess.run([str(dcy), "source", str(db), str(entity_id), generation, "1048576"],
                           capture_output=True, text=True, check=False)
        if p.returncode:
            raise RuntimeError(f"source E{entity_id}: {p.stderr.strip()}")
        cache[entity_id] = p.stdout
    return cache[entity_id]


def refs(text, names):
    return {names[int(i)] for i in re.findall(r"\[ref=E(\d+)\]", text) if int(i) in names}


def rank(values):
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        value = (i + 1 + j) / 2.0
        for k in range(i, j):
            result[ordered[k][0]] = value
        i = j
    return result


def spearman(xs, ys):
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else 0.0


def route_score(row, metric, lam=0.0, ranges=None):
    if metric == "sigma":
        return row["sigma"]
    if metric == "average":
        return row["average"]
    if metric == "-cost":
        return -row["cost"]
    if metric == "J":
        return row["sigma"] - lam * row["cost"]
    if metric == "J_norm":
        if ranges is None:
            raise ValueError("J_norm requires normalization ranges")
        sigma_min, sigma_max, cost_min, cost_max = ranges
        sigma = (row["sigma"] - sigma_min) / (sigma_max - sigma_min or 1.0)
        cost = (row["cost"] - cost_min) / (cost_max - cost_min or 1.0)
        return sigma - lam * cost
    if metric == "random":
        return row["random_score"]
    raise ValueError(metric)


def agreement(rows, metric, lam=0.0, ranges=None):
    """Tie-safe route-choice agreement.

    Python's max() breaks ties by returning the first maximal element in
    iteration order; routes are always generated in the same fixed order
    (focused, redundant, distracted, cheap-wrong, broad), so a naive
    max()-based comparison systematically favors "focused" whenever scores
    tie on either side. This computes the full set of tied winners on the
    score side and on the observed-utility side and reports three distinct
    numbers instead of collapsing them into one silently tie-biased figure:

    strict_agreement: only counts a task when the metric has a single
      undisputed winner and it matches an observed winner.
    tie_aware_agreement: counts a task whenever any metric-tied winner
      overlaps any observed-tied winner (credit, not proof of preference).
    expected_agreement_random_tiebreak: the probability of matching if ties
      on the metric side were broken uniformly at random.
    """
    groups = {}
    for row in rows:
        groups.setdefault(row["task"], []).append(row)
    strict_hits, tie_aware_hits, expected_hits = 0, 0, 0.0
    for candidates in groups.values():
        scores = {row["route"]: route_score(row, metric, lam, ranges) for row in candidates}
        best_score = max(scores.values())
        winners = [route for route, s in scores.items() if abs(s - best_score) <= 1e-12]

        utilities = {row["route"]: row["observed_utility"] for row in candidates}
        best_utility = max(utilities.values())
        observed_winners = {route for route, u in utilities.items() if abs(u - best_utility) <= 1e-12}

        overlap = set(winners) & observed_winners
        tie_aware_hits += bool(overlap)
        strict_hits += 1 if (len(winners) == 1 and winners[0] in observed_winners) else 0
        expected_hits += (len(overlap) / len(winners)) if winners else 0.0
    n = len(groups)
    return {
        "strict_agreement": strict_hits / n,
        "tie_aware_agreement": tie_aware_hits / n,
        "expected_agreement_random_tiebreak": expected_hits / n,
    }


def evaluate(rows, metric, lam=0.0, ranges=None):
    xs = [route_score(row, metric, lam, ranges) for row in rows]
    ys = [row["observed_utility"] for row in rows]
    return {**agreement(rows, metric, lam, ranges), "rho": spearman(xs, ys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcy", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--budget-bytes", type=int, default=512)
    ap.add_argument("--virtual-tokens", type=float, required=True)
    ap.add_argument("--partitions", type=float, required=True)
    ap.add_argument("--external", action="store_true",
                    help="verify required symbols through dcy source before scoring success")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    connection = sqlite3.connect(args.db)
    names = {i: f"{p}:{n}" for i, p, n in connection.execute(
        "SELECT e.id,f.path,e.name FROM entities e JOIN files f ON f.id=e.file_id")}
    ids = {value: key for key, value in names.items()}
    generation = connection.execute("SELECT value FROM meta WHERE key='generation'").fetchone()[0]
    connection.close()
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    rng = random.Random(args.seed)
    records = []
    source_cache = {}

    for task in tasks:
        required = set(task["gold"])
        base = task["goal"]
        routes = [
            ("focused", [base]),
            ("redundant", [base, base]),
            ("distracted", ["database connection", "logging", base]),
            ("cheap-wrong", ["database connection"]),
            ("broad", [base, "source implementation", "function definition"]),
        ]
        for route_id, steps in routes:
            measurements, union = [], set()
            total_cost = 0.0
            for step_goal in steps:
                rendered, latency = query(args.dcy, args.db, step_goal, args.budget_bytes)
                selected = refs(rendered, names)
                relevant = selected & required
                union |= relevant
                c_goal = len(relevant) / len(selected) if selected else 0.0
                db_utility = len(relevant) / len(required) if required else 0.0
                working_set = args.virtual_tokens / args.partitions
                local = args.virtual_tokens * c_goal * db_utility / (working_set * working_set)
                prompt_words = len(re.findall(r"\S+", rendered))
                cost = prompt_words + latency / 10.0 + 1.0
                total_cost += cost
                measurements.append({"goal": step_goal, "selected": len(selected),
                    "relevant": len(relevant), "C_goal": c_goal, "DB": db_utility,
                    "E_DCY": local, "prompt_words": prompt_words,
                    "latency_ms": latency, "cost": cost})
            sigma = sum(m["E_DCY"] for m in measurements)
            success = union >= required
            if args.external and success:
                assertions = task.get("assertions", {})
                if set(assertions) != required:
                    raise SystemExit(f"{task['id']}: assertions must cover required symbols")
                success = all(
                    entity_id := ids.get(symbol) is not None and
                    re.search(pattern, source(args.dcy, args.db, ids[symbol], generation, source_cache))
                    for symbol, pattern in assertions.items()
                )
            records.append({
                "task": task["id"], "split": task.get("split", "design"),
                "route": route_id, "success": success, "steps": len(steps),
                "sigma": sigma, "average": sigma / len(steps), "cost": total_cost,
                "observed_utility": (1.0 / total_cost) if success else 0.0,
                "measurements": measurements,
            })

    # Stable random control, generated once and reused for every comparison.
    for row in records:
        row["random_score"] = rng.random()

    design = [r for r in records if r["split"] == "design"]
    heldout = [r for r in records if r["split"] == "held-out"]
    if not design or not heldout:
        raise SystemExit("manifest needs both design and held-out routes")
    sigma_range = (min(r["sigma"] for r in design), max(r["sigma"] for r in design))
    cost_range = (min(r["cost"] for r in design), max(r["cost"] for r in design))
    ranges_design = sigma_range + cost_range

    # Calibrate lambda only on design, using strict (tie-free) agreement so
    # tie credit cannot silently inflate the chosen lambda. Grid is declared
    # before seeing results.
    grid = [0.0, 0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    candidates = []
    for lam in grid:
        result = evaluate(design, "J", lam)
        candidates.append((result["strict_agreement"], result["rho"], -lam, lam))
    _, _, _, best_lambda = max(candidates)

    # Normalized lambda is calibrated separately, still design-only.
    norm_candidates = []
    for lam in grid:
        result = evaluate(design, "J_norm", lam, ranges_design)
        norm_candidates.append((result["strict_agreement"], result["rho"], -lam, lam))
    _, _, _, best_norm_lambda = max(norm_candidates)

    metrics = {
        "sigma": (0.0, None), "average": (0.0, None), "-cost": (0.0, None),
        "J": (best_lambda, None), "J_norm": (best_norm_lambda, ranges_design),
        "random": (0.0, None),
    }
    evaluations = {}
    for metric, (lam, ranges) in metrics.items():
        evaluations[metric] = {
            "lambda_design": lam if metric in ("J", "J_norm") else None,
            "design": evaluate(design, metric, lam, ranges),
            "held_out": evaluate(heldout, metric, lam, ranges),
        }

    report = {
        "kind": "external_objective_comparison_e5" if args.external else "objective_calibration_e3", "tasks": len(tasks),
        "trajectories": len(records), "budget_bytes": args.budget_bytes,
        "virtual_tokens": args.virtual_tokens, "partitions": args.partitions,
        "lambda_grid": grid, "best_lambda": best_lambda,
        "best_norm_lambda": best_norm_lambda,
        "design_tasks": len({r["task"] for r in design}),
        "held_out_tasks": len({r["task"] for r in heldout}),
        "evaluations": evaluations, "records": records,
        "limitations": [
            "success is externally checked source/assertion coverage, not hidden test success" if args.external else "success is gold-evidence coverage, not hidden test success",
            "routes are deterministic generated controls, not planner proposals",
            "lambda and normalization are selected on design only, using strict tie-free agreement",
            "fixture costs use prompt words and normalized latency, not billing units",
            "C_goal and DB are computed from the task's gold set, and success is also computed from the same gold set: "
            "sigma/E_DCY and success share a data source, so this rho/agreement is not evidence the metric predicts "
            "success from independent signal, only that both are consistent functions of the same gold labels. "
            "See E6.1 for observable-only features that do not read task['gold'].",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"tasks={len(tasks)} trajectories={len(records)} design={len(design)} held-out={len(heldout)}")
    print(f"lambda={best_lambda} normalized_lambda={best_norm_lambda}")
    print(f"{'metric':10s} {'design_strict':>13s} {'design_tie':>11s} {'design_rho':>11s} "
          f"{'held_strict':>12s} {'held_tie':>9s} {'held_rho':>9s}")
    for metric, values in evaluations.items():
        d, h = values["design"], values["held_out"]
        print(f"{metric:10s} {d['strict_agreement']:13.3f} {d['tie_aware_agreement']:11.3f} {d['rho']:11.4f} "
              f"{h['strict_agreement']:12.3f} {h['tie_aware_agreement']:9.3f} {h['rho']:9.4f}")


if __name__ == "__main__":
    main()
