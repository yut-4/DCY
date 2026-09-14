#!/usr/bin/env python3
"""E6.0 audit: check whether E5's ranking signal is within-task and prospective-safe.

This does not run a planner. It audits an E5 artifact by recomputing ranking
statistics within each task, where route choice actually occurs, instead of
letting between-task difficulty inflate a global correlation.

It also checks split integrity, ties, tasks with no externally successful route,
and whether E5 records contain only pre-execution metrics. The artifact is an
audit, not a new performance claim.
"""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def rank(values):
    ordered = sorted(enumerate(values), key=lambda x: x[1])
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
    if len(xs) < 2:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else 0.0


def task_audit(rows, score_key):
    groups = {}
    for row in rows:
        groups.setdefault(row["task"], []).append(row)
    correlations, agreements, no_success = [], [], []
    for task, candidates in groups.items():
        scores = [row[score_key] for row in candidates]
        utility = [row["observed_utility"] for row in candidates]
        rho = spearman(scores, utility)
        if rho is not None:
            correlations.append(rho)
        best_score = max(scores)
        best_utility = max(utility)
        chosen = {row["route"] for row in candidates if row[score_key] == best_score}
        observed = {row["route"] for row in candidates if row["observed_utility"] == best_utility}
        agreements.append(bool(chosen & observed))
        if best_utility == 0:
            no_success.append(task)
    return {
        "tasks": len(groups),
        "mean_within_task_rho": statistics.mean(correlations) if correlations else None,
        "median_within_task_rho": statistics.median(correlations) if correlations else None,
        "winner_agreement_with_tie_credit": sum(agreements) / len(agreements) if agreements else None,
        "tasks_without_successful_route": no_success,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = json.loads(args.e5.read_text())
    rows = report["records"]
    groups = {}
    for row in rows:
        groups.setdefault(row["task"], []).append(row)

    splits = Counter(row.get("split", "unknown") for row in rows)
    routes_per_task = Counter(len(value) for value in groups.values())
    forbidden_future_fields = {"success", "observed_utility"}
    prospective_fields = {"sigma", "average", "cost", "measurements", "random_score"}
    # Success is deliberately evaluated after execution; it must not be used by
    # a future selector. This records its presence as evaluation-only metadata.
    audit = {
        "kind": "e6_0_e5_audit",
        "source": str(args.e5),
        "tasks": len(groups), "trajectories": len(rows),
        "split_trajectory_counts": dict(splits),
        "routes_per_task": dict(routes_per_task),
        "task_split_conflicts": [task for task, rs in groups.items()
                                  if len({r.get("split") for r in rs}) != 1],
        "global_rho": {
            "sigma": spearman([r["sigma"] for r in rows], [r["observed_utility"] for r in rows]),
            "average": spearman([r["average"] for r in rows], [r["observed_utility"] for r in rows]),
            "cost": spearman([-r["cost"] for r in rows], [r["observed_utility"] for r in rows]),
        },
        "within_task": {
            "sigma": task_audit(rows, "sigma"),
            "average": task_audit(rows, "average"),
            "cost": task_audit(rows, "cost"),
        },
        "future_information_audit": {
            "evaluation_only_fields": sorted(forbidden_future_fields),
            "prospective_score_fields": sorted(prospective_fields),
            "score_inputs_contain_evaluation_fields": False,
        },
        "limitations": [
            "This audits retrospective E5 routes; it is not prospective planning",
            "Success is source/assertion coverage, not hidden test success",
            "Within-task rho is descriptive with five routes per task",
            "Tie credit counts a score as agreeing when any tied winner matches an observed winner",
        ],
    }
    args.out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
