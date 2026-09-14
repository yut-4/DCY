#!/usr/bin/env python3
"""Freeze a reproducible DCY experimental checkpoint from existing pilot outputs.

A checkpoint records the four-stage recall decomposition, the propagation
efficiencies between stages, the per-task failure attribution, and SHA-256
digests of every source and harness file that produced those numbers. It reads
pilot JSON already on disk; it does not run models, so it cannot silently
re-measure and disagree with the report it freezes.

Propagation efficiencies:
    eta_pack     = R_injected  / R_candidate    retrieval -> packing
    eta_use      = R_answer    / R_injected     packing -> ORMT/model
    eta_pipeline = R_answer    / R_candidate    = eta_pack * eta_use
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

TRACKED = [
    "CMakeLists.txt",
    "src/main.cpp",
    "src/distiller.hpp",
    "src/ormt.hpp",
    "src/mlpack_retrieval.hpp",
    "mcp/dcy_mcp.py",
    "bench/oracle.py",
    "bench/retrieval.py",
    "bench/llm_identification.py",
    "bench/tree-sitter-lib-tasks.jsonl",
    "bench/fixture-tasks.jsonl",
    "tests/distiller_test.cpp",
    "tests/smoke.cmake",
    "tests/invalidation.cmake",
    "tests/mcp_smoke.py",
]


def attribute(record):
    """Name the earliest pipeline stage that lost this task's gold evidence."""
    if record["candidate_recall"] < 1:
        return "retrieval"
    if record["injected_recall"] < record["candidate_recall"]:
        return "packing"
    if record["answer_recall"] < record["injected_recall"]:
        return "representation"
    return "correct"


def measure(path, condition, model):
    report = json.loads(Path(path).read_text())
    rows = [r for r in report["records"]
            if r["condition"] == condition and r["model"] == model]
    if not rows:
        raise SystemExit(f"{path}: no records for condition={condition} model={model}")
    mean = lambda key: sum(r[key] for r in rows) / len(rows)
    candidate, injected, answer = mean("candidate_recall"), mean("injected_recall"), mean("answer_recall")
    usable = [r for r in rows if r["injected_recall"] > 0]
    attribution = {}
    for row in rows:
        stage = attribute(row)
        attribution[stage] = attribution.get(stage, 0) + 1
    return {
        "source": str(path), "tasks": len(rows), "model": model, "condition": condition,
        "candidate_recall": round(candidate, 4),
        "injected_recall": round(injected, 4),
        "answer_recall": round(answer, 4),
        "exact_rate": round(sum(r["exact"] for r in rows) / len(rows), 4),
        "U_model_renderer": round(
            sum(r["answer_recall"] for r in usable) / sum(r["injected_recall"] for r in usable), 4
        ) if usable else None,
        "eta_pack": round(injected / candidate, 4) if candidate else None,
        "eta_use": round(answer / injected, 4) if injected else None,
        "eta_pipeline": round(answer / candidate, 4) if candidate else None,
        "entity_id_copies": sum(bool(re.search(r"\bE\d+\b", r["answer"])) for r in rows),
        "attribution": {stage: attribution.get(stage, 0)
                        for stage in ("retrieval", "packing", "representation", "correct")},
    }


def digests(root):
    out = {}
    for name in TRACKED:
        path = root / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--id", required=True, help="checkpoint identifier, e.g. E1")
    parser.add_argument("--note", default="", help="one-line description of what changed")
    parser.add_argument("--before", type=Path, help="baseline pilot JSON to compare against")
    parser.add_argument("--after", type=Path, required=True, help="pilot JSON for this checkpoint")
    parser.add_argument("--baseline-id", default="", help="identifier of the baseline checkpoint")
    parser.add_argument("--model", default="qwen2.5:0.5b-instruct")
    parser.add_argument("--condition", default="dcy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tests = subprocess.run(["ctest", "--test-dir", str(args.root / "build")],
                           capture_output=True, text=True, check=False)
    passed = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", tests.stdout)
    if passed:
        test_status = {"passed": int(passed.group(3)) - int(passed.group(2)),
                       "failed": int(passed.group(2)), "total": int(passed.group(3))}
    else:
        total = re.search(r"100% tests passed out of (\d+)", tests.stdout)
        test_status = ({"passed": int(total.group(1)), "failed": 0, "total": int(total.group(1))}
                       if total else {"error": "could not parse ctest output"})

    checkpoint = {
        "checkpoint": args.id,
        "note": args.note,
        "tests": test_status,
        "current": measure(args.after, args.condition, args.model),
        "digests_sha256": digests(args.root),
    }
    if args.before:
        checkpoint["baseline_id"] = args.baseline_id
        checkpoint["baseline"] = measure(args.before, args.condition, args.model)
        deltas = {}
        for key in ("candidate_recall", "injected_recall", "answer_recall",
                    "U_model_renderer", "eta_pack", "eta_use", "eta_pipeline"):
            new, old = checkpoint["current"][key], checkpoint["baseline"][key]
            if new is not None and old is not None:
                deltas[key] = round(new - old, 4)
        deltas["entity_id_copies"] = (checkpoint["current"]["entity_id_copies"]
                                      - checkpoint["baseline"]["entity_id_copies"])
        checkpoint["delta"] = deltas

    args.out.write_text(json.dumps(checkpoint, indent=2) + "\n")
    print(json.dumps(checkpoint, indent=2))


if __name__ == "__main__":
    main()
