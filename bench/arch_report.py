#!/usr/bin/env python3
"""Aggregate arch_comparison.py records into the tables that matter.

Keeps the creador's two budget experiments separate — they are different
claims and blending them would hide which one the data supports:
  A. equal TOTAL tokens  : static-2048 vs dcy-v2
  B. equal INSTANT window: static-512  vs dcy-v2

Also splits R_c from R_a per architecture, because a low R_a with a high R_c
is a model limit while a low R_c is a retriever limit, and only the first says
anything about the model.

Usage: python3 bench/arch_report.py build/arch-qwen3-0.6b.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "build/arch-comparison.json")
    recs = [r for r in json.loads(path.read_text()) if not r.get("error")]
    errs = [r for r in json.loads(path.read_text()) if r.get("error")]
    if not recs:
        print("no successful records")
        return

    model = recs[0]["model"]
    arches = []
    for r in recs:
        if r["architecture"] not in arches:
            arches.append(r["architecture"])
    scored = [r for r in recs if not r["tier"].startswith("control")]
    tasks = sorted({r["task"] for r in scored})

    print(f"model = {model}   scored tasks = {len(tasks)}   "
          f"records = {len(recs)}" + (f"   errors = {len(errs)}" if errs else ""))
    print()
    print("MAIN TABLE — all scored tasks (T01-T12, P01)")
    print("-" * 94)
    print(f"{'architecture':<13} {'R_c':>6} {'R_a':>6} {'prec':>6} {'F1':>6} "
          f"{'halluc':>7} {'med_PT':>7} {'med_s':>7} {'calls':>6} {'solved':>7}")
    print("-" * 94)
    table = {}
    for a in arches:
        rs = [r for r in scored if r["architecture"] == a]
        if not rs:
            continue
        # "solved" = every required entity named; the strictest per-task bar.
        solved = sum(1 for r in rs if r.get("answer_recall_Ra") == 1.0)
        row = {
            "Rc": mean(r.get("evidence_recall_Rc") for r in rs),
            "Ra": mean(r.get("answer_recall_Ra") for r in rs),
            "prec": mean(r.get("entity_precision") for r in rs),
            "F1": mean(r.get("entity_f1") for r in rs),
            "halluc": mean(r.get("n_false_entities") for r in rs),
            "PT": med(r.get("physical_input_tokens") for r in rs),
            "secs": med((r.get("wall_ms") or 0) / 1000 for r in rs),
            "calls": mean(r.get("llm_calls") for r in rs),
            "solved": f"{solved}/{len(rs)}",
            "n": len(rs),
        }
        table[a] = row
        print(f"{a:<13} {str(row['Rc']):>6} {str(row['Ra']):>6} {str(row['prec']):>6} "
              f"{str(row['F1']):>6} {str(row['halluc']):>7} {str(row['PT']):>7} "
              f"{str(row['secs']):>7} {str(row['calls']):>6} {row['solved']:>7}")

    print()
    print("BY TIER — answer recall R_a")
    print("-" * 70)
    tiers = ["easy", "medium", "hard"]
    print(f"{'architecture':<13} " + " ".join(f"{t:>10}" for t in tiers))
    print("-" * 70)
    for a in arches:
        cells = []
        for t in tiers:
            rs = [r for r in scored
                  if r["architecture"] == a and r["tier"] == t]
            cells.append(str(mean(r.get("answer_recall_Ra") for r in rs)))
        print(f"{a:<13} " + " ".join(f"{c:>10}" for c in cells))

    print()
    print("EXPERIMENT A — equal TOTAL physical tokens (static-2048 vs dcy-v2)")
    print("  question: is navigation better than the same total context, flat?")
    print("-" * 70)
    for a in ("static-2048", "dcy-v2"):
        if a in table:
            r = table[a]
            print(f"  {a:<12} R_a={r['Ra']}  F1={r['F1']}  "
                  f"median_PT={r['PT']}  median_s={r['secs']}  solved={r['solved']}")
    if "static-2048" in table and "dcy-v2" in table:
        d = round((table["dcy-v2"]["Ra"] or 0) - (table["static-2048"]["Ra"] or 0), 3)
        print(f"  delta R_a (dcy - static2048) = {d:+}")

    print()
    print("EXPERIMENT B — equal INSTANT window (static-512 vs dcy-v2)")
    print("  question: can DCY traverse more while keeping the window small?")
    print("-" * 70)
    for a in ("static-512", "dcy-v2"):
        if a in table:
            r = table[a]
            print(f"  {a:<12} R_a={r['Ra']}  F1={r['F1']}  "
                  f"median_PT={r['PT']}  median_s={r['secs']}  solved={r['solved']}")
    if "static-512" in table and "dcy-v2" in table:
        d = round((table["dcy-v2"]["Ra"] or 0) - (table["static-512"]["Ra"] or 0), 3)
        print(f"  delta R_a (dcy - static512)  = {d:+}")

    print()
    print("CEILING DECOMPOSITION — where the loss happens")
    print("-" * 70)
    for a in arches:
        rs = [r for r in scored if r["architecture"] == a]
        if not rs:
            continue
        rc, ra = mean(r.get("evidence_recall_Rc") for r in rs), mean(
            r.get("answer_recall_Ra") for r in rs)
        if rc:
            # eta_use: of the gold the architecture surfaced, how much did the
            # model actually put in its answer? Isolates model from retriever.
            print(f"  {a:<12} retriever surfaced {rc:.0%} of gold; "
                  f"model named {ra:.0%}; eta_use = {ra / rc:.0%}")
        else:
            print(f"  {a:<12} retriever surfaced {rc}; model named {ra}")

    ctrl = [r for r in recs if r["tier"] == "control-absent"]
    if ctrl:
        print()
        print("CONTROL N01 — absent symbol (ts_subtree_quantum_edit)")
        print("-" * 70)
        for r in ctrl:
            verdict = ("ABSTAINED (good)" if r.get("control_abstained")
                       else "FABRICATED (bad)" if r.get("control_fabricated")
                       else "unclear")
            print(f"  {r['architecture']:<12} {verdict:<18} "
                  f"| {' '.join((r.get('answer') or '').split())[:60]}")

    para = [r for r in recs if r["tier"] == "control-paraphrase"]
    if para:
        print()
        print("CONTROL P01 — paraphrase of T02 (no symbol names in the question)")
        print("-" * 70)
        for r in para:
            t2 = next((x for x in recs if x["task"] == "T02"
                       and x["architecture"] == r["architecture"]), None)
            base_ra = t2.get("answer_recall_Ra") if t2 else None
            print(f"  {r['architecture']:<12} P01 R_a={r.get('answer_recall_Ra')} "
                  f"vs T02 R_a={base_ra}")

    if errs:
        print()
        print(f"ERRORS ({len(errs)})")
        for e in errs[:10]:
            print(f"  {e['task']:<5} {e['architecture']:<12} {e['error'][:70]}")


if __name__ == "__main__":
    main()
