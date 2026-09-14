#!/usr/bin/env python3
"""Stage attribution pilot: separates retrieval, packing, representation and model capacity.

For each task it reports four stages:
  candidate_recall  gold present in the retriever output at an effectively unbounded budget
  injected_recall   gold surviving distiller packing at the real byte budget
  answer_recall     gold names produced by the model from that packed view
  oracle_*          same model given the gold entity directly, as compact IR or as prose

A gap between candidate_recall and injected_recall blames packing; a gap between
injected_recall and answer_recall blames representation or model capacity; a low
oracle_nl score means the model cannot do the task at all and DCY is not the cause.
"""

import argparse
import json
import re
import sqlite3
import subprocess
import urllib.request
from pathlib import Path

PROMPT = (
    "Identify the C function names required by this repository goal. "
    "Output only function names, separated by commas. "
    "Never output entity reference tags such as [ref=E12] or bare E-numbers. "
    "Example: if the context says 'function tree_edit [ref=E7]', answer 'tree_edit'. "
    "If you cannot determine the function name, output UNKNOWN.\n"
    "Goal: {goal}\nRepository context:\n{context}\nAnswer:"
)
UNBOUNDED = 1_000_000
# Answers are scored against the gold symbol names themselves rather than a
# corpus-specific prefix, so the same harness works on tree-sitter, redis and
# curl without a per-corpus regex.
IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


def predicted_names(answer, gold_names):
    """Names the model actually produced, scored against this task's gold set.

    Exact match is computed over the full identifier set the model emitted, so
    padding an answer with extra symbols is penalised rather than rewarded.
    """
    tokens = {t for t in IDENTIFIER.findall(answer) if not re.fullmatch(r"E\d+", t)}
    return tokens


def view(dcy, db, mode, goal, budget):
    command = [str(dcy), mode, str(db), goal, str(budget)]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode:
        raise RuntimeError(f"{command}: {process.stderr.strip()}")
    return process.stdout


def emitted(rendered, names):
    return {names[int(i)] for i in re.findall(r"\[ref=E(\d+)\]", rendered) if int(i) in names}


def generate(endpoint, model, prompt, window, num_predict=48):
    payload = {
        "model": model, "prompt": prompt, "stream": False, "think": False, "keep_alive": "15m",
        "options": {"num_ctx": window, "num_predict": num_predict,
                    "temperature": 0, "seed": 42},
    }
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def oracle_ir(task, entities):
    lines = ["DCY/0.1 generation=oracle", "GOAL " + task["goal"]]
    for symbol in task["gold"]:
        entity_id, kind, start, end = entities[symbol]
        path, name = symbol.rsplit(":", 1)
        lines.append(f"E{entity_id} {kind} {name}\n  source={path}:{start}-{end} score=100")
    return "\n".join(lines) + "\n"


# Entity-ID placement ablation. Small models copy a leading E<id> as if it were
# the answer, so placement is varied with everything else held constant.
ID_PLACEMENTS = {
    "id_prefix": lambda i, kind, name: f"E{i} {kind} {name}",
    "id_suffix": lambda i, kind, name: f"{kind} {name} [ref=E{i}]",
    "id_separate": lambda i, kind, name: f"{kind} {name}\n  ref=E{i}",
    "no_id": lambda _i, kind, name: f"{kind} {name}",
}


def oracle_placement(task, entities, placement, distractors=()):
    render = ID_PLACEMENTS[placement]
    lines = ["DCY/0.1 generation=oracle", "GOAL " + task["goal"]]
    shown = list(task["gold"]) + [s for s in distractors if s not in set(task["gold"])]
    for symbol in shown:
        entity_id, kind, start, end = entities[symbol]
        path, name = symbol.rsplit(":", 1)
        lines.append(render(entity_id, kind, name) + f"\n  source={path}:{start}-{end} score=100")
    return "\n".join(lines) + "\n"


def oracle_prose(task, _entities):
    return " ".join(
        f"The function {symbol.rsplit(':', 1)[1]} in {symbol.rsplit(':', 1)[0]} implements this."
        for symbol in task["gold"]
    ) + "\n"


def dense_distractors(task, args, names, pool, count=8):
    """Non-gold symbols the retriever actually returns for this goal.

    Falls back to the indexed symbol pool if retrieval yields too few, so the
    placement conditions always compare the same number of competing lines.
    """
    gold = set(task["gold"])
    rendered = view(args.dcy, args.db, args.mode, task["goal"], UNBOUNDED)
    retrieved = [s for s in emitted(rendered, names) if s not in gold]
    retrieved.sort()
    extra = [s for s in pool if s not in gold and s not in set(retrieved)]
    return (retrieved + extra)[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcy", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["qwen2.5:0.5b-instruct"])
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--mode", default="query", choices=["query", "query-fts"])
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--context-bytes", type=int, default=512)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--split", choices=["design", "held-out", "all"], default="all")
    parser.add_argument("--num-predict", type=int, default=48)
    parser.add_argument("--conditions", nargs="+",
                        default=["none", "dcy", "oracle_ir", "oracle_prose"],
                        choices=["none", "dcy", "oracle_ir", "oracle_prose"],
                        help="conditions to run; use fewer for larger scale pilots")
    parser.add_argument("--id-ablation", action="store_true",
                        help="add id_prefix/id_suffix/id_separate/no_id oracle conditions")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if args.split != "all":
        tasks = [t for t in tasks if t.get("split", "design") == args.split]
    selected, counts = [], {}
    for task in tasks:
        category = task.get("category", "uncategorized")
        if counts.get(category, 0) >= args.per_category:
            continue
        selected.append(task)
        counts[category] = counts.get(category, 0) + 1
    if args.repo:
        subprocess.run([str(args.dcy), "index", str(args.repo), str(args.db)], check=True,
                       capture_output=True, text=True)
    connection = sqlite3.connect(args.db)
    rows = list(connection.execute(
        "SELECT e.id,f.path,e.name,e.kind,e.start_byte,e.end_byte "
        "FROM entities e JOIN files f ON f.id=e.file_id"
    ))
    connection.close()
    names = {row[0]: f"{row[1]}:{row[2]}" for row in rows}
    entities = {f"{row[1]}:{row[2]}": (row[0], row[3], row[4], row[5]) for row in rows}
    for task in selected:
        missing = [symbol for symbol in task["gold"] if symbol not in entities]
        if missing:
            raise RuntimeError(f"gold symbols missing in index for {task['id']}: {missing}")

    all_conditions = {
        "none": lambda task: "",
        "dcy": lambda task: view(args.dcy, args.db, args.mode, task["goal"], args.context_bytes),
        "oracle_ir": lambda task: oracle_ir(task, entities),
        "oracle_prose": lambda task: oracle_prose(task, entities),
    }
    conditions = {name: all_conditions[name] for name in args.conditions}
    if args.id_ablation:
        # A gold-only oracle has one or two lines; the real view has ~10 entities
        # competing for attention. Placement is therefore also probed at realistic
        # density, using retrieved non-gold candidates as distractors.
        pool = sorted(entities)
        for placement in ID_PLACEMENTS:
            conditions[placement] = (
                lambda task, p=placement: oracle_placement(task, entities, p)
            )
            conditions[placement + "_dense"] = (
                lambda task, p=placement: oracle_placement(
                    task, entities, p,
                    distractors=dense_distractors(task, args, names, pool),
                )
            )

    records = []
    for task in selected:
        gold_symbols = set(task["gold"])
        gold_names = {symbol.rsplit(":", 1)[1] for symbol in gold_symbols}
        candidates = emitted(view(args.dcy, args.db, args.mode, task["goal"], UNBOUNDED), names)
        packed_view = view(args.dcy, args.db, args.mode, task["goal"], args.context_bytes)
        injected = emitted(packed_view, names)
        for model in args.models:
            for condition, build in conditions.items():
                context = build(task)
                response = generate(
                    args.endpoint, model,
                    PROMPT.format(goal=task["goal"], context=context or "(none)"),
                    args.window, args.num_predict,
                )
                answer = response.get("response", "").strip()
                predicted = predicted_names(answer, gold_names)
                output_tokens = response.get("eval_count", 0)
                records.append({
                    "task": task["id"], "category": task.get("category"), "model": model,
                    "condition": condition,
                    "candidate_recall": len(gold_symbols & candidates) / len(gold_symbols),
                    "injected_recall": len(gold_symbols & injected) / len(gold_symbols),
                    "answer_recall": len(gold_names & predicted) / len(gold_names),
                    "exact": predicted == gold_names,
                    "candidates": len(candidates), "injected": len(injected),
                    "context_bytes": len(context.encode()),
                    "input_tokens": response.get("prompt_eval_count", 0),
                    "output_tokens": output_tokens,
                    "output_truncated": output_tokens >= args.num_predict,
                    "answer": answer,
                    "copied_entity_id": bool(re.search(r"\bE\d+\b", answer)),
                })

    summary = []
    for model in args.models:
        for condition in conditions:
            subset = [r for r in records if r["model"] == model and r["condition"] == condition]
            summary.append({
                "model": model, "condition": condition, "n": len(subset),
                "mean_candidate_recall": sum(r["candidate_recall"] for r in subset) / len(subset),
                "mean_injected_recall": sum(r["injected_recall"] for r in subset) / len(subset),
                "mean_answer_recall": sum(r["answer_recall"] for r in subset) / len(subset),
                "exact_rate": sum(r["exact"] for r in subset) / len(subset),
                "mean_input_tokens": sum(r["input_tokens"] for r in subset) / len(subset),
                "entity_id_copy_rate": sum(r["copied_entity_id"] for r in subset) / len(subset),
                "output_truncated_rate": sum(r["output_truncated"] for r in subset) / len(subset),
            })
    report = {
        "kind": "stage_attribution_pilot", "mode": args.mode, "window": args.window,
        "context_byte_budget": args.context_bytes, "tasks": len(selected),
        "summary": summary, "records": records,
        "limitations": [
            "oracle conditions hand-build context and are an upper bound, not a DCY result",
            "name identification is not code repair",
            "single seed per task/condition is not a confidence interval",
            "byte budgets are not model token budgets",
        ],
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
