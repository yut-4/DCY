#!/usr/bin/env python3
"""Retrieval feature ablation: does structural ranking beat plain lexical FTS?

This is a Python prototype over the same SQLite index the C++ binary builds. It
exists so feature combinations can be compared before any of them is committed
to `src/main.cpp`. It reports candidate recall (gold reachable by the retriever,
before packing), so the packer cannot mask a retrieval change.

Features are discrete and are switched on or off as a set, rather than being
hand-tuned against the task list, which would overfit an 18-task manifest:

    fts       normalized BM25 rank over name/path/body
    exact     goal mentions the symbol name, or a distinctive part of it
    defn      entity is a definition with a body, not a bare declaration
    kind      entity kind matches what the goal asks for ("function", "struct")
    graph     one-hop neighbour of a high-scoring seed
    path      goal words overlap the file/module name

The lexical ceiling is reported alongside: tasks whose gold shares no term with
the goal cannot be recovered by any reweighting of lexical evidence, and a
combination that appears to fix them is measuring something else.
"""

import argparse
import itertools
import json
import re
import sqlite3
from pathlib import Path

STOP = {
    "the", "that", "this", "a", "an", "of", "to", "in", "on", "for", "with",
    "and", "or", "is", "are", "does", "do", "where", "which", "what", "how",
    "find", "locate", "trace", "function", "routine", "returns", "return",
    "its", "it", "be", "by", "from", "into", "after", "when", "same",
}
KIND_WORDS = {
    "function": {"function", "routine", "method", "call", "caller", "callback"},
    "struct": {"struct", "structure", "type", "record"},
}


def terms(text):
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]{2,}", text)]


def content_terms(text):
    return [t for t in terms(text) if t not in STOP]


def name_parts(name):
    """ts_language_abi_version -> {ts, language, abi, version} plus camel splits."""
    parts = set()
    for chunk in name.split("_"):
        if not chunk:
            continue
        parts.add(chunk.lower())
        for piece in re.findall(r"[A-Z]?[a-z0-9]+", chunk):
            parts.add(piece.lower())
    return parts


def lexical_candidates(connection, goal, limit):
    query = " OR ".join(f'"{t}"' for t in terms(goal)[:12])
    if not query:
        return []
    rows = connection.execute(
        "SELECT rowid FROM entity_fts WHERE entity_fts MATCH ? "
        "ORDER BY bm25(entity_fts),rowid LIMIT ?", (query, limit)
    )
    return [row[0] for row in rows]


def lexical_reachable(connection, goal):
    """Every entity the FTS query matches at any depth: the lexical ceiling."""
    query = " OR ".join(f'"{t}"' for t in terms(goal)[:12])
    if not query:
        return set()
    return {row[0] for row in connection.execute(
        "SELECT rowid FROM entity_fts WHERE entity_fts MATCH ?", (query,))}


def score(entity, goal, rank, total, features):
    kind, name, path, has_body = entity
    goal_terms = set(content_terms(goal))
    value = 0.0
    if "fts" in features:
        value += 0.40 * (1.0 - rank / max(total, 1))
    if "exact" in features:
        parts = name_parts(name) - {"ts"}
        overlap = parts & goal_terms
        if name.lower() in goal.lower():
            value += 0.25
        elif overlap:
            value += 0.25 * (len(overlap) / max(len(parts), 1))
    if "defn" in features and has_body:
        value += 0.15
    if "kind" in features:
        for entity_kind, words in KIND_WORDS.items():
            if kind == entity_kind and goal_terms & words:
                value += 0.10
    if "path" in features:
        stem = Path(path).stem.lower()
        if stem in goal_terms or any(stem in t or t in stem for t in goal_terms):
            value += 0.10
    return value


def retrieve(connection, entities, goal, features, cap, graph_seeds=0.5):
    ranked = lexical_candidates(connection, goal, cap)
    total = len(ranked)
    scored = {}
    for rank, entity_id in enumerate(ranked):
        if entity_id not in entities:
            continue
        scored[entity_id] = score(entities[entity_id], goal, rank, total, features)
    if "graph" in features and scored:
        threshold = max(scored.values()) * graph_seeds
        seeds = [i for i, s in scored.items() if s >= threshold]
        for seed in seeds:
            rows = connection.execute(
                "SELECT target_id FROM relations WHERE source_id=? "
                "UNION SELECT source_id FROM relations WHERE target_id=? LIMIT 32",
                (seed, seed))
            for (neighbour,) in rows:
                if neighbour in entities:
                    scored[neighbour] = max(scored.get(neighbour, 0.0), 0.20)
    return sorted(scored, key=lambda i: (-scored[i], i))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=24,
                        help="FTS candidate cap; the C++ retriever currently uses 24")
    parser.add_argument("--top", type=int, default=0,
                        help="truncate the ranked list to this many entities (0 = no limit)")
    parser.add_argument("--split", choices=["design", "held-out", "all"], default="all",
                        help="score only one split of a task manifest that defines one")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if args.split != "all":
        tasks = [t for t in tasks if t.get("split", "design") == args.split]
    if not tasks:
        parser.error(f"no tasks in split {args.split}")
    connection = sqlite3.connect(args.db)
    entities, names = {}, {}
    for entity_id, kind, name, path, start, end in connection.execute(
        "SELECT e.id,e.kind,e.name,f.path,e.start_byte,e.end_byte "
        "FROM entities e JOIN files f ON f.id=e.file_id"
    ):
        entities[entity_id] = (kind, name, path, (end - start) > 80)
        names[entity_id] = f"{path}:{name}"

    ceiling_hits, unreachable = 0, []
    for task in tasks:
        reachable = {names[i] for i in lexical_reachable(connection, task["goal"]) if i in names}
        if set(task["gold"]) <= reachable:
            ceiling_hits += 1
        else:
            unreachable.append(task["id"])

    combos = [
        ("fts",),
        ("fts", "exact"),
        ("fts", "defn"),
        ("fts", "kind"),
        ("fts", "path"),
        ("fts", "exact", "path"),
        ("fts", "exact", "defn"),
        ("fts", "exact", "defn", "kind"),
        ("fts", "exact", "defn", "kind", "path"),
        ("fts", "exact", "path", "graph"),
        ("fts", "graph"),
        ("fts", "exact", "graph"),
        ("fts", "exact", "defn", "kind", "path", "graph"),
    ]
    results = []
    for features in combos:
        recalls, sizes, misses = [], [], []
        for task in tasks:
            gold = set(task["gold"])
            ranked = retrieve(connection, entities, task["goal"], set(features), args.cap)
            if args.top:
                ranked = ranked[:args.top]
            found = {names[i] for i in ranked if i in names}
            recalls.append(len(gold & found) / len(gold))
            sizes.append(len(ranked))
            if not gold <= found:
                misses.append(task["id"])
        results.append({
            "features": "+".join(features),
            "candidate_recall": round(sum(recalls) / len(recalls), 4),
            "full_gold_rate": round(sum(1 for r in recalls if r == 1) / len(recalls), 4),
            "mean_candidates": round(sum(sizes) / len(sizes), 1),
            "missed": misses,
        })

    report = {
        "kind": "retrieval_feature_ablation",
        "cap": args.cap, "top": args.top or None, "tasks": len(tasks),
        "lexical_ceiling": round(ceiling_hits / len(tasks), 4),
        "lexically_unreachable": unreachable,
        "results": results,
        "limitations": [
            "Python prototype over the same index; the C++ retriever is unchanged",
            "candidate recall is not packed recall and not task success",
            "18 hand-written tasks; feature sets are discrete and untuned, but the "
            "manifest is small enough that any ranking claim is provisional",
            "tasks whose gold is lexically unreachable bound every row in this table",
        ],
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(f"lexical ceiling: {report['lexical_ceiling']:.1%} "
          f"(unreachable: {', '.join(unreachable) or 'none'})\n")
    print(f"{'features':44s} {'recall':>7s} {'full':>7s} {'cands':>7s}")
    for row in results:
        print(f"{row['features']:44s} {row['candidate_recall']:7.3f} "
              f"{row['full_gold_rate']:7.3f} {row['mean_candidates']:7.1f}")
    connection.close()


if __name__ == "__main__":
    main()
