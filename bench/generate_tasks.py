#!/usr/bin/env python3
"""Generate retrieval tasks from an index, with a held-out split.

Hand-writing tasks after seeing retriever behaviour is how a benchmark gets
overfitted. These are derived mechanically from indexed symbols, in strata that
probe different retrieval abilities, and split into design/held-out halves by a
hash of the task id so the split is stable across regenerations.

Strata:
  direct      the goal names the symbol's own words ("set the allocator")
  symbol      the goal quotes the exact symbol name
  path        the goal names the module the symbol lives in
  paraphrase  the goal uses a synonym mapping that avoids the symbol's words
  multi-hop   the goal needs a caller and its callee together

Paraphrase goals rewrite symbol words through a fixed synonym table, which makes
them lexically distant by construction. That is the point: the paraphrase
stratum is meant to be unreachable by term overlap, so it measures semantic
addressability rather than ranking.
"""

import argparse
import hashlib
import json
import random
import re
import sqlite3
from pathlib import Path

# Deliberately avoids the source word, so the goal cannot match the symbol
# lexically. Chosen from ordinary English, not from this corpus's vocabulary.
SYNONYMS = {
    "get": "obtain", "set": "configure", "new": "construct", "free": "release",
    "delete": "destroy", "init": "prepare", "read": "consume", "write": "emit",
    "add": "append", "remove": "discard", "find": "locate", "parse": "interpret",
    "size": "magnitude", "len": "extent", "length": "extent", "count": "tally",
    "next": "subsequent", "prev": "preceding", "start": "commencement",
    "end": "termination", "copy": "duplicate", "move": "relocate",
    "check": "validate", "test": "probe", "error": "fault", "abi": "compatibility",
    "version": "revision number", "name": "label", "id": "identifier",
    "field": "member role", "node": "element", "tree": "hierarchy",
    "cursor": "position marker", "query": "pattern request", "lexer": "scanner",
    "buffer": "staging area", "list": "sequence", "table": "lookup structure",
    "hash": "digest", "string": "text value", "file": "document",
    "client": "consumer", "server": "provider", "connection": "link",
    "command": "instruction", "reply": "response", "event": "occurrence",
    "config": "settings", "range": "span", "point": "coordinate",
    "state": "condition", "cache": "stored copy", "pool": "reservoir",
    "alloc": "memory reservation", "allocator": "memory provider",
}
STOPWORDS = {"ts", "sds", "redis", "curl", "sqlite", "internal", "impl", "static"}


def words(symbol):
    parts = []
    for chunk in symbol.split("_"):
        for piece in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", chunk):
            lower = piece.lower()
            if lower and lower not in STOPWORDS and not lower.isdigit():
                parts.append(lower)
    return parts


def phrase(parts):
    return " ".join(parts)


def paraphrase(parts):
    """Rewrite through the synonym table; return None if nothing can be replaced."""
    out, replaced = [], 0
    for part in parts:
        if part in SYNONYMS:
            out.append(SYNONYMS[part])
            replaced += 1
        else:
            out.append(part)
    return phrase(out) if replaced else None


def held_out(task_id, fraction):
    digest = hashlib.sha256(task_id.encode()).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0 < fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=20)
    parser.add_argument("--held-out-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--min-body-bytes", type=int, default=120)
    args = parser.parse_args()

    connection = sqlite3.connect(args.db)
    rows = [
        (entity_id, name, path, end - start)
        for entity_id, kind, name, path, start, end in connection.execute(
            "SELECT e.id,e.kind,e.name,f.path,e.start_byte,e.end_byte "
            "FROM entities e JOIN files f ON f.id=e.file_id WHERE e.kind='function'"
        )
        if (end - start) >= args.min_body_bytes
    ]
    # Unique names only: a duplicated symbol name has no single gold answer.
    seen = {}
    for entity_id, name, path, size in rows:
        seen.setdefault(name, []).append((entity_id, path, size))
    unique = [(name, entries[0]) for name, entries in seen.items() if len(entries) == 1]

    relations = {}
    for source_id, target_id in connection.execute(
        "SELECT source_id,target_id FROM relations"):
        relations.setdefault(source_id, []).append(target_id)
    names_by_id = {entity_id: (name, path)
                   for entity_id, name, path in connection.execute(
                       "SELECT e.id,e.name,f.path FROM entities e JOIN files f ON f.id=e.file_id")}
    connection.close()

    rng = random.Random(args.seed)
    rng.shuffle(unique)

    tasks, used = [], set()

    def emit(task_id, category, goal, gold):
        if task_id in used or not goal:
            return False
        used.add(task_id)
        tasks.append({"id": task_id, "category": category, "goal": goal,
                      "gold": gold, "split": "held-out" if held_out(task_id, args.held_out_fraction)
                      else "design"})
        return True

    for name, (entity_id, path, _size) in unique:
        parts = words(name)
        if len(parts) < 2:
            continue
        gold = [f"{path}:{name}"]
        stem = Path(path).stem

        counts = {c: sum(1 for t in tasks if t["category"] == c)
                  for c in ("direct", "symbol", "path", "paraphrase", "multi-hop")}
        if counts["direct"] < args.per_stratum:
            emit(f"g-direct-{name}", "direct",
                 f"Find the function that handles {phrase(parts)}", gold)
        elif counts["symbol"] < args.per_stratum:
            emit(f"g-symbol-{name}", "symbol",
                 f"Where is {name} defined?", gold)
        elif counts["path"] < args.per_stratum:
            emit(f"g-path-{name}", "path",
                 f"In the {stem} module, which function deals with {phrase(parts[-2:])}?", gold)
        elif counts["paraphrase"] < args.per_stratum:
            rewritten = paraphrase(parts)
            emit(f"g-para-{name}", "paraphrase",
                 f"Which routine is responsible for {rewritten}?" if rewritten else None, gold)
        elif counts["multi-hop"] < args.per_stratum:
            callees = [t for t in relations.get(entity_id, []) if t in names_by_id]
            if callees:
                callee_name, callee_path = names_by_id[callees[0]]
                if callee_name != name:
                    emit(f"g-multi-{name}", "multi-hop",
                         f"Trace how {phrase(parts)} delegates to its helper",
                         gold + [f"{callee_path}:{callee_name}"])
        if all(counts[c] >= args.per_stratum for c in counts):
            break

    with args.out.open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")

    summary = {}
    for task in tasks:
        key = (task["category"], task["split"])
        summary[key] = summary.get(key, 0) + 1
    print(f"wrote {len(tasks)} tasks to {args.out}")
    for category in ("direct", "symbol", "path", "paraphrase", "multi-hop"):
        design = summary.get((category, "design"), 0)
        hidden = summary.get((category, "held-out"), 0)
        print(f"  {category:12s} design={design:3d} held-out={hidden:3d}")


if __name__ == "__main__":
    main()
