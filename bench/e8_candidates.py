#!/usr/bin/env python3
"""E8 Phase 1 — candidate generation / semantic addressability. No LLM.

Separates two failures that a single recall number conflates:

  R_candidate : was the gold entity generated as a candidate AT ALL?
  R_topk      : did it survive into the top-k the packer actually sees?

  R_candidate high + R_topk low  -> ranking problem
  R_candidate low  + R_topk low  -> addressability problem (no ranker can fix it)

E7 found R_c = 0.000 on P01 (a paraphrase naming no literal symbols) while
oracle held at 1.000, which points at candidate generation rather than ranking.
This harness tests that directly and cheaply: pure SQLite, no model, so a
variant can be evaluated in seconds instead of CPU-minutes.

The `lexical-current` arm is a Python re-implementation of src/main.cpp's
fts_query + retrieve. It is verified against the real `dcy query` binary by
--verify-baseline before any comparison is trusted; a baseline that does not
reproduce the C++ output makes every delta meaningless.

Run:
  python3 bench/e8_candidates.py --verify-baseline
  python3 bench/e8_candidates.py --out build/e8-phase1.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
WORD_RE = re.compile(r"[A-Za-z0-9_]+")

# Query-side stopwords. These are ordinary English function words that carry no
# addressing information but currently enter the FTS disjunction with the same
# weight as a symbol name, so they dominate a natural-language paraphrase.
# Deliberately does NOT include domain words (tree, edit, decay, arena...).
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "there", "here", "when", "where", "which", "who", "whom",
    "what", "why", "how", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having", "can",
    "could", "should", "would", "may", "might", "must", "shall", "will",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into", "onto",
    "over", "under", "about", "as", "it", "its", "his", "her", "their", "our",
    "your", "my", "me", "him", "them", "us", "you", "we", "they", "he", "she",
    "i", "not", "no", "so", "such", "only", "also", "very", "more", "most",
    "some", "any", "all", "each", "both", "few", "many", "much", "other",
    "same", "own", "just", "now", "up", "out", "off", "down", "again",
    "further", "once", "because", "while", "during", "before", "after",
    "above", "below", "between", "through", "across", "against",
    # benchmark-question phrasing that addresses nothing in the corpus
    "find", "identify", "explain", "describe", "trace", "role", "part",
    "parts", "piece", "pieces", "thing", "things", "code", "function",
    "functions", "implementation", "implemented", "used", "use", "uses",
    "handle", "handles", "handling", "work", "works", "happen", "happens",
    "key", "main", "important", "required", "directly", "participate",
    "participates", "involved", "difference", "relationship", "understand",
    "connected", "leads", "lead", "carry", "carries", "change", "changes",
    "represent", "representation", "pieces", "settings", "behavior",
}


def split_identifier(tok: str) -> list[str]:
    """ts_subtree_edit -> [ts, subtree, edit]; TSNodeEdit -> [ts, node, edit].

    Splits on underscores and camelCase boundaries. This is what makes a
    natural-language query able to address a symbol it never names literally.
    """
    parts: list[str] = []
    for chunk in tok.split("_"):
        if not chunk:
            continue
        # camelCase / PascalCase / ACRONYMWord
        for piece in re.findall(
                r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+|[0-9]+", chunk):
            parts.append(piece.lower())
    return [p for p in parts if len(p) >= 2]


# --- morphological variants ------------------------------------------------
# A question says "edited" / "editing"; the symbol says "edit". Without this a
# paraphrase cannot reach the entity at all. Kept crude on purpose: a real
# stemmer is a dependency and this is the cheapest thing that could work.

def morph_variants(word: str) -> set[str]:
    out = {word}
    for suf, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""),
                      ("s", "")):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            stem = word[: len(word) - len(suf)] + repl
            out.add(stem)
            # editing -> edit (doubled consonant: running -> run)
            if suf in ("ing", "ed") and len(stem) >= 4 and stem[-1] == stem[-2]:
                out.add(stem[:-1])
    return {w for w in out if len(w) >= 2}


# ---------------------------------------------------------------------------
# Arms. Each returns an ORDERED candidate list of entity ids (best first).
# ---------------------------------------------------------------------------

def fts_query_current(goal: str) -> str:
    """Faithful port of src/main.cpp:302 fts_query."""
    terms, current = [], ""
    for ch in goal:
        if ch.isalnum() or ch == "_":
            current += ch
        elif len(current) >= 2:
            terms.append(current)
            current = ""
        else:
            current = ""
    if len(current) >= 2:
        terms.append(current)
    return " OR ".join(f'"{t}"' for t in terms[:12])


def _fts_ids(conn, query: str, limit: int) -> list[int]:
    if not query:
        return []
    cur = conn.execute(
        "SELECT rowid FROM entity_fts WHERE entity_fts MATCH ? "
        "ORDER BY bm25(entity_fts), rowid LIMIT ?", (query, limit))
    return [r[0] for r in cur]


def _graph_expand(conn, seeds: list[int], per_seed: int = 32) -> list[int]:
    """1-hop call/reference expansion, mirroring retrieve()'s use_graph."""
    out: list[int] = []
    for sid in seeds:
        cur = conn.execute(
            "SELECT target_id FROM relations WHERE source_id=? "
            "UNION SELECT source_id FROM relations WHERE target_id=? LIMIT ?",
            (sid, sid, per_seed))
        out.extend(r[0] for r in cur)
    return out


def arm_lexical_current(conn, goal: str, k: int) -> list[int]:
    """Port of the shipped retrieve(): FTS top-24, then graph expansion of
    entities scoring >= 65 (i.e. the first 18 FTS hits: 100 - rank*2 >= 65)."""
    ids = _fts_ids(conn, fts_query_current(goal), 24)
    scores: dict[int, int] = {}
    for rank, eid in enumerate(ids):
        scores[eid] = max(scores.get(eid, 0), 100 - rank * 2)
    seeds = [i for i, s in scores.items() if s >= 65]
    for eid in _graph_expand(conn, seeds):
        scores[eid] = max(scores.get(eid, 0), 40)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _query_terms(goal: str, split: bool, morph: bool, drop_stop: bool):
    raw = [w.lower() for w in WORD_RE.findall(goal)]
    terms: list[str] = []
    for w in raw:
        pieces = split_identifier(w) if split else [w]
        # keep the literal identifier too: exact matches must not get worse
        if split and "_" in w:
            pieces = [w.lower()] + pieces
        terms.extend(pieces)
    if drop_stop:
        terms = [t for t in terms if t not in STOPWORDS]
    if morph:
        expanded: list[str] = []
        for t in terms:
            expanded.extend(sorted(morph_variants(t)))
        terms = expanded
    seen, out = set(), []
    for t in terms:
        if len(t) >= 2 and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _weighted_fts(conn, terms: list[str], limit: int) -> dict[int, float]:
    """Score entities by how many distinct query terms they match, with name
    matches worth more than body matches.

    Counting distinct term hits is the point: a single OR disjunction lets one
    very common term flood the result set, which is what makes the current arm
    collapse on a paraphrase. An entity matching three query terms should beat
    one matching a single frequent term regardless of bm25 on the body.
    """
    scores: dict[int, float] = {}
    for term in terms[:24]:
        q = f'"{term}"'
        for rank, eid in enumerate(_fts_ids(conn, q, limit)):
            scores[eid] = scores.get(eid, 0.0) + max(1.0, 6.0 - rank * 0.2)
        for rank, eid in enumerate(_fts_ids(conn, f'name:{q}', limit)):
            scores[eid] = scores.get(eid, 0.0) + max(2.0, 14.0 - rank * 0.4)
    return scores


_DF_CACHE: dict[tuple[int, str], int] = {}


def _doc_freq(conn, term: str) -> int:
    """How many entities match this term at all. Cached per connection."""
    key = (id(conn), term)
    if key not in _DF_CACHE:
        row = conn.execute(
            "SELECT count(*) FROM entity_fts WHERE entity_fts MATCH ?",
            (f'"{term}"',)).fetchone()
        _DF_CACHE[key] = row[0] if row else 0
    return _DF_CACHE[key]


def _idf_weighted_fts(conn, terms: list[str], limit: int,
                      total: int) -> dict[int, float]:
    """Same distinct-term accumulation, but each term contributes in
    proportion to its inverse document frequency.

    Phase 1 showed the un-weighted variant reaches R_candidate = 1.0 while
    losing 0.269 before top-k: every gold entity IS generated, then a flood of
    entities matching only a common split token (`ts`, `arena`, `edit`) outranks
    it. Rare terms are the ones that actually address a symbol, so weight by
    `log(N / df)` and let a single rare hit outweigh several common ones.
    """
    scores: dict[int, float] = {}
    import math
    for term in terms[:24]:
        df = _doc_freq(conn, term)
        if df == 0:
            continue
        idf = math.log((total + 1) / (df + 1)) + 0.25
        if idf <= 0:
            continue
        q = f'"{term}"'
        for rank, eid in enumerate(_fts_ids(conn, q, limit)):
            scores[eid] = scores.get(eid, 0.0) + idf * max(0.4, 2.0 - rank * 0.05)
        for rank, eid in enumerate(_fts_ids(conn, f'name:{q}', limit)):
            scores[eid] = scores.get(eid, 0.0) + idf * max(1.0, 5.0 - rank * 0.15)
    return scores


def make_idf_arm(graph: bool, fts_limit: int = 64):
    def arm(conn, goal: str, k: int) -> list[int]:
        total = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
        terms = _query_terms(goal, split=True, morph=True, drop_stop=True)
        scores = _idf_weighted_fts(conn, terms, fts_limit, total)
        if graph and scores:
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:12]
            best = top[0][1] if top else 1.0
            for eid in _graph_expand(conn, [i for i, _ in top]):
                # Expansion must not outrank direct lexical evidence, so cap
                # its contribution well below the strongest seed's score.
                scores[eid] = scores.get(eid, 0.0) + best * 0.08
        return [i for i, _ in sorted(scores.items(),
                                     key=lambda kv: (-kv[1], kv[0]))]
    return arm


def make_arm(split: bool, morph: bool, drop_stop: bool, graph: bool,
             fts_limit: int = 48):
    def arm(conn, goal: str, k: int) -> list[int]:
        terms = _query_terms(goal, split, morph, drop_stop)
        scores = _weighted_fts(conn, terms, fts_limit)
        if graph and scores:
            top = sorted(scores.items(), key=lambda kv: -kv[1])[:12]
            for eid in _graph_expand(conn, [i for i, _ in top]):
                scores[eid] = scores.get(eid, 0.0) + 1.5
        return [i for i, _ in sorted(scores.items(),
                                     key=lambda kv: (-kv[1], kv[0]))]
    return arm


ARMS = {
    "lexical-current": arm_lexical_current,
    "lexical-stop": make_arm(split=False, morph=False, drop_stop=True, graph=True),
    "lexical-split": make_arm(split=True, morph=False, drop_stop=True, graph=True),
    "lexical-morph": make_arm(split=True, morph=True, drop_stop=True, graph=True),
    "hybrid-graph": make_arm(split=True, morph=True, drop_stop=True, graph=True,
                             fts_limit=96),
    "idf": make_idf_arm(graph=False),
    "idf-graph": make_idf_arm(graph=True),
}


# ---------------------------------------------------------------------------

def gold_ids(conn, names: list[str]) -> dict[str, int]:
    out = {}
    for n in names:
        r = conn.execute("SELECT id FROM entities WHERE name=? LIMIT 1",
                         (n,)).fetchone()
        if r:
            out[n] = r[0]
    return out


def verify_baseline(dcy: Path, db: Path, conn, tasks: list[dict]) -> bool:
    """The Python `lexical-current` must reproduce the C++ binary's page.

    Compares the symbol names the real `dcy query` renders against the top of
    the Python candidate list. Exact set equality is not expected (the C++ side
    also packs under a byte budget), so the check is that the binary's rendered
    symbols are a subset of the Python candidates — if the port missed an
    entity the binary found, the baseline is wrong.
    """
    print("verifying lexical-current against the dcy binary")
    ok = True
    for t in tasks[:6]:
        proc = subprocess.run([str(dcy), "query", str(db), t["question"], "512"],
                              capture_output=True, text=True, timeout=120)
        rendered = set()
        for line in proc.stdout.splitlines():
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+\[ref=", line)
            if m:
                rendered.add(m.group(1))
        py_ids = arm_lexical_current(conn, t["question"], 64)
        py_names = set()
        for eid in py_ids:
            r = conn.execute("SELECT name FROM entities WHERE id=?",
                             (eid,)).fetchone()
            if r:
                py_names.add(r[0])
        missing = rendered - py_names
        status = "ok" if not missing else f"MISSING {sorted(missing)[:5]}"
        if missing:
            ok = False
        print(f"  {t['id']:<5} binary={len(rendered):>3} python={len(py_names):>3}  {status}")
    print("baseline faithful\n" if ok else "BASELINE MISMATCH — deltas not trustworthy\n")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("build/corpus-1m.sqlite"))
    ap.add_argument("--dcy", type=Path, default=Path("build/dcy"))
    ap.add_argument("--tasks", type=Path,
                    default=Path("bench/arch-suite-tasks.jsonl"))
    ap.add_argument("--k", type=int, default=24,
                    help="top-k window the packer sees")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--verify-baseline", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("build/e8-phase1.json"))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]

    if args.verify_baseline:
        if not verify_baseline(args.dcy, args.db, conn, tasks):
            sys.exit(1)

    scored = [t for t in tasks if not t["tier"].startswith("control")]
    records = []

    for arm_name in args.arms:
        arm = ARMS[arm_name]
        for t in tasks:
            gold = gold_ids(conn, t["required"])
            want = set(gold.values())
            cands = arm(conn, t["question"], args.k)
            topk = cands[: args.k]

            rec = {
                "arm": arm_name, "task": t["id"], "tier": t["tier"],
                "domain": t["domain"],
                "n_candidates": len(cands),
                "R_candidate": (len(want & set(cands)) / len(want)) if want else None,
                "R_topk": (len(want & set(topk)) / len(want)) if want else None,
            }
            # MRR over the required set: 1/rank of the first gold hit.
            rr = 0.0
            for rank, eid in enumerate(cands, 1):
                if eid in want:
                    rr = 1.0 / rank
                    break
            rec["reciprocal_rank"] = round(rr, 4)
            if t.get("absent_symbol"):
                # N01 guards against "fix the paraphrase by matching anything":
                # an absent symbol must not start pulling a big candidate set.
                rec["n_candidates_absent_query"] = len(cands)
            records.append(rec)

    def agg(arm_name, rows, key):
        vals = [r[key] for r in rows if r["arm"] == arm_name and r[key] is not None]
        return round(statistics.mean(vals), 3) if vals else None

    print(f"E8 Phase 1 — candidate generation, no LLM.  k={args.k}  "
          f"db={args.db}  scored_tasks={len(scored)}")
    print("=" * 96)
    print(f"{'arm':<18} {'R_cand':>8} {'R_topk':>8} {'MRR':>7} {'med_cands':>10} "
          f"{'easy':>7} {'medium':>7} {'hard':>7} {'P01':>6} {'N01_cands':>10}")
    print("-" * 96)
    summary = {}
    for arm_name in args.arms:
        rows = [r for r in records if r["arm"] == arm_name]
        s = [r for r in rows if not r["tier"].startswith("control")]
        tiers = {}
        for tier in ("easy", "medium", "hard"):
            v = [r["R_topk"] for r in s if r["tier"] == tier and r["R_topk"] is not None]
            tiers[tier] = round(statistics.mean(v), 3) if v else None
        p01 = next((r["R_topk"] for r in rows if r["task"] == "P01"), None)
        n01 = next((r.get("n_candidates_absent_query") for r in rows
                    if r["task"] == "N01"), None)
        row = {
            "R_candidate": agg(arm_name, records, "R_candidate"),
            "R_topk": agg(arm_name, records, "R_topk"),
            "MRR": agg(arm_name, records, "reciprocal_rank"),
            "median_candidates": round(statistics.median(
                r["n_candidates"] for r in rows), 1),
            "by_tier": tiers, "P01_R_topk": p01, "N01_candidates": n01,
        }
        summary[arm_name] = row
        print(f"{arm_name:<18} {str(row['R_candidate']):>8} {str(row['R_topk']):>8} "
              f"{str(row['MRR']):>7} {str(row['median_candidates']):>10} "
              f"{str(tiers['easy']):>7} {str(tiers['medium']):>7} "
              f"{str(tiers['hard']):>7} {str(p01):>6} {str(n01):>10}")

    print()
    print("DIAGNOSIS — addressability vs ranking")
    print("-" * 96)
    for arm_name in args.arms:
        r = summary[arm_name]
        rc, rk = r["R_candidate"], r["R_topk"]
        if rc is None or rk is None:
            continue
        lost = round(rc - rk, 3)
        verdict = ("addressability-bound (gold never generated)" if rc < 0.8
                   else "ranking-bound (generated, cut before top-k)" if lost > 0.05
                   else "clean")
        print(f"  {arm_name:<18} R_cand={rc:<6} R_topk={rk:<6} "
              f"lost_to_ranking={lost:<6} {verdict}")

    args.out.write_text(json.dumps(
        {"k": args.k, "db": str(args.db), "summary": summary,
         "records": records}, indent=2))
    print(f"\nraw artifact: {args.out}")


if __name__ == "__main__":
    main()
