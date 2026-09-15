#!/usr/bin/env python3
"""Architecture comparison on ONE model: BASE / STATIC / DCY-V2 / ORACLE.

Design follows the creador's spec. The point is NOT "does context help"
(trivially yes) but "does structured navigation beat a flat window of the
same size", so the headline pair is STATIC-512 vs DCY-V2: both start from a
small window; only DCY can page.

Architectures
  base        question only, no context           -> pure parametric recall
  static-512  one retrieval, ~512B injected flat  -> simple RAG baseline
  static-2048 one retrieval, ~2048B injected flat -> "is more context enough?"
  dcy-v2      DCY goal chain, ~512B pages + carry -> navigation under a small window
  oracle      gold entity source injected         -> capability ceiling of this model

Two budget experiments are reported separately because they answer different
questions and must not be blended:
  A. equal TOTAL physical tokens : static-2048 vs dcy-v2  (is navigation better
     than the same total context delivered flat?)
  B. equal INSTANTANEOUS window  : static-512  vs dcy-v2  (can DCY traverse more
     information while keeping the visible working set small?)

Scoring is entity-level, never a bare correct/incorrect, and separates
  R_c (evidence recall — did the architecture SURFACE the gold?)
  R_a (answer recall   — did the MODEL put it in the answer?)
because "DCY never found it" and "DCY found it, 0.6B could not use it" are
different failures and only the second is a model result.

Every arm is plain text in / plain text out. No JSON contract is imposed on
the model: a 0.6B failing to emit JSON would be measuring protocol compliance,
not task ability.

Run:
  python3 bench/arch_comparison.py --model qwen3:0.6b
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover
    # Only the LLM-calling arms need httpx. The scoring functions are pure and
    # are imported by tests/arch_scoring_test.py, which must run under CTest's
    # interpreter without the benchmark's optional dependencies installed.
    httpx = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")

READER_PROMPT = """You are answering a question about a C codebase.

{context_block}QUESTION: {question}

Answer in at most 4 sentences. Name the specific C functions involved.
Use only the information available to you. If the evidence is insufficient,
say so explicitly. Do not invent function names."""

CONTEXT_BLOCK = """REPOSITORY CONTEXT:
{ctx}

"""


def ollama(base: str, model: str, prompt: str, num_ctx: int,
           num_predict: int, timeout: float) -> tuple[str, float, dict]:
    if httpx is None:
        raise RuntimeError("httpx is required to run LLM arms: pip install httpx")
    t0 = time.perf_counter()
    r = httpx.post(f"{base}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict,
                    "num_ctx": num_ctx},
    }, timeout=timeout)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    text = d.get("response", "") or d.get("thinking", "") or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    ptok = d.get("prompt_eval_count") or 0
    return text, wall, {
        "physical_input_tokens": ptok,
        "generated_tokens": d.get("eval_count") or 0,
        # Ollama silently drops prompt tokens past num_ctx; if we are at the
        # window edge the model did not see everything we measured against.
        "ctx_truncated": ptok >= num_ctx - 8,
    }


def dcy_run(dcy: Path, db: Path, args: list[str]) -> tuple[str, float]:
    t0 = time.perf_counter()
    p = subprocess.run([str(dcy), *args], capture_output=True, text=True,
                       timeout=180)
    return p.stdout, time.perf_counter() - t0


def oracle_context(conn: sqlite3.Connection, symbols: list[str],
                   limit: int) -> str:
    parts = []
    for name in symbols:
        row = conn.execute(
            "SELECT f.path, f.content, e.start_byte, e.end_byte "
            "FROM entities e JOIN files f ON f.id=e.file_id "
            "WHERE e.name=? LIMIT 1", (name,)).fetchone()
        if not row:
            continue
        content = row[1]
        if isinstance(content, bytes):
            content = content.decode("utf-8", "replace")
        parts.append(f"===== {row[0]} : {name} =====\n"
                     f"{content[row[2]:row[3]]}")
    return "\n\n".join(parts)[:limit]


def score_answer(answer: str, ctx: str, task: dict) -> dict:
    """Entity-level P/R/F1 plus the R_c vs R_a split."""
    required = set(task["required"])
    allowed = required | set(task["helpful"])

    ans_syms = set(SYMBOL_RE.findall(answer))
    ctx_syms = set(SYMBOL_RE.findall(ctx))

    # Predicted = answer symbols that look like codebase identifiers. We
    # restrict false-positive counting to snake_case C-ish names so ordinary
    # English words in prose are not scored as hallucinated symbols.
    pred = {s for s in ans_syms if "_" in s}

    hit = pred & required
    # Evidence recall: did the ARCHITECTURE surface the gold at all?
    rc = len(required & ctx_syms) / len(required) if required else None
    # Answer recall: did the MODEL name it?
    ra = len(hit) / len(required) if required else None
    prec = len(pred & allowed) / len(pred) if pred else (1.0 if not required else 0.0)
    f1 = (2 * prec * ra / (prec + ra)) if (ra and prec and (prec + ra)) else 0.0

    # Hallucinated = snake_case identifier in the answer that does NOT appear
    # in the context it was given. For the base arm (no context) every symbol
    # is by definition unsupported, which is exactly the number we want.
    false_entities = sorted(pred - ctx_syms - allowed)

    out = {
        "entities_found": sorted(hit),
        "false_entities": false_entities,
        "n_false_entities": len(false_entities),
        "evidence_recall_Rc": None if rc is None else round(rc, 3),
        "answer_recall_Ra": None if ra is None else round(ra, 3),
        "entity_precision": round(prec, 3),
        "entity_f1": round(f1, 3),
    }
    if task.get("absent_symbol"):
        low = answer.lower()
        # A keyword match alone is NOT abstention: models happily write
        # "X is not explicitly defined, however it is used by foo_bar()" and
        # a naive substring check scores that as a clean refusal. Require the
        # hedge AND the absence of invented supporting symbols, and treat any
        # confident role assertion ("is to <verb>", "is used to", "is likely
        # to") as fabrication regardless of later hedging.
        hedged = any(k in low for k in (
            "not find", "no such", "not exist", "insufficient", "could not",
            "cannot find", "does not appear", "no information", "not present",
            "unable to find", "not available", "no function", "not defined",
            "not explicitly", "unclear"))
        asserts_role = any(k in low for k in (
            "is to ", "is used", "is likely", "is a function", "its role is",
            "function is responsible", "appears to be", "it manages",
            "it optimizes", "it handles"))
        out["control_hedged"] = hedged
        out["control_asserts_role"] = asserts_role
        out["control_invented_symbols"] = len(false_entities)
        # Clean only when it hedges, asserts nothing, and invents nothing.
        out["control_abstained"] = hedged and not asserts_role and not false_entities
        out["control_fabricated"] = asserts_role or bool(false_entities)
    return out


def run_dcy_v2(dcy: Path, db: Path, task: dict, budget: int, max_goals: int,
               model: str, base: str, num_ctx: int, num_predict: int,
               timeout: float) -> dict:
    """The v2 paging runtime, reusing the shipped consumer."""
    from dcy_consumer import DCYCli, run_paging_runtime

    cli = DCYCli(dcy, db)
    pages_ctx: list[str] = []
    calls = {"n": 0, "ptok": 0, "gtok": 0, "trunc": 0}

    def llm(prompt: str) -> str:
        # Capture what the model was actually shown so evidence recall is
        # computed on real pages, not on a re-query we do afterwards.
        pages_ctx.append(prompt)
        txt, _w, tim = ollama(base, model, prompt, num_ctx, num_predict, timeout)
        calls["n"] += 1
        calls["ptok"] += tim["physical_input_tokens"]
        calls["gtok"] += tim["generated_tokens"]
        calls["trunc"] += int(tim["ctx_truncated"])
        return txt

    t0 = time.perf_counter()
    res = run_paging_runtime(task=task["question"], dcy=cli, llm=llm,
                             budget=budget, max_goals=max_goals)
    wall = time.perf_counter() - t0
    return {
        "answer": res.answer,
        "context_seen": "\n".join(pages_ctx),
        "wall_ms": round(wall * 1000, 1),
        "physical_input_tokens": calls["ptok"],
        "generated_tokens": calls["gtok"],
        "llm_calls": calls["n"],
        "pages": res.total_pages,
        "unique_entities": res.unique_entities,
        "ctx_truncated": bool(calls["trunc"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcy", type=Path, default=Path("build/dcy"))
    ap.add_argument("--db", type=Path, default=Path("build/corpus-1m.sqlite"))
    ap.add_argument("--tasks", type=Path, default=Path("bench/arch-suite-tasks.jsonl"))
    ap.add_argument("--model", default="qwen3:0.6b")
    ap.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    ap.add_argument("--architectures", nargs="+",
                    default=["base", "static-512", "static-2048", "dcy-v2", "oracle"])
    ap.add_argument("--dcy-budget", type=int, default=512)
    ap.add_argument("--max-goals", type=int, default=4)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--only", nargs="*", help="task ids to run")
    ap.add_argument("--out", type=Path, default=Path("build/arch-comparison.json"))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    if args.only:
        tasks = [t for t in tasks if t["id"] in args.only]

    print(f"model={args.model}  db={args.db}  tasks={len(tasks)}  "
          f"num_ctx={args.num_ctx}  temp=0")
    print(f"architectures: {', '.join(args.architectures)}\n")

    out_path = args.out
    records = []
    # Append incrementally: a timeout mid-suite must not lose finished rows.
    for t in tasks:
        print(f"[{t['id']}/{t['tier']}] {t['question'][:66]}")
        for arch in args.architectures:
            try:
                if arch == "base":
                    ctx = ""
                    prompt = READER_PROMPT.format(context_block="",
                                                  question=t["question"])
                    ans, wall, tim = ollama(args.ollama_base, args.model, prompt,
                                            args.num_ctx, args.num_predict,
                                            args.timeout)
                    rec = {"answer": ans, "wall_ms": round(wall * 1000, 1),
                           "llm_calls": 1, "pages": 0, **tim}
                elif arch.startswith("static-"):
                    budget = int(arch.split("-")[1])
                    ctx, r_secs = dcy_run(args.dcy, args.db,
                                          ["query", str(args.db), t["question"],
                                           str(budget)])
                    prompt = READER_PROMPT.format(
                        context_block=CONTEXT_BLOCK.format(ctx=ctx),
                        question=t["question"])
                    ans, wall, tim = ollama(args.ollama_base, args.model, prompt,
                                            args.num_ctx, args.num_predict,
                                            args.timeout)
                    rec = {"answer": ans, "wall_ms": round((wall + r_secs) * 1000, 1),
                           "retrieval_ms": round(r_secs * 1000, 1),
                           "llm_calls": 1, "pages": 1, **tim}
                elif arch == "oracle":
                    ctx = oracle_context(conn, t["required"] + t["helpful"], 6000)
                    prompt = READER_PROMPT.format(
                        context_block=CONTEXT_BLOCK.format(ctx=ctx),
                        question=t["question"])
                    ans, wall, tim = ollama(args.ollama_base, args.model, prompt,
                                            args.num_ctx, args.num_predict,
                                            args.timeout)
                    rec = {"answer": ans, "wall_ms": round(wall * 1000, 1),
                           "llm_calls": 1, "pages": 1, **tim}
                elif arch == "dcy-v2":
                    r = run_dcy_v2(args.dcy, args.db, t, args.dcy_budget,
                                   args.max_goals, args.model, args.ollama_base,
                                   args.num_ctx, args.num_predict, args.timeout)
                    ctx = r.pop("context_seen")
                    ans = r["answer"]
                    rec = r
                else:
                    raise SystemExit(f"unknown architecture {arch}")

                rec.update(score_answer(ans, ctx, t))
                rec.update({"task": t["id"], "tier": t["tier"],
                            "domain": t["domain"], "model": args.model,
                            "architecture": arch, "context_chars": len(ctx),
                            "answered": bool(ans.strip()), "error": None})
            except Exception as exc:  # keep the suite alive, record the failure
                rec = {"task": t["id"], "tier": t["tier"], "domain": t["domain"],
                       "model": args.model, "architecture": arch,
                       "error": f"{type(exc).__name__}: {exc}"[:200],
                       "answered": False}
            records.append(rec)
            out_path.write_text(json.dumps(records, indent=2))

            if rec.get("error"):
                print(f"   {arch:<12} ERROR {rec['error'][:60]}")
            else:
                ra = rec.get("answer_recall_Ra")
                rc = rec.get("evidence_recall_Rc")
                print(f"   {arch:<12} {rec['wall_ms'] / 1000:6.1f}s "
                      f"{rec.get('physical_input_tokens', 0):>6}PT "
                      f"Rc={rc if rc is not None else '-':<5} "
                      f"Ra={ra if ra is not None else '-':<5} "
                      f"F1={rec.get('entity_f1', 0):<5} "
                      f"halluc={rec.get('n_false_entities', 0)}")
        print()

    print(f"\nraw artifact: {out_path}")


if __name__ == "__main__":
    main()
