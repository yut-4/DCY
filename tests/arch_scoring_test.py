#!/usr/bin/env python3
"""Tests for the architecture-comparison scoring logic.

No LLM and no database: every case feeds a fixed answer string and a fixed
context string through the real scoring functions, so a regression shows up as
a failing assertion rather than as a plausible-looking benchmark number.

Two of these tests are regressions for bugs that produced WRONG published
numbers before they were caught, and both are the dangerous kind — they do not
crash, they silently report a better result than reality:

  1. abstention-by-substring: the first N01 detector accepted "X is not
     explicitly defined, however it is used by foo_bar()" as a clean refusal,
     scoring 3 of 5 architectures ABSTAINED when all 5 had in fact fabricated.
  2. context truncation: Ollama defaults to a 4096-token window regardless of
     the model's advertised context, so a "64kB dump" arm can measure a prompt
     the model never received. The tell is two differently-sized arms reporting
     an identical prompt_eval_count.

Run:
  python3 tests/arch_scoring_test.py
  ctest --test-dir build -R dcy_arch_scoring --output-on-failure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

from arch_comparison import score_answer  # noqa: E402

FAILURES: list[str] = []
PASSED = 0


def check(name: str, got, want) -> None:
    global PASSED
    if got == want:
        PASSED += 1
    else:
        FAILURES.append(f"{name}\n      got:  {got!r}\n      want: {want!r}")


TASK = {
    "id": "TX", "tier": "easy", "domain": "tree-sitter",
    "question": "q",
    "required": ["ts_tree_edit", "ts_subtree_edit"],
    "helpful": ["ts_range_edit"],
}
CTX = ("ts_tree_edit [ref=E1] ts_subtree_edit [ref=E2] ts_range_edit [ref=E3] "
       "ts_subtree_pool_new [ref=E4]")


# --- R_c vs R_a separation ------------------------------------------------
# The whole point of the split: evidence recall describes the ARCHITECTURE,
# answer recall describes the MODEL. They must move independently.

def test_perfect_answer():
    s = score_answer("ts_tree_edit calls ts_subtree_edit.", CTX, TASK)
    check("perfect: Ra", s["answer_recall_Ra"], 1.0)
    check("perfect: Rc", s["evidence_recall_Rc"], 1.0)
    check("perfect: precision", s["entity_precision"], 1.0)
    check("perfect: no hallucination", s["n_false_entities"], 0)


def test_evidence_present_but_model_silent():
    # Retriever did its job; model named nothing. Rc high, Ra zero.
    s = score_answer("I cannot determine this.", CTX, TASK)
    check("silent model: Rc stays high", s["evidence_recall_Rc"], 1.0)
    check("silent model: Ra zero", s["answer_recall_Ra"], 0.0)


def test_model_right_but_evidence_absent():
    # Parametric recall with an empty page: Ra high, Rc zero. This is the
    # `base` arm's signature and must NOT be credited to retrieval.
    s = score_answer("ts_tree_edit and ts_subtree_edit.", "", TASK)
    check("no context: Rc zero", s["evidence_recall_Rc"], 0.0)
    check("no context: Ra full", s["answer_recall_Ra"], 1.0)


def test_partial_recall():
    s = score_answer("Only ts_tree_edit matters.", CTX, TASK)
    check("partial: Ra half", s["answer_recall_Ra"], 0.5)


# --- hallucination accounting --------------------------------------------

def test_hallucinated_symbol_counted():
    s = score_answer("ts_tree_edit, ts_subtree_edit and ts_quantum_flux.",
                     CTX, TASK)
    check("halluc: counted", s["n_false_entities"], 1)
    check("halluc: named", s["false_entities"], ["ts_quantum_flux"])
    check("halluc: Ra unaffected", s["answer_recall_Ra"], 1.0)
    # Precision must drop even though recall is perfect — this is exactly the
    # T05 case where an arm scored Ra=1.0 with F1=0.211.
    if not s["entity_precision"] < 1.0:
        FAILURES.append("halluc: precision should drop below 1.0")


def test_helpful_symbols_not_penalized():
    s = score_answer("ts_tree_edit, ts_subtree_edit, ts_range_edit.", CTX, TASK)
    check("helpful: not hallucination", s["n_false_entities"], 0)
    check("helpful: precision intact", s["entity_precision"], 1.0)


def test_prose_words_are_not_symbols():
    # English prose must not be scored as invented identifiers, or every
    # verbose answer would look like a hallucinating one.
    s = score_answer("The function ts_tree_edit propagates changes downward "
                     "through the tree structure recursively.", CTX, TASK)
    check("prose: no false symbols", s["n_false_entities"], 0)


# --- REGRESSION: abstention must not be a substring match -----------------

def test_hedge_plus_invention_is_fabrication():
    """The verbatim answer that fooled the first detector.

    This is static-2048's real N01 output from build/arch-qwen3-0.6b.json.
    The original detector matched the substring "insufficient" and scored it
    ABSTAINED, publishing "3 of 5 architectures abstained" when in fact all
    five fabricated. Keep the real string: a synthetic paraphrase may miss the
    exact keyword that triggered the bug and the test then proves nothing.
    """
    absent = dict(TASK, required=[], helpful=[],
                  tier="control-absent", absent_symbol="ts_subtree_quantum_edit")
    answer = ("The role of `ts_subtree_quantum_edit` is not explicitly defined "
              "in the provided context. However, it is mentioned in the "
              "`tsd.h` enum and referenced in various C functions like "
              "`hpa_try_alloc_one_no_grow`, `check_match`, and others. "
              "The evidence is insufficient to determine its specific role.")
    s = score_answer(answer, "unrelated context", absent)
    check("N01 regression: not abstained", s["control_abstained"], False)
    check("N01 regression: fabricated", s["control_fabricated"], True)
    # The invented supporting symbols are what make it a fabrication despite
    # the hedge, so they must be visible in the record, not just the verdict.
    if s["control_invented_symbols"] < 2:
        FAILURES.append("N01 regression: invented symbols not counted")

    # And prove the ORIGINAL rule really did pass this string, so this test
    # cannot silently stop being a regression if someone reintroduces it.
    legacy_keywords = ("not find", "no such", "not exist", "insufficient",
                       "could not", "cannot find", "does not appear",
                       "no information", "not present", "unable to find",
                       "not available", "no function")
    legacy_verdict = any(k in answer.lower() for k in legacy_keywords)
    check("N01 regression: legacy rule would have passed it",
          legacy_verdict, True)


def test_confident_role_assertion_is_fabrication():
    absent = dict(TASK, required=[], helpful=[],
                  tier="control-absent", absent_symbol="ts_quantum_edit")
    # No invented symbols at all, but it still asserts a role for something
    # that does not exist. Must not pass as abstention.
    s = score_answer("The role of ts_quantum_edit is to optimize processing.",
                     "unrelated context", absent)
    check("N01 assert: fabricated", s["control_fabricated"], True)
    check("N01 assert: not abstained", s["control_abstained"], False)


def test_true_abstention_recognised():
    absent = dict(TASK, required=[], helpful=[],
                  tier="control-absent", absent_symbol="ts_quantum_edit")
    s = score_answer("I could not find this symbol in the provided source.",
                     "unrelated context", absent)
    check("N01 clean: abstained", s["control_abstained"], True)
    check("N01 clean: not fabricated", s["control_fabricated"], False)


# --- suite integrity ------------------------------------------------------

def test_gold_suite_is_wellformed():
    path = REPO / "bench" / "arch-suite-tasks.jsonl"
    tasks = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    check("suite: task count", len(tasks), 14)
    ids = [t["id"] for t in tasks]
    check("suite: unique ids", len(set(ids)), len(ids))
    for t in tasks:
        for key in ("id", "tier", "domain", "question", "required", "helpful"):
            if key not in t:
                FAILURES.append(f"suite: {t.get('id')} missing {key}")
        # required must be disjoint from helpful, else precision double-counts
        if set(t["required"]) & set(t["helpful"]):
            FAILURES.append(f"suite: {t['id']} required/helpful overlap")
        if not t["tier"].startswith("control") and not t["required"]:
            FAILURES.append(f"suite: {t['id']} scored task with empty required")
    ctrl = [t for t in tasks if t["tier"] == "control-absent"]
    check("suite: has absent control", len(ctrl), 1)
    if ctrl and not ctrl[0].get("absent_symbol"):
        FAILURES.append("suite: absent control lacks absent_symbol")


def test_gold_symbols_exist_in_corpus():
    """Skipped unless the corpus is built — gold that is not indexed makes
    every R_c meaningless, so it is checked whenever the DB is available."""
    import sqlite3
    db = REPO / "build" / "corpus-1m.sqlite"
    if not db.is_file():
        print("  (skip) corpus-1m.sqlite absent — gold/index check not run")
        return
    conn = sqlite3.connect(db)
    path = REPO / "bench" / "arch-suite-tasks.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        for key in ("required", "helpful"):
            for sym in t[key]:
                n = conn.execute("SELECT count(*) FROM entities WHERE name=?",
                                 (sym,)).fetchone()[0]
                if n == 0:
                    FAILURES.append(f"gold: {t['id']}.{key} '{sym}' NOT indexed")
        absent = t.get("absent_symbol")
        if absent:
            n = conn.execute("SELECT count(*) FROM entities WHERE name=?",
                             (absent,)).fetchone()[0]
            if n:
                FAILURES.append(
                    f"control broken: '{absent}' exists in the index")
    global PASSED
    PASSED += 1


# --- REGRESSION: silent context truncation --------------------------------

def test_truncation_flag_logic():
    """Mirrors the ctx_truncated rule used by both harnesses.

    Ollama honours num_ctx and silently drops the overflow, so a prompt at the
    window edge means the model never saw the tail we scored against.
    """
    def truncated(prompt_eval_count: int, num_ctx: int) -> bool:
        return bool(prompt_eval_count and prompt_eval_count >= num_ctx - 8)

    check("trunc: well under window", truncated(300, 4096), False)
    check("trunc: at the edge", truncated(4090, 4096), True)
    check("trunc: exactly at window", truncated(4096, 4096), True)
    check("trunc: raised window clears it", truncated(4090, 16384), False)
    # The real signature from the killed run: two different dump sizes that
    # both report the same prompt token count are both truncated.
    check("trunc: 16kB arm", truncated(2050, 2048), True)
    check("trunc: 64kB arm same count", truncated(2050, 2048), True)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} test groups from {Path(__file__).name}\n")
    for t in tests:
        print(f"  {t.__name__}")
        t()
    print()
    if FAILURES:
        print(f"FAILED — {len(FAILURES)} assertion(s), {PASSED} passed\n")
        for f in FAILURES:
            print(f"  ✗ {f}")
        return 1
    print(f"OK — {PASSED} assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
