#!/usr/bin/env python3
"""DCY semantic paging runtime v2 — DCY drives, LLM consumes.

The LLM does NOT control paging. It does not emit JSON commands, choose
goals, or manage protocol. It receives a distilled context page and returns
a plain-text observation. DCY's goal engine decides what to materialize next.

Architecture:

    Task
     ↓
    DCY Goal Engine (deterministic)
     ├── g0.1  locate relevant subsystem
     ├── g0.2  follow important relations
     ├── g0.3  materialize exact source
     └── g0.4  synthesize final capsule
     ↓
    For each sub-goal:
      materialize page (excluding seen entities at same resolution)
       → LLM consumes page, returns plain-text observation
       → DCY records observation, updates seen set
       → cut → distill capsule → advance to next sub-goal
     ↓
    Final capsule + task → LLM produces answer

The LLM minimum capability is: read text, write text. No JSON, no protocol,
no tool calls. A 0.6B model that can read and summarize is sufficient.

Usage:

    python3 api/dcy_consumer.py \\
        --dcy build/dcy --db build/corpus-1m.sqlite \\
        --task "what functions handle tree edit propagation?" \\
        --budget 512 --max-goals 4 \\
        --ollama gemma3:4b
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ContextPage:
    """One materialized page of distilled context."""
    text: str
    tokens: int
    entities: list[str] = field(default_factory=list)
    generation: str = ""
    goal_id: int = 0


@dataclass
class GoalCapsule:
    """Distilled carry-state from a completed sub-goal."""
    goal: str
    observation: str
    entities_seen: list[str] = field(default_factory=list)


@dataclass
class PagingResult:
    """Final output of the runtime."""
    answer: str
    goals_completed: int
    total_pages: int
    total_physical_tokens: int
    total_entities_seen: int
    unique_entities: int
    wall_time_ms: float = 0.0
    capsules: list[GoalCapsule] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Seen-set: prevents same entity at same resolution from re-occupying a page
# ---------------------------------------------------------------------------

class SeenSet:
    """Tracks (entity, resolution) pairs already materialized.

    Rules (from the design doc):
      same information state  → block
      higher resolution       → allow
    Resolution order: REF < COMPACT < DETAIL < SOURCE
    """

    RESOLUTION_ORDER = {"REF": 0, "COMPACT": 1, "DETAIL": 2, "SOURCE": 3}

    def __init__(self):
        self._seen: dict[str, int] = {}  # entity_id → max resolution seen

    def should_penalize(self, entity_id: str, resolution: str = "DETAIL") -> bool:
        level = self.RESOLUTION_ORDER.get(resolution, 1)
        return self._seen.get(entity_id, -1) >= level

    def mark_seen(self, entity_id: str, resolution: str = "DETAIL") -> None:
        level = self.RESOLUTION_ORDER.get(resolution, 1)
        self._seen[entity_id] = max(self._seen.get(entity_id, -1), level)

    @property
    def count(self) -> int:
        return len(self._seen)

    @property
    def all_ids(self) -> list[str]:
        return list(self._seen.keys())


# ---------------------------------------------------------------------------
# Goal engine — deterministic sub-goal decomposition
# ---------------------------------------------------------------------------

def decompose_task(task: str) -> list[str]:
    """Break a task into deterministic sub-goals.

    This is deliberately mechanical, not LLM-driven. The sub-goal sequence
    is the same for every task: locate → follow relations → read source →
    synthesize. The task text parameterizes the first sub-goal's query.
    """
    return [
        task,                                           # g0.1: locate
        f"callers and callees of functions related to: {task}",  # g0.2: relations
        f"implementation details of: {task}",           # g0.3: source
        f"summarize findings about: {task}",            # g0.4: synthesize
    ]


# ---------------------------------------------------------------------------
# DCY CLI interface
# ---------------------------------------------------------------------------

class DCYCli:
    """Shells out to the C++ dcy binary."""

    def __init__(self, dcy: Path, db: Path):
        self.dcy = str(dcy.resolve())
        self.db = str(db.resolve())
        self._goal_cache: dict[str, tuple[int, str]] = {}

    def _cli(self, *args) -> str:
        result = subprocess.run(
            [self.dcy, *map(str, args)],
            capture_output=True, timeout=30, check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", "replace")[:240].strip()
                               or "DCY command failed")
        return result.stdout.decode("utf-8")

    def create_goal(self, goal_text: str) -> tuple[int, str]:
        """Create a fresh goal and return (goal_id, generation).

        Always creates — never reuses a cached goal. Goals get finished
        (`resolved`) at the end of each sub-goal, and `dcy context` refuses
        finished goals ("goal is not active"), so a cross-run cache would
        hand back dead goal_ids. Fresh goals cost one extra `dcy goal`
        call per sub-goal; correctness first.
        """
        created = self._cli("goal", self.db, goal_text).strip()
        goal_id = int(created.split()[0][1:])
        generation = created.split("generation=", 1)[1]
        self._goal_cache[goal_text] = (goal_id, generation)
        return goal_id, generation

    def materialize_page(self, goal_id: int, budget: int) -> ContextPage:
        """Get a distilled context page for a goal under budget."""
        view = self._cli("context", self.db, goal_id, budget)
        view_bytes = len(view.encode("utf-8"))
        entities = re.findall(r"\[ref=E(\d+)\]", view)
        return ContextPage(
            text=view,
            tokens=view_bytes,
            entities=[f"E{eid}" for eid in entities],
        )

    def record(self, goal_id: int, kind: str, claim: str) -> None:
        """Record an observation for a goal."""
        self._cli("record", self.db, goal_id, kind, claim[:512])

    def finish_goal(self, goal_id: int, status: str = "resolved") -> None:
        """Mark a goal as finished."""
        try:
            self._cli("finish", self.db, goal_id, status)
        except RuntimeError:
            pass  # non-fatal if already finished


# ---------------------------------------------------------------------------
# Ollama LLM — the model only reads and writes plain text
# ---------------------------------------------------------------------------

def make_ollama_llm(model: str, base_url: str = "http://localhost:11434"):
    """Returns a callable(prompt) -> str that talks to Ollama."""
    import httpx

    def generate(prompt: str) -> str:
        response = httpx.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256},
            },
            timeout=180,
        )
        response.raise_for_status()
        content = response.json().get("response", "")
        # Strip thinking tags if present (qwen3)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content

    return generate


def stub_llm(prompt: str) -> str:
    """Deterministic stub for testing without a real model."""
    if "What do you observe" in prompt:
        return "I see functions related to the query. Key entities appear to handle the core logic."
    return "Based on the accumulated context, this is the synthesized answer from the stub."


# ---------------------------------------------------------------------------
# The paging runtime — DCY drives everything
# ---------------------------------------------------------------------------

OBSERVE_PROMPT = """You are reading a page of source code context about a codebase.

{carry_state}

--- CONTEXT PAGE ---
{page_text}
--- END PAGE ---

What do you observe? Summarize the key findings in 2-3 sentences. Focus on:
- What functions/symbols are relevant
- What relationships you notice (callers, callees, data flow)
- What is still unclear

Be concise. Plain text only."""


ANSWER_PROMPT = """You are answering a question about a codebase. You have accumulated
observations from reading multiple context pages.

Task: {task}

Accumulated observations:
{capsule_text}

Based ONLY on these observations, answer the task. If you cannot answer fully,
say what you found and what is missing. Plain text only."""


def run_paging_runtime(
    task: str,
    dcy: DCYCli,
    llm: callable,  # (prompt: str) -> str
    *,
    budget: int = 512,
    max_goals: int = 4,
) -> PagingResult:
    """DCY-driven paging loop. The LLM never controls navigation.

    For each sub-goal:
      1. DCY materializes a page (filtering seen entities)
      2. LLM reads the page and returns a plain-text observation
      3. DCY records the observation, updates seen set
      4. Cut → distill capsule → advance to next sub-goal
    After all sub-goals, LLM synthesizes a final answer from capsules.
    """
    start = time.perf_counter()
    sub_goals = decompose_task(task)[:max_goals]
    seen = SeenSet()
    capsules: list[GoalCapsule] = []
    total_pages = 0
    total_tokens = 0

    for i, goal_text in enumerate(sub_goals):
        goal_id, generation = dcy.create_goal(goal_text)

        # Materialize page for this sub-goal
        page = dcy.materialize_page(goal_id, budget)
        total_pages += 1
        total_tokens += page.tokens

        # Filter: mark new entities, note which are truly new
        new_entities = []
        for eid in page.entities:
            if not seen.should_penalize(eid):
                new_entities.append(eid)
            seen.mark_seen(eid)

        # Build carry state from previous capsules
        carry_lines = []
        for cap in capsules:
            carry_lines.append(f"Previous finding ({cap.goal[:80]}): {cap.observation}")
        carry_state = "\n".join(carry_lines) if carry_lines else "This is the first page. No previous observations."

        # LLM consumes the page — just reads and observes, no protocol
        observe_prompt = OBSERVE_PROMPT.format(
            carry_state=carry_state,
            page_text=page.text,
        )
        observation = llm(observe_prompt)

        # Record the observation in DCY's state
        try:
            dcy.record(goal_id, "hypothesis", observation[:512])
        except RuntimeError:
            pass

        # Cut → distill into capsule
        capsule = GoalCapsule(
            goal=goal_text,
            observation=observation,
            entities_seen=new_entities,
        )
        capsules.append(capsule)

        # Finish this sub-goal
        dcy.finish_goal(goal_id)

    # Final synthesis: LLM produces the answer from accumulated capsules
    capsule_text = ""
    for j, cap in enumerate(capsules):
        capsule_text += f"\n[Page {j+1}] Goal: {cap.goal[:100]}\n"
        capsule_text += f"Observation: {cap.observation}\n"
        capsule_text += f"Entities: {', '.join(cap.entities_seen[:10])}\n"

    answer_prompt = ANSWER_PROMPT.format(task=task, capsule_text=capsule_text)
    answer = llm(answer_prompt)

    elapsed = (time.perf_counter() - start) * 1000

    return PagingResult(
        answer=answer,
        goals_completed=len(capsules),
        total_pages=total_pages,
        total_physical_tokens=total_tokens,
        total_entities_seen=sum(len(c.entities_seen) for c in capsules),
        unique_entities=seen.count,
        wall_time_ms=elapsed,
        capsules=capsules,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DCY semantic paging runtime — DCY drives, LLM consumes")
    parser.add_argument("--dcy", type=Path, default=Path("build/dcy"))
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--budget", type=int, default=512,
                        help="physical token budget per page (bytes)")
    parser.add_argument("--max-goals", type=int, default=4,
                        help="maximum sub-goals to execute")
    parser.add_argument("--ollama", type=str, default=None, metavar="MODEL")
    parser.add_argument("--stub", action="store_true")
    args = parser.parse_args()

    if not args.dcy.is_file():
        parser.error(f"DCY binary not found: {args.dcy}")
    if not args.db.is_file():
        parser.error(f"Database not found: {args.db}")

    dcy = DCYCli(args.dcy, args.db)

    if args.stub:
        llm = stub_llm
    elif args.ollama:
        llm = make_ollama_llm(model=args.ollama)
    else:
        parser.error("specify --stub or --ollama MODEL")

    result = run_paging_runtime(
        task=args.task,
        dcy=dcy,
        llm=llm,
        budget=args.budget,
        max_goals=args.max_goals,
    )

    output = {
        "answer": result.answer,
        "usage": {
            "physical_tokens_total": result.total_physical_tokens,
            "pages": result.total_pages,
            "goals_completed": result.goals_completed,
            "entities_seen": result.total_entities_seen,
            "unique_entities": result.unique_entities,
            "wall_time_ms": round(result.wall_time_ms, 1),
        },
        "capsules": [
            {
                "goal": c.goal[:100],
                "observation": c.observation[:200],
                "new_entities": len(c.entities_seen),
            }
            for c in result.capsules
        ],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
