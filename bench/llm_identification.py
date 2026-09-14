#!/usr/bin/env python3
"""Small local-model pilot: identify gold C function names, not solve code tasks."""

import argparse
import json
import re
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def cli(command):
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"{command}: {proc.stderr.strip()}")
    return proc.stdout


def generate(endpoint, payload):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result, (time.perf_counter_ns() - start) / 1_000_000


class McpClient:
    def __init__(self, server, dcy, db, cap):
        self.process = subprocess.Popen(
            [sys.executable, str(server), "--dcy", str(dcy), "--db", str(db),
             "--max-context-bytes", str(cap)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        init = self.request(1, "initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "dcy-benchmark", "version": "0.1"},
        })
        if "error" in init:
            raise RuntimeError(init["error"])
        self.process.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        self.process.stdin.flush()
        self.next_id = 2

    def request(self, ident, method, params):
        request = {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("MCP server terminated before reply")
        return json.loads(line)

    def view(self, goal, budget):
        response = self.request(self.next_id, "tools/call", {
            "name": "dcy_view", "arguments": {"goal": goal, "max_bytes": budget},
        })
        self.next_id += 1
        if "error" in response or response["result"].get("isError"):
            raise RuntimeError(response)
        return response["result"]["content"][0]["text"]

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcy", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--context-bytes", type=int, default=512)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["none", "fts", "dcy"])
    parser.add_argument("--mcp-server", type=Path,
                        default=Path(__file__).resolve().parents[1] / "mcp" / "dcy_mcp.py")
    args = parser.parse_args()

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    selected, counts = [], {}
    for task in tasks:
        category = task.get("category", "uncategorized")
        if counts.get(category, 0) >= args.per_category:
            continue
        selected.append(task)
        counts[category] = counts.get(category, 0) + 1
    cli([str(args.dcy), "index", str(args.repo), str(args.db)])
    with sqlite3.connect(args.db) as connection:
        available = {
            f"{path}:{name}" for path, name in connection.execute(
                "SELECT f.path,e.name FROM entities e JOIN files f ON f.id=e.file_id"
            )
        }
    for task in selected:
        if not set(task["gold"]) <= available:
            raise RuntimeError(f"gold symbols missing in index for {task['id']}")

    mcp = McpClient(args.mcp_server, args.dcy, args.db, args.context_bytes) if "dcy-mcp" in args.modes else None
    records = []
    try:
      for task in selected:
        gold = {name.rsplit(":", 1)[1] for name in task["gold"]}
        for mode in args.modes:
            if mode == "none":
                context = ""
            elif mode in {"fts", "dcy"}:
                command = "query-fts" if mode == "fts" else "query"
                context = cli([str(args.dcy), command, str(args.db), task["goal"], str(args.context_bytes)])
            elif mode == "dcy-mcp":
                context = mcp.view(task["goal"], args.context_bytes)
            else:
                parser.error(f"unsupported mode: {mode}")
            prompt = (
                "Identify the C function names required by this repository goal. "
                "Output only function names beginning with ts_, separated by commas. "
                "Never output E-number entity references. "
                "Example: if the context says 'E7 function ts_tree_edit', answer 'ts_tree_edit'. "
                "If you cannot determine the function name, output UNKNOWN.\n"
                f"Goal: {task['goal']}\n"
                f"Repository context:\n{context if context else '(none)'}\nAnswer:"
            )
            response, wall_ms = generate(args.endpoint, {
                "model": args.model, "prompt": prompt, "stream": False, "think": False,
                "keep_alive": "15m",
                "options": {"num_ctx": args.window, "num_predict": 48, "temperature": 0, "seed": 42},
            })
            answer = response.get("response", "").strip()
            predicted = set(re.findall(r"\bts_[A-Za-z0-9_]+\b", answer))
            input_tokens = response.get("prompt_eval_count", 0)
            output_tokens = response.get("eval_count", 0)
            records.append({
                "task": task["id"], "category": task.get("category"), "mode": mode,
                "gold": sorted(gold), "answer": answer, "predicted": sorted(predicted),
                "exact": predicted == gold, "recall": len(gold & predicted) / len(gold),
                "context_bytes": len(context.encode()), "input_tokens": input_tokens,
                "output_tokens": output_tokens, "window_overflow": input_tokens + output_tokens > args.window,
                "wall_ms": wall_ms, "total_ms": response.get("total_duration", 0) / 1_000_000,
                "load_ms": response.get("load_duration", 0) / 1_000_000,
                "prompt_eval_ms": response.get("prompt_eval_duration", 0) / 1_000_000,
                "decode_ms": response.get("eval_duration", 0) / 1_000_000,
                "done_reason": response.get("done_reason"),
            })
    finally:
        if mcp:
            mcp.close()

    summary = []
    for mode in args.modes:
        rows = [row for row in records if row["mode"] == mode]
        summary.append({
            "mode": mode, "n": len(rows),
            "exact_rate": sum(row["exact"] for row in rows) / len(rows),
            "mean_recall": statistics.mean(row["recall"] for row in rows),
            "mean_input_tokens": statistics.mean(row["input_tokens"] for row in rows),
            "mean_output_tokens": statistics.mean(row["output_tokens"] for row in rows),
            "median_total_ms_excluding_load": statistics.median(row["total_ms"] - row["load_ms"] for row in rows),
            "overflow_count": sum(row["window_overflow"] for row in rows),
        })
    print(json.dumps({
        "kind": "function_identification_pilot", "model": args.model,
        "window": args.window, "context_byte_budget": args.context_bytes,
        "tasks": len(selected), "summary": summary, "records": records,
        "limitations": [
            "Function-name identification is not code repair or multi-step reasoning",
            "Context byte budget differs from model token budget",
            "The model may know Tree-sitter function names from pretraining",
            "One deterministic seed per task/mode is not a confidence interval",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
