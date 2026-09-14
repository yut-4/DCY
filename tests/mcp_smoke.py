#!/usr/bin/env python3
"""Protocol-level smoke test for DCY's stdio MCP boundary."""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DCY = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build" / "dcy"
SERVER = ROOT / "mcp" / "dcy_mcp.py"


def exchange(process, request):
    process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    assert line, process.stderr.read()
    return json.loads(line)


def call(process, ident, name, args):
    return exchange(process, {"jsonrpc": "2.0", "id": ident, "method": "tools/call",
                              "params": {"name": name, "arguments": args}})["result"]


def main():
    with tempfile.TemporaryDirectory(prefix="dcy-mcp-") as temp:
        db = Path(temp) / "index.sqlite"
        subprocess.run([DCY, "index", ROOT / "tests" / "fixture", db], check=True, capture_output=True)
        process = subprocess.Popen([sys.executable, SERVER, "--dcy", DCY, "--db", db,
                                    "--max-context-bytes", "512", "--max-source-bytes", "1024"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            init = exchange(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                      "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                                                 "clientInfo": {"name": "smoke", "version": "1"}}})
            assert init["result"]["protocolVersion"] == "2025-11-25"
            process.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            process.stdin.flush()
            listed = exchange(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            assert {tool["name"] for tool in listed["result"]["tools"]} == {
                "dcy_view", "dcy_source", "dcy_record", "dcy_finish"}
            view = call(process, 3, "dcy_view", {"goal": "Find login authentication", "max_bytes": 512})
            assert not view.get("isError")
            assert len(view["content"][0]["text"].encode()) <= 512
            assert "[ref=E" in view["content"][0]["text"]
            assert "view" not in view["structuredContent"]  # no duplicate payload
            goal_id = view["structuredContent"]["goal_id"]
            generation = view["structuredContent"]["generation"]
            too_big = call(process, 4, "dcy_view", {"goal": "login", "max_bytes": 513})
            assert too_big["isError"]
            match = re.search(r"\[ref=E(\d+)\]", view["content"][0]["text"])
            assert match
            entity = int(match.group(1))
            source = call(process, 5, "dcy_source", {"entity_id": entity, "generation": generation,
                                                     "max_bytes": 1024})
            assert not source.get("isError")
            assert source["content"][0]["text"].startswith(f"SOURCE E{entity}")
            stale = call(process, 6, "dcy_source", {"entity_id": entity, "generation": "0" * 64,
                                                    "max_bytes": 1024})
            assert stale["isError"]
            recorded = call(process, 7, "dcy_record", {"goal_id": goal_id, "kind": "fact",
                                                      "claim": "Verified source", "entity_id": entity})
            assert not recorded.get("isError")
            finished = call(process, 8, "dcy_finish", {"goal_id": goal_id, "status": "resolved"})
            assert finished["structuredContent"]["status"] == "resolved"
            print("MCP smoke passed: lifecycle, tool list, bounded view, exact source, stale source, state")
        finally:
            process.stdin.close()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
