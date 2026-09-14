#!/usr/bin/env python3
"""Small MCP stdio boundary for the C++ DCY CLI; no model or API calls."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROTOCOL_VERSION = "2025-11-25"
MAX_MESSAGE_BYTES = 65536


def schema(properties, required):
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


TOOLS = [
    {
        "name": "dcy_view",
        "description": "Create one concrete goal and return its byte-bounded DCY context view. Host should call before model inference.",
        "inputSchema": schema({
            "goal": {"type": "string", "minLength": 1, "maxLength": 512},
            "max_bytes": {"type": "integer", "minimum": 128, "maximum": 8192},
        }, ["goal", "max_bytes"]),
    },
    {
        "name": "dcy_source",
        "description": "Retrieve one exact source span by entity ID and generation, with a hard byte ceiling.",
        "inputSchema": schema({
            "entity_id": {"type": "integer", "minimum": 1},
            "generation": {"type": "string", "minLength": 64, "maxLength": 64},
            "max_bytes": {"type": "integer", "minimum": 1, "maximum": 8192},
        }, ["entity_id", "generation", "max_bytes"]),
    },
    {
        "name": "dcy_record",
        "description": "Persist one bounded observation for an existing goal; a fact needs an entity ID.",
        "inputSchema": schema({
            "goal_id": {"type": "integer", "minimum": 1},
            "kind": {"type": "string", "enum": ["hypothesis", "fact", "rejected", "result"]},
            "claim": {"type": "string", "minLength": 1, "maxLength": 512},
            "entity_id": {"type": "integer", "minimum": 1},
        }, ["goal_id", "kind", "claim"]),
    },
    {
        "name": "dcy_finish",
        "description": "Mark an existing goal resolved, failed, or blocked.",
        "inputSchema": schema({
            "goal_id": {"type": "integer", "minimum": 1},
            "status": {"type": "string", "enum": ["resolved", "failed", "blocked"]},
        }, ["goal_id", "status"]),
    },
]


def integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")
    return value


def bounded_text(value, name, maximum=512):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise ValueError(f"invalid {name}")
    return value


class Server:
    def __init__(self, dcy, db, max_context_bytes, max_source_bytes):
        self.dcy = str(dcy)
        self.db = str(db)
        self.max_context_bytes = max_context_bytes
        self.max_source_bytes = max_source_bytes
        self.initialized = False
        self.ready = False

    def cli(self, *args):
        try:
            result = subprocess.run([self.dcy, *map(str, args)], capture_output=True,
                                    timeout=30, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("DCY command timed out") from exc
        if result.returncode:
            raise ValueError(result.stderr.decode("utf-8", "replace")[:240].strip() or "DCY command failed")
        try:
            return result.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("source is not UTF-8; MCP text cannot preserve it byte-exactly") from exc

    def call(self, name, args):
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        if name == "dcy_view":
            goal = bounded_text(args.get("goal"), "goal")
            budget = integer(args.get("max_bytes"), "max_bytes", 128, self.max_context_bytes)
            created = self.cli("goal", self.db, goal).strip()
            goal_id = integer(int(created.split()[0][1:]), "goal_id")
            generation = created.split("generation=", 1)[1]
            view = self.cli("context", self.db, goal_id, budget)
            if len(view.encode("utf-8")) > budget:
                raise ValueError("DCY context exceeded byte budget")
            return view, {"goal_id": goal_id, "generation": generation,
                          "view_bytes": len(view.encode("utf-8")), "max_bytes": budget}
        if name == "dcy_source":
            entity_id = integer(args.get("entity_id"), "entity_id")
            generation = bounded_text(args.get("generation"), "generation", 64)
            if len(generation) != 64 or any(c not in "0123456789abcdef" for c in generation):
                raise ValueError("invalid generation")
            budget = integer(args.get("max_bytes"), "max_bytes", 1, self.max_source_bytes)
            source = self.cli("source", self.db, entity_id, generation, budget)
            if len(source.encode("utf-8")) > budget + 256:
                raise ValueError("DCY source exceeded transport limit")
            return source, {"entity_id": entity_id, "generation": generation,
                            "source_bytes_with_header": len(source.encode("utf-8"))}
        if name == "dcy_record":
            goal_id = integer(args.get("goal_id"), "goal_id")
            kind = args.get("kind")
            if kind not in ("hypothesis", "fact", "rejected", "result"):
                raise ValueError("invalid kind")
            claim = bounded_text(args.get("claim"), "claim")
            entity = args.get("entity_id")
            if kind == "fact" and entity is None:
                raise ValueError("fact requires entity_id")
            command = ["record", self.db, goal_id, kind, claim]
            if entity is not None:
                command.append(integer(entity, "entity_id"))
            result = self.cli(*command).strip()
            return result, {"observation_id": int(result[1:])}
        if name == "dcy_finish":
            goal_id = integer(args.get("goal_id"), "goal_id")
            status = args.get("status")
            if status not in ("resolved", "failed", "blocked"):
                raise ValueError("invalid status")
            result = self.cli("finish", self.db, goal_id, status).strip()
            return result, {"goal_id": goal_id, "status": status}
        raise KeyError("unknown tool")

    def handle(self, message):
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise RpcError(-32600, "invalid JSON-RPC request")
        method = message.get("method")
        if not isinstance(method, str):
            raise RpcError(-32600, "missing method")
        if "id" not in message:
            if method == "notifications/initialized" and self.initialized:
                self.ready = True
            return None
        if method == "initialize":
            if self.initialized or not isinstance(message.get("params"), dict):
                raise RpcError(-32602, "invalid initialization")
            self.initialized = True
            requested = message["params"].get("protocolVersion")
            version = requested if requested == PROTOCOL_VERSION else PROTOCOL_VERSION
            return {"protocolVersion": version, "capabilities": {"tools": {}},
                    "serverInfo": {"name": "dcy", "version": "0.1.0"},
                    "instructions": "Host-mediated context only. Do not forward raw tool metadata or duplicate context to the model."}
        if not self.ready:
            raise RpcError(-32000, "server not initialized")
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            params = message.get("params")
            if not isinstance(params, dict) or not isinstance(params.get("name"), str):
                raise RpcError(-32602, "invalid tool call")
            try:
                body, metadata = self.call(params["name"], params.get("arguments", {}))
            except KeyError as exc:
                raise RpcError(-32602, str(exc)) from exc
            except (ValueError, IndexError, subprocess.SubprocessError) as exc:
                return {"content": [{"type": "text", "text": str(exc)[:256]}], "isError": True}
            return {"content": [{"type": "text", "text": body}], "structuredContent": metadata}
        raise RpcError(-32601, "method not found")


class RpcError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def serve(server):
    for raw in sys.stdin.buffer:
        if len(raw) > MAX_MESSAGE_BYTES:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32600, "message": "request too large"}}) + "\n")
            sys.stdout.flush()
            continue
        request = None
        try:
            request = json.loads(raw)
            result = server.handle(request)
            if result is None:
                continue
            response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except RpcError as exc:
            response = {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None,
                        "error": {"code": exc.code, "message": str(exc)}}
        except (ValueError, json.JSONDecodeError) as exc:
            response = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": str(exc)[:256]}}
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="DCY MCP stdio server")
    parser.add_argument("--dcy", type=Path, default=Path("build/dcy"))
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--max-context-bytes", type=int, default=2048)
    parser.add_argument("--max-source-bytes", type=int, default=4096)
    args = parser.parse_args()
    if not args.dcy.is_file() or not args.db.is_file():
        parser.error("DCY executable and an indexed DB must exist")
    if not 128 <= args.max_context_bytes <= 8192 or not 1 <= args.max_source_bytes <= 8192:
        parser.error("invalid server-side byte caps")
    serve(Server(args.dcy.resolve(), args.db.resolve(), args.max_context_bytes, args.max_source_bytes))


if __name__ == "__main__":
    main()
