#!/usr/bin/env python3
"""DCY context runtime HTTP API — Ollama consumes DCY through here.

Endpoints (all JSON):
  POST /session          {"budget":512,"max_goals":4,"model":"qwen2.5:0.5b-instruct","use_stub":false} -> {"session":"1",...}
  POST /chat             {"session":"1","message":"why does ...?"} -> {"answer":"...","usage":{...}}
  GET  /session/{id}     -> {"session":"1","config":{...},"history":[...]}
  DELETE /session/{id}   -> {"deleted":"1"}
  GET  /health           -> {"ok":true}

The server owns one shared DCYCli (C++ `dcy` binary). Each chat runs the
full v2 paging loop: DCY goal engine decomposes the task, materializes pages
under budget, the LLM (Ollama, plain text in/out) observes each page, DCY
records the observation and advances. The LLM never sees gold, scores, or VT.

Run:
  python3 api/dcy_server.py --db build/corpus-1m.sqlite --port 8765
Test:
  curl -s -X POST localhost:8765/session -H 'Content-Type: application/json' \\
    -d '{"use_stub":true}'
  curl -s -X POST localhost:8765/chat -H 'Content-Type: application/json' \\
    -d '{"session":"1","message":"what functions handle tree edit propagation?"}'
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    from dcy_consumer import DCYCli, make_ollama_llm, run_paging_runtime, stub_llm
except ImportError:  # allow `python3 api/dcy_server.py` from repo root
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dcy_consumer import DCYCli, make_ollama_llm, run_paging_runtime, stub_llm


class RuntimeState:
    def __init__(self, dcy: DCYCli, default_model: str, ollama_base: str,
                 default_budget: int, default_max_goals: int, default_stub: bool):
        self.dcy = dcy
        self.default_model = default_model
        self.ollama_base = ollama_base
        self.default_budget = default_budget
        self.default_max_goals = default_max_goals
        self.default_stub = default_stub
        self.lock = threading.Lock()
        self.counter = itertools.count(1)
        self.sessions: dict[str, dict] = {}


def make_llm_for_session(state: RuntimeState, cfg: dict):
    if cfg.get("use_stub"):
        return stub_llm
    return make_ollama_llm(model=cfg.get("model", state.default_model),
                           base_url=cfg.get("ollama_base", state.ollama_base))


class Handler(BaseHTTPRequestHandler):
    state: RuntimeState  # set by serve()

    def log_message(self, format, *args):  # quieter logs
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 1_000_000:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send(200, {"ok": True})
            return
        if parsed.path.startswith("/session/"):
            sid = parsed.path[len("/session/"):].strip("/")
            with self.state.lock:
                sess = self.state.sessions.get(sid)
            if sess is None:
                self._send(404, {"error": "unknown session"})
                return
            self._send(200, {"session": sid, "config": sess["config"],
                             "history": sess["history"]})
            return
        self._send(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/session/"):
            sid = parsed.path[len("/session/"):].strip("/")
            with self.state.lock:
                gone = self.state.sessions.pop(sid, None)
            if gone is None:
                self._send(404, {"error": "unknown session"})
                return
            self._send(200, {"deleted": sid})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/session":
            data = self._read_json()
            with self.state.lock:
                sid = str(next(self.state.counter))
                cfg = {
                    "budget": int(data.get("budget", self.state.default_budget)),
                    "max_goals": int(data.get("max_goals", self.state.default_max_goals)),
                    "model": str(data.get("model", self.state.default_model)),
                    "ollama_base": str(data.get("ollama_base", self.state.ollama_base)),
                    "use_stub": bool(data.get("use_stub", self.state.default_stub)),
                }
                cfg["budget"] = max(128, min(cfg["budget"], 8192))
                cfg["max_goals"] = max(1, min(cfg["max_goals"], 8))
                self.state.sessions[sid] = {"config": cfg, "history": []}
            self._send(200, {"session": sid, **cfg})
            return
        if parsed.path == "/chat":
            data = self._read_json()
            sid = str(data.get("session", ""))
            message = str(data.get("message", "")).strip()
            if not sid or not message:
                self._send(400, {"error": "need {session, message}"})
                return
            with self.state.lock:
                sess = self.state.sessions.get(sid)
            if sess is None:
                self._send(404, {"error": "unknown session"})
                return
            cfg = sess["config"]
            llm = make_llm_for_session(self.state, cfg)
            try:
                result = run_paging_runtime(
                    task=message[:2000],
                    dcy=self.state.dcy,
                    llm=llm,
                    budget=cfg["budget"],
                    max_goals=cfg["max_goals"],
                )
            except Exception as exc:  # noqa: BLE001 — report backend failures as JSON
                self._send(502, {"error": f"paging loop failed: {exc}"[:300]})
                return
            entry = {"message": message, "answer": result.answer,
                     "pages": result.total_pages,
                     "physical_tokens": result.total_physical_tokens,
                     "unique_entities": result.unique_entities}
            with self.state.lock:
                sess["history"].append(entry)
            self._send(200, {
                "answer": result.answer,
                "usage": {
                    "physical_tokens": result.total_physical_tokens,
                    "context_pages": result.total_pages,
                    "goals_completed": result.goals_completed,
                    "unique_entities": result.unique_entities,
                    "wall_time_ms": round(result.wall_time_ms, 1),
                },
            })
            return
        self._send(404, {"error": "not found"})


def main() -> None:
    ap = argparse.ArgumentParser(description="DCY context runtime HTTP API")
    ap.add_argument("--dcy", type=Path, default=Path("build/dcy"))
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--default-model", default="qwen2.5:0.5b-instruct")
    ap.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--max-goals", type=int, default=4)
    ap.add_argument("--stub", action="store_true",
                    help="default new sessions to the stub LLM")
    args = ap.parse_args()
    if not args.dcy.is_file():
        ap.error(f"DCY binary not found: {args.dcy}")
    if not args.db.is_file():
        ap.error(f"Database not found: {args.db}")

    dcy = DCYCli(args.dcy, args.db)
    Handler.state = RuntimeState(dcy, args.default_model, args.ollama_base,
                                 args.budget, args.max_goals, args.stub)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dcy-server listening on {args.host}:{args.port} "
          f"db={args.db} model={args.default_model}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
