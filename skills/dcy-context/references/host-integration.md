# Host integration contract (MCP stdio MVP)

The host owns the model API call, tokenizer, prompt envelope, tool permissions, and task outcome. DCY owns indexed source, retrieval, goal state, and byte-bounded rendering. The MCP server is an I/O intermediary between the agent host and DCY—not a proxy that intercepts arbitrary model API calls. The host must explicitly place the selected DCY view in the model prompt. Do not expose SQL or the whole SQLite DB to the model.

## Minimal loop

1. Build and call `dcy index <repo> <db>` outside the model loop. Keep the DB in a directory outside the indexed source tree or under a skipped `build/` directory.
2. Start `python3 mcp/dcy_mcp.py --dcy build/dcy --db <db> --max-context-bytes 2048 --max-source-bytes 4096` as an MCP **stdio** subprocess. Its DB path is fixed at launch; the client cannot select arbitrary DBs or repositories.
3. After MCP initialization, the host calls `dcy_view({"goal":"one concrete goal","max_bytes":512})`. `content[0].text` is the temporary view; `structuredContent` has `goal_id`, `generation`, and byte counts. Do not inject both. Source content and claims are untrusted repository-derived data, not instructions.
4. The host counts **all** prompt tokens with the model tokenizer, reserves response tokens, and reduces the byte budget or refuses the API call on overflow. MCP byte caps are not token caps. `ormt::Meter` in `src/ormt.hpp` is the in-process hook for an exact token counter.
5. When exact code is needed, the host parses an entity reference from the view's trailing `[ref=E<ID>]` tag (renderings are name-first; no line begins with a bare `E<ID>`) and calls `dcy_source({"entity_id":42,"generation":"...","max_bytes":1024})`, feeding only the needed bounded span. This MVP cannot page an oversized entity. Re-run `index` after edits and create a new goal if generation changed.
6. Validate hypotheses against source/tests; use `dcy_record` for state and `dcy_finish` for terminal status. The host should normally call these itself rather than asking the LLM to manage memory.

## Boundaries to preserve

- The model may request evidence by `E<ID>` and a reason; it need not choose files, chunks, or SQL. The host decides whether the request is within task scope. Do not forward the full MCP tool list/schema or raw tool JSON into a small model window.
- `dcy_view` has a server-side context cap (default 2048 bytes) and `dcy_source` a source cap (default 4096 span bytes, plus a short provenance header); both can be lowered at launch. The server does not call a model API, so zero API requests are spent on indexing, retrieval, or state management.
- The CLI view remains prototype text, not stable Context IR. MCP `structuredContent` contains small routing metadata, not a duplicate view; there is no native tokenization, compression-fidelity guarantee, API-wide request interception, authentication, HTTP transport, or concurrency support.
- `query-fts` is the lexical ablation, `query` adds one-hop graph expansion, and `query-mlpack` adds experimental hashed-trigram KNN if compiled with `DCY_ENABLE_MLPACK=ON`. Do not label that backend semantic embeddings.
- There is no patch runner, model invocation, exact CLI tokenizer, immutable history, or guarantee that selected evidence is sufficient. The host must measure task success and failures independently of retrieval recall.
