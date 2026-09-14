---
name: dcy-context
description: Use DCY through its bounded MCP stdio server or CLI to give an agent goal-specific context from a C/C++ repository, retrieve exact source by entity ID, and persist evidence. Use for repository analysis or code repair when DCY is available; not for unsupported languages or as a substitute for model token accounting.
---

# DCY context for coding agents

DCY's SQLite index is external knowledge. The text returned by `context` is a temporary view for one goal, not a copy of the repository. Keep the agent's task and authority unchanged.

From the DCY project root, build once with `cmake -S . -B build && cmake --build build`. For a C/C++ target repository, use an index path under its build or temporary directory, then run `build/dcy index TARGET_REPO INDEX_DB`. Index again after source changes; a changed generation makes prior goals and entity references stale.

Prefer the MCP host integration: launch `python3 mcp/dcy_mcp.py --dcy build/dcy --db INDEX_DB`. The host calls `dcy_view` with one concrete goal and `max_bytes`, then injects only its `content[0].text` after tokenizer accounting. Its `structuredContent` carries goal ID, generation and byte counts for orchestration, **not** a second prompt payload. If a decision needs original code, the host calls `dcy_source` with `entity_id`, generation and a bounded `max_bytes`; this checks generation and stored source hash. Entity IDs appear in the view as a trailing `[ref=E<ID>]` tag on each rendered line, never as a line prefix; parse that tag to get the `entity_id`. The agent may cite `[ref=E<ID>]` and explain why evidence is missing, but should not search files, SQL or chunks through MCP.

The host can use `dcy_record` and `dcy_finish` for external state; `fact` requires an entity reference, but DCY does not prove the claim. Do not record unverified interpretations as confirmed facts. Keep maintenance tools and MCP schemas out of the model prompt unless a specific task needs them. If MCP is unavailable, use the equivalent CLI `goal`, `context`, `source`, `record`, and `finish` commands described in [host integration](references/host-integration.md).

Both MCP and CLI budgets are **bytes**, not physical model tokens. Before an API call, the host must count the complete prompt with the actual tokenizer and reserve output tokens; reduce `max_bytes` or stop on overflow. DCY currently indexes C/C++ only; mlpack retrieval is optional and experimental. For wiring and limits, read [host integration](references/host-integration.md) and [README](../../README.md).
