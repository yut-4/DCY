# Repository Guidelines

## Project Structure & Module Organization

This repository is a C++20 MVP of DCY, which builds a SQLite/FTS5 index of C/C++ source and presents bounded context for a goal. `src/main.cpp` owns the CLI, database, libclang indexing, and retrieval. `src/distiller.hpp` chooses which candidate representations fit a budget; `src/ormt.hpp` renders each chosen representation. `src/mlpack_retrieval.hpp` is an optional experimental lexical-neighbor backend. The architecture proposal is in `DCY-v0.1.md`; `README.md` distinguishes that proposal from implemented behavior. The agent-facing workflow is in `skills/dcy-context/SKILL.md`.

## Build, Test, and Development Commands

Run `cmake -S . -B build`, then `cmake --build build`. The build needs SQLite3, OpenSSL, and libclang development files. Run all CTest checks with `ctest --test-dir build --output-on-failure`; run only the source and indexing smoke check with `ctest --test-dir build -R dcy_smoke --output-on-failure`. The benchmark scoring logic has its own dependency-free check, `ctest --test-dir build -R dcy_arch_scoring --output-on-failure`, which needs no Ollama, database, or network. The retrieval-only pilot is `python3 bench/retrieval.py --dcy build/dcy --repo tests/fixture --db build/retrieval-pilot.sqlite --tasks bench/fixture-tasks.jsonl`. Its gold-symbol recall is not task success. Enable optional mlpack only with `-DDCY_ENABLE_MLPACK=ON` and its dependencies installed.

## Agent Context Workflow

When a C/C++ task benefits from goal-specific repository context, read `skills/dcy-context/SKILL.md`. Index current source, then prefer the bounded MCP stdio intermediary (`mcp/dcy_mcp.py`) when the host supports it; use the CLI fallback otherwise. Request a concrete goal view and exact source by entity ID/generation when needed. Reindex after edits; prior goals then become stale. MCP and CLI budgets are bytes, so the host must count the full prompt with its model tokenizer and reserve output tokens. Do not report this prototype as an infinite transformer window or a validated repair benchmark.
