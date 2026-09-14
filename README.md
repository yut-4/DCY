<p align="center">
  <img
    src="https://github.com/user-attachments/assets/0bf57c3d-7ee5-40a3-9d8e-13fa5f4208e3"
    alt="DCY Logo"
    width="220"
  />
</p>

<h1 align="center">DCY</h1>

<p align="center">
  <strong>Dynamic Context Injection</strong>
</p>

<p align="center">
  Goal-driven context virtualization and token-efficient context delivery for LLMs.
</p>

---

# DCY core MVP


This repository contains the first executable slice of [DCY v0.1](DCY-v0.1.md). It indexes C/C++ definitions and resolved calls with libclang, stores file bytes and graph data in SQLite/FTS5, retrieves context by goal, renders it through ORMT, and returns byte-exact source spans. No LLM is invoked by the core: `context` is the handoff point to a model adapter.

## Modelo conceptual

**DCY no resume el repositorio entero.** Construye una DB dinámica y, para cada goal, produce una vista temporal con la evidencia que parece suficiente para la siguiente inferencia:

```text
Repository → Project Context Image → Session DB → Goal Working Set
           → Context Distiller → ORMT → Physical Tokens → Transformer
```

Formalmente:

$$DB_s = BuildDB(Repo, Scope, Session)$$

$$C_t = Distill(DB_s, G_t, B, M)$$

$$\min_{C_t}\ tokens_M(C_t)\quad\text{sujeto a}\quad Fidelity(C_t,G_t)\ge\tau,\quad tokens_M(C_t)\le B$$

Aquí `B` es el presupuesto físico y `M` incluye el tokenizador. **La fidelidad real no se conoce en línea**: DCY sólo puede usar señales como cobertura de evidencia, confianza de relaciones y disponibilidad de source. El benchmark debe medir retención de hechos necesarios y éxito final; si no, “mínimo suficiente” sería una afirmación sin verificar.

La misma entidad puede ofrecer distintas resoluciones: `SOURCE` (bytes originales), `DETAIL`, `SEMANTIC`, `IR` y `REF` (ID recuperable). Cambiar de goal cambia la selección y la resolución de cada entidad. Esto no equivale a comprimir arbitrariamente un millón de tokens en 500: sólo funciona cuando la evidencia necesaria para el goal es mucho menor que el corpus y puede presentarse o procesarse en pasos acotados. Una tarea que requiere muchos hechos simultáneos puede superar la ventana física.

Los roles propuestos son: **Goal Engine** fija qué evidencia buscar; **Context Distiller** decide qué candidatos y cuánta resolución conservar; **ORMT** expresa cada entidad en el nivel elegido. En el código actual, FTS+grafo recupera candidatos, `src/distiller.hpp` selecciona resolución bajo presupuesto y `src/ormt.hpp` implementa `DETAIL`, compacto y referencia mínima. El `SOURCE` exacto se obtiene con `dcy source`; no hay todavía una capa `SEMANTIC` validada ni selección automática de source por un modelo.

$$TDR = \frac{VT_{goal}}{PT_{injected}}$$

TDR puede ser una métrica descriptiva **si** `VT_goal` se define como los tokens de un conjunto gold de evidencia por tarea. Por sí sola se puede inflar y no mide utilidad. Debe acompañarse de `FidelityRetention` operacional: proporción de hechos gold necesarios preservados y, sobre todo, tasa de resolución de la tarea al mismo costo. Un TDR alto con baja retención es un fallo, no un logro.

## Build

Requirements: C++20 compiler, CMake, SQLite3 development headers, OpenSSL development headers, and libclang development headers/library.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

`mlpack` is an optional **experimental** lexical-neighbor retrieval backend. When mlpack, Armadillo, ensmallen, and cereal are installed, build with:

```sh
cmake -S . -B build-mlpack -DDCY_ENABLE_MLPACK=ON
cmake --build build-mlpack
```

The backend hashes character trigrams of symbol names/paths and uses mlpack KNN. It is not a semantic embedding model; `query-mlpack` merges its candidates with FTS and must be benchmarked against the default before enabling it in a goal loop. Its current one-shot CLI rebuilds the feature matrix for each query, so this path is **not** the scalable ANN design for million-token repositories. The default build has no mlpack dependency. [mlpack's C++ documentation](https://mlpack.org/doc/user/methods/knn.html) describes the KNN API.

## CLI

```sh
build/dcy index /path/to/cpp-repository /path/to/dcy.sqlite
build/dcy goal /path/to/dcy.sqlite "Investigate login returning 403"
build/dcy context /path/to/dcy.sqlite 1 2048
build/dcy finish /path/to/dcy.sqlite 1 resolved
build/dcy query /path/to/dcy.sqlite "login" 512
build/dcy query-fts /path/to/dcy.sqlite "login" 512
build/dcy source /path/to/dcy.sqlite 42 GENERATION 8192
build/dcy record /path/to/dcy.sqlite 1 hypothesis "expiry comparator is inverted" 42
build/dcy query-mlpack /path/to/dcy.sqlite "login" 512
```

`index` prints a generation SHA-256 derived from paths and content hashes. `context` refuses stale goals after a repository update. `source` checks the generation, verifies the stored file's SHA-256, validates the span, and enforces `MAX_BYTES`. `record` accepts `hypothesis`, `fact`, `rejected`, or `result`; a `fact` requires an entity ID. Recent observations are injected by `context` as external state.

The CLI's ORMT budget bounds the emitted context **in bytes**, not model tokens. This is intentional: no tokenizer/model was chosen, so the executable cannot truthfully guarantee a 512-token physical window. ORMT accepts a `Meter` callback and measures each complete candidate prompt, allowing a model adapter to provide its exact tokenizer. That adapter must count the *entire* prompt, reserve output tokens, and reject overflow before inference. The CLI byte budget still makes selection/rendering measurable and lets ORMT choose detailed, compact, or minimal representations. `query` is read-only and does not create a goal.

## MCP intermediary

After indexing, an agent host can launch a local MCP stdio server:

```sh
python3 mcp/dcy_mcp.py --dcy build/dcy --db /path/to/dcy.sqlite \
  --max-context-bytes 2048 --max-source-bytes 4096
```

The host calls `dcy_view(goal,max_bytes)` before inference. DCY creates the goal and returns one bounded ORMT view; small `structuredContent` metadata carries its goal ID and generation. `dcy_source(entity_id,generation,max_bytes)` fetches exact UTF-8 source with provenance when needed. `dcy_record` and `dcy_finish` maintain external state. The DB path is fixed by the host; tools cannot browse arbitrary DBs. The server makes **no model API calls**. The host sends only the selected text view, not raw MCP output or the tool schemas, to the model API and must check the full prompt with the actual tokenizer. A byte cap alone cannot enforce a physical token window. This is an MCP server for a cooperating host, **not** a transparent proxy for arbitrary LLM API traffic. See the [host contract](skills/dcy-context/references/host-integration.md).

## Retrieval pilot

```sh
python3 bench/retrieval.py --dcy build/dcy --repo tests/fixture \
  --db build/retrieval-pilot.sqlite --tasks bench/fixture-tasks.jsonl \
  --budgets 512 1024 2048 --modes query-fts query --repeats 3
```

The JSON output records index time, per-query CLI latency, context bytes, and gold-symbol recall. This is a retrieval ablation, **not** an LLM task-success benchmark. `query-fts` excludes graph expansion; `query` includes it. Add `query-mlpack` to `--modes` only in an mlpack-enabled build. The fixture tasks are a pipeline check, not evidence that DCY beats another method.

The [2026-09-14 exploratory results](bench/results-2026-09-14.md) include an 18-task Tree-sitter retrieval ablation and a small local-model identification pilot. At 512 bytes, FTS and FTS+graph both reached only 44.4% mean target-symbol recall; the model pilot had no exact task matches. These are negative/limited early results, not a claim of superior reasoning or million-token addressability.

## Skill para agentes

La skill [dcy-context](skills/dcy-context/SKILL.md) explica cuándo y cómo un agente usa DCY vía MCP o CLI: crea un goal, pide una vista acotada, recupera source exacto y guarda observaciones. [AGENTS.md](AGENTS.md) la señala a los agentes que trabajan en este repositorio. Para integrar DCY en el bucle de otro agente, sigue el [contrato de host](skills/dcy-context/references/host-integration.md): el host conserva el control del modelo y cuenta los tokens reales; DCY aporta recuperación y representación. La skill vive en este repo y puede copiarse a la ubicación de skills del host si éste no descubre skills locales automáticamente.

## Current boundaries

- The index is a mutable current generation. Changed files are reparsed; unchanged files are skipped. Resolved call relations are rebuilt from stored mentions after updates. Old source versions are not retained; a stale generation is rejected. Immutable historical snapshots remain future work.
- C/C++ are the only indexed languages. libclang resolves calls as far as its parse configuration permits; build flags and project-specific include paths are not yet read from `compile_commands.json`. Unresolved calls are kept as mentions but cannot become graph edges.
- FTS and graph expansion are deterministic. The current policy is one hop, a fixed candidate cap, and a stable tie-break; relevance and source fidelity remain empirical questions.
- The distiller packs coverage-first: it admits as many candidates as possible at the minimal level, then spends leftover budget upgrading them. An earlier greedy packer gave each candidate in turn the richest level that still fit, which let the top candidate consume the budget; a pilot measured that as 44.4% versus 63.9% gold retention at 512 bytes. See [bench/results-2026-09-14.md](bench/results-2026-09-14.md).
- ORMT is a separate renderer in `src/ormt.hpp`. Renderings are name-first, with the entity ID as a trailing `[ref=E<id>]` tag; hosts parse that tag, not a line prefix. Leading with the ID measurably caused a small model to answer with entity IDs instead of symbol names. Minimal lines omit paths and hashes but preserve the reference tag, which resolves to exact source in the current generation. They are lossy representations.
- There is no patch runner, test-result capture, prefetch, embedding store, native tokenization, or validated end-to-end repair benchmark yet. The MCP adapter is stdio-only and handles one request at a time. The [specification](DCY-v0.1.md) distinguishes later components and the required experimental controls.
