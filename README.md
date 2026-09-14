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


```text
Repository → Project Context Image → Session DB → Goal Working Set
           → Context Distiller → ORMT → Physical Tokens → Transformer
```
## DCY Experimental Equation

The original intuition behind DCY came from trying to represent how much useful external context can remain addressable while reducing the physical working set processed by the model.

Start with the simplified quadratic attention term for a context of size `N`:

```math
C \propto N^2
```

If the context can be divided into `G` goal-addressable regions, the idealized working set required for one goal becomes:

```math
K = \frac{N}{G}
```

Therefore:

```math
K^2 = \left(\frac{N}{G}\right)^2
```

DCY then introduces three quantities:

* `VT` — addressable **Virtual Tokens**
* `C_goal` — usefulness or continuity of context for the active goal
* `DB` — effective utility of the dynamic external context database

This leads to the original experimental DCY equation:

```math
\boxed{
E_{DCY}
=
\frac{
VT \cdot C_{goal} \cdot DB
}{
(N/G)^2
}
}
```

The derivation can be summarized as:

```math
N^2
```

↓ goal-addressed partitioning

```math
\left(\frac{N}{G}\right)^2
```

↓ external virtual context, goal relevance, and dynamic database

```math
\boxed{
E_{DCY}
=
\frac{
VT \cdot C_{goal} \cdot DB
}{
(N/G)^2
}
}
```

Equivalently:

```math
\boxed{
E_{DCY}
=
\frac{
VT \cdot C_{goal} \cdot DB \cdot G^2
}{
N^2
}
}
```

This equation is an **experimental heuristic for DCY**, not a claim that DCY changes the asymptotic complexity of the transformer itself.

### Per-goal and session-aggregate efficiency

The equation above is local: it describes one goal and its working set. A stateful DCY session is a trajectory of goals:

```math
\pi = (g_1,g_2,\ldots,g_T)
```

For each step, retain the original local definition:

```math
\boxed{
E_{DCY}(g_t)
=
\frac{VT_t \cdot C_{goal,t} \cdot DB_t}
{(N/G_t)^2}
}
```

The accumulated efficiency of the whole trajectory is then:

```math
\boxed{
E_{DCY}^{\Sigma}(\pi)
=
\sum_{t=1}^{T} E_{DCY}(g_t)
=
\sum_{t=1}^{T}
\frac{VT_t \cdot C_{goal,t} \cdot DB_t}
{(N/G_t)^2}
}
```

`E_DCY(g_t)` is the efficiency of one goal/working set. `E_DCY^Σ(π)` is the total contextual utility produced across a stateful session, long task, or goal trajectory. Because a sum tends to increase when more goals are added, comparisons between sessions should also report the per-goal average:

```math
\boxed{
\bar E_{DCY}(\pi)
=
\frac{1}{T}\sum_{t=1}^{T}E_{DCY}(g_t)
}
```

The Goal Engine is not merely choosing independent queries. Each goal updates state and influences the next one:

```math
S_{t+1}=Update(S_t,g_t,R_t)
```

```math
g_{t+1}=Select(G,S_{t+1})
```

A useful operational interpretation is that goal continuity depends on the current state:

```math
C_{goal,t}=Rel(g_t,S_{t-1})
```

This makes a goal that is unrelated to the evidence already established contribute little to the aggregate, even if its isolated retrieval looks good. In implementation terms, a trajectory might move from `locate → inspect → decide → retrieve source → validate → result`, with observations and validated facts carried through the session.

For optimization, maximizing the raw sum would reward unnecessarily long trajectories. The practical objective therefore subtracts the cost of each step:

```math
\boxed{
J_{DCY}(\pi)
=
E_{DCY}^{\Sigma}(\pi)-\lambda Cost(\pi)
}
```

where a first operational cost model is:

```math
Cost(\pi)=\sum_{t=1}^{T}(PT_t+L_t+A_t)
```

with `PT_t` physical tokens used, `L_t` latency, and `A_t` model/API cost. The intended optimization is consequently:

```math
\boxed{
\pi^*
=
\arg\max_{\pi}
\left[
\sum_{t=1}^{T}E_{DCY}(g_t)-\lambda Cost(\pi)
\right]
}
```

These aggregate equations are experimental definitions, and `src/efficiency.hpp` now provides an isolated metric utility for evaluating them. The current MVP still does not connect this utility to automatic trajectory planning or goal selection, and it does not optimize `J_DCY`; `tests/efficiency_test.cpp` verifies the local equation, sigma aggregate, per-goal average, continuity effect, cost objective, and invalid-input guards. They formalize the distinction between local signal preservation and session-level goal continuity.

### E2 — Goal Trajectory Validation

`bench/trajectory_benchmark.py` is the first offline test of whether `J_DCY` orders candidate goal trajectories like observed utility. It does not implement a planner: the manifest supplies candidate routes and externally declared success labels, while the runner executes each step through the real DCY CLI and derives observable proxies:

```text
C_goal = relevant selected evidence / selected evidence
DB     = relevant selected evidence / required evidence
```

For each route it computes `E_DCY(g_t)`, `E_DCY^Σ(π)`, `J_DCY(π)`, total step cost, and observed success per cost. On the initial authentication fixture, two tasks and six candidate trajectories produced:

```text
best route selected by J_DCY: 2/2 tasks
Spearman rho(J_DCY, observed utility): 0.4928
```

This is a useful but weak first result: `J_DCY` selected the best observed route in both fixture tasks, but `ρ=0.4928` is far from predictive validation. The route labels are manifest inputs in this fixture, not hidden test outcomes, and cost currently uses prompt word count plus normalized latency/API terms rather than billing units. E2 therefore remains an offline methodology test; larger trajectory manifests with test-verified success are required before implementing a Goal Planner.

### E3 — Objective Calibration & Generalization

`bench/objective_calibration.py` scales the offline test to 100 generated tasks and 500 deterministic candidate trajectories (five routes per task), with a stable design/held-out split. It calibrates `λ` only on design and freezes it on held-out. The route success proxy is gold-evidence coverage across the route; it is not hidden-test success.

The comparison is deliberately broader than `J_DCY`:

```text
E_DCY^Σ       accumulated contextual utility
Ē_DCY         per-goal average
-Cost         cheapest-route control
J_DCY         EΣ - λ Cost
J_norm        normalized EΣ - λ normalized Cost
Random        deterministic negative control
```

Results:

| Metric | Design agreement | Design ρ | Held-out agreement | Held-out ρ |
|---|---:|---:|---:|---:|
| `E_DCY^Σ` | 0.283 | 0.8303 | 0.175 | 0.8029 |
| `Ē_DCY` | 0.950 | 0.8713 | **0.975** | **0.8825** |
| `-Cost` | 0.050 | -0.1160 | 0.175 | 0.0004 |
| `J_DCY` | 0.500 | 0.1221 | 0.725 | 0.2683 |
| `J_norm` | 0.567 | 0.5026 | 0.475 | 0.5883 |
| `Random` | 0.217 | -0.0228 | 0.125 | 0.0286 |

The current E3 hypothesis — that `J_DCY` ranks held-out trajectories better than `EΣ`, `-Cost`, and random — is **not supported**. `Ē_DCY` wins this fixture decisively, while `J_DCY` underperforms it (`ρ=0.2683` versus `0.8825` held-out). This does not yet prove that the formula is wrong: the benchmark's observed utility is success divided by cost, while `J_DCY` is an additive utility-minus-penalty objective, and the cost components are normalized proxies. It does prove that the current `λ`/cost geometry cannot be used to justify an automatic planner.

E3 is therefore frozen as a negative/diagnostic result, not an optimization milestone. Before changing the equation, the next experiment should hold the route labels and costs constant while comparing alternative objective geometries — ratio, additive, normalized additive — and use test-verified success on real multi-step tasks. No Goal Planner is implemented.

Run it with:

```sh
python3 bench/objective_calibration.py --dcy build/dcy \
  --db build/corpus-1m.sqlite --tasks bench/corpus-1m-tasks.jsonl \
  --budget-bytes 512 --virtual-tokens 1002132 --partitions 100 \
  --out build/e3-objective-calibration.json
```

### E4 — External Utility Validation

E4 removes route-level success labels from the test. `bench/trajectory_benchmark.py --external` retrieves the entity references, calls `dcy source`, and checks independent source assertions such as `return 403` and `active && !expired`. The metric never participates in defining success.

On the authentication fixture, two tasks and six routes produced:

```text
J_DCY vs external success/cost: Spearman rho = 0.5429
best route agreement:            2/2
```

This improves on the E2 diagnostic but remains a tiny fixture, not predictive validation. The external verifier changes the result: a route with distractor steps can still succeed if its final step retrieves all required evidence. E4 is therefore a methodology checkpoint showing that success can be measured outside the objective; it does not justify a planner yet. The next valid scale-up is 50–100 trajectories with independently verified tests or source assertions, then compare `EΣ`, `Ē`, ratio, additive `J`, normalized `J`, cost-only and random under a frozen design/held-out split.

### E5 — External Objective Comparison

E5 applies the external-success boundary to the larger objective comparison: 100 tasks, 500 deterministic trajectories, 300 design and 200 held-out. Each required symbol must be retrieved, resolved with `dcy source`, and pass an independently specified source assertion. `λ` and normalization are selected on design only.

| Metric | Design agreement | Design ρ | Held-out agreement | Held-out ρ |
|---|---:|---:|---:|---:|
| `E_DCY^Σ` | 0.233 | 0.8313 | 0.175 | 0.8032 |
| `Ē_DCY` | **1.000** | 0.8732 | **0.975** | **0.8827** |
| `-Cost` | 0.067 | -0.1037 | 0.150 | 0.0018 |
| `J_DCY` | 0.633 | 0.1875 | 0.725 | 0.2691 |
| `J_norm` | 0.733 | 0.4706 | 0.675 | 0.5288 |
| `Random` | 0.250 | -0.0116 | 0.125 | 0.0313 |

`H_E5` is supported on this benchmark: `Ē_DCY` ranks externally validated trajectories better than `EΣ`, cost-only, additive `J`, normalized `J`, and random. Held-out `ρ=0.8827` and 0.975 route-choice agreement are strong for this fixture, but not universal validation: the tasks are generated, the source assertions are mechanical name/body checks rather than hidden tests, and all routes share the same cost proxy. The practical result is a constraint-first direction — require external correctness for evaluation, then compare efficiency among valid routes — rather than wiring the current `J_DCY` into a planner.

This is retrospective trajectory ranking, not prospective planning. In evaluation, `Success_external(π)` may be observed after executing a preconstructed route. A future planner cannot use that value before execution; it must estimate the next step from current state:

```math
\boxed{
\pi^* = \arg\max_{\pi}\widehat{\bar E}_{DCY}(\pi)
\quad\text{s.t.}\quad Cost(\pi)\le B
}
```

External success is then measured after execution:

```math
Success_{external}(\pi^*)\in\{0,1\}
```

The first prospective policy experiment is reserved for **E6 — Prospective Planning**: generate candidate next goals from `S_t`, estimate `\widehat{\bar E}_{DCY}(g\mid S_t)`, apply resource limits, execute one choice, update state, and repeat. Compare against fixed-plan, greedy-relevance, and random policies under identical model, task, token and step budgets, using `Success@Budget`. No E6 planner is implemented yet.

The first E5 run exposed and fixed a harness boundary: `dcy source` correctly rejected a gold span larger than the verifier's initial 4 KiB limit. The rerun uses an explicit 1 MiB benchmark limit and records the original failure rather than hiding it.

Reproduce:

```sh
python3 bench/objective_calibration.py --dcy build/dcy \
  --db build/corpus-1m.sqlite --tasks bench/corpus-1m-external-tasks.jsonl \
  --budget-bytes 512 --virtual-tokens 1002132 --partitions 100 --external \
  --out build/e5-external-objective.json
```


The denominator

```math
\left(\frac{N}{G}\right)^2
```


represents the original intuition of reducing the physical working set through goal-addressed partitioning, while `VT`, `C_goal`, and `DB` represent the useful external context that DCY attempts to keep addressable.

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

The [2026-09-14 exploratory results](bench/results-2026-09-14.md) include the full ablations, raw checkpoint artifacts, and scale pilots. The current experimental checkpoint is **E1**: coverage-first packing and name-first ORMT changed injected recall from 0.500 to 0.611, answer recall from 0.111 to 0.500, and `U(model,renderer)` from 0.222 to 0.818 on the same nine-task 0.5B pilot; candidate recall stayed at 0.778, so the gain came from propagation rather than retrieval. `ctest` passes 4/4. E1 artifacts are under `bench/checkpoints/`.

## 1M virtual tokens / 500-token model context

A scale pilot built a **1,002,132-token** corpus using `cl100k_base`, indexed 431 C/C++ files into an 8.5 MB SQLite database, and served it through DCY to `qwen2.5:0.5b-instruct` configured with Ollama `num_ctx=500`. The physical DCY view remained bounded to 512 bytes; its mean model input was 334.2 tokens and the maximum recorded input plus requested output was 406 tokens. On 20 held-out generated tasks, candidate recall was 0.875, injected recall 0.725, and answer recall 0.525. Five generations hit the intentionally conservative 64-token output cap, so this is a bounded-context/transport result, **not** a claim that arbitrary tasks can be solved in 500 tokens.

Matched-scale runs with the same 16 tasks across 100k → 1M VT measured prompt growth of 0.5% at a 512-byte budget and 6.6% at 2,048 bytes. A distractor-controlled run kept the 100k Tree-sitter base files and gold tasks fixed while adding unrelated Redis/curl/SQLite files: the 512-byte prompt grew 3.9%, but recall fell from 0.688 to 0.625 (`DR=0.908`). Thus DCY currently demonstrates **scale isolation of physical prompt size**, while distractor resistance remains an open retrieval problem. VCR is an addressability ratio, not comprehension.

These are experimental results, not evidence of infinite context, superior reasoning, or validated code repair. See the [reproducible results](bench/results-2026-09-14.md), `bench/matched_scale.py`, `bench/build_distractor_corpus.py`, and `bench/scale.py`.

## Skill para agentes

The skill [dcy-context](skills/dcy-context/SKILL.md) explains when and how an agent uses DCY via MCP or CLI: it creates a goal, requests a scoped view, retrieves the exact source, and saves observations. [AGENTS.md](AGENTS.md) points it out to the agents working in this repository. To integrate DCY into another agent's loop, follow the [host contract](skills/dcy-context/references/host-integration.md): the host keeps control of the model and counts the actual tokens; DCY provides retrieval and representation. The skill lives in this repo and can be copied to the host's skills location if it doesn't automatically discover local skills.

## Current boundaries

- The index is a mutable current generation. Changed files are reparsed; unchanged files are skipped. Resolved call relations are rebuilt from stored mentions after updates. Old source versions are not retained; a stale generation is rejected. Immutable historical snapshots remain future work.
- C/C++ are the only indexed languages. libclang resolves calls as far as its parse configuration permits; build flags and project-specific include paths are not yet read from `compile_commands.json`. Unresolved calls are kept as mentions but cannot become graph edges.
- FTS and graph expansion are deterministic. The current policy is one hop, a fixed candidate cap, and a stable tie-break; relevance and source fidelity remain empirical questions.
- The distiller packs coverage-first: it admits as many candidates as possible at the minimal level, then spends leftover budget upgrading them. An earlier greedy packer gave each candidate in turn the richest level that still fit, which let the top candidate consume the budget; a pilot measured that as 44.4% versus 63.9% gold retention at 512 bytes. See [bench/results-2026-09-14.md](bench/results-2026-09-14.md).
- ORMT is a separate renderer in `src/ormt.hpp`. Renderings are name-first, with the entity ID as a trailing `[ref=E<id>]` tag; hosts parse that tag, not a line prefix. Leading with the ID measurably caused a small model to answer with entity IDs instead of symbol names. Minimal lines omit paths and hashes but preserve the reference tag, which resolves to exact source in the current generation. They are lossy representations.
- There is no patch runner, test-result capture, prefetch, embedding store, native tokenization, or validated end-to-end repair benchmark yet. The MCP adapter is stdio-only and handles one request at a time. The [specification](DCY-v0.1.md) distinguishes later components and the required experimental controls.
