<p align="center">
  <img
    src="https://github.com/user-attachments/assets/0bf57c3d-7ee5-40a3-9d8e-13fa5f4208e3"
    alt="DCY Logo"
    width="220"
  />
</p>

<h1 align="center">DCY</h1>

<p align="center">
  <strong>change acronym cause a random guy remember me this exist :v</strong>
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

### Finite-horizon action value

A trajectory score evaluates a route after it exists. A prospective selector needs a different quantity: expected progress after taking an action from the current state. The transition and one-step progress are:

```math
\boxed{
S_{t+1}=T(S_t,a_t,o_t)
}
```

```math
\boxed{
\Delta\Phi_t
=
\Phi(S_{t+1})-\Phi(S_t)
}
```

For expected per-step progress deltas `ΔΦ`, a finite-horizon evaluator is:

```math
\boxed{
Q_{DCY}^{(h)}(a_t\mid S_t)
=
\mathbb E
\left[
\sum_{k=0}^{h-1}
\gamma^k\Delta\Phi_{t+k}
\;\middle\vert\;
S_t,a_t
\right]
}
```

The isolated utility `finite_horizon_q` in `src/efficiency.hpp` evaluates a supplied sequence of expected deltas; it is a discounted-return calculator and does not generate futures. The same header now contains a callback-based prospective evaluator:

```math
\boxed{
Q_{DCY}^{(h)}(S_t,a_t)
=
\mathbb E_{o_t\sim P(\cdot\mid S_t,a_t)}
\left[
\Delta\Phi(S_t,a_t,o_t)
+
\gamma V_{DCY}^{(h-1)}(S_{t+1})
\right]
}
```

with:

```math
\boxed{
V_{DCY}^{(h)}(S)
=
\max_{a\in\mathcal A(S)}Q_{DCY}^{(h)}(S,a)
}
```

and `S_{t+1}=T(S_t,a_t,o_t)`, `V^(0)=0`. `prospective_q` receives action generation, outcome probabilities, transition, progress and constraint callbacks; it knows neither SQLite nor an LLM and does not estimate outcomes by itself. `select_best_action` performs deterministic argmax over scores. The callbacks are the future E6 experiment: no callback may use gold future evidence or success observed after execution.

`tests/efficiency_test.cpp` verifies both levels. With immediate deltas `[0.30,0.02,0]`, horizon one prefers the immediate action; with unlocking deltas `[0.05,0.50,0.30]`, horizon two and `γ=0.8` select the initially weaker action because its Bellman value is `0.45` versus `0.316`. This is a tested finite-horizon evaluator, not evidence that DCY can currently construct useful futures or run an autonomous planner.

The planned prospective decision is therefore distinct from retrospective trajectory ranking:

```math
\boxed{
a_t^*
=
\arg\max_{a\in\mathcal A(S_t)}
\widehat Q_{DCY}^{(h)}(a\mid S_t)
}
```

The hat matters: `C_goal`, `DB`, future deltas and reachability must be estimated from information available at `S_t`, without using external success observed after execution. E6 will first compare one-step action selection and state updates; short-horizon lookahead is a later ablation.

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

### E5 — External Objective Comparison — retrospective benchmark later found to contain target leakage and tie-order bias

E5 applies the external-success boundary to the larger objective comparison: 100 tasks, 500 deterministic trajectories, 300 design and 200 held-out. Each required symbol must be retrieved, resolved with `dcy source`, and pass an independently specified source assertion. `λ` and normalization are selected on design only.

**Two bugs were found in `bench/objective_calibration.py` after the fact, confirmed against the code and re-run, not assumed:**

1. **Target leakage.** `C_goal` and `DB` are computed as `relevant/selected` and `relevant/required` where `relevant = selected ∩ task["gold"]` (`objective_calibration.py:138,153-156`), and `success` is computed from the same `task["gold"]` set (`union >= required`, line 167). `sigma`/`Ē_DCY` and `success` are therefore both direct functions of the same gold labels, not independent signals — a high correlation between them is partly tautological and does not show the metric predicts success from information a planner would actually have at decision time.
2. **Tie-order bias.** `agreement()` originally used Python's `max(candidates, key=...)`, which returns the first maximal element in iteration order. Routes are always generated in the fixed order `focused, redundant, distracted, cheap-wrong, broad`, so ties on either the score side or the observed-utility side were silently resolved in favor of `focused` every time. This has been fixed: `agreement()` now reports three separate numbers — `strict_agreement` (single undisputed winner matches), `tie_aware_agreement` (any tied winner overlaps, the old number's tie-crediting behavior made explicit), and `expected_agreement_random_tiebreak` (probability of matching under a uniform random tie-break).

Re-running the same fixture with the tie fix (`bench/checkpoints/E5-external-objective-tiefix.json`):

| Metric | Design strict | Design tie-aware | Design ρ | Held-out strict | Held-out tie-aware | Held-out ρ |
|---|---:|---:|---:|---:|---:|---:|
| `E_DCY^Σ` (sigma) | 0.150 | 0.383 | 0.8266 | 0.100 | 0.300 | 0.7946 |
| `Ē_DCY` (average) | **0.000** | 1.000 | 0.8701 | **0.025** | 0.975 | 0.8769 |
| `-Cost` | 0.400 | 0.400 | -0.0819 | 0.400 | 0.400 | -0.0114 |
| `J_DCY` | 0.883 | 0.883 | 0.1862 | 0.925 | 0.925 | 0.2654 |
| `J_norm` | 0.917 | 0.917 | 0.4665 | 0.850 | 0.850 | 0.5235 |
| `Random` | 0.533 | 0.533 | -0.0117 | 0.375 | 0.375 | 0.0389 |

The originally reported "0.975 held-out agreement" for `Ē_DCY` was `tie_aware_agreement`, effectively `focused` winning on tie by construction: `strict_agreement` for the same metric is **0.025**, meaning `Ē_DCY` almost never uniquely picks the observed-best route once ties are not credited to it. `J_DCY` and `J_norm`, which include the cost term and rarely tie, keep essentially the same strict and tie-aware numbers (0.883–0.925), so their earlier agreement figures were not inflated by this bug — but their ρ against `observed_utility` is far weaker (0.19–0.53) than `Ē_DCY`'s reported 0.87–0.88, and that ρ is itself compromised by leakage (bug 1). No metric here should be read as validated route selection.

`H_E5` is **not** supported by this benchmark once both bugs are accounted for: the strong `Ē_DCY` numbers were a mix of gold-label leakage inflating ρ and tie-order bias inflating agreement. The corrected, still-imperfect picture is that `J_DCY`/`J_norm` (which do use tie-safe scores because cost breaks ties) show moderate strict agreement with weak-to-moderate ρ, while `Ē_DCY` shows strong ρ but that ρ is not independently earned and its strict agreement collapses. The practical result is unchanged in direction — require external correctness for evaluation, then compare efficiency among valid routes — but the specific numeric claim of "`Ē_DCY` outperforms" is retracted pending a leakage-free feature set (E6.1's observable-only state, which does not read `task["gold"]`).

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

### E6.0 — Audit before prospective planning

Before building a selector, `bench/e6_audit.py` audits E5 at the unit where a planner would actually decide: **within each task**, not across all routes from all tasks. This prevents global correlation from being inflated by between-task difficulty. It also checks split integrity, routes per task, ties, tasks with no successful route, and future-information leakage. The leakage result is derived from the score fields present in the artifact; it is not a hard-coded pass/fail assertion.

The audit found:

```text
tasks:                         100
trajectories:                  500
routes per task:               5 for all tasks
split conflicts:               0
success-free tasks:            34/100
```

Within-task results against externally checked success-per-cost utility:

| Score | Mean within-task ρ | Median within-task ρ | Winner agreement |
|---|---:|---:|---:|
| `E_DCY^Σ` | 0.4391 | 0.6708 | 0.350 |
| `Ē_DCY` | 0.6164 | 0.9487 | **0.990** |
| `-Cost` | -0.0249 | 0.0000 | 0.340 |

The earlier global E5 correlation for `Ē_DCY` was 0.8827; the within-task mean is 0.6164. The difference is a methodological warning, not a contradiction: the global statistic included between-task variation, while a selector needs within-task discrimination. `Ē_DCY` still chooses an observed winning route in 99% of tasks when ties receive credit, but its correlation is substantially weaker once the task is held constant. The 34 tasks with no successful route also cannot provide a meaningful winner comparison and must remain in the denominator rather than being silently filtered. The mandatory `always-focused` baseline succeeds on 27/40 held-out tasks (67.5%) and agrees with the observed best utility on 38/40 (95%). The E5/E6 route order is therefore not a fair claim of broad planning gain: `Ē_DCY` reaches 39/40 agreement with tie credit, but the direct improvement over always-focused is only one held-out task under this artifact.

This changes E6's order. E6.1 must first define a prospective action environment: concrete actions (`search`, `expand`, `read_source`, `increase_resolution`, `verify`, `finish`), state fields for pending obligations, evidence, hypotheses, contradictions, history and remaining resources, plus a candidate generator that uses only the current state, task and accessible index. E6.2 then compares fixed-plan, immediate relevance, marginal obligation coverage, `\widehat{\bar E}_{DCY}`, and random under identical budgets. The selector is evaluated after execution with:

```math
Success@B
=\frac{1}{n}\sum_{i=1}^{n}
\mathbf{1}[solved_i \land usage_i\preceq B]
```

All attempts remain in the denominator; candidate generation, scoring calls, retries, source reads and model calls count against independent token, call, latency and step limits. Only after that audit passes should a greedy prospective selector be implemented. A short-horizon planner is a later experiment, not part of initial E6.

Run the audit with:

```sh
python3 bench/e6_audit.py --e5 build/e5-external-objective.json \
  --out build/e6-0-audit.json
```

### E6.1 — Prospective environment contract

The Bellman evaluator is now generic, but the planner world remains explicit and domain-independent in `src/planning.hpp`. Its minimum state contains:

```text
obligations       pending work and statuses
evidence          entity, generation, source span, claim, confidence
hypotheses        unverified claims plus supporting entities
contradictions    incompatible evidence awaiting resolution
history           actions, repetition and newly produced evidence
remaining        physical tokens, calls, steps and latency
```

Actions are closed values — `Search`, `Expand`, `ReadSource`, `Verify`, `IncreaseResolution`, `Finish` — rather than unconstrained text. Each action carries a vector `ResourceCost` over physical tokens, model calls, latency and steps; it is legal only when it fits the corresponding remaining `Budget` component by component. `consume()` is the tested budget transition primitive.

The environment and the selector have separate outcome contracts:

```text
EnvironmentOutcome  = what actually happened after execution
EstimatedOutcome    = what the selector predicted before execution
```

Only estimated outcomes may enter `prospective_q`; external success, hidden assertions and future gold evidence stay outside the selector. A hypothesis starts as `Unverified`; model output cannot silently become confirmed evidence. The prospective callbacks must use only `S_t` and `a_t` for estimated outcomes.

`prospective_q` enforces the current mathematical contract: `gamma ∈ [0,1]` and finite, legal actions only, finite non-negative progress and outcome probabilities, probabilities summing to one, and at least one modeled outcome. `finite_horizon_q`, `local`, costs and action scores reject non-finite values such as `NaN` and `Inf`. A terminal state or state with no legal next action contributes zero future value; finite horizon guarantees termination. Budget propagation belongs in the callback transition, which must decrement `State.remaining`. Memoization by `(StateHash,horizon)` is deliberately not present until the concrete state has a stable hash. The test also covers a genuinely probabilistic action with two valid outcomes, not only deterministic branches.

E6.1 is therefore a contract and evaluator test, not a planner result. The remaining empirical objects are `\widehat P(o\mid S,a)` and `\Phi(S)`. E6.2 must first compare next-action policies under a common candidate generator — fixed, immediate relevance, marginal obligation coverage, estimated `Q`, and random — and record unavailable necessary actions separately from bad selections.

### E6.2 — Softmax action selection (ready primitive, not a planner default)

`select_best_action` performs deterministic argmax over `Q_DCY^(h)`. Softmax is a ready, tested selection policy on top of the same score, not a replacement for it:

```math
\pi_{DCY}(a\mid S_t)=\frac{e^{\widehat{Q}_{DCY}^{(h)}(S_t,a)/\tau}}{\sum_{a'\in A(S_t)}e^{\widehat{Q}_{DCY}^{(h)}(S_t,a')/\tau}}
```

The motivation is that argmax collapses near-equal candidate actions to a single winner even when `Q_hat` is noisy, while a low-`\tau` softmax stays close to greedy and a high-`\tau` softmax flattens the distribution and admits exploration. As `\tau\to 0`, `argmax_a \pi_{DCY}(a\mid S_t) \to argmax_a \widehat{Q}_{DCY}^{(h)}(S_t,a)`, so softmax is a strict generalization of the existing selector rather than a divergent one, but that limit equivalence is not itself evidence that softmax helps at any finite `\tau`.

This is sequenced after, not inside, E6.2's baseline comparison: `select_best_action` stays the greedy default, and softmax is the stochastic alternative the E6.2-A harness invokes explicitly. The primitive is ready today — `src/efficiency.hpp` ships `softmax_distribution`, `sample_from_distribution`, and `softmax_select_action`, all covered by `tests/efficiency_test.cpp::test_softmax_action_distribution` (build + `ctest --test-dir build` green). What is still open is the empirical comparison, not the code: an E6 prospective run must report `Success@Budget` for at least argmax `Q`, softmax `Q` at more than one `tau`, random, immediate relevance, and marginal obligation coverage, on the same tasks and budgets. The open questions are whether softmax improves `Success@Budget` over plain argmax, what `tau` generalizes rather than overfits the design split, whether stochastic selection helps specifically when `Q_hat` is noisy, whether it hurts on deterministic tasks where one action strictly dominates, and whether it is worth the added variance at all compared to the simpler baselines already in E6.2's list.

Ready status: `softmax_distribution` normalizes with max-subtraction for numerical stability, `sample_from_distribution` takes a caller-supplied uniform draw (seeded RNG in harnesses, no hidden random state), and `softmax_select_action` composes both for one-shot candidate selection. None of the three overrides `select_best_action` or any CLI path — the harness chooses argmax vs softmax explicitly per run. `tests/efficiency_test.cpp::test_softmax_action_distribution` checks that the distribution sums to one, that a near-zero `tau` reproduces `select_best_action`'s winner, that a large `tau` flattens toward uniform, and that non-finite/non-positive temperatures and out-of-range draws are rejected the same way the rest of this header rejects invalid input.

### E6.2 candidate — h-explosion mitigation (pruning, memoization, receding horizon)

`prospective_q`'s branching is `O(actions × outcomes)^h`. With 10 actions and 3 outcomes each (`b≈30`), a synthetic benchmark using the same branching shape (not a DCY task) shows the outcome-model call count at `h=1..4`:

```text
EXACT      h=1  calls=1       h=2  calls=31      h=3  calls=931      h=4  calls=27931
```

which matches the `30^(h-1)`-ish growth this section warned about — not an estimate, a measured call count from a synthetic branching-30 tree. Three independent primitives exist in `src/efficiency.hpp`/`src/planning.hpp` to address this; none is wired into a planner default:

- `top_k_actions(state, actions, cheap_score, k)` — prunes the candidate action set with a caller-supplied cheap heuristic *before* Bellman recursion runs. On the same synthetic tree, pruning to the top 4 actions per node cuts `h=4` calls from 27,931 to 1,885 (14.8× fewer), but the resulting value estimate differs from the exact one (1.166 vs 1.284 in that run) — pruning is not free, and its effect on `Success@Budget` must be measured, not assumed harmless, per the earlier note on `prune_outcomes`.
- `prune_outcomes(outcomes, keep_threshold)` — keeps the highest-probability outcomes until their cumulative probability reaches `keep_threshold`, drops the rest, and rescales the kept set back to sum to one (`prospective_q` requires normalized probabilities). Always keeps at least one outcome.
- `memoized_prospective_q(..., state_hash, action_hash, cache)` — a caching wrapper around the same Bellman recursion, keyed by `(StateHash, ActionHash, horizon)` via `combine_hash`. On the synthetic tree, memoization alone reduced `h=4` calls from 27,931 to 16,161 (~42%); the gain depends entirely on how often the caller's `state_hash` collides across different action sequences, which for a real DCY state (E6.1's `State`, still without a stable hash) is unknown until measured. `cache.size()` after a call is a usable `NodesExpanded` proxy for the E6.2-A experiment below.
- `dcy::planning::drop_no_progress_repeats(history, candidates, action_equal)` — removes a candidate action from the generator's output if the same action already ran in `history` and produced no new evidence, using a caller-supplied equality (e.g. same `ActionType` and `target`). This targets exactly the `SEARCH foo / SEARCH foo / SEARCH foo` loop pattern, not general repetition.

All four are tested in `tests/efficiency_test.cpp` (`test_action_pruning_before_bellman`, `test_outcome_pruning_by_cumulative_probability`, `test_memoized_prospective_q_matches_exact`, `test_drop_no_progress_repeats`); the memoization test explicitly checks the memoized and unmemoized Bellman value agree exactly, not just that the cache fills.

The architectural direction implied by these numbers — and not yet built — is receding-horizon (MPC-style) planning: keep `h` small (2–3), plan from `S_t`, execute one action, observe the real outcome, replan from `S_{t+1}`, rather than planning an entire trajectory from `S_0` at one horizon. This requires a terminal value estimate `\widehat V(S_{t+h})` so a small horizon does not ignore reward that lands just past it — `finite_horizon_q` and `prospective_q` already support this shape (`gamma^h` weighting a future term), but no `\widehat V` estimator exists yet. The next real experiment (**E6.2-A**, not yet run) is comparing `Random`, `Fixed`, `immediate relevance`, `marginal obligation coverage`, and `Q_DCY` at `h=1,2,3` under one shared `ActionGenerator`, identical budgets, on identical tasks, reporting `Success@Budget` *and* `NodesExpanded`/planner latency together — not assuming `h=3 > h=2`, since the compute cost may not be worth it. That comparison is the actual test of `h* = argmax_h Success(h)/Compute(h)`, and it does not exist in this repository yet.

### E6.2 candidate — observable-only terminal value estimate (V-hat)

Two rules are adopted for this project going forward, in response to the E5 leakage finding above:

```text
No metric used for planning may depend on gold or hidden evaluation data.
Never increase horizon before proving the previous horizon improves Success@Budget.
```

`src/planning.hpp` implements `\widehat V(S)` as a deliberately boring, unweighted average over observable ratios computed only from `dcy::planning::State` — no networks, no training, no gold:

$$
O(S)=\frac{\text{obligations resolved}}{\text{obligations total}}
\quad
V(S)=\frac{\text{verified evidence}}{\max(1,\text{all evidence})}
\quad
X(S)=\frac{\text{contradictions resolved}}{\max(1,\text{contradictions discovered})}
$$

$$
R(S)=\frac{\text{no-progress repeated actions}}{\max(1,\text{actions executed})}
\qquad
B(S)=\frac{\text{remaining budget}}{\text{initial budget}}
$$

$$
\boxed{
\widehat V_0(S)=\frac{O(S)+V(S)+X(S)+B(S)-R(S)}{4}
}
$$

clamped to `[0,1]`. This is a hypothesis-only baseline, not a claimed-correct formula — it exists to be falsified by an ablation. `Evidence` gained a `verified` field (default `false`) so `V(S)` has something observable to read; it is set by whatever cross-checks evidence within the state, never by a hidden/gold label. `obligations_resolved_ratio`, `verified_evidence_ratio`, `contradictions_resolved_ratio`, `no_progress_ratio`, and `budget_remaining_ratio` are each independently computable and independently tested, so an ablation can add them one at a time via `estimate_value(state, initial_budget, level)`, where `level` is `ValueEstimateLevel::V0Obligations` through `V4Repetition`:

```text
V0: obligations only
V1: + verified evidence
V2: + contradictions
V3: + remaining budget
V4: + repetition penalty (the full boxed formula)
```

`budget_exhausted(remaining)` formalizes one of the recursion's stopping conditions (any single resource component at zero); `is_repeated_without_progress`/`drop_no_progress_repeats` from the previous section formalize the other. A receding-horizon recursion is expected to look like:

```cpp
if (horizon == 0) return estimate_value(state, initial_budget);
if (budget_exhausted(state.remaining)) return estimate_value(state, initial_budget);
// caller checks is_repeated_without_progress per candidate before recursing
```

None of `estimate_value`, `budget_exhausted`, or the individual ratio functions is called from `prospective_q`, `select_best_action`, or any planner default — they are primitives for the E6.2-A ablation below, tested in `tests/efficiency_test.cpp::test_observable_value_estimate_ablation_ladder` (each ratio checked independently, each ablation level checked against the exact arithmetic of the boxed formula — not assumed to cancel out, since the first draft of this test asserted the repetition penalty would net back to the pre-penalty value and a real run caught that it does not: subtracting `R(S)/4` from a 3-term average produces a different number than a 4-term average, exactly as the formula specifies).

E6.2-A itself — the actual comparison of `Random`, `Fixed`, `immediate relevance`, `marginal obligation coverage`, and `Q_DCY` at `h=1,2,3` with and without `\widehat V`, on real DCY tasks, reporting `Success@Budget`, `NodesExpanded`, planner wall time, and a `PlanningEfficiency = Success@Budget / NodesExpanded` engineering metric — has not been run. It is the next concrete step, not a result to report yet.

### E7 — Paging architecture vs flat retrieval on a small model — virtualization claim not supported

E7 tests the semantic-paging claim of the next section directly: does DCY-driven navigation under a small window beat a single flat retrieval of comparable size, on the same model and the same tasks? The headline comparison is `static-512` vs `dcy-v2`, chosen because both begin from a small physical window and differ only in whether the runtime may page. `qwen3:0.6b` at `temperature=0`, `num_ctx=8192`, 12 scored tasks plus two controls, over `corpus-1m.sqlite` (redis, jemalloc, tree-sitter). Every gold symbol in `bench/arch-suite-tasks.jsonl` was validated against the index before the run; two proposed entries were dropped because they are macros rather than indexed entities. All five architectures share one plain-text reader prompt — no JSON contract — so no arm is penalized for protocol compliance rather than task ability. Harness `bench/arch_comparison.py`, aggregation `bench/arch_report.py`, artifact `build/arch-qwen3-0.6b.json` (70 records).

| Architecture | `R_c` | `R_a` | Precision | `F1` | Hallucinated/task | Median PT | Median s | LLM calls | Solved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` (no context) | 0.000 | 0.250 | 0.103 | 0.135 | 1.75 | 74 | 15.1 | 1 | 2/12 |
| `static-512` | 0.625 | 0.583 | 0.360 | 0.390 | 0.58 | 301 | 16.3 | 1 | 6/12 |
| `static-2048` | 0.792 | 0.597 | 0.396 | 0.432 | 0.25 | 955 | 30.1 | 1 | 6/12 |
| `dcy-v2` | 0.667 | 0.583 | 0.409 | 0.432 | **0.00** | 2342 | 78.2 | 5 | 6/12 |
| `oracle` | 1.000 | 0.875 | 0.872 | 0.868 | 0.08 | 1512 | 37.3 | 1 | 9/12 |

`oracle` is a ceiling instrument, not a competing system: it injects the source of the task's own gold symbols and therefore has `R_c = 1.000` by construction. It is present only to separate "the retriever did not supply the evidence" from "the model could not use it", and must never be read as a retrieval result.

Two budget questions are reported separately because they are distinct claims. At **equal total physical tokens** (`static-2048` vs `dcy-v2`) the difference in answer recall is `-0.014`, within noise and in the wrong direction. At **equal instantaneous window** (`static-512` vs `dcy-v2`) it is exactly `+0.000`, with `dcy-v2` consuming 7.8× the prompt tokens and 4.8× the wall time to reach the same six solved tasks. `H_E7` — that structured navigation under a bounded window outperforms flat retrieval of that window size — is **not supported by this benchmark**. This is recorded as a negative result; the paging runtime remains implemented and correct, but its measured advantage on this suite is confined to precision rather than capability.

The signal split localizes the loss. `eta_use = R_a / R_c` is 0.93, 0.75 and 0.87 for `static-512`, `static-2048` and `dcy-v2` respectively, so the 0.6B model converts nearly all evidence it is shown into named entities and is not the binding constraint; the ceiling is `R_c`, where `dcy-v2` (0.667) improves only marginally on a single 512-byte query (0.625) and stays below a single 2048-byte query (0.792). By tier the gap is starkest on the hard multi-hop jemalloc tasks, exactly where navigation was expected to pay: `dcy-v2` reaches `R_a = 0.1` against `oracle` at `0.8`, and on T12 all four retrieval arms record `R_c = 0.000` while `oracle` records `1.000`. The model can answer these questions; the ranking is not supplying the evidence. The one unambiguous `dcy-v2` win is **zero** hallucinated symbols per task against 1.75 for the no-context arm and 0.58 for `static-512` — carry-state paging suppresses invention even where it does not raise capability.

Two controls constrain the interpretation. N01 probes a symbol confirmed absent from the index (`ts_subtree_quantum_edit`); all five architectures fabricate a role for it, `oracle` included, and none abstains — so none of these numbers should be read as evidence of calibrated refusal. P01 is a lexical paraphrase of T02 naming no symbols; every retrieval arm falls from `R_a` of 0.667–1.000 to **0.000** while `oracle` holds at 1.000, indicating candidate generation keys on literal identifier overlap with the query rather than on semantic intent. That mechanism is the most plausible explanation for the depressed `R_c` on the hard tier, and it makes ranking — not paging strategy, budget policy, or model size — the next subsystem to measure.

**Two harness bugs were found and fixed during E7, both of the silent kind that inflates a result rather than crashing.** First, Ollama honours a default `num_ctx` of 4096 regardless of the model's advertised context (`qwen3:0.6b` reports 40960), so a long-context arm can score against a prompt the model never received; the tell was a 16 kB and a 64 kB dump arm both reporting exactly 2050 prompt tokens. Both harnesses now set `num_ctx` explicitly and flag any row where `prompt_eval_count >= num_ctx - 8`. Second, the N01 abstention detector originally matched hedge keywords as substrings and scored `The role of X is not explicitly defined... However, it is referenced in hpa_try_alloc_one_no_grow, check_match` as a clean refusal, publishing "3 of 5 abstained" when all five had fabricated; abstention now requires the hedge **and** no asserted role **and** zero invented symbols, with all three sub-flags recorded. Both are covered by regression tests in `tests/arch_scoring_test.py` (CTest target `dcy_arch_scoring`), the first using the verbatim model output that defeated the original rule, together with an assertion that the superseded rule would indeed have passed it — so the test cannot quietly stop being a regression.

A separate naming hazard is worth recording because it caused a result to be read backwards in review. `bench/long_context_vs_dcy.py` compares raw source dumps against a **single** `dcy query` plus one inference, and that arm was originally called plain `dcy`; it is flat retrieval and is now named `dcy-query-<budget>`. Its 1.4× speed and 4.3× token advantage over a 4 kB raw dump is a real result about DCY's distillation, and says nothing about the v2 paging runtime, which issues roughly five inferences and is the slowest arm in the table above. The C++ retrieval itself costs 0.02–0.1 s throughout; the cost in `dcy-v2` is LLM calls, not DCY.

## Design principle — semantic paging

> **DCY virtualizes context by repeatedly materializing distilled regions of a larger externally addressable token space while preserving task state across windows.**

The analogy to virtual memory is deliberate:

```text
virtual memory:                    DCY:
virtual address                    context need (goal + state)
→ page fault                       → context fault: PT lacks info for next reasoning step
→ load page from disk              → retrieve + distill region from DB/index
→ CPU continues with same state    → LLM continues with same externalized state
```

Formally, each step materializes a physical context from a virtual token space:

```math
C_t = D(VT, S_t, a_t, B)
```

where `VT` is the full externally addressable token space (the indexed repository), `S_t` is the persistent task state (obligations, evidence, hypotheses, contradictions, history, remaining budget — the `dcy::planning::State` struct), `a_t` is the selected action (which region/goal to focus on), `B` is the physical token budget, and `C_t` is the distilled context the LLM actually sees. The LLM's output is then:

```math
o_t = LLM(C_t, M_t)
```

where `M_t` is a small continuity capsule from the previous step (in the current code: the `STATE` lines that `render()` injects from the `observations` table, line 375-385 of `main.cpp`). The key property is that `C_t + M_t` should make the model experience continuity equivalent to having seen the relevant parts of `VT`, even though the raw source is no longer physically present.

This is not RAG. RAG is `query → retrieve docs → answer` — a single retrieval step without persistent state or multi-step context management. DCY is:

```text
persistent task state S_t
→ context fault (current PT lacks evidence for next step)
→ select region (goal engine / action)
→ distill (distiller packs entities under budget)
→ LLM reasons within physical window
→ externalize state update (dcy record / observations table)
→ next context fault
→ ...
```

The difference is that each distilled window becomes part of the cognitive continuity of the process, even though it does not persist physically. This is what makes VT meaningful: it is not a compression claim, but a measure of the externally addressable space that DCY can bring into a bounded window on demand, step by step, while the LLM's task state survives across windows via the external `State`.

**Implementation map** — what exists and what does not:

| Semantic paging concept | Code | Status |
|---|---|---|
| Virtual token space (VT) | SQLite index built by `dcy index` | Implemented, used |
| Context fault (need → retrieve) | `retrieve()` in `main.cpp:321` (FTS + graph expansion) | Implemented, used |
| Distill region under budget | `distiller::make_view()` with ref-first two-pass packing | Implemented, used |
| Multi-resolution encoding | `ormt::encode()` at `Detail`/`Compact`/`Ref` levels | Implemented, used |
| Exact source on demand | `dcy source` (verified hash, bounded span) | Implemented, used |
| Continuity capsule M_t | `STATE` lines from `observations` table, injected by `render()` | Implemented, used (3 most recent observations) |
| Externalize state update | `dcy record` (hypotheses, facts, results per goal) | Implemented, used |
| Context fault detection | `api/dcy_consumer.py` v2: deterministic sub-goal engine drives paging, LLM only reads/writes plain text | Ready runtime (`--stub` green; Ollama `gemma3:4b` answered, `qwen3:0.6b` produced correct symbols) |
| Goal-driven action selection | `dcy::planning::ActionType` enum (Search, Expand, ReadSource, Verify, IncreaseResolution, Finish) + `decompose_task()` locate → relations → source → synthesize in `api/dcy_consumer.py` | Runtime wired (deterministic decomposition; selector policies still explicit per-run) |
| Prospective action scoring | `prospective_q` / `memoized_prospective_q` / `softmax_distribution` | Ready primitives, tested, harness-selected (not a CLI default) |
| Terminal value estimate | `estimate_value()` with ablation ladder V0-V4 | Ready primitive, tested, harness-selected (not a CLI default) |
| Receding-horizon loop | `run_paging_runtime()` in `api/dcy_consumer.py`: materialize page → LLM observes → `dcy record` → `SeenSet` update → cut → distill capsule → advance sub-goal → final synthesis | Ready runtime (deterministic goals; full `plan(h) → act → replan` with learned policies is still E6.2-A) |

The remaining gap is E6.2-A measurement, not missing runtime code: the paging loop runs today (`python3 api/dcy_consumer.py --dcy build/dcy --db build/corpus-1m.sqlite --task "..." --budget 512 --max-goals 4 --stub`), the planner primitives are tested, but no run yet proves which selector/horizon/`tau`/V-hat level improves `Success@Budget` over the baselines. That comparison is the next concrete step.

### Runtime HTTP API — Ollama consumes DCY through here

`api/dcy_server.py` exposes the v2 paging runtime over HTTP (stdlib only, no web framework), so any model client — Ollama, Hermes, curl — consumes virtualized context without knowing DCY exists:

```text
POST /session          {"budget":512,"max_goals":4,"model":"qwen2.5:0.5b-instruct"} -> {"session":"1",...}
POST /chat             {"session":"1","message":"what functions handle tree edit propagation?"} -> {"answer":"...","usage":{...}}
GET  /session/{id}     -> config + chat history
DELETE /session/{id}   -> removes the session
GET  /health           -> {"ok":true}
```

Run it with:

```sh
python3 api/dcy_server.py --db build/corpus-1m.sqlite --port 8765
```

Verified live against local Ollama (this repo's own test run, not a claim): `qwen2.5:0.5b-instruct` answered "what functions handle tree edit propagation?" through `POST /chat` with `ts_tree_edit`, `unmarshal_edit`, `ts_subtree_edit`, `ts_range_edit` grounded in 14 unique entities over 2 goals and 993 physical bytes. A second chat on the same session, session history (`history_n: 2`), delete + unknown-session-after-delete all verified in the same run.

Two honest notes from that live run: (1) it caught a real bug — a cross-run goal cache handed back finished goal_ids (`dcy: goal is not active`), so `DCYCli.create_goal` now always creates fresh goals; repeat CLI runs are byte-identical after the fix (2 pages, 993 PT, 14 uniq, twice in a row). (2) On mixed-corpus queries the retriever can blend subsystems (a "callers of ts_subtree_edit" answer mentioned jemalloc's `large_ralloc`), which is retrieval behavior, not API behavior — recorded here so it is not mistaken for a clean result.

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

The [2026-09-14 exploratory results](bench/results-2026-09-14.md) include the full ablations, raw checkpoint artifacts, and scale pilots. The current experimental checkpoint is **E1**: coverage-first packing and name-first ORMT changed injected recall from 0.500 to 0.611, answer recall from 0.111 to 0.500, and `U(model,renderer)` from 0.222 to 0.818 on the same nine-task 0.5B pilot; candidate recall stayed at 0.778, so the gain came from propagation rather than retrieval. `ctest` passes 6/6. E1 artifacts are under `bench/checkpoints/`.

## Architecture comparison benchmark (E7)

Compares `base`, `static-512`, `static-2048`, `dcy-v2` and `oracle` on one model, scoring entities rather than a bare correct/incorrect, and separating evidence recall `R_c` from answer recall `R_a`:

```sh
python3 bench/arch_comparison.py --model qwen3:0.6b \
  --db build/corpus-1m.sqlite --tasks bench/arch-suite-tasks.jsonl \
  --out build/arch-qwen3-0.6b.json
python3 bench/arch_report.py build/arch-qwen3-0.6b.json
```

Requires `httpx` and a running Ollama. Roughly 2.5 minutes per task on CPU for a 0.6B model — run it in the background; records are appended to the artifact after every arm, so a timeout loses at most one row. `--only T01 T02` restricts the task set and `--architectures static-512 dcy-v2` restricts the arms.

Two auxiliary A/B harnesses: `bench/ab_dcy_vs_nodcy.py` (with-context vs no-context on the same model, reporting gold-in-page, consumption and correctness separately) and `bench/long_context_vs_dcy.py` (raw source dumps of increasing size vs a single distilled query — set `--num-ctx` above the largest dump or Ollama silently truncates and the arm measures nothing).

The scoring logic these three share is covered by `tests/arch_scoring_test.py`, which runs under `ctest -R dcy_arch_scoring` without Ollama, a database, or network access. Read `E7` above before quoting any number from these harnesses: the headline result is negative, and `oracle` is a ceiling instrument rather than a competing retrieval system.

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
