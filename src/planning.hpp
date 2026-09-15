#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace dcy::planning {

using EntityId=long long;
using Generation=std::string;

struct SourceSpan {
  std::string path;
  std::size_t start=0;
  std::size_t end=0;
};

enum class Status { Pending, Active, Resolved, Unverified, Supported, Rejected };

struct Obligation {
  std::string id;
  std::string description;
  Status status=Status::Pending;
};

struct Evidence {
  EntityId entity=0;
  Generation generation;
  SourceSpan source;
  std::string claim;
  double confidence=0.0;
  // Independently corroborated within the observable state (e.g. cross-
  // checked by dcy source), never derived from a hidden/gold label. Default
  // false: evidence starts unverified until something observable confirms it.
  bool verified=false;
};

struct Hypothesis {
  std::string claim;
  std::vector<EntityId> supporting_entities;
  Status status=Status::Unverified;
};

struct Contradiction {
  std::string id;
  std::string description;
  std::vector<EntityId> evidence;
  Status status=Status::Pending;
};

enum class ActionType { Search, Expand, ReadSource, Verify, IncreaseResolution, Finish };

// Resource use is a vector: a legal action must fit every remaining limit.
struct ResourceCost {
  std::size_t physical_tokens=0;
  std::size_t model_calls=0;
  double latency_ms=0.0;
  std::size_t steps=1;
};

struct Action {
  ActionType type=ActionType::Search;
  std::string target;
  ResourceCost estimated_cost;
};

struct Budget {
  std::size_t physical_tokens=0;
  std::size_t model_calls=0;
  double latency_ms=0.0;
  std::size_t steps=0;
};

inline bool fits(const ResourceCost& cost,const Budget& remaining) {
  return cost.physical_tokens<=remaining.physical_tokens &&
         cost.model_calls<=remaining.model_calls &&
         cost.latency_ms<=remaining.latency_ms &&
         cost.steps<=remaining.steps;
}

inline Budget consume(Budget remaining,const ResourceCost& cost) {
  if (!fits(cost,remaining)) throw std::invalid_argument("action exceeds remaining budget");
  remaining.physical_tokens-=cost.physical_tokens;
  remaining.model_calls-=cost.model_calls;
  remaining.latency_ms-=cost.latency_ms;
  remaining.steps-=cost.steps;
  return remaining;
}

struct ActionRecord {
  Action action;
  bool repeated=false;
  bool produced_new_evidence=false;
};

struct State {
  std::vector<Obligation> obligations;
  std::vector<Evidence> evidence;
  std::vector<Hypothesis> hypotheses;
  std::vector<Contradiction> contradictions;
  std::vector<ActionRecord> history;
  Budget remaining;
};

// Environment outcomes are observed after execution. Estimated outcomes are
// the selector's prediction and must never contain gold/future evaluator data.
struct EnvironmentOutcome {
  std::string observation;
  bool terminal=false;
};

struct EstimatedOutcome {
  double probability=0.0;
  std::string observation;
};

struct ProgressComponents {
  double obligations=0.0;
  double validated_evidence=0.0;
  double contradictions=0.0;
  double repeated_no_progress=0.0;
};

// Loop-cutting primitive (README section 11 candidate): if the same action
// already ran and produced no new evidence, a candidate generator should not
// propose it again. ActionEqual compares two actions for the caller's notion
// of "the same action" (e.g. same ActionType and target); this function does
// not itself hash or compare State, matching E6.1's note that State has no
// stable hash yet. This filters candidate actions; it does not change
// prospective_q's contract and is not called from any candidate generator
// by default.
template <typename ActionEqual>
bool is_repeated_without_progress(const std::vector<ActionRecord>& history,
                                  const Action& candidate,const ActionEqual& action_equal) {
  for (const auto& record:history) {
    if (record.repeated && !record.produced_new_evidence && action_equal(record.action,candidate)) {
      return true;
    }
  }
  return false;
}

template <typename ActionEqual>
std::vector<Action> drop_no_progress_repeats(const std::vector<ActionRecord>& history,
                                             std::vector<Action> candidates,
                                             const ActionEqual& action_equal) {
  candidates.erase(
      std::remove_if(candidates.begin(),candidates.end(),
                     [&](const Action& candidate) {
                       return is_repeated_without_progress(history,candidate,action_equal);
                     }),
      candidates.end());
  return candidates;
}

// --- Observable-only terminal value estimate V-hat(S) --------------------
//
// These features read ONLY dcy::planning::State (obligations, evidence,
// contradictions, history, remaining budget). None of them may read a task's
// gold set, hidden assertions, or post-execution success — that is the E5
// leakage bug this header exists to avoid repeating. Every ratio is clamped
// to [0,1] by construction (numerator bounded by denominator, or explicit
// clamp for the repetition penalty which is subtracted).
//
// This is a hypothesis-only estimator (V0 below is a naive unweighted
// average, not a validated model) meant for an ablation ladder: V0 through
// V4 add one observable component at a time so each can be tested for
// whether it actually improves Success@Budget, rather than assuming a
// hand-picked linear combination is correct. Nothing here is wired into
// prospective_q's `progress`/outcome callbacks by default.

inline double obligations_resolved_ratio(const State& state) {
  if (state.obligations.empty()) return 0.0;
  std::size_t resolved=0;
  for (const auto& obligation:state.obligations) {
    if (obligation.status==Status::Resolved) ++resolved;
  }
  return static_cast<double>(resolved)/static_cast<double>(state.obligations.size());
}

inline double verified_evidence_ratio(const State& state) {
  const std::size_t total=std::max<std::size_t>(1,state.evidence.size());
  std::size_t verified=0;
  for (const auto& evidence:state.evidence) {
    if (evidence.verified) ++verified;
  }
  return static_cast<double>(verified)/static_cast<double>(total);
}

inline double contradictions_resolved_ratio(const State& state) {
  const std::size_t total=std::max<std::size_t>(1,state.contradictions.size());
  std::size_t resolved=0;
  for (const auto& contradiction:state.contradictions) {
    if (contradiction.status==Status::Resolved) ++resolved;
  }
  return static_cast<double>(resolved)/static_cast<double>(total);
}

// Fraction of executed actions that were repeats producing no new evidence.
// This is a penalty term (subtracted in v0_naive_average), not a reward.
inline double no_progress_ratio(const State& state) {
  const std::size_t total=std::max<std::size_t>(1,state.history.size());
  std::size_t stalled=0;
  for (const auto& record:state.history) {
    if (record.repeated && !record.produced_new_evidence) ++stalled;
  }
  return static_cast<double>(stalled)/static_cast<double>(total);
}

// Remaining budget as a fraction of the initial budget the planner started
// with. `initial` must be the Budget at S_0 for this action sequence, not
// the constant-zero default; passing a zero-valued component divides by one
// component being unconstrained and returns 1.0 for that component (an
// unconstrained resource is always "fully available").
inline double budget_remaining_ratio(const Budget& remaining,const Budget& initial) {
  const auto ratio=[](std::size_t left,std::size_t start) {
    return start==0 ? 1.0 : static_cast<double>(left)/static_cast<double>(start);
  };
  const double latency_ratio=initial.latency_ms<=0.0 ? 1.0 : remaining.latency_ms/initial.latency_ms;
  const double tokens=ratio(remaining.physical_tokens,initial.physical_tokens);
  const double calls=ratio(remaining.model_calls,initial.model_calls);
  const double steps=ratio(remaining.steps,initial.steps);
  // Average across resource components rather than picking one, since a
  // planner can exhaust any single component first.
  return (tokens+calls+latency_ratio+steps)/4.0;
}

enum class ValueEstimateLevel { V0Obligations, V1Evidence, V2Contradictions, V3Budget, V4Repetition };

// Ablation ladder (README: "define it deliberately boring, no networks, no
// training, no gold, then ablate"). Each level adds exactly one observable
// component so E6.2-A can compare Success@Budget per level and find out
// which components actually help, instead of assuming a hand-tuned
// combination works. `initial_budget` must be the Budget the trajectory
// started with (see budget_remaining_ratio).
inline double estimate_value(const State& state,const Budget& initial_budget,
                             ValueEstimateLevel level=ValueEstimateLevel::V4Repetition) {
  double sum=obligations_resolved_ratio(state);
  double terms=1.0;
  if (level>=ValueEstimateLevel::V1Evidence) { sum+=verified_evidence_ratio(state); terms+=1.0; }
  if (level>=ValueEstimateLevel::V2Contradictions) { sum+=contradictions_resolved_ratio(state); terms+=1.0; }
  if (level>=ValueEstimateLevel::V3Budget) { sum+=budget_remaining_ratio(state.remaining,initial_budget); terms+=1.0; }
  double value=sum/terms;
  if (level>=ValueEstimateLevel::V4Repetition) {
    value-=no_progress_ratio(state)/terms;
  }
  return std::clamp(value,0.0,1.0);
}

// Formalized cutoff conditions for a receding-horizon Bellman recursion
// (README: "your switch break already becomes formalized"). A caller-side
// recursion should fall back to estimate_value() instead of recursing
// further when any of these hold; none of them reads gold or hidden success.
inline bool budget_exhausted(const Budget& remaining) {
  return remaining.physical_tokens==0 || remaining.model_calls==0 || remaining.steps==0;
}

} // namespace dcy::planning
