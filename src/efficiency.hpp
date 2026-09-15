#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace dcy::efficiency {

// Experimental metric model. Units are intentionally caller-defined: VT and
// PT may be tokenizer tokens, while latency/API cost must be normalized before
// combining them in cost() or objective(). This header does not affect runtime
// retrieval or goal lifecycle behaviour yet.
struct GoalMeasurement {
  double virtual_tokens=0.0;
  double goal_continuity=0.0;
  double database_utility=0.0;
  double corpus_tokens=0.0;
  double partitions=0.0;
};

struct GoalCost {
  double physical_tokens=0.0;
  double latency=0.0;
  double api_cost=0.0;
};

inline double local(const GoalMeasurement& measurement) {
  if (!std::isfinite(measurement.virtual_tokens) ||
      !std::isfinite(measurement.goal_continuity) ||
      !std::isfinite(measurement.database_utility) ||
      !std::isfinite(measurement.corpus_tokens) ||
      !std::isfinite(measurement.partitions) ||
      measurement.corpus_tokens < 0.0 || measurement.partitions <= 0.0 ||
      measurement.virtual_tokens < 0.0 || measurement.goal_continuity < 0.0 ||
      measurement.database_utility < 0.0) {
    throw std::invalid_argument("efficiency measurements must be finite and non-negative");
  }
  if (measurement.corpus_tokens == 0.0) {
    throw std::invalid_argument("corpus token count must be positive");
  }
  const double working_set=measurement.corpus_tokens/measurement.partitions;
  return measurement.virtual_tokens*measurement.goal_continuity*
         measurement.database_utility/(working_set*working_set);
}

inline double aggregate(const std::vector<GoalMeasurement>& goals) {
  return std::accumulate(goals.begin(),goals.end(),0.0,
                         [](double total,const auto& goal) { return total+local(goal); });
}

inline double average(const std::vector<GoalMeasurement>& goals) {
  if (goals.empty()) throw std::invalid_argument("cannot average an empty trajectory");
  return aggregate(goals)/static_cast<double>(goals.size());
}

// Finite-horizon action value over already estimated expected progress deltas.
// This evaluates Q_DCY^(h); it does not generate futures or implement a planner.
inline double finite_horizon_q(const std::vector<double>& expected_progress,
                               double gamma) {
  if (!std::isfinite(gamma) || gamma < 0.0 || gamma > 1.0) {
    throw std::invalid_argument("discount factor must be finite and between zero and one");
  }
  double value=0.0;
  double discount=1.0;
  for (double delta:expected_progress) {
    if (!std::isfinite(delta)) throw std::invalid_argument("progress delta must be finite");
    value+=discount*delta;
    discount*=gamma;
  }
  return value;
}

// Expected finite-horizon value for one action. The caller supplies all domain
// knowledge through callbacks; this layer knows neither DCY storage nor models.
// Outcome ranges must expose `probability`; transition receives the outcome.
template <typename State, typename Action,
          typename ActionGenerator, typename OutcomeModel, typename Transition,
          typename Progress, typename Allowed>
double prospective_q(const State& state,const Action& action,size_t horizon,double gamma,
                     const ActionGenerator& actions,const OutcomeModel& outcomes,
                     const Transition& transition,const Progress& progress,
                     const Allowed& allowed) {
  if (horizon==0) return 0.0;
  if (!std::isfinite(gamma) || gamma<0.0 || gamma>1.0) {
    throw std::invalid_argument("discount factor must be finite and between zero and one");
  }
  if (!allowed(state,action)) throw std::invalid_argument("action exceeds constraints");

  double expected=0.0;
  double probability_sum=0.0;
  bool has_outcome=false;
  for (const auto& outcome:outcomes(state,action)) {
    if (!std::isfinite(outcome.probability) || outcome.probability<0.0) {
      throw std::invalid_argument("outcome probability must be finite and non-negative");
    }
    has_outcome=true;
    probability_sum+=outcome.probability;
    const State next=transition(state,action,outcome);
    double future=0.0;
    if (horizon>1) {
      bool found=false;
      double best=-std::numeric_limits<double>::infinity();
      for (const auto& next_action:actions(next)) {
        if (!allowed(next,next_action)) continue;
        found=true;
        best=std::max(best,prospective_q(next,next_action,horizon-1,gamma,actions,
                                         outcomes,transition,progress,allowed));
      }
      if (found) future=best;
    }
    const double delta=progress(state,next,action,outcome);
    if (!std::isfinite(delta)) throw std::invalid_argument("progress value must be finite");
    expected+=outcome.probability*(delta+gamma*future);
  }
  if (!has_outcome) throw std::invalid_argument("action has no modeled outcomes");
  if (std::abs(probability_sum-1.0)>1e-9) {
    throw std::invalid_argument("outcome probabilities must sum to one");
  }
  return expected;
}

template <typename State,typename Action,typename Score>
std::optional<Action> select_best_action(const State& state,const std::vector<Action>& actions,
                                         const Score& score) {
  if (actions.empty()) return std::nullopt;
  const Action* best_action=nullptr;
  double best_score=-std::numeric_limits<double>::infinity();
  for (const auto& action:actions) {
    const double value=score(state,action);
    if (!std::isfinite(value)) throw std::invalid_argument("action score must be finite");
    if (best_action==nullptr || value>best_score) {
      best_action=&action;
      best_score=value;
    }
  }
  return *best_action;
}

// Cheap pre-Bellman action pruning (h-explosion mitigation, README section 8
// candidate). Branching factor b in prospective_q is O(actions x outcomes),
// so shrinking the candidate action set before the expensive recursion runs
// is the first lever, independent of any Bellman scoring. The caller
// supplies an inexpensive heuristic (e.g. lexical relevance); this function
// does not call prospective_q or replace its scoring. stable_sort keeps
// original relative order on ties so pruning stays deterministic for
// reproducible benchmark runs. This is a candidate primitive for the E6.2
// horizon/compute comparison, not wired into any planner default.
template <typename State,typename Action,typename CheapScore>
std::vector<Action> top_k_actions(const State& state,std::vector<Action> actions,
                                  const CheapScore& cheap_score,std::size_t k) {
  std::stable_sort(actions.begin(),actions.end(),[&](const Action& a,const Action& b) {
    return cheap_score(state,a)>cheap_score(state,b);
  });
  if (actions.size()>k) actions.resize(k);
  return actions;
}

// Outcome pruning by cumulative probability (h-explosion mitigation, README
// section 9 candidate). Keeps the highest-probability outcomes until their
// cumulative probability reaches keep_threshold, drops the remainder, then
// rescales the kept outcomes so they still sum to one — prospective_q
// requires normalized probabilities and would otherwise reject the pruned
// set. Always keeps at least one outcome so a legal action never ends up
// with zero modeled outcomes. This changes runtime behaviour, not scoring
// semantics; it should be measured for its effect on Success@Budget
// separately from the exact Bellman evaluation, not assumed harmless.
template <typename Outcome>
std::vector<Outcome> prune_outcomes(std::vector<Outcome> outcomes,double keep_threshold) {
  if (!std::isfinite(keep_threshold) || keep_threshold<=0.0 || keep_threshold>1.0) {
    throw std::invalid_argument("outcome keep threshold must be in (0,1]");
  }
  if (outcomes.empty()) return outcomes;
  std::stable_sort(outcomes.begin(),outcomes.end(),[](const Outcome& a,const Outcome& b) {
    return a.probability>b.probability;
  });
  double cumulative=0.0;
  std::size_t keep=0;
  while (keep<outcomes.size() && cumulative<keep_threshold) {
    cumulative+=outcomes[keep].probability;
    ++keep;
  }
  if (keep==0) keep=1;
  outcomes.resize(keep);
  double kept_sum=0.0;
  for (const auto& outcome:outcomes) kept_sum+=outcome.probability;
  if (kept_sum<=0.0) throw std::invalid_argument("pruned outcomes have no positive probability mass");
  for (auto& outcome:outcomes) outcome.probability/=kept_sum;
  return outcomes;
}

// Deterministic hash mixer (boost::hash_combine's mixing constant) used by
// memoized_prospective_q's cache key. No random seed: identical inputs must
// produce identical keys across runs so cached benchmark results stay
// reproducible.
inline std::size_t combine_hash(std::size_t seed,std::size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed<<6) + (seed>>2));
}

// Memoized variant of prospective_q (h-explosion mitigation, README sections
// 10-11 candidate). The same (state, action, horizon) triple can be reached
// through different action sequences within one search; caching avoids
// recomputing its Bellman value. StateHash/ActionHash are caller-supplied
// because this header has no concrete State/Action of its own — planning.hpp's
// State has no stable hash yet (see README E6.1). `cache` is owned by the
// caller so it can be inspected (cache.size() as a NodesExpanded proxy) or
// discarded between distinct root decisions in a receding-horizon loop.
// This mirrors prospective_q's validation exactly; it is a caching wrapper,
// not a different mathematical contract.
template <typename State, typename Action,
          typename ActionGenerator, typename OutcomeModel, typename Transition,
          typename Progress, typename Allowed, typename StateHash, typename ActionHash>
double memoized_prospective_q(const State& state,const Action& action,size_t horizon,double gamma,
                              const ActionGenerator& actions,const OutcomeModel& outcomes,
                              const Transition& transition,const Progress& progress,
                              const Allowed& allowed,const StateHash& state_hash,
                              const ActionHash& action_hash,
                              std::unordered_map<std::size_t,double>& cache) {
  if (horizon==0) return 0.0;
  const std::size_t key=combine_hash(combine_hash(state_hash(state),action_hash(action)),horizon);
  if (auto it=cache.find(key); it!=cache.end()) return it->second;

  if (!std::isfinite(gamma) || gamma<0.0 || gamma>1.0) {
    throw std::invalid_argument("discount factor must be finite and between zero and one");
  }
  if (!allowed(state,action)) throw std::invalid_argument("action exceeds constraints");

  double expected=0.0;
  double probability_sum=0.0;
  bool has_outcome=false;
  for (const auto& outcome:outcomes(state,action)) {
    if (!std::isfinite(outcome.probability) || outcome.probability<0.0) {
      throw std::invalid_argument("outcome probability must be finite and non-negative");
    }
    has_outcome=true;
    probability_sum+=outcome.probability;
    const State next=transition(state,action,outcome);
    double future=0.0;
    if (horizon>1) {
      bool found=false;
      double best=-std::numeric_limits<double>::infinity();
      for (const auto& next_action:actions(next)) {
        if (!allowed(next,next_action)) continue;
        found=true;
        best=std::max(best,memoized_prospective_q(next,next_action,horizon-1,gamma,actions,
                                                   outcomes,transition,progress,allowed,
                                                   state_hash,action_hash,cache));
      }
      if (found) future=best;
    }
    const double delta=progress(state,next,action,outcome);
    if (!std::isfinite(delta)) throw std::invalid_argument("progress value must be finite");
    expected+=outcome.probability*(delta+gamma*future);
  }
  if (!has_outcome) throw std::invalid_argument("action has no modeled outcomes");
  if (std::abs(probability_sum-1.0)>1e-9) {
    throw std::invalid_argument("outcome probabilities must sum to one");
  }
  cache.emplace(key,expected);
  return expected;
}

// Softmax action-selection distribution (DCY v2 hypothesis, README E6.2
// candidate). This is a candidate POLICY for the E6 prospective benchmark
// comparison alongside argmax, random, immediate relevance and marginal
// obligation coverage; it does not replace select_best_action and nothing
// wires it into a planner path. tau -> 0 concentrates the distribution on
// the same action select_best_action would return (see
// test_softmax_action_distribution); tau must be finite and > 0 — use
// select_best_action directly for pure greedy selection instead of tau=0.
template <typename State,typename Action,typename Score>
std::vector<std::pair<Action,double>> softmax_distribution(
    const State& state,const std::vector<Action>& actions,const Score& score,double tau) {
  if (!std::isfinite(tau) || tau<=0.0) {
    throw std::invalid_argument("softmax temperature must be finite and positive");
  }
  if (actions.empty()) return {};
  std::vector<double> scores;
  scores.reserve(actions.size());
  double max_score=-std::numeric_limits<double>::infinity();
  for (const auto& action:actions) {
    const double value=score(state,action);
    if (!std::isfinite(value)) throw std::invalid_argument("action score must be finite");
    scores.push_back(value);
    max_score=std::max(max_score,value);
  }
  std::vector<double> weights(scores.size());
  double sum=0.0;
  for (std::size_t i=0;i<scores.size();++i) {
    // Subtract max_score before exponentiating for numerical stability;
    // it cancels in the normalized ratio so the distribution is unchanged.
    weights[i]=std::exp((scores[i]-max_score)/tau);
    sum+=weights[i];
  }
  std::vector<std::pair<Action,double>> distribution;
  distribution.reserve(actions.size());
  for (std::size_t i=0;i<actions.size();++i) {
    distribution.emplace_back(actions[i],weights[i]/sum);
  }
  return distribution;
}

// Reproducible sampling primitive: the caller supplies a uniform draw in
// [0,1) (e.g. from a seeded RNG in a benchmark harness) instead of this
// header owning random state, so selections stay replayable in experiments.
template <typename Action>
std::size_t sample_from_distribution(const std::vector<std::pair<Action,double>>& distribution,
                                     double uniform_draw) {
  if (distribution.empty()) throw std::invalid_argument("cannot sample an empty distribution");
  if (!std::isfinite(uniform_draw) || uniform_draw<0.0 || uniform_draw>=1.0) {
    throw std::invalid_argument("uniform draw must be in [0,1)");
  }
  double cumulative=0.0;
  for (std::size_t i=0;i<distribution.size();++i) {
    cumulative+=distribution[i].second;
    if (uniform_draw<cumulative) return i;
  }
  return distribution.size()-1; // guards floating-point rounding at the tail
}

// Convenience wrapper combining softmax_distribution and
// sample_from_distribution into one candidate selection call for the E6.2
// comparison harness. Still not a planner default; see the header comment
// on softmax_distribution.
template <typename State,typename Action,typename Score>
std::optional<Action> softmax_select_action(const State& state,const std::vector<Action>& actions,
                                            const Score& score,double tau,double uniform_draw) {
  const auto distribution=softmax_distribution(state,actions,score,tau);
  if (distribution.empty()) return std::nullopt;
  return distribution[sample_from_distribution(distribution,uniform_draw)].first;
}

inline double cost(const GoalCost& step) {
  if (!std::isfinite(step.latency) || !std::isfinite(step.api_cost) ||
      step.physical_tokens < 0.0 || step.latency < 0.0 || step.api_cost < 0.0) {
    throw std::invalid_argument("goal costs must be finite and non-negative");
  }
  return step.physical_tokens+step.latency+step.api_cost;
}

inline double objective(const std::vector<GoalMeasurement>& goals,
                        const std::vector<GoalCost>& costs,double lambda) {
  if (!std::isfinite(lambda) || lambda < 0.0) throw std::invalid_argument("lambda must be finite and non-negative");
  if (goals.size()!=costs.size()) throw std::invalid_argument("goals and costs must align");
  double total_cost=0.0;
  for (const auto& step:costs) total_cost+=cost(step);
  return aggregate(goals)-lambda*total_cost;
}

} // namespace dcy::efficiency
