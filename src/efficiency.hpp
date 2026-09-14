#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
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
  if (measurement.corpus_tokens < 0.0 || measurement.partitions <= 0.0 ||
      measurement.virtual_tokens < 0.0 || measurement.goal_continuity < 0.0 ||
      measurement.database_utility < 0.0) {
    throw std::invalid_argument("efficiency measurements must be non-negative");
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
  if (gamma < 0.0 || gamma > 1.0) {
    throw std::invalid_argument("discount factor must be between zero and one");
  }
  double value=0.0;
  double discount=1.0;
  for (double delta:expected_progress) {
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
  if (gamma<0.0 || gamma>1.0) throw std::invalid_argument("discount factor must be between zero and one");
  if (!allowed(state,action)) throw std::invalid_argument("action exceeds constraints");

  double expected=0.0;
  double probability_sum=0.0;
  bool has_outcome=false;
  for (const auto& outcome:outcomes(state,action)) {
    if (outcome.probability<0.0) throw std::invalid_argument("outcome probability must be non-negative");
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
    expected+=outcome.probability*(progress(state,next,action,outcome)+gamma*future);
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
    if (best_action==nullptr || value>best_score) {
      best_action=&action;
      best_score=value;
    }
  }
  return *best_action;
}

inline double cost(const GoalCost& step) {
  if (step.physical_tokens < 0.0 || step.latency < 0.0 || step.api_cost < 0.0) {
    throw std::invalid_argument("goal costs must be non-negative");
  }
  return step.physical_tokens+step.latency+step.api_cost;
}

inline double objective(const std::vector<GoalMeasurement>& goals,
                        const std::vector<GoalCost>& costs,double lambda) {
  if (lambda < 0.0) throw std::invalid_argument("lambda must be non-negative");
  if (goals.size()!=costs.size()) throw std::invalid_argument("goals and costs must align");
  double total_cost=0.0;
  for (const auto& step:costs) total_cost+=cost(step);
  return aggregate(goals)-lambda*total_cost;
}

} // namespace dcy::efficiency
