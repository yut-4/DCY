#pragma once

#include <cstddef>
#include <numeric>
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
