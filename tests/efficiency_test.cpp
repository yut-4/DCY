#include "../src/efficiency.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
int failures=0;
void check(bool condition,const char* message) {
  if (condition) return;
  std::fprintf(stderr,"FAIL: %s\n",message);
  ++failures;
}
void close_to(double actual,double expected,const char* message) {
  check(std::abs(actual-expected)<1e-9,message);
}

dcy::efficiency::GoalMeasurement goal(double vt,double continuity,double utility,
                                      double corpus=1000.0,double partitions=10.0) {
  return {vt,continuity,utility,corpus,partitions};
}

void test_original_local_equation() {
  // (100 * .8 * .5) / (1000/10)^2 = .004
  close_to(dcy::efficiency::local(goal(100,.8,.5)),.004,
           "local efficiency preserves the original equation");
}

void test_sigma_and_average() {
  std::vector goals{goal(100,.8,.5),goal(200,.5,.5)};
  close_to(dcy::efficiency::aggregate(goals),.009,
           "sigma sums per-goal efficiencies");
  close_to(dcy::efficiency::average(goals),.0045,
           "average normalizes sigma by trajectory length");
}

void test_cost_objective() {
  std::vector goals{goal(100,.8,.5),goal(200,.5,.5)};
  std::vector<dcy::efficiency::GoalCost> costs{{10,2,1},{20,3,2}};
  close_to(dcy::efficiency::cost(costs[0]),13.0,
           "step cost sums physical tokens latency and API cost");
  close_to(dcy::efficiency::objective(goals,costs,.001),-.029,
           "objective subtracts normalized trajectory cost");
}

void test_continuity_is_caller_signal() {
  auto useful=goal(100,.9,.8);
  auto disconnected=goal(100,.0,.8);
  check(dcy::efficiency::local(useful)>dcy::efficiency::local(disconnected),
        "lower goal continuity lowers local contribution");
}

void test_invalid_trajectories() {
  bool empty_average=false;
  try { dcy::efficiency::average({}); } catch (const std::invalid_argument&) { empty_average=true; }
  check(empty_average,"empty trajectory average is rejected");

  bool misaligned=false;
  try {
    dcy::efficiency::objective({goal(100,.8,.5)},{{1,1,1},{1,1,1}},.1);
  } catch (const std::invalid_argument&) { misaligned=true; }
  check(misaligned,"misaligned goal and cost trajectories are rejected");

  bool negative=false;
  try { dcy::efficiency::local(goal(-1,.8,.5)); } catch (const std::invalid_argument&) { negative=true; }
  check(negative,"negative measurement is rejected");
}

void test_more_goals_does_not_fake_average() {
  std::vector one{goal(100,.8,.5)};
  auto two=one;
  two.push_back(one[0]);
  check(dcy::efficiency::aggregate(two)==2*dcy::efficiency::aggregate(one),
        "sigma records accumulated utility");
  close_to(dcy::efficiency::average(two),dcy::efficiency::average(one),
           "average prevents trajectory length from inflating comparison");
}

void test_average_increment_rule() {
  std::vector prior{goal(100,.8,.5),goal(100,.6,.5)};
  auto high=prior;
  high.push_back(goal(100,.9,.5));
  auto low=prior;
  low.push_back(goal(100,.1,.5));
  check(dcy::efficiency::average(high)>dcy::efficiency::average(prior),
        "a goal above the prior average improves average efficiency");
  check(dcy::efficiency::average(low)<dcy::efficiency::average(prior),
        "a goal below the prior average lowers average efficiency");
}
}

int main() {
  test_original_local_equation();
  test_sigma_and_average();
  test_cost_objective();
  test_continuity_is_caller_signal();
  test_invalid_trajectories();
  test_more_goals_does_not_fake_average();
  test_average_increment_rule();
  if (failures) {
    std::fprintf(stderr,"%d efficiency check(s) failed\n",failures);
    return 1;
  }
  std::printf("all efficiency checks passed\n");
  return 0;
}
