#include "../src/efficiency.hpp"
#include "../src/planning.hpp"

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

void test_finite_horizon_action_value() {
  const std::vector<double> unlocks{.05,.50,.30};
  const std::vector<double> immediate{.30,.02,0.0};
  close_to(dcy::efficiency::finite_horizon_q(unlocks,.8),.642,
           "discounted horizon sums future unlock progress");
  close_to(dcy::efficiency::finite_horizon_q(immediate,.8),.316,
           "discounted horizon values immediate progress");
  check(dcy::efficiency::finite_horizon_q(unlocks,.8)>
        dcy::efficiency::finite_horizon_q(immediate,.8),
        "horizon three prefers the action with better downstream progress");
  check(dcy::efficiency::finite_horizon_q(unlocks,0.0)<
        dcy::efficiency::finite_horizon_q(immediate,0.0),
        "horizon one prefers the action with better immediate progress");

  bool invalid=false;
  try { dcy::efficiency::finite_horizon_q(unlocks,1.1); }
  catch (const std::invalid_argument&) { invalid=true; }
  check(invalid,"discount factor outside zero to one is rejected");
}

struct Outcome {
  double probability;
  int next_state;
};

void test_prospective_bellman_value() {
  const auto actions=[](int state) {
    return state==0 ? std::vector<int>{1,2} : std::vector<int>{3};
  };
  const auto outcomes=[](int state,int action) {
    if (state==0 && action==1) return std::vector<Outcome>{{1.0,1}};
    if (state==0 && action==2) return std::vector<Outcome>{{1.0,2}};
    if (state==1 && action==3) return std::vector<Outcome>{{1.0,3}};
    if (state==2 && action==3) return std::vector<Outcome>{{1.0,4}};
    return std::vector<Outcome>{};
  };
  const auto transition=[](int /*state*/,int /*action*/,const Outcome& outcome) {
    return outcome.next_state;
  };
  const auto progress=[](int /*state*/,int next,int action,const Outcome& /*outcome*/) {
    if (action==1) return .05;  // mediocre now, unlocks a strong next action
    if (action==2) return .30;  // strong now, then a weak continuation
    return next==3 ? .50 : .02;
  };
  const auto allowed=[](int /*state*/,int action) { return action!=99; };

  auto q1= [&](int state,int action) {
    return dcy::efficiency::prospective_q(state,action,1,.8,actions,outcomes,
                                           transition,progress,allowed);
  };
  auto q2= [&](int state,int action) {
    return dcy::efficiency::prospective_q(state,action,2,.8,actions,outcomes,
                                           transition,progress,allowed);
  };
  close_to(q1(0,1),.05,"Bellman horizon one evaluates immediate progress");
  close_to(q1(0,2),.30,"Bellman horizon one sees the immediate action");
  close_to(q2(0,1),.45,"Bellman horizon two includes the best future action");
  close_to(q2(0,2),.316,"Bellman horizon two discounts the weak continuation");
  check(q1(0,2)>q1(0,1),"horizon one selects immediate action");
  check(q2(0,1)>q2(0,2),"horizon two selects unlocking action");

  auto selected=dcy::efficiency::select_best_action(0,std::vector<int>{1,2},q2);
  check(selected.has_value() && *selected==1,"selector returns Bellman-best action");

  bool blocked=false;
  try { q1(0,99); } catch (const std::invalid_argument&) { blocked=true; }
  check(blocked,"Bellman rejects an action outside constraints");
  bool empty=false;
  try { dcy::efficiency::prospective_q(0,7,1,.8,actions,outcomes,transition,progress,allowed); }
  catch (const std::invalid_argument&) { empty=true; }
  check(empty,"Bellman rejects an action with no modeled outcomes");

  const auto bad_outcomes=[](int /*state*/,int /*action*/) {
    return std::vector<Outcome>{{.4,1}};
  };
  bool unnormalized=false;
  try { dcy::efficiency::prospective_q(0,1,1,.8,actions,bad_outcomes,transition,progress,allowed); }
  catch (const std::invalid_argument&) { unnormalized=true; }
  check(unnormalized,"Bellman rejects outcomes that do not sum to one");
}

void test_planning_state_contract() {
  dcy::planning::State state;
  state.obligations.push_back({"auth", "verify authentication", dcy::planning::Status::Pending});
  state.evidence.push_back({42, "generation", {"auth.cpp", 10, 20}, "login returns 403", .99});
  state.hypotheses.push_back({"expired subscription", {42}, dcy::planning::Status::Unverified});
  state.contradictions.push_back({"c1", "status disagrees", {42}, dcy::planning::Status::Pending});
  state.history.push_back({{dcy::planning::ActionType::ReadSource, "E42", 1024}, false, true});
  state.remaining={500,2,3,100.0};
  check(state.obligations.size()==1 && state.evidence[0].entity==42,
        "planning state stores obligation and provenance evidence");
  check(state.hypotheses[0].status==dcy::planning::Status::Unverified,
        "hypotheses remain unverified until externally supported");
  check(state.history[0].action.type==dcy::planning::ActionType::ReadSource,
        "planning actions use a closed action type");
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
  test_finite_horizon_action_value();
  test_prospective_bellman_value();
  test_planning_state_contract();
  if (failures) {
    std::fprintf(stderr,"%d efficiency check(s) failed\n",failures);
    return 1;
  }
  std::printf("all efficiency checks passed\n");
  return 0;
}
