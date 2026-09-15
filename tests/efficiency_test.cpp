#include "../src/efficiency.hpp"
#include "../src/planning.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
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
  bool nan_invalid=false;
  try { dcy::efficiency::finite_horizon_q(unlocks,std::numeric_limits<double>::quiet_NaN()); }
  catch (const std::invalid_argument&) { nan_invalid=true; }
  check(nan_invalid,"finite horizon rejects NaN discount factor");
  try { dcy::efficiency::finite_horizon_q({std::numeric_limits<double>::infinity()},.8); }
  catch (const std::invalid_argument&) { nan_invalid=true; }
  check(nan_invalid,"finite horizon rejects non-finite progress");
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
  bool nan_score=false;
  try {
    dcy::efficiency::select_best_action(0,std::vector<int>{1,2},
      [](int /*state*/,int /*action*/) { return std::numeric_limits<double>::quiet_NaN(); });
  } catch (const std::invalid_argument&) { nan_score=true; }
  check(nan_score,"selector rejects non-finite scores");

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

  const auto probabilistic_outcomes=[](int state,int action) {
    if (state==0 && action==1) return std::vector<Outcome>{{.5,1},{.5,2}};
    if (state==1 && action==3) return std::vector<Outcome>{{1.0,3}};
    if (state==2 && action==3) return std::vector<Outcome>{{1.0,4}};
    return std::vector<Outcome>{};
  };
  const auto stochastic_progress=[](int /*state*/,int next,int action,const Outcome& /*outcome*/) {
    if (action==1) return .10;
    return next==3 ? .80 : .20;
  };
  auto stochastic=dcy::efficiency::prospective_q(0,1,2,.8,actions,
      probabilistic_outcomes,transition,stochastic_progress,allowed);
  close_to(stochastic,.50,"Bellman weights multiple outcomes by probability");

  const double nan=std::numeric_limits<double>::quiet_NaN();
  bool nan_probability=false;
  const auto nan_outcomes=[nan](int /*state*/,int /*action*/) {
    return std::vector<Outcome>{{nan,1}};
  };
  try { dcy::efficiency::prospective_q(0,1,1,.8,actions,nan_outcomes,transition,progress,allowed); }
  catch (const std::invalid_argument&) { nan_probability=true; }
  check(nan_probability,"Bellman rejects NaN probabilities");
  bool nan_delta=false;
  const auto nan_progress=[nan](int /*state*/,int /*next*/,int /*action*/,const Outcome& /*outcome*/) { return nan; };
  try { dcy::efficiency::prospective_q(0,1,1,.8,actions,outcomes,transition,nan_progress,allowed); }
  catch (const std::invalid_argument&) { nan_delta=true; }
  check(nan_delta,"Bellman rejects NaN progress");
}

void test_planning_state_contract() {
  dcy::planning::State state;
  state.obligations.push_back({"auth", "verify authentication", dcy::planning::Status::Pending});
  state.evidence.push_back({42, "generation", {"auth.cpp", 10, 20}, "login returns 403", .99});
  state.hypotheses.push_back({"expired subscription", {42}, dcy::planning::Status::Unverified});
  state.contradictions.push_back({"c1", "status disagrees", {42}, dcy::planning::Status::Pending});
  state.history.push_back({{dcy::planning::ActionType::ReadSource, "E42", {200,1,0.5,1}}, false, true});
  state.remaining={500,2,100.0,3};
  check(state.obligations.size()==1 && state.evidence[0].entity==42,
        "planning state stores obligation and provenance evidence");
  check(state.hypotheses[0].status==dcy::planning::Status::Unverified,
        "hypotheses remain unverified until externally supported");
  check(state.history[0].action.type==dcy::planning::ActionType::ReadSource,
        "planning actions use a closed action type");
  const dcy::planning::ResourceCost legal{100,1,10.0,1};
  check(dcy::planning::fits(legal,state.remaining),
        "resource vector accepts an action within every limit");
  auto after=dcy::planning::consume(state.remaining,legal);
  check(after.physical_tokens==400 && after.model_calls==1 && after.steps==2,
        "transition budget consumption decrements every resource");

  const dcy::planning::ResourceCost too_many_calls{1,2,0.0,1};
  check(!dcy::planning::fits(too_many_calls,after),
        "resource vector rejects an action exceeding one component");

  dcy::planning::EnvironmentOutcome actual{"source returned",true};
  dcy::planning::EstimatedOutcome estimate{0.8,"predicted source"};
  check(actual.observation!="" && estimate.probability==.8,
        "environment and estimated outcomes remain distinct contracts");
}
void test_softmax_action_distribution() {
  const auto scores=[](int /*state*/,int action) {
    if (action==1) return .30;
    if (action==2) return .10;
    return .05;
  };
  const std::vector<int> actions{1,2,3};

  // Distribution sums to one for an arbitrary finite temperature.
  auto dist=dcy::efficiency::softmax_distribution(0,actions,scores,.1);
  double total=0.0;
  for (const auto& [action,probability]:dist) total+=probability;
  close_to(total,1.0,"softmax distribution normalizes to one");
  check(dist[0].second>dist[1].second && dist[1].second>dist[2].second,
        "softmax preserves the score ranking as probability ranking");

  // As tau -> 0, softmax concentrates on the argmax action (same winner as
  // select_best_action), matching the README's stated limit equivalence.
  auto low_tau=dcy::efficiency::softmax_distribution(0,actions,scores,1e-6);
  check(low_tau[0].second>0.999,
        "softmax with tau near zero concentrates on the best action");
  auto greedy=dcy::efficiency::select_best_action(0,actions,scores);
  check(greedy.has_value() && *greedy==1,
        "select_best_action agrees with the low-tau softmax winner");

  // A high tau flattens the distribution toward uniform.
  auto high_tau=dcy::efficiency::softmax_distribution(0,actions,scores,1000.0);
  check(std::abs(high_tau[0].second-high_tau[2].second)<0.01,
        "softmax with high tau flattens toward a uniform distribution");

  bool bad_tau=false;
  try { dcy::efficiency::softmax_distribution(0,actions,scores,0.0); }
  catch (const std::invalid_argument&) { bad_tau=true; }
  check(bad_tau,"softmax rejects a non-positive temperature");
  bool negative_tau=false;
  try { dcy::efficiency::softmax_distribution(0,actions,scores,-1.0); }
  catch (const std::invalid_argument&) { negative_tau=true; }
  check(negative_tau,"softmax rejects a negative temperature");
  bool nan_tau=false;
  try { dcy::efficiency::softmax_distribution(0,actions,scores,std::numeric_limits<double>::quiet_NaN()); }
  catch (const std::invalid_argument&) { nan_tau=true; }
  check(nan_tau,"softmax rejects a NaN temperature");

  bool nan_softmax_score=false;
  const auto nan_scores=[](int /*state*/,int /*action*/) { return std::numeric_limits<double>::quiet_NaN(); };
  try { dcy::efficiency::softmax_distribution(0,actions,nan_scores,.5); }
  catch (const std::invalid_argument&) { nan_softmax_score=true; }
  check(nan_softmax_score,"softmax rejects a non-finite action score");

  check(dcy::efficiency::softmax_distribution(0,std::vector<int>{},scores,.5).empty(),
        "softmax over no candidate actions returns an empty distribution");

  // Sampling is a deterministic function of the supplied uniform draw, so
  // benchmark runs stay reproducible under a seeded RNG.
  check(dcy::efficiency::sample_from_distribution(dist,0.0)==0,
        "sampling at draw zero returns the first action");
  check(dcy::efficiency::sample_from_distribution(dist,0.999)==2,
        "sampling near one returns the last action");
  auto same_draw_a=dcy::efficiency::sample_from_distribution(dist,.42);
  auto same_draw_b=dcy::efficiency::sample_from_distribution(dist,.42);
  check(same_draw_a==same_draw_b,"sampling is deterministic for a fixed draw");

  bool bad_draw=false;
  try { dcy::efficiency::sample_from_distribution(dist,1.0); }
  catch (const std::invalid_argument&) { bad_draw=true; }
  check(bad_draw,"sampling rejects a draw outside [0,1)");

  auto selected_low_tau=dcy::efficiency::softmax_select_action(0,actions,scores,1e-6,.5);
  check(selected_low_tau.has_value() && *selected_low_tau==1,
        "softmax_select_action matches the greedy winner at low tau");
}

void test_action_pruning_before_bellman() {
  const auto cheap_score=[](int /*state*/,int action) { return -action; }; // lower id = higher score
  auto pruned=dcy::efficiency::top_k_actions(0,std::vector<int>{5,1,3,2,4},cheap_score,std::size_t{3});
  check((pruned==std::vector<int>{1,2,3}),
        "top_k_actions keeps the k highest cheap-scoring actions");
  auto unchanged=dcy::efficiency::top_k_actions(0,std::vector<int>{1,2},cheap_score,std::size_t{5});
  check((unchanged==std::vector<int>{1,2}),
        "top_k_actions is a no-op when there are fewer actions than k");

  const auto tie_score=[](int /*state*/,int /*action*/) { return 0.0; };
  auto stable=dcy::efficiency::top_k_actions(0,std::vector<int>{9,8,7},tie_score,std::size_t{2});
  check((stable==std::vector<int>{9,8}),
        "top_k_actions keeps original order among tied cheap scores");
}

void test_outcome_pruning_by_cumulative_probability() {
  std::vector<Outcome> five{{.72,1},{.22,2},{.04,3},{.01,4},{.01,5}};
  auto kept=dcy::efficiency::prune_outcomes(five,.95);
  check(kept.size()==3,"prune_outcomes drops low-mass outcomes past the cumulative threshold");
  double total=0.0;
  for (const auto& outcome:kept) total+=outcome.probability;
  close_to(total,1.0,"prune_outcomes rescales kept outcomes back to a sum of one");
  check(kept[0].next_state==1 && kept[1].next_state==2 && kept[2].next_state==3,
        "prune_outcomes keeps the highest-probability outcomes in order");

  auto keep_all=dcy::efficiency::prune_outcomes(five,1.0);
  check(keep_all.size()==5,"prune_outcomes with threshold one keeps every outcome");

  std::vector<Outcome> single{{1.0,1}};
  auto solo=dcy::efficiency::prune_outcomes(single,.5);
  check(solo.size()==1,"prune_outcomes never drops below one outcome");

  bool bad_threshold=false;
  try { dcy::efficiency::prune_outcomes(five,0.0); }
  catch (const std::invalid_argument&) { bad_threshold=true; }
  check(bad_threshold,"prune_outcomes rejects a non-positive threshold");
  bool over_threshold=false;
  try { dcy::efficiency::prune_outcomes(five,1.5); }
  catch (const std::invalid_argument&) { over_threshold=true; }
  check(over_threshold,"prune_outcomes rejects a threshold above one");
}

void test_memoized_prospective_q_matches_exact() {
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
    if (action==1) return .05;
    if (action==2) return .30;
    return next==3 ? .50 : .02;
  };
  const auto allowed=[](int /*state*/,int action) { return action!=99; };
  const auto state_hash=[](int state) { return static_cast<std::size_t>(state); };
  const auto action_hash=[](int action) { return static_cast<std::size_t>(action); };

  std::unordered_map<std::size_t,double> cache;
  const double memoized=dcy::efficiency::memoized_prospective_q(
      0,1,2,.8,actions,outcomes,transition,progress,allowed,state_hash,action_hash,cache);
  const double exact=dcy::efficiency::prospective_q(0,1,2,.8,actions,outcomes,transition,progress,allowed);
  close_to(memoized,exact,"memoized_prospective_q matches the unmemoized Bellman value");
  check(!cache.empty(),"memoized_prospective_q populates the caller-owned cache");

  const std::size_t size_after_first=cache.size();
  const double memoized_again=dcy::efficiency::memoized_prospective_q(
      0,1,2,.8,actions,outcomes,transition,progress,allowed,state_hash,action_hash,cache);
  close_to(memoized_again,exact,"a cached call returns the same value on repeat");
  check(cache.size()==size_after_first,"a fully cached repeat call adds no new cache entries");

  bool invalid=false;
  try {
    std::unordered_map<std::size_t,double> fresh_cache;
    dcy::efficiency::memoized_prospective_q(0,99,1,.8,actions,outcomes,transition,progress,allowed,
                                            state_hash,action_hash,fresh_cache);
  } catch (const std::invalid_argument&) { invalid=true; }
  check(invalid,"memoized_prospective_q rejects an action outside constraints, like prospective_q");
}

void test_drop_no_progress_repeats() {
  const auto same_target=[](const dcy::planning::Action& a,const dcy::planning::Action& b) {
    return a.type==b.type && a.target==b.target;
  };
  std::vector<dcy::planning::ActionRecord> history;
  history.push_back({{dcy::planning::ActionType::Search,"foo",{}},true,false});
  history.push_back({{dcy::planning::ActionType::Search,"bar",{}},true,true});

  const dcy::planning::Action repeat_no_progress{dcy::planning::ActionType::Search,"foo",{}};
  const dcy::planning::Action repeat_with_progress{dcy::planning::ActionType::Search,"bar",{}};
  const dcy::planning::Action novel{dcy::planning::ActionType::Search,"baz",{}};

  check(dcy::planning::is_repeated_without_progress(history,repeat_no_progress,same_target),
        "a repeated action that produced no new evidence is flagged");
  check(!dcy::planning::is_repeated_without_progress(history,repeat_with_progress,same_target),
        "a repeated action that did produce new evidence is not flagged");
  check(!dcy::planning::is_repeated_without_progress(history,novel,same_target),
        "a novel action is never flagged as a no-progress repeat");

  std::vector<dcy::planning::Action> candidates{repeat_no_progress,repeat_with_progress,novel};
  auto filtered=dcy::planning::drop_no_progress_repeats(history,candidates,same_target);
  check(filtered.size()==2,"drop_no_progress_repeats removes only the stalled repeat");
  bool kept_novel=false, kept_progress=false, kept_stalled=false;
  for (const auto& action:filtered) {
    if (action.target=="baz") kept_novel=true;
    if (action.target=="bar") kept_progress=true;
    if (action.target=="foo") kept_stalled=true;
  }
  check(kept_novel && kept_progress && !kept_stalled,
        "drop_no_progress_repeats keeps novel and progressing actions, drops stalled ones");
}

void test_observable_value_estimate_ablation_ladder() {
  dcy::planning::State state;
  state.obligations.push_back({"o1","resolve auth",dcy::planning::Status::Resolved});
  state.obligations.push_back({"o2","resolve billing",dcy::planning::Status::Pending});
  state.evidence.push_back({1,"g",{"a.cpp",0,1},"claim1",.9,true});
  state.evidence.push_back({2,"g",{"b.cpp",0,1},"claim2",.9,false});
  state.contradictions.push_back({"c1","x",{1},dcy::planning::Status::Resolved});
  state.contradictions.push_back({"c2","y",{2},dcy::planning::Status::Pending});
  state.history.push_back({{dcy::planning::ActionType::Search,"foo",{}},true,false}); // stalled
  state.history.push_back({{dcy::planning::ActionType::Search,"bar",{}},false,true}); // progress
  state.remaining={50,1,50.0,5};
  const dcy::planning::Budget initial{100,2,100.0,10};

  close_to(dcy::planning::obligations_resolved_ratio(state),.5,
           "obligations_resolved_ratio counts resolved over total obligations");
  close_to(dcy::planning::verified_evidence_ratio(state),.5,
           "verified_evidence_ratio counts verified over total evidence");
  close_to(dcy::planning::contradictions_resolved_ratio(state),.5,
           "contradictions_resolved_ratio counts resolved over total contradictions");
  close_to(dcy::planning::no_progress_ratio(state),.5,
           "no_progress_ratio counts stalled repeats over total history");
  close_to(dcy::planning::budget_remaining_ratio(state.remaining,initial),.5,
           "budget_remaining_ratio averages the remaining fraction across resource components");

  // Every feature here happens to sit at exactly .5, so every ablation level
  // (a plain average, then minus the .5 repetition penalty) must also be .5.
  using Level=dcy::planning::ValueEstimateLevel;
  close_to(dcy::planning::estimate_value(state,initial,Level::V0Obligations),.5,
           "V0 (obligations only) matches the single observable component");
  close_to(dcy::planning::estimate_value(state,initial,Level::V1Evidence),.5,
           "V1 (obligations+evidence) averages two equal components to the same value");
  close_to(dcy::planning::estimate_value(state,initial,Level::V2Contradictions),.5,
           "V2 (+contradictions) still averages to the same value when components agree");
  close_to(dcy::planning::estimate_value(state,initial,Level::V3Budget),.5,
           "V3 (+budget) still averages to the same value when components agree");
  close_to(dcy::planning::estimate_value(state,initial,Level::V4Repetition),.375,
           "V4 (+repetition penalty) subtracts R/4 from the V3 average per the boxed formula (O+V+X+B-R)/4");

  dcy::planning::State empty_state;
  close_to(dcy::planning::estimate_value(empty_state,initial,Level::V4Repetition),0.0,
           "an empty state (no obligations, evidence, history) estimates zero value");

  dcy::planning::State perfect;
  perfect.obligations.push_back({"o1","x",dcy::planning::Status::Resolved});
  perfect.evidence.push_back({1,"g",{"a.cpp",0,1},"c",.9,true});
  perfect.contradictions.push_back({"c1","x",{1},dcy::planning::Status::Resolved});
  perfect.remaining=initial;
  close_to(dcy::planning::estimate_value(perfect,initial,Level::V4Repetition),1.0,
           "a fully resolved, fully funded, non-repeating state estimates value one");

  check(!dcy::planning::budget_exhausted(state.remaining),
        "a state with resources left in every component is not exhausted");
  dcy::planning::Budget zero_calls=state.remaining;
  zero_calls.model_calls=0;
  check(dcy::planning::budget_exhausted(zero_calls),
        "exhausting a single budget component is enough to trigger budget_exhausted");
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
  test_softmax_action_distribution();
  test_action_pruning_before_bellman();
  test_outcome_pruning_by_cumulative_probability();
  test_memoized_prospective_q_matches_exact();
  test_drop_no_progress_repeats();
  test_observable_value_estimate_ablation_ladder();
  if (failures) {
    std::fprintf(stderr,"%d efficiency check(s) failed\n",failures);
    return 1;
  }
  std::printf("all efficiency checks passed\n");
  return 0;
}
