#pragma once

#include <cstddef>
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

struct Action {
  ActionType type=ActionType::Search;
  std::string target;
  std::size_t requested_bytes=0;
};

struct Budget {
  std::size_t physical_tokens=0;
  std::size_t model_calls=0;
  std::size_t steps=0;
  double latency_ms=0.0;
};

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

struct OutcomeProbability {
  double probability=0.0;
};

} // namespace dcy::planning
