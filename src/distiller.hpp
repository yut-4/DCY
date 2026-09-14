#pragma once

#include "ormt.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace distiller {

struct View {
  std::string text;
  std::vector<long long> included;
  std::vector<long long> omitted;
  size_t measured_units=0;
};

// Selects evidence and resolution. ORMT only serializes an entity at the
// selected level; it does not assert that the resulting view is sufficient.
//
// Packing is reference-first, not greedy-per-entity. A greedy pass that gives
// each candidate in turn the richest level that still fits lets the first few
// candidates spend the whole budget on Detail lines, so high-scoring evidence
// further down the list is dropped entirely. Pass one therefore admits every
// candidate it can at Level::Ref; pass two spends whatever budget is left
// upgrading admitted entities in score order. The meter is arbitrary and need
// not be additive, so each trial is measured on the fully assembled view.
inline View make_view(const std::string& generation,const std::string& goal,
                      const std::string& state,const std::vector<ormt::Entity>& candidates,
                      size_t budget,const ormt::Meter& meter=ormt::byte_meter) {
  View result;
  const std::string head=ormt::header(generation,goal);
  if(meter(head)>budget) throw std::runtime_error("budget too small for goal and protocol");
  const std::string prefix=(!state.empty() && meter(head+state)<=budget) ? head+state : head;

  std::vector<std::string> lines;
  auto assemble=[&prefix](const std::vector<std::string>& parts) {
    std::string text=prefix;
    for(const auto& part:parts) text+=part;
    return text;
  };

  std::vector<size_t> admitted;
  for(size_t i=0;i<candidates.size();++i) {
    std::vector<std::string> trial=lines;
    trial.push_back(ormt::encode(candidates[i],ormt::Level::Ref));
    if(meter(assemble(trial))>budget) { result.omitted.push_back(candidates[i].id); continue; }
    lines=std::move(trial);
    admitted.push_back(i);
  }
  for(size_t slot=0;slot<admitted.size();++slot) {
    for(auto level:{ormt::Level::Detail,ormt::Level::Compact}) {
      std::vector<std::string> trial=lines;
      trial[slot]=ormt::encode(candidates[admitted[slot]],level);
      if(meter(assemble(trial))>budget) continue;
      lines=std::move(trial);
      break;
    }
  }
  for(auto index:admitted) result.included.push_back(candidates[index].id);

  result.text=assemble(lines);
  if(result.included.empty() && meter(result.text+"INSUFFICIENT\n")<=budget) result.text+="INSUFFICIENT\n";
  result.measured_units=meter(result.text);
  return result;
}

} // namespace distiller
