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
inline View make_view(const std::string& generation,const std::string& goal,
                      const std::string& state,const std::vector<ormt::Entity>& candidates,
                      size_t budget,const ormt::Meter& meter=ormt::byte_meter) {
  View result;
  result.text=ormt::header(generation,goal);
  if(meter(result.text)>budget) throw std::runtime_error("budget too small for goal and protocol");
  if(!state.empty() && meter(result.text+state)<=budget) result.text+=state;
  for(const auto& entity:candidates) {
    bool selected=false;
    for(auto level:{ormt::Level::Detail,ormt::Level::Compact,ormt::Level::Ref}) {
      std::string line=ormt::encode(entity,level);
      if(meter(result.text+line)>budget) continue;
      result.text+=line;
      result.included.push_back(entity.id);
      selected=true;
      break;
    }
    if(!selected) result.omitted.push_back(entity.id);
  }
  if(candidates.empty() && meter(result.text+"INSUFFICIENT\n")<=budget) result.text+="INSUFFICIENT\n";
  result.measured_units=meter(result.text);
  return result;
}

} // namespace distiller
