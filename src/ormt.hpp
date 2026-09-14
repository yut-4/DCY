#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace ormt {

struct Entity {
  long long id;
  std::string kind;
  std::string name;
  std::string path;
  std::string sha256;
  int start;
  int end;
  int score;
  std::vector<std::string> relations;
};

using Meter = std::function<size_t(const std::string&)>;
inline size_t byte_meter(const std::string& text) { return text.size(); }
enum class Level { Detail, Compact, Ref };

inline std::string relation_text(const Entity& e, size_t limit) {
  std::string out;
  for (size_t i=0;i<e.relations.size() && i<limit;++i) {
    if (i) out+=',';
    out+=e.relations[i];
  }
  if (e.relations.size()>limit) out+="+"+std::to_string(e.relations.size()-limit);
  return out;
}

inline std::string detailed(const Entity& e) {
  std::string line="E"+std::to_string(e.id)+" "+e.kind+" "+e.name+"\n  source="+e.path+":"+
    std::to_string(e.start)+"-"+std::to_string(e.end)+" sha="+e.sha256.substr(0,12)+
    " score="+std::to_string(e.score);
  if (!e.relations.empty()) line+="\n  relations="+relation_text(e,4);
  return line+"\n";
}

inline std::string compact(const Entity& e) {
  std::string line="E"+std::to_string(e.id)+" "+e.name+" @"+e.path+":"+
    std::to_string(e.start)+"-"+std::to_string(e.end);
  if (!e.relations.empty()) line+=" >"+relation_text(e,2);
  return line+"\n";
}

inline std::string minimal(const Entity& e) {
  return "E"+std::to_string(e.id)+" "+e.name+"\n";
}

inline std::string header(const std::string& generation,const std::string& goal) {
  return "DCY/0.1 generation="+generation+"\nGOAL "+goal+"\n";
}

inline std::string encode(const Entity& e,Level level) {
  switch(level) {
    case Level::Detail: return detailed(e);
    case Level::Compact: return compact(e);
    case Level::Ref: return minimal(e);
  }
  return {};
}

} // namespace ormt
