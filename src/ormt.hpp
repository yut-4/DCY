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

// Renderings are name-first. An entity ID placed at the start of every line is
// a visually privileged token that also looks like a valid answer, and small
// models copy it instead of the symbol name. The name carries the semantics;
// the ID is routing metadata for `dcy source`, so it is rendered as a trailing
// [ref=E<id>] tag. Hosts parse the tag, not the line prefix.
inline std::string reference(const Entity& e) {
  return "[ref=E"+std::to_string(e.id)+"]";
}

inline std::string detailed(const Entity& e) {
  std::string line=e.kind+" "+e.name+" "+reference(e)+"\n  source="+e.path+":"+
    std::to_string(e.start)+"-"+std::to_string(e.end)+" sha="+e.sha256.substr(0,12)+
    " score="+std::to_string(e.score);
  if (!e.relations.empty()) line+="\n  relations="+relation_text(e,4);
  return line+"\n";
}

inline std::string compact(const Entity& e) {
  std::string line=e.name+" @"+e.path+":"+
    std::to_string(e.start)+"-"+std::to_string(e.end)+" "+reference(e);
  if (!e.relations.empty()) line+=" >"+relation_text(e,2);
  return line+"\n";
}

inline std::string minimal(const Entity& e) {
  return e.name+" "+reference(e)+"\n";
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
