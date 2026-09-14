#include <clang-c/Index.h>
#include <openssl/sha.h>
#include <sqlite3.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ormt.hpp"
#include "distiller.hpp"
#ifdef DCY_ENABLE_MLPACK
#include "mlpack_retrieval.hpp"
#endif

namespace fs = std::filesystem;

static void fail(const std::string& message) { throw std::runtime_error(message); }

static std::string sha256(std::string_view bytes) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), digest);
  std::ostringstream out;
  for (auto c : digest) out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c);
  return out.str();
}

static std::string read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

struct Db {
  sqlite3* p = nullptr;
  explicit Db(const fs::path& path) {
    if (sqlite3_open(path.string().c_str(), &p) != SQLITE_OK) fail("cannot open database: " + std::string(sqlite3_errmsg(p)));
    exec("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;");
  }
  ~Db() { if (p) sqlite3_close(p); }
  Db(const Db&) = delete;
  Db& operator=(const Db&) = delete;
  void exec(const char* sql) {
    char* error = nullptr;
    if (sqlite3_exec(p, sql, nullptr, nullptr, &error) != SQLITE_OK) {
      std::string message = error ? error : sqlite3_errmsg(p);
      sqlite3_free(error);
      fail(message);
    }
  }
};

struct Stmt {
  Db& db;
  sqlite3_stmt* p = nullptr;
  Stmt(Db& db_, const char* sql) : db(db_) {
    if (sqlite3_prepare_v2(db.p, sql, -1, &p, nullptr) != SQLITE_OK) fail(sqlite3_errmsg(db.p));
  }
  ~Stmt() { sqlite3_finalize(p); }
  void str(int i, const std::string& s) { sqlite3_bind_text(p, i, s.data(), static_cast<int>(s.size()), SQLITE_TRANSIENT); }
  void integer(int i, sqlite3_int64 n) { sqlite3_bind_int64(p, i, n); }
  void blob(int i, const std::string& s) { sqlite3_bind_blob(p, i, s.data(), static_cast<int>(s.size()), SQLITE_TRANSIENT); }
  bool row() {
    int rc = sqlite3_step(p);
    if (rc == SQLITE_ROW) return true;
    if (rc == SQLITE_DONE) return false;
    fail(sqlite3_errmsg(db.p));
    return false;
  }
  void done() { if (row()) fail("unexpected query row"); }
  sqlite3_int64 id() const { return sqlite3_last_insert_rowid(db.p); }
  sqlite3_int64 i(int column) const { return sqlite3_column_int64(p, column); }
  std::string s(int column) const {
    auto* value = reinterpret_cast<const char*>(sqlite3_column_text(p, column));
    return value ? std::string(value) : std::string();
  }
  std::string b(int column) const {
    auto* value = reinterpret_cast<const char*>(sqlite3_column_blob(p, column));
    int size = sqlite3_column_bytes(p, column);
    return value && size ? std::string(value, value + size) : std::string();
  }
};

static void init(Db& db) {
  db.exec(R"sql(
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS files(
      id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL,
      content BLOB NOT NULL, indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS entities(
      id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
      kind TEXT NOT NULL, name TEXT NOT NULL, usr TEXT NOT NULL,
      start_byte INTEGER NOT NULL, end_byte INTEGER NOT NULL,
      CHECK(start_byte>=0 AND end_byte>start_byte)
    );
    CREATE TABLE IF NOT EXISTS mentions(
      id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
      target_usr TEXT NOT NULL, kind TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS relations(
      source_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
      target_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
      kind TEXT NOT NULL, PRIMARY KEY(source_id,target_id,kind)
    );
    CREATE TABLE IF NOT EXISTS goals(
      id INTEGER PRIMARY KEY, statement TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
      generation TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS observations(
      id INTEGER PRIMARY KEY, goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
      kind TEXT NOT NULL, claim TEXT NOT NULL,
      entity_id INTEGER REFERENCES entities(id) ON DELETE SET NULL,
      evidence_sha256 TEXT, evidence_path TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5(name,path,body);
    CREATE TRIGGER IF NOT EXISTS entity_ai AFTER INSERT ON entities BEGIN
      INSERT INTO entity_fts(rowid,name,path,body)
      SELECT new.id,new.name,f.path,
        CAST(substr(f.content,new.start_byte+1,new.end_byte-new.start_byte) AS TEXT)
      FROM files f WHERE f.id=new.file_id;
    END;
    CREATE TRIGGER IF NOT EXISTS entity_ad AFTER DELETE ON entities BEGIN
      DELETE FROM entity_fts WHERE rowid=old.id;
    END;
    CREATE INDEX IF NOT EXISTS entity_usr ON entities(usr);
    CREATE INDEX IF NOT EXISTS relation_target ON relations(target_id);
  )sql");
}

static std::string meta(Db& db, const std::string& key) {
  Stmt q(db, "SELECT value FROM meta WHERE key=?"); q.str(1,key);
  return q.row() ? q.s(0) : "";
}

static void set_meta(Db& db, const std::string& key, const std::string& value) {
  Stmt q(db, "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value");
  q.str(1,key); q.str(2,value); q.done();
}

static std::string cx(CXString value) {
  const char* s = clang_getCString(value);
  std::string result = s ? s : "";
  clang_disposeString(value);
  return result;
}

struct Walker {
  Db& db;
  sqlite3_int64 file_id;
  fs::path path;
  size_t size;
  int entities = 0;
  int mentions = 0;
};

static bool in_file(CXCursor cursor, const fs::path& path, unsigned& begin, unsigned& end) {
  auto range = clang_getCursorExtent(cursor);
  CXFile first = nullptr, last = nullptr;
  clang_getSpellingLocation(clang_getRangeStart(range), &first, nullptr, nullptr, &begin);
  clang_getSpellingLocation(clang_getRangeEnd(range), &last, nullptr, nullptr, &end);
  if (!first || !last || end <= begin) return false;
  return cx(clang_getFileName(first)) == path.string() && cx(clang_getFileName(last)) == path.string();
}

static bool symbol_kind(CXCursorKind kind, std::string& name) {
  switch (kind) {
    case CXCursor_FunctionDecl: name="function"; return true;
    case CXCursor_CXXMethod: name="method"; return true;
    case CXCursor_Constructor: name="constructor"; return true;
    case CXCursor_Destructor: name="destructor"; return true;
    case CXCursor_StructDecl: name="struct"; return true;
    case CXCursor_ClassDecl: name="class"; return true;
    case CXCursor_EnumDecl: name="enum"; return true;
    default: return false;
  }
}

struct Visit { Walker* walker; sqlite3_int64 owner; };

static CXChildVisitResult visit(CXCursor cursor, CXCursor, CXClientData raw) {
  auto* state = static_cast<Visit*>(raw);
  Walker& w = *state->walker;
  unsigned begin=0,end=0;
  if (!in_file(cursor,w.path,begin,end)) return CXChildVisit_Continue;
  auto kind = clang_getCursorKind(cursor);
  sqlite3_int64 owner = state->owner;
  std::string label;
  if (symbol_kind(kind,label) && clang_isCursorDefinition(cursor) && end <= w.size) {
    std::string name = cx(clang_getCursorSpelling(cursor));
    std::string usr = cx(clang_getCursorUSR(cursor));
    if (!name.empty() && !usr.empty()) {
      Stmt q(w.db,"INSERT INTO entities(file_id,kind,name,usr,start_byte,end_byte) VALUES(?,?,?,?,?,?)");
      q.integer(1,w.file_id); q.str(2,label); q.str(3,name); q.str(4,usr);
      q.integer(5,begin); q.integer(6,end); q.done();
      owner=q.id(); ++w.entities;
    }
  }
  if (kind == CXCursor_CallExpr && owner != 0) {
    std::string target = cx(clang_getCursorUSR(clang_getCursorReferenced(cursor)));
    if (!target.empty()) {
      Stmt q(w.db,"INSERT INTO mentions(source_id,target_usr,kind) VALUES(?,?,'calls')");
      q.integer(1,owner); q.str(2,target); q.done(); ++w.mentions;
    }
  }
  Visit child{&w,owner};
  clang_visitChildren(cursor,visit,&child);
  return CXChildVisit_Continue;
}

static std::pair<int,int> parse_file(Db& db, sqlite3_int64 file_id, const fs::path& path,
                                      const fs::path& root, const std::string& content) {
  CXIndex index = clang_createIndex(0,0);
  std::string extension = path.extension().string();
  const char* language = extension == ".c" ? "c" : "c++";
  std::string include_root="-I"+root.string();
  std::string include_dir="-I"+(root/"include").string();
  std::vector<const char*> args={"-x",language};
  if(extension != ".c") args.push_back("-std=c++20");
  args.push_back(include_root.c_str()); args.push_back(include_dir.c_str());
  std::string filename=path.string();
  CXUnsavedFile unsaved{filename.data(),const_cast<char*>(content.data()),static_cast<unsigned long>(content.size())};
  CXTranslationUnit tu = clang_parseTranslationUnit(index,filename.c_str(),args.data(),
    static_cast<int>(args.size()),&unsaved,1,0);
  if (!tu) { clang_disposeIndex(index); fail("libclang could not parse " + path.string()); }
  Walker walker{db,file_id,path,content.size()};
  Visit state{&walker,0};
  clang_visitChildren(clang_getTranslationUnitCursor(tu),visit,&state);
  clang_disposeTranslationUnit(tu);
  clang_disposeIndex(index);
  return {walker.entities,walker.mentions};
}

static bool eligible(const fs::path& path) {
  static const std::set<std::string> extensions={".c",".cc",".cpp",".cxx",".h",".hh",".hpp",".hxx"};
  return extensions.count(path.extension().string()) != 0;
}

static void index_repo(const fs::path& root_arg, const fs::path& db_path) {
  fs::path root=fs::canonical(root_arg);
  fs::create_directories(db_path.parent_path().empty()?fs::path("."):db_path.parent_path());
  Db db(db_path); init(db);
  std::string old_root=meta(db,"root");
  if (!old_root.empty() && old_root != root.string()) fail("database belongs to another repository");
  db.exec("BEGIN IMMEDIATE");
  try {
    set_meta(db,"root",root.string());
    std::map<std::string,std::string> seen;
    int changed=0,unchanged=0,removed=0,symbols=0,calls=0;
    fs::recursive_directory_iterator it(root,fs::directory_options::skip_permission_denied), end;
    for (;it!=end;++it) {
      if (it->is_directory()) {
        auto n=it->path().filename().string();
        if (n==".git" || n==".dcy" || n=="build" || n==".venv") it.disable_recursion_pending();
        continue;
      }
      if (!it->is_regular_file() || it->is_symlink() || !eligible(it->path())) continue;
      auto path=fs::canonical(it->path());
      auto rel=fs::relative(path,root).generic_string();
      std::string content=read_file(path), digest=sha256(content);
      seen[rel]=digest;
      Stmt existing(db,"SELECT id,sha256 FROM files WHERE path=?"); existing.str(1,rel);
      if (existing.row() && existing.s(1)==digest) { ++unchanged; continue; }
      sqlite3_int64 old_id=0;
      Stmt find(db,"SELECT id FROM files WHERE path=?"); find.str(1,rel);
      if (find.row()) old_id=find.i(0);
      if (old_id) {
        Stmt q(db,"DELETE FROM files WHERE id=?"); q.integer(1,old_id); q.done();
      }
      Stmt insert(db,"INSERT INTO files(path,sha256,content) VALUES(?,?,?)");
      insert.str(1,rel); insert.str(2,digest); insert.blob(3,content); insert.done();
      auto [n,m]=parse_file(db,insert.id(),path,root,content);
      symbols+=n; calls+=m; ++changed;
    }
    std::vector<sqlite3_int64> stale;
    Stmt all(db,"SELECT id,path FROM files");
    while(all.row()) if (!seen.count(all.s(1))) stale.push_back(all.i(0));
    for (auto id:stale) { Stmt q(db,"DELETE FROM files WHERE id=?"); q.integer(1,id); q.done(); ++removed; }
    db.exec("DELETE FROM relations;");
    db.exec("INSERT OR IGNORE INTO relations(source_id,target_id,kind) SELECT m.source_id,e.id,m.kind FROM mentions m JOIN entities e ON e.usr=m.target_usr WHERE m.source_id<>e.id;");
    std::string material;
    for (auto& [path,digest]:seen) { material+=path; material+='\0'; material+=digest; material+='\n'; }
    std::string generation=sha256(material);
    set_meta(db,"generation",generation);
    db.exec("COMMIT");
    std::cout<<"generation "<<generation<<"\nchanged "<<changed<<" unchanged "<<unchanged<<" removed "<<removed
             <<" symbols_new "<<symbols<<" calls_new "<<calls<<"\n";
  } catch (...) { db.exec("ROLLBACK"); throw; }
}

static std::string fts_query(std::string_view input) {
  std::vector<std::string> terms;
  std::string current;
  for (unsigned char c:input) {
    if (std::isalnum(c) || c=='_') current+=static_cast<char>(c);
    else if (current.size()>=2) { terms.push_back(current); current.clear(); }
    else current.clear();
  }
  if (current.size()>=2) terms.push_back(current);
  std::string result;
  for (size_t i=0;i<terms.size() && i<12;++i) {
    if (i) result+=" OR ";
    result+='"'; result+=terms[i]; result+='"';
  }
  return result;
}

struct Candidate { sqlite3_int64 id; std::string kind,name,path,sha; int start,end; int score; };

static std::vector<Candidate> retrieve(Db& db, const std::string& goal, bool use_mlpack=false,
                                       bool use_graph=true) {
  std::map<sqlite3_int64,int> scores;
  auto query=fts_query(goal);
  if (!query.empty()) {
    Stmt q(db,"SELECT rowid FROM entity_fts WHERE entity_fts MATCH ? ORDER BY bm25(entity_fts),rowid LIMIT 24");
    q.str(1,query); int rank=0;
    while(q.row()) scores[q.i(0)]=std::max(scores[q.i(0)],100-rank++*2);
  }
  if(use_mlpack) {
#ifdef DCY_ENABLE_MLPACK
    std::vector<std::pair<long long,std::string>> symbols;
    Stmt all(db,"SELECT e.id,e.name || ' ' || f.path FROM entities e JOIN files f ON f.id=e.file_id ORDER BY e.id");
    while(all.row()) symbols.emplace_back(all.i(0),all.s(1));
    for(auto [id,score]:mlpack_retrieval::nearest(symbols,goal)) scores[id]=std::max(scores[id],score);
#else
    fail("mlpack backend is not built; configure with -DDCY_ENABLE_MLPACK=ON");
#endif
  }
  if(use_graph) {
    std::vector<sqlite3_int64> seeds;
    for (auto& [id,score]:scores) if(score>=65) seeds.push_back(id);
    for (auto id:seeds) {
      Stmt q(db,"SELECT target_id FROM relations WHERE source_id=? UNION SELECT source_id FROM relations WHERE target_id=? LIMIT 32");
      q.integer(1,id); q.integer(2,id);
      while(q.row()) scores[q.i(0)]=std::max(scores[q.i(0)],40);
    }
  }
  std::vector<Candidate> result;
  for (auto& [id,score]:scores) {
    Stmt q(db,"SELECT e.kind,e.name,f.path,f.sha256,e.start_byte,e.end_byte FROM entities e JOIN files f ON f.id=e.file_id WHERE e.id=?");
    q.integer(1,id);
    if (q.row()) result.push_back({id,q.s(0),q.s(1),q.s(2),q.s(3),static_cast<int>(q.i(4)),static_cast<int>(q.i(5)),score});
  }
  std::sort(result.begin(),result.end(),[](const auto& a,const auto& b) {
    if (a.score!=b.score) return a.score>b.score;
    if (a.path!=b.path) return a.path<b.path;
    if (a.start!=b.start) return a.start<b.start;
    return a.id<b.id;
  });
  return result;
}

static std::string render(Db& db, const std::string& generation, const std::string& goal,
                          const std::vector<Candidate>& candidates, size_t budget,
                          sqlite3_int64 goal_id=0) {
  std::vector<ormt::Entity> items;
  for (auto& c:candidates) {
    ormt::Entity e{c.id,c.kind,c.name,c.path,c.sha,c.start,c.end,c.score,{}};
    Stmt q(db,"SELECT 'CALL>E'||target_id FROM relations WHERE source_id=? ORDER BY target_id LIMIT 6");
    q.integer(1,c.id);
    while(q.row()) e.relations.push_back(q.s(0));
    items.push_back(std::move(e));
  }
  std::string state;
  if(goal_id) {
    Stmt q(db,"SELECT kind,claim,entity_id FROM observations WHERE goal_id=? ORDER BY id DESC LIMIT 3");
    q.integer(1,goal_id);
    while(q.row()) {
      std::string claim=q.s(1);
      if(claim.size()>100) claim=claim.substr(0,100)+"...";
      state+="STATE "+q.s(0)+": "+claim;
      if(q.i(2)) state+=" E"+std::to_string(q.i(2));
      state+='\n';
    }
  }
  return distiller::make_view(generation,goal,state,items,budget).text;
}

static void source(Db& db, sqlite3_int64 id, const std::string& generation, size_t max_bytes) {
  if (meta(db,"generation")!=generation) fail("stale generation");
  Stmt q(db,"SELECT f.content,f.sha256,e.start_byte,e.end_byte,f.path FROM entities e JOIN files f ON f.id=e.file_id WHERE e.id=?");
  q.integer(1,id);
  if (!q.row()) fail("unknown entity");
  auto content=q.b(0), sha=q.s(1);
  if (sha256(content)!=sha) fail("source hash mismatch");
  size_t start=static_cast<size_t>(q.i(2)), end=static_cast<size_t>(q.i(3));
  if (end>content.size() || end<=start) fail("invalid source span");
  if (end-start>max_bytes) fail("source span exceeds max-bytes; request a larger explicit limit");
  std::cout<<"SOURCE E"<<id<<" "<<q.s(4)<<":"<<start<<"-"<<end<<" sha="<<sha<<"\n";
  std::cout.write(content.data()+start,static_cast<std::streamsize>(end-start));
  std::cout<<"\n";
}

static sqlite3_int64 number(const std::string& s) {
  size_t pos=0; long long value=0;
  try { value=std::stoll(s,&pos); } catch (...) { fail("invalid number: "+s); }
  if(pos!=s.size() || value<0) fail("invalid number: "+s);
  return value;
}

static void usage() {
  std::cerr<<"usage:\n  dcy index REPO DB\n  dcy goal DB STATEMENT\n  dcy context DB GOAL_ID BUDGET_BYTES\n  dcy finish DB GOAL_ID resolved|failed|blocked\n  dcy query DB STATEMENT BUDGET_BYTES\n  dcy query-fts DB STATEMENT BUDGET_BYTES\n  dcy query-mlpack DB STATEMENT BUDGET_BYTES\n  dcy source DB ENTITY_ID GENERATION MAX_BYTES\n  dcy record DB GOAL_ID KIND CLAIM [ENTITY_ID]\n";
}

int main(int argc,char** argv) {
  try {
    if (argc<2) { usage(); return 2; }
    std::string op=argv[1];
    if(op=="index" && argc==4) { index_repo(argv[2],argv[3]); return 0; }
    if(argc<4) { usage(); return 2; }
    Db db(argv[2]); init(db);
    if(op=="goal" && argc==4) {
      std::string generation=meta(db,"generation");
      if(generation.empty()) fail("index a repository first");
      Stmt q(db,"INSERT INTO goals(statement,generation) VALUES(?,?)"); q.str(1,argv[3]); q.str(2,generation); q.done();
      std::cout<<"G"<<q.id()<<" generation="<<generation<<"\n"; return 0;
    }
    if((op=="query" || op=="query-fts" || op=="query-mlpack") && argc==5) {
      std::string generation=meta(db,"generation");
      if(generation.empty()) fail("index a repository first");
      std::cout<<render(db,generation,argv[3],retrieve(db,argv[3],op=="query-mlpack",op!="query-fts"),number(argv[4])); return 0;
    }
    if(op=="context" && argc==5) {
      Stmt q(db,"SELECT statement,generation,status FROM goals WHERE id=?"); q.integer(1,number(argv[3]));
      if(!q.row()) fail("unknown goal");
      std::string statement=q.s(0),generation=q.s(1),status=q.s(2);
      if(generation!=meta(db,"generation")) fail("goal belongs to stale generation");
      if(status!="active") fail("goal is not active");
      std::cout<<render(db,generation,statement,retrieve(db,statement),number(argv[4]),number(argv[3])); return 0;
    }
    if(op=="finish" && argc==5) {
      std::string status=argv[4];
      if(status!="resolved" && status!="failed" && status!="blocked") fail("invalid goal status");
      Stmt q(db,"UPDATE goals SET status=? WHERE id=? AND generation=? AND status='active'");
      q.str(1,status); q.integer(2,number(argv[3])); q.str(3,meta(db,"generation")); q.done();
      if(sqlite3_changes(db.p)!=1) fail("unknown, stale, or already finished goal");
      std::cout<<"G"<<argv[3]<<" "<<status<<"\n"; return 0;
    }
    if(op=="source" && argc==6) { source(db,number(argv[3]),argv[4],number(argv[5])); return 0; }
    if(op=="record" && (argc==6 || argc==7)) {
      std::string kind=argv[4];
      if(kind!="hypothesis" && kind!="fact" && kind!="rejected" && kind!="result") fail("invalid observation kind");
      sqlite3_int64 goal_id=number(argv[3]);
      Stmt g(db,"SELECT generation FROM goals WHERE id=?"); g.integer(1,goal_id);
      if(!g.row() || g.s(0)!=meta(db,"generation")) fail("unknown or stale goal");
      sqlite3_int64 entity_id=0;
      if(argc==7) {
        entity_id=number(argv[6]);
        Stmt e(db,"SELECT id FROM entities WHERE id=?"); e.integer(1,entity_id);
        if(!e.row()) fail("unknown entity");
      }
      if(kind=="fact" && !entity_id) fail("fact requires entity evidence");
      Stmt q(db,"INSERT INTO observations(goal_id,kind,claim,entity_id,evidence_sha256,evidence_path) VALUES(?,?,?,?,?,?)");
      q.integer(1,goal_id); q.str(2,kind); q.str(3,argv[5]);
      if(entity_id) {
        q.integer(4,entity_id);
        Stmt evidence(db,"SELECT f.sha256,f.path FROM entities e JOIN files f ON f.id=e.file_id WHERE e.id=?");
        evidence.integer(1,entity_id); evidence.row();
        q.str(5,evidence.s(0)); q.str(6,evidence.s(1));
      }
      q.done();
      std::cout<<"R"<<q.id()<<"\n"; return 0;
    }
    usage(); return 2;
  } catch(const std::exception& e) { std::cerr<<"dcy: "<<e.what()<<"\n"; return 1; }
}
