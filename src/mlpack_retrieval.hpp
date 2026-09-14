#pragma once

// Experimental lexical-neighbor backend. It is intentionally not called
// "semantic": hashed character trigrams cannot substitute for embeddings.

#include <mlpack.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mlpack_retrieval {

inline std::uint64_t hash3(unsigned char a,unsigned char b,unsigned char c) {
  std::uint64_t h=1469598103934665603ULL;
  for (unsigned char x:{a,b,c}) { h^=x; h*=1099511628211ULL; }
  return h;
}

inline void encode(const std::string& text, arma::mat& matrix, size_t column) {
  std::string lower;
  for(unsigned char c:text) lower+=static_cast<char>(std::tolower(c));
  if(lower.size()<3) return;
  for(size_t i=0;i+2<lower.size();++i) {
    size_t slot=hash3(static_cast<unsigned char>(lower[i]),static_cast<unsigned char>(lower[i+1]),static_cast<unsigned char>(lower[i+2]))%128;
    matrix(slot,column)+=1.0;
  }
  double norm=arma::norm(matrix.col(column),2);
  if(norm>0) matrix.col(column)/=norm;
}

inline std::vector<std::pair<long long,int>> nearest(
    const std::vector<std::pair<long long,std::string>>& symbols,
    const std::string& goal, size_t limit=24) {
  if(symbols.empty()) return {};
  arma::mat reference(128,symbols.size(),arma::fill::zeros);
  for(size_t i=0;i<symbols.size();++i) encode(symbols[i].second,reference,i);
  arma::mat query(128,1,arma::fill::zeros);
  encode(goal,query,0);
  if(arma::norm(query.col(0),2)==0) return {};
  mlpack::KNN knn(std::move(reference),mlpack::SINGLE_TREE);
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  knn.Search(query,std::min(limit,symbols.size()),neighbors,distances);
  std::vector<std::pair<long long,int>> result;
  for(size_t i=0;i<neighbors.n_rows;++i) {
    auto column=neighbors(i,0);
    if(column>=symbols.size()) continue;
    // Unit vectors: squared distance > 1.6 is weak lexical evidence.
    if(distances(i,0)>1.6) continue;
    result.emplace_back(symbols[column].first,80-static_cast<int>(i)*2);
  }
  return result;
}

} // namespace mlpack_retrieval
