// Unit tests for the distiller packing contract, independent of SQLite, libclang
// and the CLI. These pin the behaviour that a pilot measured as a 44.4% -> 63.9%
// injected-recall change: packing must be coverage-first, not greedy-per-entity.

#include "../src/distiller.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
  if (condition) return;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++failures;
}

ormt::Entity entity(long long id, const std::string& name, int score) {
  return ormt::Entity{id, "function", name, "src/big/path/module.c",
                      "0123456789abcdef0123456789abcdef", 1000, 2000, score, {}};
}

std::vector<ormt::Entity> candidates(size_t count) {
  std::vector<ormt::Entity> result;
  for (size_t i = 0; i < count; ++i) {
    result.push_back(entity(static_cast<long long>(i + 1),
                            "ts_symbol_number_" + std::to_string(i), 100 - static_cast<int>(i)));
  }
  return result;
}

bool contains(const std::vector<long long>& haystack, long long needle) {
  return std::find(haystack.begin(), haystack.end(), needle) != haystack.end();
}

const std::string kGeneration(64, 'a');

// A Detail rendering of these entities is well over 100 bytes, so a greedy
// packer spends the whole evidence budget on the first two or three candidates.
// Coverage-first must admit far more of them at Ref level instead.
void test_coverage_beats_greedy() {
  auto items = candidates(40);
  auto view = distiller::make_view(kGeneration, "locate the symbol", "", items, 512);
  check(view.measured_units <= 512, "coverage view respects the byte budget");
  check(view.included.size() >= 8,
        "coverage-first admits many candidates, got " + std::to_string(view.included.size()));
  check(view.included.size() + view.omitted.size() == items.size(),
        "every candidate is either included or omitted");
}

// The specific failure found in the pilot: gold ranked below the top candidates
// was dropped because earlier entities took Detail renderings.
void test_low_ranked_gold_survives() {
  auto items = candidates(40);
  auto view = distiller::make_view(kGeneration, "locate the symbol", "", items, 512);
  check(contains(view.included, 8), "the 8th-ranked candidate survives packing");
}

// Budget must still bind, and the highest-scoring candidate must still be first.
void test_budget_and_order() {
  auto items = candidates(40);
  for (size_t budget : {256u, 512u, 1024u, 4096u}) {
    auto view = distiller::make_view(kGeneration, "locate the symbol", "", items, budget);
    check(view.measured_units <= budget,
          "budget " + std::to_string(budget) + " respected");
    if (!view.included.empty()) {
      check(view.included.front() == 1,
            "highest-scoring candidate stays first at budget " + std::to_string(budget));
    }
  }
}

// Leftover budget must be spent upgrading, otherwise coverage-first would
// throw away the detail a large budget can afford.
void test_upgrade_uses_leftover_budget() {
  auto items = candidates(2);
  auto view = distiller::make_view(kGeneration, "locate the symbol", "", items, 4096);
  check(view.included.size() == 2, "both candidates admitted at a large budget");
  check(view.text.find("source=src/big/path/module.c") != std::string::npos,
        "leftover budget promotes an entity to a Detail rendering");
}

// Every rendering level must expose the entity ID as a trailing [ref=E<id>] tag
// and must not lead with it, because leading IDs get copied as answers.
void test_reference_is_not_a_line_prefix() {
  auto items = candidates(1);
  for (size_t budget : {112u, 200u, 4096u}) {
    auto view = distiller::make_view(kGeneration, "locate the symbol", "", items, budget);
    if (view.included.empty()) continue;
    check(view.text.find("[ref=E1]") != std::string::npos,
          "entity reference tag present at budget " + std::to_string(budget));
    check(view.text.find("\nE1 ") == std::string::npos,
          "no line starts with a bare entity ID at budget " + std::to_string(budget));
    check(view.text.find("ts_symbol_number_0") != std::string::npos,
          "symbol name present at budget " + std::to_string(budget));
  }
}

void test_header_and_state() {
  auto items = candidates(3);
  auto view = distiller::make_view(kGeneration, "locate the symbol", "STATE fact: prior\n", items, 512);
  check(view.text.rfind("DCY/0.1 generation=", 0) == 0, "header comes first");
  check(view.text.find("STATE fact: prior") != std::string::npos,
        "goal state survives when it fits");

  auto tight = distiller::make_view(kGeneration, "g", std::string(400, 'x') + "\n", items, 200);
  check(tight.text.find('x') == std::string::npos, "oversized state is dropped, not truncated");

  bool threw = false;
  try {
    distiller::make_view(kGeneration, "locate the symbol", "", items, 80);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "a budget below the protocol header is rejected");
}

void test_insufficient_marker() {
  auto empty = distiller::make_view(kGeneration, "locate the symbol", "", {}, 512);
  check(empty.text.find("INSUFFICIENT") != std::string::npos,
        "no candidates yields an INSUFFICIENT marker");

  // Candidates exist but none fit: the view must say so rather than look complete.
  // The header here is 390 bytes, "INSUFFICIENT\n" is 13, and the cheapest Ref
  // line is larger than the remainder, so the marker fits and no evidence does.
  auto items = candidates(3);
  auto starved = distiller::make_view(kGeneration, std::string(300, 'g'), "", items, 405);
  check(starved.included.empty(), "no candidate fits the starved budget");
  check(starved.text.find("INSUFFICIENT") != std::string::npos,
        "candidates that all fail to fit yield an INSUFFICIENT marker");
  check(starved.measured_units <= 405, "starved view still respects the budget");

  // When the marker cannot fit either, the budget still binds.
  auto tighter = distiller::make_view(kGeneration, std::string(300, 'g'), "", items, 395);
  check(tighter.measured_units <= 395, "unmarked starved view still respects the budget");
}

// The meter is a host-supplied callable and need not be additive; a token-like
// meter must not break the budget contract.
void test_non_additive_meter() {
  auto items = candidates(20);
  auto words = [](const std::string& text) {
    size_t count = 0;
    bool inside = false;
    for (char c : text) {
      bool space = (c == ' ' || c == '\n');
      if (!space && !inside) ++count;
      inside = !space;
    }
    return count;
  };
  auto view = distiller::make_view(kGeneration, "locate the symbol", "", items, 40, words);
  check(words(view.text) <= 40, "non-additive meter budget respected");
  check(!view.included.empty(), "non-additive meter still admits candidates");
}

}  // namespace

int main() {
  test_coverage_beats_greedy();
  test_low_ranked_gold_survives();
  test_budget_and_order();
  test_upgrade_uses_leftover_budget();
  test_reference_is_not_a_line_prefix();
  test_header_and_state();
  test_insufficient_marker();
  test_non_additive_meter();
  if (failures) {
    std::fprintf(stderr, "%d distiller check(s) failed\n", failures);
    return 1;
  }
  std::printf("all distiller checks passed\n");
  return 0;
}
