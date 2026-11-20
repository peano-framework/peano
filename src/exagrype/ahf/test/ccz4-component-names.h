#ifndef APPARENT_HORIZON_FINDER_TEST_CCZ4_COMPONENT_NAMES_H_
#define APPARENT_HORIZON_FINDER_TEST_CCZ4_COMPONENT_NAMES_H_

#include <string>
#include <utility>
#include <vector>

namespace exahype2::ccz4::ahf::tests {

// Component names as Peano's patch-file converter writes them: one name per
// unknown, v[0], ..., v[k-1] for a variable of multiplicity k, and a trailing
// subdomain component.
inline std::vector<std::string> UnknownNames(
  const std::vector<std::pair<std::string, int>>& unknowns
) {
  std::vector<std::string> names;
  for (const auto& [name, multiplicity] : unknowns) {
    if (multiplicity == 1) {
      names.push_back(name);
    } else {
      for (int i = 0; i < multiplicity; ++i) {
        names.push_back(name + "[" + std::to_string(i) + "]");
      }
    }
  }
  names.push_back("subdomain");
  return names;
}

// FOCCZ4Solver.Unknowns (src/exagrype/api/FOCCZ4Solver.py).
inline std::vector<std::string> ExaGRyPEComponentNames() {
  return UnknownNames({
    {"G", 6}, {"K", 6}, {"theta", 1}, {"Z", 3}, {"lapse", 1}, {"shift", 3},
    {"b", 3}, {"traceK", 1}, {"phi", 1}, {"dLapse", 3}, {"dxShift", 3},
    {"dyShift", 3}, {"dzShift", 3}, {"dxG", 6}, {"dyG", 6}, {"dzG", 6},
    {"P", 3}, {"K0", 1}
  });
}

// The unknowns of the legacy applications/exahype2/ccz4/ccz4.py.
inline std::vector<std::string> LegacyCCZ4ComponentNames() {
  return UnknownNames({
    {"G", 6}, {"K", 6}, {"theta", 1}, {"Z", 3}, {"lapse", 1}, {"shift", 3},
    {"b", 3}, {"dLapse", 3}, {"dxShift", 3}, {"dyShift", 3}, {"dzShift", 3},
    {"dxG", 6}, {"dyG", 6}, {"dzG", 6}, {"traceK", 1}, {"phi", 1},
    {"P", 3}, {"K0", 1}
  });
}

}  // namespace exahype2::ccz4::ahf::tests

#endif  // APPARENT_HORIZON_FINDER_TEST_CCZ4_COMPONENT_NAMES_H_
