#ifndef APPARENT_HORIZON_FINDER_TEST_BRILL_LINDQUIST_GRID_H_
#define APPARENT_HORIZON_FINDER_TEST_BRILL_LINDQUIST_GRID_H_

#include "../ahf-data.h"
#include "../spacetimes.h"

#include <array>
#include <stdexcept>

namespace exahype2::ccz4::ahf::tests {

/**
 * Brill-Lindquist data at the cell centres of a regular grid on
 * [-halfWidth, halfWidth]^3: gammaTilde = delta, ATilde = 0, trace K = 0 and
 * phi = psi^-2. TwoPunctures produces exactly this for punctures without
 * momentum or spin, so these grids stand in for ExaGRyPE binary snapshots
 * while the analytic BrillLindquist spacetime gives the exact horizons.
 *
 * Use an even cellsPerAxis and punctures off the cell centres: psi is
 * singular at a puncture.
 */
inline GridData MakeBrillLindquistGrid(
  const BrillLindquist& spacetime,
  double                halfWidth,
  int                   cellsPerAxis
) {
  if (!(halfWidth > 0.0) || cellsPerAxis < 2) {
    throw std::invalid_argument("MakeBrillLindquistGrid: invalid grid");
  }
  const double spacing = 2.0 * halfWidth / cellsPerAxis;
  GridData grid;
  const std::size_t cells = static_cast<std::size_t>(cellsPerAxis);
  grid.coordinates.reserve(cells * cells * cells);
  for (int i = 0; i < cellsPerAxis; ++i) {
    for (int j = 0; j < cellsPerAxis; ++j) {
      for (int k = 0; k < cellsPerAxis; ++k) {
        const FieldVector<double> x{
          -halfWidth + (i + 0.5) * spacing,
          -halfWidth + (j + 0.5) * spacing,
          -halfWidth + (k + 0.5) * spacing};
        const double psi = spacetime.psi(x);
        grid.coordinates.push_back({x[0], x[1], x[2]});
        const std::array<double, NumberOfCompactFields> row{
          1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 1.0 / (psi * psi)};
        grid.values.insert(grid.values.end(), row.begin(), row.end());
      }
    }
  }
  return grid;
}

}  // namespace exahype2::ccz4::ahf::tests

#endif
