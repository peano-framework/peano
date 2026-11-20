#include "../require-asserts.h"

#include <array>
#include <cassert>
#include <iostream>
#include <string>

#include "../..//apparent-horizon-finder.h"

namespace exahype2::ccz4::ahf::tests::iteration {

/** Flat space (gammaTilde = identity, phi = 1, no curvature) on [-1, 1]^3. */
GridData flatGrid() {
  GridData grid;
  for (int ix = -2; ix <= 2; ++ix) {
    for (int iy = -2; iy <= 2; ++iy) {
      for (int iz = -2; iz <= 2; ++iz) {
        grid.coordinates.push_back({0.5 * ix, 0.5 * iy, 0.5 * iz});
        const std::array<double, NumberOfCompactFields> row{
          1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 1.0
        };
        grid.values.insert(grid.values.end(), row.begin(), row.end());
      }
    }
  }
  return grid;
}

// An initial surface that leaves the data is rejected before any Newton
// solve, with a message naming the point and the bounds. Without the check the
// solve would start and fail on a non-finite residual instead.
void testSolvePDEforAH() {
  const GridData grid = flatGrid();
  const CartesianPointInterpolator interpolator(
    grid,
    {.method = CartesianToSurfaceMethod::NearestNeighbour, .numberOfNeighbours = 1});
  const SpectralBasis basis = BuildSpectralBasis(2);

  const SpectralNewtonResult outside =
    SolvePDEforAH(basis, interpolator, {0.0, 0.0, 0.0}, 3.0);
  assert(!outside.converged);
  assert(outside.iterations == 0);
  assert(outside.message.find("the initial guess leaves the sampled domain")
         != std::string::npos);
  assert(outside.message.find("[-1, 1] x [-1, 1] x [-1, 1]") != std::string::npos);
  assert(outside.coefficients == MakeSphericalInitialCoefficients(basis, 3.0));

  // Shifting the reference point takes part of a small sphere out of the data.
  const SpectralNewtonResult shifted =
    SolvePDEforAH(basis, interpolator, {0.0, 0.0, 0.95}, 0.3);
  assert(!shifted.converged);
  assert(shifted.message.find("leaves the sampled domain") != std::string::npos);

  // A surface inside the data goes on to the solve. Flat space has no
  // horizon, so the solve fails, but not at the domain check.
  const SpectralNewtonResult inside =
    SolvePDEforAH(basis, interpolator, {0.0, 0.0, 0.0}, 0.5);
  assert(inside.message.find("leaves the sampled domain") == std::string::npos);

  std::cout << "SolvePDEforAH rejects an initial surface that leaves the data: \""
            << outside.message.substr(0, 60) << "...\"" << std::endl;
}

void testCheckCommonHorizonOfBBH() {
  std::cout << "Placeholder test for CheckCommonHorizonOfBBH" << std::endl;
}

}  // namespace exahype2::ccz4::ahf::tests::iteration

int main() {
  using namespace exahype2::ccz4::ahf::tests::iteration;

  std::cout << "Apparent Horizon Finder iteration tests begin..." << std::endl;

  testSolvePDEforAH();
  testCheckCommonHorizonOfBBH();

  std::cout << "All iteration tests completed." << std::endl;
  return 0;
}
