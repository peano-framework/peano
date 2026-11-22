#include "../require-asserts.h"

#include "../../initial-guess.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace exahype2::ccz4::ahf::tests::initial_guess {
namespace {

bool ApproximatelyEqual(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

void TestSphericalInitialCoefficients() {
  constexpr double radius = 2.5;
  constexpr double tolerance = 2.0e-12;
  for (int lmax : {0, 1, 3, 6}) {
    const SpectralBasis basis = BuildSpectralBasis(lmax);
    const std::vector<double> coefficients =
      MakeSphericalInitialCoefficients(basis, radius);
    assert(coefficients.size() == basis.nModes);

    const SynthesizedField field = Synthesize(basis, coefficients);
    for (std::size_t point = 0; point < basis.nPoints; ++point) {
      assert(ApproximatelyEqual(field.h[point], radius, tolerance));
      assert(ApproximatelyEqual(field.hRho[point], 0.0, tolerance));
      assert(ApproximatelyEqual(field.hSigma[point], 0.0, tolerance));
      assert(ApproximatelyEqual(field.hRhoRho[point], 0.0, tolerance));
      assert(ApproximatelyEqual(field.hRhoSigma[point], 0.0, tolerance));
      assert(ApproximatelyEqual(field.hSigmaSigma[point], 0.0, tolerance));
    }
  }
}

void TestInvalidRadii() {
  const SpectralBasis basis = BuildSpectralBasis(2);
  for (double radius : {
         0.0,
         -1.0,
         std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::infinity()
       }) {
    bool rejected = false;
    try {
      (void)MakeSphericalInitialCoefficients(basis, radius);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
}

void TestDistortedInitialCoefficients() {
  const SpectralBasis basis = BuildSpectralBasis(3);
  std::vector<double> radii(basis.nPoints);
  for (std::size_t p = 0; p < basis.nPoints; ++p) {
    radii[p] = 1.2 + 0.15 * EvalMode(
      2, 1, basis.rho[p], basis.sigma[p]).Y;
  }
  const std::vector<double> coefficients =
    MakeDistortedInitialCoefficients(basis, radii);
  const SynthesizedField field = Synthesize(basis, coefficients);
  assert(coefficients.size() == basis.nModes);
  for (std::size_t p = 0; p < basis.nPoints; ++p) {
    assert(ApproximatelyEqual(field.h[p], radii[p], 3.0e-12));
  }

  bool rejected = false;
  try { (void)MakeDistortedInitialCoefficients(basis, {1.0}); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  radii[4] = 0.0;
  rejected = false;
  try { (void)MakeDistortedInitialCoefficients(basis, radii); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests::initial_guess

int main() {
  using namespace exahype2::ccz4::ahf::tests::initial_guess;
  TestSphericalInitialCoefficients();
  TestInvalidRadii();
  TestDistortedInitialCoefficients();
  std::cout << "All initial-guess tests completed.\n";
}
