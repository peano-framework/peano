#include "../require-asserts.h"

#include "../../apparent-horizon-finder.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace exahype2::ccz4::ahf::tests::analytic_horizon {

  namespace {

    bool approxEqual(double a, double b, double tol) {
      return std::abs(a - b) <= tol;
    }

    const ReferencePoint origin{0.0, 0.0, 0.0};

    // Drive a full damped-Newton horizon find on an analytic spacetime from a
    // coordinate sphere of radius r0.
    template <class Spacetime>
    SpectralNewtonResult solve(
      const SpectralBasis& basis,
      const Spacetime&     spacetime,
      double               r0
    ) {
      const SpectralResidual residual = MakeAnalyticResidual(
        basis,
        spacetime,
        origin
      );
      const std::vector<double> initial = SphereCoefficients(basis, r0);
      return SolveSpectralNewton(basis, residual, initial);
    }

  } // namespace

  // Schwarzschild isotropic: coordinate radius M/2, areal radius 2M.
  void testSchwarzschildIsotropic() {
    const SpectralBasis          basis = BuildSpectralBasis(4);
    const SchwarzschildIsotropic spacetime{1.0};
    const SpectralNewtonResult   result = solve(basis, spacetime, 0.8);
    assert(result.converged);

    const double meanRadius = MeanCoordinateRadius(result.coefficients);
    const double area       = HorizonArea(
      basis,
      result.coefficients,
      spacetime,
      origin
    );
    assert(approxEqual(meanRadius, 0.5, 1.0e-6)); // coordinate radius M/2
    assert(approxEqual(ArealRadius(area), 2.0, 1.0e-6)); // areal radius 2M

    std::cout
      << "Schwarzschild isotropic: coord radius " << meanRadius
      << ", areal radius " << ArealRadius(area) << " in " << result.iterations
      << " iterations" << std::endl;
  }

  // Schwarzschild Kerr-Schild: coordinate radius 2M, areal radius 2M.
  void testSchwarzschildKerrSchild() {
    const SpectralBasis           basis = BuildSpectralBasis(4);
    const SchwarzschildKerrSchild spacetime{1.0};
    const SpectralNewtonResult    result = solve(basis, spacetime, 3.0);
    assert(result.converged);

    const double meanRadius = MeanCoordinateRadius(result.coefficients);
    const double area       = HorizonArea(
      basis,
      result.coefficients,
      spacetime,
      origin
    );
    assert(approxEqual(meanRadius, 2.0, 1.0e-6));        // coordinate radius 2M
    assert(approxEqual(ArealRadius(area), 2.0, 1.0e-6)); // areal radius 2M

    std::cout
      << "Schwarzschild Kerr-Schild: coord radius " << meanRadius
      << ", areal radius " << ArealRadius(area) << " in " << result.iterations
      << " iterations" << std::endl;
  }

  // Both Schwarzschild slicings agree on the (gauge-invariant) areal radius.
  void testBothSlicingsAgreeOnArealRadius() {
    const SpectralBasis           basis = BuildSpectralBasis(4);
    const SchwarzschildIsotropic  isotropic{1.0};
    const SchwarzschildKerrSchild kerrSchild{1.0};

    const SpectralNewtonResult isotropicResult  = solve(basis, isotropic, 0.8);
    const SpectralNewtonResult kerrSchildResult = solve(basis, kerrSchild, 3.0);
    assert(isotropicResult.converged && kerrSchildResult.converged);

    const double isotropicAreal = ArealRadius(
      HorizonArea(basis, isotropicResult.coefficients, isotropic, origin)
    );
    const double kerrSchildAreal = ArealRadius(
      HorizonArea(basis, kerrSchildResult.coefficients, kerrSchild, origin)
    );
    assert(approxEqual(isotropicAreal, kerrSchildAreal, 1.0e-6));

    std::cout
      << "Both Schwarzschild slicings agree on areal radius (" << isotropicAreal
      << " vs " << kerrSchildAreal << ")" << std::endl;
  }

  // Kerr: areal radius matches sqrt(horizon area / 4 pi), and the spinning
  // horizon is oblate (its l = 2 harmonic power is non-zero).
  void testKerrArealRadiusAndOblateness() {
    const SpectralBasis basis = BuildSpectralBasis(8);
    for (double spin : {0.3, 0.6, 0.9}) {
      const KerrKerrSchild       spacetime{1.0, spin};
      const SpectralNewtonResult result = solve(basis, spacetime, 2.5);
      assert(result.converged);

      const double area = HorizonArea(
        basis,
        result.coefficients,
        spacetime,
        origin
      );
      const double exactAreal = ArealRadius(spacetime.horizonArea());
      assert(approxEqual(ArealRadius(area), exactAreal, 1.0e-4));

      const std::vector<double>
        power = PowerSpectrum(basis, result.coefficients);
      assert(power[2] > 1.0e-6); // oblate: l = 2 excited

      std::cout
        << "Kerr a = " << spin << ": areal radius " << ArealRadius(area)
        << " (exact " << exactAreal << "), l=2 power " << power[2] << std::endl;
    }
  }

} // namespace exahype2::ccz4::ahf::tests::analytic_horizon

int main() {
  using namespace exahype2::ccz4::ahf::tests::analytic_horizon;

  std::cout
    << "Apparent Horizon Finder analytic-horizon tests begin..." << std::endl;

  testSchwarzschildIsotropic();
  testSchwarzschildKerrSchild();
  testBothSlicingsAgreeOnArealRadius();
  testKerrArealRadiusAndOblateness();

  std::cout << "All analytic-horizon tests completed." << std::endl;
  return 0;
}
