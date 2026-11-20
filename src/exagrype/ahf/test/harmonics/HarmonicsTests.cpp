#include "../require-asserts.h"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

#include "../../harmonics.h"

namespace exahype2::ccz4::ahf::tests::harmonics {

namespace {

bool approxEqual(double a, double b, double tol) {
  return std::abs(a - b) <= tol;
}

double Y(int l, int m, double rho, double sigma) {
  return EvalMode(l, m, rho, sigma).Y;
}

}  // namespace

// Gauss-Legendre nodes/weights integrate polynomials up to degree 2n-1 exactly:
// int_{-1}^{1} mu^k dmu = 0 (odd k), 2/(k+1) (even k).
void testGaussLegendreExactness() {
  for (int n : {3, 5, 8}) {
    const auto [nodes, weights] = GaussLegendre(n);
    for (int k = 0; k <= 2 * n - 1; ++k) {
      double quadrature = 0.0;
      for (int i = 0; i < n; ++i) {
        quadrature += weights[i] * std::pow(nodes[i], k);
      }
      const double exact = (k % 2 == 0) ? 2.0 / (k + 1) : 0.0;
      assert(approxEqual(quadrature, exact, 1.0e-13));
    }
  }
  std::cout << "Gauss-Legendre integrates polynomials to degree 2n-1 exactly" << std::endl;
}

// Discrete transform is the identity: sum_pts w Y_lm Y_l'm' = delta.
void testOrthonormality() {
  const SpectralBasis basis = BuildSpectralBasis(6);
  double maxDeviation = 0.0;
  for (std::size_t a = 0; a < basis.nModes; ++a) {
    for (std::size_t b = 0; b < basis.nModes; ++b) {
      double gram = 0.0;
      for (std::size_t p = 0; p < basis.nPoints; ++p) {
        gram += basis.weight[p]
              * basis.Y[p * basis.nModes + a]
              * basis.Y[p * basis.nModes + b];
      }
      const double expected = (a == b) ? 1.0 : 0.0;
      maxDeviation = std::max(maxDeviation, std::abs(gram - expected));
    }
  }
  assert(maxDeviation < 1.0e-12);
  std::cout << "Orthonormality: max |Gram - I| = " << maxDeviation << std::endl;
}

// project(synthesize(a)) == a for a band-limited coefficient vector.
void testSynthesisAnalysisRoundtrip() {
  const SpectralBasis basis = BuildSpectralBasis(5);
  std::mt19937 rng(1);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<double> a(basis.nModes);
  for (double& value : a) {
    value = normal(rng);
  }

  const SynthesizedField field = Synthesize(basis, a);
  const std::vector<double> recovered = Project(basis, field.h);

  double maxError = 0.0;
  for (std::size_t j = 0; j < basis.nModes; ++j) {
    maxError = std::max(maxError, std::abs(recovered[j] - a[j]));
  }
  assert(maxError < 1.0e-12);
  std::cout << "Synthesize/Project round-trip: max coefficient error = " << maxError << std::endl;
}

// A coordinate sphere synthesizes to a constant radius, and a_00 = r sqrt(4 pi).
void testSphereCoefficients() {
  const SpectralBasis basis = BuildSpectralBasis(4);
  const double radius = 1.7;
  const std::vector<double> a = SphereCoefficients(basis, radius);

  assert(approxEqual(a[0], radius * std::sqrt(4.0 * std::numbers::pi), 1.0e-13));
  for (std::size_t j = 1; j < basis.nModes; ++j) {
    assert(a[j] == 0.0);
  }

  const SynthesizedField field = Synthesize(basis, a);
  for (double h : field.h) {
    assert(approxEqual(h, radius, 1.0e-12));
  }
  std::cout << "SphereCoefficients synthesize to a constant radius" << std::endl;
}

// First rho/sigma derivatives from EvalMode match central finite differences.
void testFirstDerivativesVsFiniteDifference() {
  const std::array<std::pair<int, int>, 4> modes = {{{1, 0}, {2, 1}, {4, -3}, {5, 2}}};
  const std::array<std::pair<double, double>, 4> points = {{
    {0.7, 0.3}, {2.1, -0.6}, {4.0, 0.9}, {5.5, -0.2}
  }};
  const double e = 1.0e-6;

  for (const auto& [l, m] : modes) {
    for (const auto& [rho, sigma] : points) {
      const ModeValues v = EvalMode(l, m, rho, sigma);
      const double dYrho = (Y(l, m, rho + e, sigma) - Y(l, m, rho - e, sigma)) / (2 * e);
      const double dYsigma = (Y(l, m, rho, sigma + e) - Y(l, m, rho, sigma - e)) / (2 * e);
      assert(approxEqual(v.Yrho, dYrho, 1.0e-7));
      assert(approxEqual(v.Ysigma, dYsigma, 1.0e-7));
    }
  }
  std::cout << "EvalMode first derivatives match finite differences" << std::endl;
}

// Second rho/sigma/mixed derivatives from EvalMode match central finite differences.
void testSecondDerivativesVsFiniteDifference() {
  const std::array<std::pair<int, int>, 3> modes = {{{2, 0}, {3, 2}, {4, -1}}};
  const std::array<std::pair<double, double>, 3> points = {{
    {0.9, 0.4}, {2.6, -0.5}, {5.0, 0.1}
  }};
  const double e = 1.0e-4;

  for (const auto& [l, m] : modes) {
    for (const auto& [rho, sigma] : points) {
      const ModeValues v = EvalMode(l, m, rho, sigma);
      const double y0 = Y(l, m, rho, sigma);

      const double d2rho =
        (Y(l, m, rho + e, sigma) - 2 * y0 + Y(l, m, rho - e, sigma)) / (e * e);
      const double d2sigma =
        (Y(l, m, rho, sigma + e) - 2 * y0 + Y(l, m, rho, sigma - e)) / (e * e);
      const double d2mixed =
        (Y(l, m, rho + e, sigma + e) - Y(l, m, rho + e, sigma - e)
         - Y(l, m, rho - e, sigma + e) + Y(l, m, rho - e, sigma - e)) / (4 * e * e);

      assert(approxEqual(v.Yrhorho, d2rho, 1.0e-4));
      assert(approxEqual(v.Ysigmasigma, d2sigma, 1.0e-4));
      assert(approxEqual(v.Yrhosigma, d2mixed, 1.0e-4));
    }
  }
  std::cout << "EvalMode second derivatives match finite differences" << std::endl;
}

}  // namespace exahype2::ccz4::ahf::tests::harmonics

int main() {
  using namespace exahype2::ccz4::ahf::tests::harmonics;

  std::cout << "Apparent Horizon Finder harmonics tests begin..." << std::endl;

  testGaussLegendreExactness();
  testOrthonormality();
  testSynthesisAnalysisRoundtrip();
  testSphereCoefficients();
  testFirstDerivativesVsFiniteDifference();
  testSecondDerivativesVsFiniteDifference();

  std::cout << "All harmonics tests completed." << std::endl;
  return 0;
}
