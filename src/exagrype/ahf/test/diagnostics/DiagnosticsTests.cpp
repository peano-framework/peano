#include "../require-asserts.h"

#include "../../apparent-horizon-finder.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace exahype2::ccz4::ahf::tests::diagnostics {

  namespace {

    constexpr double pi = std::numbers::pi;

    bool approxEqual(double a, double b, double tol) {
      return std::abs(a - b) <= tol;
    }

    std::vector<Tensor3> constantMetric(const SpectralBasis& basis, const Tensor3& gamma) {
      return std::vector<Tensor3>(basis.nPoints, gamma);
    }

    Tensor3 scaledIdentity(double factor) {
      return {{{factor, 0.0, 0.0}, {0.0, factor, 0.0}, {0.0, 0.0, factor}}};
    }

    Vector3 radialDirection(const SpectralBasis& basis, std::size_t p) {
      const double cosSigma = std::cos(basis.sigma[p]);
      return {
        cosSigma * std::cos(basis.rho[p]),
        cosSigma * std::sin(basis.rho[p]),
        std::sin(basis.sigma[p])};
    }

    std::size_t modeIndex(const SpectralBasis& basis, int l, int m) {
      const auto mode = std::find(basis.modes.begin(), basis.modes.end(), std::pair{l, m});
      assert(mode != basis.modes.end());
      return static_cast<std::size_t>(mode - basis.modes.begin());
    }

  } // namespace

  void testArealRadiusAndIrreducibleMass() {
    // A Schwarzschild horizon of mass M has area 16 pi M^2.
    assert(approxEqual(ArealRadius(16.0 * pi), 2.0, 1.0e-15));
    assert(approxEqual(IrreducibleMass(16.0 * pi), 1.0, 1.0e-15));
    assert(approxEqual(IrreducibleMass(4.0 * pi * 9.0), 1.5, 1.0e-15));
  }

  // gamma_ij = delta_ij / phi^2 shrinks or stretches lengths by 1 / phi, so a
  // coordinate sphere of radius r has area 4 pi r^2 / phi^2.
  void testAreaOfSphereInConformallyFlatMetric() {
    const SpectralBasis basis  = BuildSpectralBasis(2);
    const double        radius = 1.5;
    const double        phi    = 0.8;
    const double        area   = HorizonArea(
      basis,
      SphereCoefficients(basis, radius),
      constantMetric(basis, scaledIdentity(1.0 / (phi * phi)))
    );
    const double expected = 4.0 * pi * radius * radius / (phi * phi);
    assert(approxEqual(area, expected, 1.0e-12 * expected));
    std::cout << "Sphere of coordinate radius " << radius << " with phi = " << phi
              << ": area " << area << " (exact " << expected << ")" << std::endl;
  }

  // A unit sphere whose centre is offset from the reference point is not a
  // coordinate sphere about it, so h varies and the h_rho and h_sigma terms of
  // the tangent vectors enter. Its flat area is still 4 pi.
  void testAreaOfOffCentreSphere() {
    const SpectralBasis basis = BuildSpectralBasis(16);
    const Vector3       centre{0.1, -0.2, 0.15};
    const double        centreSquared =
      centre[0] * centre[0] + centre[1] * centre[1] + centre[2] * centre[2];

    std::vector<double> h(basis.nPoints);
    for (std::size_t p = 0; p < basis.nPoints; ++p) {
      const Vector3 n = radialDirection(basis, p);
      const double  along = centre[0] * n[0] + centre[1] * n[1] + centre[2] * n[2];
      h[p] = along + std::sqrt(1.0 - centreSquared + along * along);
    }
    const std::vector<double> coefficients = Project(basis, h);
    assert(Synthesize(basis, coefficients).h.size() == basis.nPoints);

    const double area = HorizonArea(basis, coefficients, constantMetric(basis, scaledIdentity(1.0)));
    assert(approxEqual(area, 4.0 * pi, 1.0e-9));
    std::cout << "Off-centre unit sphere: area " << area << ", error "
              << std::abs(area - 4.0 * pi) << std::endl;
  }

  // A constant metric diag(1, 1, 4) turns a coordinate sphere of radius r into
  // a prolate spheroid with semi-axes r, r and 2r, whose area is
  // 2 pi a^2 (1 + c arcsin(e) / (a e)) with e^2 = 1 - a^2 / c^2. Rotating the
  // metric adds off-diagonal components but leaves the area unchanged.
  void testAreaWithAnisotropicMetric() {
    const SpectralBasis basis  = BuildSpectralBasis(24);
    const double        radius = 0.7;

    const double angle = 0.4;
    const double c     = std::cos(angle);
    const double s     = std::sin(angle);
    // Rotation about the x axis, then gamma = R^T diag(1, 1, 4) R.
    const Tensor3 rotation{{{1.0, 0.0, 0.0}, {0.0, c, -s}, {0.0, s, c}}};
    const std::array<double, 3> stretch{1.0, 1.0, 4.0};
    Tensor3 gamma{};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          gamma[i][j] += rotation[k][i] * stretch[k] * rotation[k][j];
        }
      }
    }
    assert(std::abs(gamma[1][2]) > 0.5);

    const double area = HorizonArea(
      basis, SphereCoefficients(basis, radius), constantMetric(basis, gamma)
    );
    const double a         = radius;
    const double cAxis     = 2.0 * radius;
    const double e         = std::sqrt(1.0 - a * a / (cAxis * cAxis));
    const double expected  = 2.0 * pi * a * a * (1.0 + cAxis * std::asin(e) / (a * e));
    assert(approxEqual(area, expected, 1.0e-9 * expected));
    std::cout << "Sphere in a rotated diag(1, 1, 4) metric: area " << area
              << " (prolate spheroid " << expected << ")" << std::endl;
  }

  // The analytic overload evaluates the metric on the surface about the
  // reference point. A puncture of mass M has its horizon on the coordinate
  // sphere of radius M/2 about it, where psi = 2 and the area is 16 pi M^2;
  // the puncture sits off every axis so that each component of the reference
  // point matters.
  void testAnalyticAreaAboutOffsetPuncture() {
    const SpectralBasis  basis = BuildSpectralBasis(2);
    const ReferencePoint centre{0.3, -0.2, 0.1};
    const BrillLindquist puncture = BrillLindquist::single(1.0, {centre.x, centre.y, centre.z});
    const double area = HorizonArea(basis, SphereCoefficients(basis, 0.5), puncture, centre);
    assert(approxEqual(area, 16.0 * pi, 1.0e-12));
    assert(approxEqual(IrreducibleMass(area), 1.0, 1.0e-14));

    const double areaAboutOrigin = HorizonArea(
      basis, SphereCoefficients(basis, 0.5), puncture, ReferencePoint{0.0, 0.0, 0.0}
    );
    assert(std::abs(areaAboutOrigin - 16.0 * pi) > 1.0);
  }

  void testAreaNeedsTheMetricAtEveryPoint() {
    const SpectralBasis        basis        = BuildSpectralBasis(2);
    const std::vector<double>  coefficients = SphereCoefficients(basis, 1.0);
    std::vector<Tensor3>       metric = constantMetric(basis, scaledIdentity(1.0));
    metric.pop_back();
    bool threw = false;
    try {
      (void)HorizonArea(basis, coefficients, metric);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    assert(threw);
  }

  // MeanCoordinateRadius is the solid-angle mean of h. The collocation points
  // are not spread evenly over the sphere, so the plain average of h at them
  // differs from it once h is not constant.
  void testMeanCoordinateRadiusIsSolidAngleMean() {
    const SpectralBasis basis        = BuildSpectralBasis(4);
    std::vector<double> coefficients = SphereCoefficients(basis, 1.0);
    coefficients[modeIndex(basis, 2, 0)] = 0.3;
    const std::vector<double> h = Synthesize(basis, coefficients).h;

    double weightedSum = 0.0;
    double totalWeight = 0.0;
    for (std::size_t p = 0; p < basis.nPoints; ++p) {
      weightedSum += basis.weight[p] * h[p];
      totalWeight += basis.weight[p];
    }
    assert(approxEqual(totalWeight, 4.0 * pi, 1.0e-12));
    const double solidAngleMean = weightedSum / totalWeight;
    const double plainMean = std::accumulate(h.begin(), h.end(), 0.0) / static_cast<double>(h.size());

    assert(approxEqual(MeanCoordinateRadius(coefficients), 1.0, 1.0e-14));
    assert(approxEqual(MeanCoordinateRadius(coefficients), solidAngleMean, 1.0e-14));
    assert(std::abs(plainMean - solidAngleMean) > 1.0e-3);
  }

} // namespace exahype2::ccz4::ahf::tests::diagnostics

int main() {
  using namespace exahype2::ccz4::ahf::tests::diagnostics;

  std::cout << "Apparent Horizon Finder diagnostics tests begin..." << std::endl;

  testArealRadiusAndIrreducibleMass();
  testAreaOfSphereInConformallyFlatMetric();
  testAreaOfOffCentreSphere();
  testAreaWithAnisotropicMetric();
  testAnalyticAreaAboutOffsetPuncture();
  testAreaNeedsTheMetricAtEveryPoint();
  testMeanCoordinateRadiusIsSolidAngleMean();

  std::cout << "All diagnostics tests completed." << std::endl;
  return 0;
}
