#include "../require-asserts.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "../../apparent-horizon-finder.h"

namespace exahype2::ccz4::ahf::tests {

bool close(double a, double b) {
  return std::abs(a - b) <= 1.0e-13 * std::max({1.0, std::abs(a), std::abs(b)});
}

void testConstantRadiusSphere() {
  constexpr ReferencePoint centre{1.2, -0.7, 2.3};
  for (const auto angles : {std::array{0.31, -0.27}, std::array{1.17, 0.42},
                            std::array{-2.04, -0.63}}) {
    InterpolatedSurfacePoint point{.r = 2.7, .rho = angles[0], .sigma = angles[1]};
    const GridCoordinate xyz = SurfacePointToCartesian(
      {point.r, point.rho, point.sigma}, centre);
    const AnalyticCoordinateGeometry geometry =
      ComputeAnalyticCoordinateGeometry(xyz, centre);
    point.geometry.T = geometry.T;
    point.geometry.coordinateGradient = geometry.gradient;
    point.geometry.coordinateHessian = geometry.hessian;
    PopulateSurfaceShapeGeometry(point, {});
    const double expected[3]{std::cos(point.sigma) * std::cos(point.rho),
                             std::cos(point.sigma) * std::sin(point.rho),
                             std::sin(point.sigma)};
    for (std::size_t i = 0; i < 3; ++i) {
      assert(close(point.surfaceNormal[i], expected[i]));
      for (std::size_t j = 0; j < 3; ++j) {
        assert(close(point.surfaceNormalDerivative[i][j],
                     point.geometry.T[i][j] / std::pow(point.r, 3)));
      }
    }
  }
}

InterpolatedSurfacePoint syntheticPoint() {
  InterpolatedSurfacePoint point{.r = 2.0, .rho = 0.37, .sigma = -0.29};
  for (std::size_t i = 0; i < 3; ++i) {
    point.geometry.coordinateGradient[0][i] = 0.11 + 0.17 * i;
    point.geometry.coordinateGradient[1][i] = -0.23 + 0.31 * i;
    for (std::size_t j = 0; j < 3; ++j) {
      point.geometry.T[i][j] = 1.3 + 0.7 * i - 0.2 * j;
      point.geometry.coordinateHessian[0][i][j] = -0.4 + 0.13 * i + 0.09 * j;
      point.geometry.coordinateHessian[1][i][j] = 0.8 - 0.21 * i + 0.16 * j;
    }
  }
  return point;
}

void testDistinctDerivativesAndMixedTerm() {
  InterpolatedSurfacePoint point = syntheticPoint();
  const SurfaceShapeDerivatives d{0.7, -1.1, 1.9, -2.3, 3.1};
  PopulateSurfaceShapeGeometry(point, d);
  const double radial[3]{std::cos(point.sigma) * std::cos(point.rho),
                         std::cos(point.sigma) * std::sin(point.rho),
                         std::sin(point.sigma)};
  for (std::size_t i = 0; i < 3; ++i) {
    const double ri = point.geometry.coordinateGradient[0][i];
    const double si = point.geometry.coordinateGradient[1][i];
    assert(close(point.surfaceNormal[i], radial[i] - ri*d.hRho - si*d.hSigma));
    for (std::size_t j = 0; j < 3; ++j) {
      const double rj = point.geometry.coordinateGradient[0][j];
      const double sj = point.geometry.coordinateGradient[1][j];
      const double expected = point.geometry.T[i][j] / 8.0
        - point.geometry.coordinateHessian[0][i][j]*d.hRho
        - point.geometry.coordinateHessian[1][i][j]*d.hSigma
        - ri*rj*d.hRhoRho - ri*sj*d.hRhoSigma
        - si*rj*d.hRhoSigma - si*sj*d.hSigmaSigma;
      assert(close(point.surfaceNormalDerivative[i][j], expected));
    }
  }

  InterpolatedSurfacePoint withoutMixed = syntheticPoint();
  SurfaceShapeDerivatives noMixed = d;
  noMixed.hRhoSigma = 0.0;
  PopulateSurfaceShapeGeometry(withoutMixed, noMixed);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      const double expectedDifference = -d.hRhoSigma * (
        point.geometry.coordinateGradient[0][i] * point.geometry.coordinateGradient[1][j]
        + point.geometry.coordinateGradient[1][i] * point.geometry.coordinateGradient[0][j]);
      assert(close(point.surfaceNormalDerivative[i][j]
                   - withoutMixed.surfaceNormalDerivative[i][j], expectedDifference));
    }
}

void expectRejected(InterpolatedSurfacePoint point, SurfaceShapeDerivatives d) {
  bool rejected = false;
  try { PopulateSurfaceShapeGeometry(point, d); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}

void testInvalidInputs() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  auto point = syntheticPoint(); point.r = 0.0; expectRejected(point, {});
  point = syntheticPoint(); point.r = -1.0; expectRejected(point, {});
  expectRejected(syntheticPoint(), {.hRho = nan});
  expectRejected(syntheticPoint(), {.hSigmaSigma = inf});
  point = syntheticPoint(); point.geometry.coordinateGradient[1][2] = inf;
  expectRejected(point, {});
  point = syntheticPoint(); point.geometry.coordinateHessian[0][1][2] = nan;
  expectRejected(point, {});
}

}  // namespace exahype2::ccz4::ahf::tests

int main() {
  using namespace exahype2::ccz4::ahf::tests;
  testConstantRadiusSphere();
  testDistinctDerivativesAndMixedTerm();
  testInvalidInputs();
  std::cout << "All surface geometry tests completed.\n";
}
