#include "../require-asserts.h"

#include "../../ah-finder-result.h"
#include "../../expansion.h"
#include "../../initial-guess.h"
#include "../../interpolated-surface-residual.h"
#include "../../iteration.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <utility>

namespace exahype2::ccz4::ahf::tests::result {
namespace {

// gammaTilde_ij = delta_ij, so the physical metric is delta_ij / phi^2.
GridData MakeSphericalHorizonGrid(double phi = 1.0) {
  GridData grid;
  for (int x = -2; x <= 2; ++x) {
    for (int y = -2; y <= 2; ++y) {
      for (int z = -2; z <= 2; ++z) {
        grid.coordinates.push_back({
          static_cast<double>(x),
          static_cast<double>(y),
          static_cast<double>(z)
        });
        const std::array<double, NumberOfCompactFields> row{
          1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
          1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
          3.0, phi
        };
        grid.values.insert(grid.values.end(), row.begin(), row.end());
      }
    }
  }
  return grid;
}

CartesianToSurfaceOptions InterpolationOptions() {
  return {
    .method = CartesianToSurfaceMethod::UnweightedPolynomial,
    .numberOfNeighbours = 64,
    .polynomialDegree = 1,
    .allowOrderReduction = false
  };
}

void TestConvergedResultAndCanonicalSurface() {
  const GridData grid = MakeSphericalHorizonGrid();
  const CartesianPointInterpolator interpolator(grid, InterpolationOptions());
  const SpectralBasis basis = BuildSpectralBasis(0);
  const ReferencePoint referencePoint{0.0, 0.0, 0.0};
  const SpectralNewtonResult solver = SolvePDEforAH(
    basis, interpolator, referencePoint, 1.3
  );
  assert(solver.converged);

  const AHFinderResult result = BuildAHFinderResult(
    basis, interpolator, referencePoint, solver
  );
  assert(result.surfaceAvailable);
  assert(result.solver.coefficients == solver.coefficients);
  assert(result.solver.residualHistory == solver.residualHistory);
  assert(result.solver.converged == solver.converged);
  assert(result.solver.iterations == solver.iterations);
  assert(result.solver.message == solver.message);
  assert(result.referencePoint.x == referencePoint.x);
  assert(result.referencePoint.y == referencePoint.y);
  assert(result.referencePoint.z == referencePoint.z);
  assert(result.lMax == basis.lmax);

  const SynthesizedField field = Synthesize(basis, solver.coefficients);
  assert(result.horizon.points.size() == basis.nPoints);
  for (std::size_t point = 0; point < basis.nPoints; ++point) {
    assert(result.horizon.points[point].r == field.h[point]);
    assert(result.horizon.points[point].rho == basis.rho[point]);
    assert(result.horizon.points[point].sigma == basis.sigma[point]);
  }

  assert(result.minRadius == field.h.front());
  assert(result.meanRadius == field.h.front());
  assert(result.maxRadius == field.h.front());
  const double radius = field.h.front();
  assert(std::abs(result.meanCoordinateRadius - radius) < 1.0e-14);
  assert(std::abs(result.area - 4.0 * std::numbers::pi * radius * radius) < 1.0e-12);
  assert(std::abs(result.arealRadius - radius) < 1.0e-12);
  assert(std::abs(result.irreducibleMass - 0.5 * radius) < 1.0e-12);
  assert(std::isfinite(result.maxAbsExpansion));
  assert(result.maxAbsExpansion < 1.0e-8);
  assert(std::isfinite(result.maxAbsExpansionOverResolved));
  assert(result.maxAbsExpansionOverResolved < 1.0e-8);
  assert(std::isfinite(result.relativeExpansionError));
  assert(result.relativeExpansionError < 1.0e-8);
  assert(std::isfinite(result.spectralResidualInfinityNorm));
  assert(result.spectralResidualInfinityNorm < 1.0e-8);
}

void TestNonConvergedButReconstructableResult() {
  const GridData grid = MakeSphericalHorizonGrid();
  const CartesianPointInterpolator interpolator(grid, InterpolationOptions());
  const SpectralBasis basis = BuildSpectralBasis(2);
  const ReferencePoint referencePoint{0.25, -0.5, 0.75};
  SpectralNewtonResult solver{
    .coefficients = MakeSphericalInitialCoefficients(basis, 1.2),
    .converged = false,
    .iterations = 3,
    .residualHistory = {2.0, 1.0, 0.5},
    .message = "controlled test failure"
  };

  const AHFinderResult result = BuildAHFinderResult(
    basis, interpolator, referencePoint, solver
  );
  assert(!result.solver.converged);
  assert(result.solver.residualHistory == solver.residualHistory);
  assert(result.solver.message == solver.message);
  assert(result.surfaceAvailable);
  assert(result.referencePoint.x == referencePoint.x);
  assert(result.referencePoint.y == referencePoint.y);
  assert(result.referencePoint.z == referencePoint.z);
  assert(std::abs(result.meanRadius - 1.2) < 1.0e-12);
  assert(std::isfinite(result.maxAbsExpansion));
  assert(std::isfinite(result.spectralResidualInfinityNorm));
  const SynthesizedField field = Synthesize(basis, solver.coefficients);
  assert(result.horizon.points.size() == basis.nPoints);
  for (std::size_t point = 0; point < basis.nPoints; ++point) {
    assert(result.horizon.points[point].r == field.h[point]);
    assert(result.horizon.points[point].rho == basis.rho[point]);
    assert(result.horizon.points[point].sigma == basis.sigma[point]);
  }
}

void TestUnavailableSurfacePreservesSolverDiagnostics() {
  const GridData grid = MakeSphericalHorizonGrid();
  const CartesianPointInterpolator interpolator(grid, InterpolationOptions());
  const SpectralBasis basis = BuildSpectralBasis(2);
  const SpectralNewtonResult solver{
    .coefficients = {},
    .converged = false,
    .iterations = 4,
    .residualHistory = {3.0, 2.0, 1.0},
    .message = "no usable coefficients"
  };

  const AHFinderResult result = BuildAHFinderResult(
    basis, interpolator, {1.0, 2.0, 3.0}, solver
  );
  assert(!result.surfaceAvailable);
  assert(result.horizon.points.empty());
  assert(result.solver.coefficients.empty());
  assert(result.solver.residualHistory == solver.residualHistory);
  assert(result.solver.message == solver.message);
  assert(std::isnan(result.minRadius));
  assert(std::isnan(result.meanRadius));
  assert(std::isnan(result.maxRadius));
  assert(std::isnan(result.maxAbsExpansion));
  assert(std::isnan(result.maxAbsExpansionOverResolved));
  assert(std::isnan(result.relativeExpansionError));
  assert(std::isnan(result.spectralResidualInfinityNorm));
  assert(std::isnan(result.meanCoordinateRadius));
  assert(std::isnan(result.area));
  assert(std::isnan(result.arealRadius));
  assert(std::isnan(result.irreducibleMass));
}

// The area is measured with the physical metric interpolated from the grid,
// not with the coordinate one: with phi = 0.5 every length doubles, so a
// coordinate sphere of radius 1.2 has areal radius 2.4.
void TestAreaUsesPhysicalMetric() {
  const GridData grid = MakeSphericalHorizonGrid(0.5);
  const CartesianPointInterpolator interpolator(grid, InterpolationOptions());
  const SpectralBasis basis = BuildSpectralBasis(2);
  const SpectralNewtonResult solver{
    .coefficients = MakeSphericalInitialCoefficients(basis, 1.2),
    .converged = false,
    .iterations = 1,
    .residualHistory = {1.0},
    .message = "controlled test failure"
  };

  const AHFinderResult result = BuildAHFinderResult(
    basis, interpolator, {0.25, -0.5, 0.75}, solver
  );
  assert(result.surfaceAvailable);
  assert(std::abs(result.meanRadius - 1.2) < 1.0e-12);
  assert(std::abs(result.area - 4.0 * std::numbers::pi * 2.4 * 2.4) < 1.0e-10);
  assert(std::abs(result.arealRadius - 2.4) < 1.0e-12);
  assert(std::abs(result.irreducibleMass - 1.2) < 1.0e-12);
}

// On a surface that is not a coordinate sphere the two mean radii differ:
// meanRadius averages h over the collocation points, meanCoordinateRadius is
// the solid-angle mean a_00 / sqrt(4 pi).
void TestMeanRadiiOnDistortedSurface() {
  const GridData grid = MakeSphericalHorizonGrid();
  const CartesianPointInterpolator interpolator(grid, InterpolationOptions());
  const SpectralBasis basis = BuildSpectralBasis(2);
  std::vector<double> coefficients = MakeSphericalInitialCoefficients(basis, 1.0);
  const auto mode = std::find(
    basis.modes.begin(), basis.modes.end(), std::pair{2, 0}
  );
  assert(mode != basis.modes.end());
  coefficients[static_cast<std::size_t>(mode - basis.modes.begin())] = 0.3;
  const SpectralNewtonResult solver{
    .coefficients = coefficients,
    .converged = false,
    .iterations = 1,
    .residualHistory = {1.0},
    .message = "controlled test failure"
  };

  const AHFinderResult result = BuildAHFinderResult(
    basis, interpolator, {0.0, 0.0, 0.0}, solver
  );
  assert(result.surfaceAvailable);
  assert(std::abs(result.meanCoordinateRadius - 1.0) < 1.0e-14);
  assert(std::abs(result.meanRadius - result.meanCoordinateRadius) > 1.0e-3);
  assert(result.minRadius < result.meanCoordinateRadius);
  assert(result.meanCoordinateRadius < result.maxRadius);

  // Theta is also sampled on a grid twice as fine in each direction, and the
  // relative error compares it with 2 / R of a sphere of the same area.
  const SpectralBasis overResolved =
    BuildSpectralBasis(2, 2 * basis.nSigma, 2 * basis.nRho);
  const InterpolatedSurface surface = BuildInterpolatedSurface(
    overResolved, coefficients, interpolator, {0.0, 0.0, 0.0}
  );
  double maxTheta = 0.0;
  for (const InterpolatedSurfacePoint& point : surface.points) {
    maxTheta = std::max(maxTheta, std::abs(computeExpansion(point)));
  }
  assert(std::abs(result.maxAbsExpansionOverResolved - maxTheta) < 1.0e-14);
  assert(std::abs(result.maxAbsExpansionOverResolved -
                  result.maxAbsExpansion) > 1.0e-3);
  assert(std::abs(result.relativeExpansionError -
                  maxTheta * result.arealRadius / 2.0) < 1.0e-14);
}

// The directional radius and enclosure test used by the planner, the
// validation and the searches' excluded points.
void TestSurfaceRadiusAndEnclosure() {
  const SpectralBasis basis = BuildSpectralBasis(2);
  AHFinderResult sphere;
  sphere.referencePoint = {1.0, -1.0, 0.5};
  sphere.lMax = 2;
  sphere.solver.coefficients = MakeSphericalInitialCoefficients(basis, 0.8);
  assert(std::abs(SurfaceRadius(sphere, 1.0, 0.3) - 0.8) < 1.0e-14);
  assert(std::abs(SurfaceRadiusTowards(sphere, {1.0, -1.0, 3.0}) - 0.8) <
         1.0e-14);
  assert(SurfaceEncloses(sphere, {1.5, -1.0, 0.5}));
  assert(!SurfaceEncloses(sphere, {1.0, -1.0, 1.4}));
  assert(SurfaceEncloses(sphere, sphere.referencePoint));

  // An off-centre sphere about the reference point is longer towards +x.
  std::vector<double> radii(basis.nPoints);
  for (std::size_t p = 0; p < radii.size(); ++p) {
    radii[p] = 0.8 + 0.1 * std::cos(basis.rho[p]) * std::cos(basis.sigma[p]);
  }
  AHFinderResult distorted = sphere;
  distorted.solver.coefficients =
    MakeDistortedInitialCoefficients(basis, radii);
  assert(SurfaceRadiusTowards(distorted, {2.0, -1.0, 0.5}) > 0.89);
  assert(SurfaceRadiusTowards(distorted, {0.0, -1.0, 0.5}) < 0.71);
  assert(SurfaceEncloses(distorted, {1.85, -1.0, 0.5}));
  assert(!SurfaceEncloses(distorted, {0.15, -1.0, 0.5}));

  bool threw = false;
  try { (void)SurfaceRadiusTowards(sphere, sphere.referencePoint); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
  sphere.lMax = 3;
  threw = false;
  try { (void)SurfaceRadius(sphere, 0.0, 0.0); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests::result

int main() {
  using namespace exahype2::ccz4::ahf::tests::result;
  TestConvergedResultAndCanonicalSurface();
  TestNonConvergedButReconstructableResult();
  TestUnavailableSurfacePreservesSolverDiagnostics();
  TestAreaUsesPhysicalMetric();
  TestMeanRadiiOnDistortedSurface();
  TestSurfaceRadiusAndEnclosure();
  std::cout << "All AHFinderResult tests completed.\n";
}
