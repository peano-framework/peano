#include "ah-finder-result.h"

#include "diagnostics.h"
#include "expansion.h"
#include "interpolated-surface-residual.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace exahype2::ccz4::ahf {

AHFinderResult BuildAHFinderResult(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  const ReferencePoint& referencePoint,
  const SpectralNewtonResult& solverResult
) {
  AHFinderResult result{
    .solver = solverResult,
    .referencePoint = referencePoint,
    .lMax = basis.lmax
  };

  if (solverResult.coefficients.size() != basis.nModes
      || basis.rho.size() != basis.nPoints
      || basis.sigma.size() != basis.nPoints) {
    return result;
  }

  try {
    const SynthesizedField field = Synthesize(
      basis, solverResult.coefficients
    );
    if (field.h.size() != basis.nPoints || field.h.empty()) return result;

    AHGuess horizon;
    horizon.points.resize(basis.nPoints);
    for (std::size_t point = 0; point < basis.nPoints; ++point) {
      if (!std::isfinite(field.h[point]) || field.h[point] <= 0.0) {
        return result;
      }
      horizon.points[point] = {
        .r = field.h[point],
        .rho = basis.rho[point],
        .sigma = basis.sigma[point]
      };
    }

    const InterpolatedSurface surface = BuildInterpolatedSurface(
      basis, solverResult.coefficients, interpolator, referencePoint
    );
    if (surface.numberOfPoints() != basis.nPoints) return result;

    std::vector<double> expansion;
    std::vector<Tensor3> spatialMetric;
    expansion.reserve(basis.nPoints);
    spatialMetric.reserve(basis.nPoints);
    double maxAbsExpansion = 0.0;
    for (const InterpolatedSurfacePoint& point : surface.points) {
      const double theta = computeExpansion(point);
      if (!std::isfinite(theta)) return result;
      expansion.push_back(theta);
      spatialMetric.push_back(point.geometry.spatialMetric);
      maxAbsExpansion = std::max(maxAbsExpansion, std::abs(theta));
    }

    const std::vector<double> spectralResidual = Project(basis, expansion);
    if (spectralResidual.size() != basis.nModes) return result;
    double spectralResidualInfinityNorm = 0.0;
    for (double component : spectralResidual) {
      if (!std::isfinite(component)) return result;
      spectralResidualInfinityNorm = std::max(
        spectralResidualInfinityNorm, std::abs(component)
      );
    }

    const auto [minimum, maximum] = std::minmax_element(
      field.h.begin(), field.h.end()
    );
    // The surface points follow the basis' collocation order, as HorizonArea
    // requires.
    const double area = HorizonArea(
      basis, solverResult.coefficients, spatialMetric
    );

    // The same surface sampled twice as densely in each direction, where
    // Newton has not forced the projection of Theta to vanish.
    const SpectralBasis overResolved = BuildSpectralBasis(
      basis.lmax, 2 * basis.nSigma, 2 * basis.nRho
    );
    const InterpolatedSurface overResolvedSurface = BuildInterpolatedSurface(
      overResolved, solverResult.coefficients, interpolator, referencePoint
    );
    if (overResolvedSurface.numberOfPoints() != overResolved.nPoints) {
      return result;
    }
    double maxAbsExpansionOverResolved = 0.0;
    for (const InterpolatedSurfacePoint& point : overResolvedSurface.points) {
      const double theta = computeExpansion(point);
      if (!std::isfinite(theta)) return result;
      maxAbsExpansionOverResolved = std::max(
        maxAbsExpansionOverResolved, std::abs(theta)
      );
    }
    result.horizon = std::move(horizon);
    result.minRadius = *minimum;
    result.meanRadius = std::accumulate(
      field.h.begin(), field.h.end(), 0.0
    ) / static_cast<double>(field.h.size());
    result.maxRadius = *maximum;
    result.meanCoordinateRadius = MeanCoordinateRadius(
      solverResult.coefficients
    );
    result.area = area;
    result.arealRadius = ArealRadius(area);
    result.irreducibleMass = IrreducibleMass(area);
    result.maxAbsExpansion = maxAbsExpansion;
    result.maxAbsExpansionOverResolved = maxAbsExpansionOverResolved;
    result.relativeExpansionError =
      maxAbsExpansionOverResolved * result.arealRadius / 2.0;
    result.spectralResidualInfinityNorm = spectralResidualInfinityNorm;
    result.surfaceAvailable = true;
  } catch (const std::exception&) {
    // A failed or stopped solve can leave coefficients whose trial surface is
    // outside the interpolation domain or otherwise unreconstructable. Solver
    // diagnostics remain available; surface diagnostics retain their NaNs.
  }

  return result;
}

double SurfaceRadius(const AHFinderResult& result, double rho, double sigma) {
  const auto modes = EnumerateModes(result.lMax);
  if (modes.size() != result.solver.coefficients.size()) {
    throw std::invalid_argument(
      "SurfaceRadius: coefficient count does not match lMax");
  }
  double radius = 0.0;
  for (std::size_t i = 0; i < modes.size(); ++i) {
    radius += result.solver.coefficients[i] *
      EvalModeValue(modes[i].first, modes[i].second, rho, sigma);
  }
  return radius;
}

double SurfaceRadiusTowards(
  const AHFinderResult& result,
  const ReferencePoint& destination
) {
  const double dx = destination.x - result.referencePoint.x;
  const double dy = destination.y - result.referencePoint.y;
  const double dz = destination.z - result.referencePoint.z;
  const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (!(distance > 0.0)) {
    throw std::invalid_argument(
      "SurfaceRadiusTowards: destination coincides with the reference point");
  }
  double rho = std::atan2(dy, dx);
  if (rho < 0.0) rho += 2.0 * std::numbers::pi;
  const double sigma = std::asin(std::clamp(dz / distance, -1.0, 1.0));
  return SurfaceRadius(result, rho, sigma);
}

bool SurfaceEncloses(
  const AHFinderResult& result,
  const ReferencePoint& point
) {
  const double distance = std::hypot(
    point.x - result.referencePoint.x,
    point.y - result.referencePoint.y,
    point.z - result.referencePoint.z);
  if (!(distance > 0.0)) return true;
  return distance < SurfaceRadiusTowards(result, point);
}

}  // namespace exahype2::ccz4::ahf
