#include "surface-residual.h"

#include "assemble-matrix.h"
#include "interpolation.h"
#include "surface-geometry.h"

#include <limits>
#include <stdexcept>

namespace exahype2::ccz4::ahf {

  InterpolatedSurface BuildSurface(
    const SpectralBasis&       basis,
    const std::vector<double>& coefficients,
    const GeometrySource&      geometryAt,
    const ReferencePoint&      referencePoint
  ) {
    const SynthesizedField field = Synthesize(basis, coefficients);
    if (field.h.size() != basis.nPoints) {
      throw std::invalid_argument(
        "BuildSurface: synthesized field does not match the basis grid"
      );
    }

    InterpolatedSurface surface;
    surface.points.resize(basis.nPoints);
    for (std::size_t p = 0; p < basis.nPoints; ++p) {
      InterpolatedSurfacePoint& point = surface.points[p];
      point.r                         = field.h[p];
      point.rho                       = basis.rho[p];
      point.sigma                     = basis.sigma[p];

      const AHGuessPoint   surfacePoint{point.r, point.rho, point.sigma};
      const GridCoordinate global = SurfacePointToCartesian(
        surfacePoint,
        referencePoint
      );

      // Coordinate-transform geometry (T, X^u_i, X^u_ij) from the appendix
      const AnalyticCoordinateGeometry coordinate
        = ComputeAnalyticCoordinateGeometry(global, referencePoint);
      point.geometry.T                  = coordinate.T;
      point.geometry.coordinateGradient = coordinate.gradient;
      point.geometry.coordinateHessian  = coordinate.hessian;

      // Physical geometry (gamma_ij, partial_k gamma_ij, K_ij, trK) from the
      // injected source, evaluated at the absolute Cartesian point.
      const Vector3                  absolute{global.x, global.y, global.z};
      const InterpolatedGeometryData physical = geometryAt(absolute);
      point.geometry.spatialMetric            = physical.spatialMetric;
      point.geometry.spatialMetricDerivative = physical.spatialMetricDerivative;
      point.geometry.spatialMetricDerivativeAvailable
        = physical.spatialMetricDerivativeAvailable;
      point.geometry.extrinsicCurvature = physical.extrinsicCurvature;
      point.geometry.traceK             = physical.traceK;
    }

    // Non-unit normal s_i and partial_i s_j from the surface shape and its
    // coordinate geometry (reads the fields filled above; requires field.h ==
    // point.r per point).
    PopulateSurfaceShapeGeometry(surface, field);
    return surface;
  }

  SpectralResidual MakeResidual(
    const SpectralBasis&  basis,
    const GeometrySource& geometryAt,
    const ReferencePoint& referencePoint
  ) {
    return
      [&basis, geometryAt, referencePoint](
        const std::vector<double>& coefficients
      ) -> std::vector<double> {
        try {
          const InterpolatedSurface surface = BuildSurface(
            basis,
            coefficients,
            geometryAt,
            referencePoint
          );
          std::vector<std::vector<double>> matrixCoefficients;
          std::vector<double>              residuals;
          AssembleGlobalMatrix(surface, basis, matrixCoefficients, residuals);
          return residuals;
        } catch (const std::exception&) {
          // An invalid trial surface (non-positive radius, degenerate geometry)
          // can arise mid-solve, e.g. a full Newton step overshoots and drives
          // h non-positive at some collocation point.
          // Return a non-finite residual rather than propagate the exception, so the solver's line
          // search backs the step off and its runaway guard can flag a genuine
          // divergence.
          return std::vector<double>(
            basis.nModes,
            std::numeric_limits<double>::quiet_NaN()
          );
        }
      };
  }

} // namespace exahype2::ccz4::ahf
