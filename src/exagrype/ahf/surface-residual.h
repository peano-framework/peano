#ifndef APPARENT_HORIZON_FINDER_SURFACE_RESIDUAL_H_
#define APPARENT_HORIZON_FINDER_SURFACE_RESIDUAL_H_

#include "ahf-data.h"
#include "harmonics.h"
#include "spectral-solver.h"

#include <functional>
#include <vector>

namespace exahype2::ccz4::ahf {

  /**
   * Source of the physical geometry (gamma_ij, partial_k gamma_ij, K_ij, trK)
   * at an absolute Cartesian point, packed into an InterpolatedGeometryData.
   * This is the one step of the residual assembly that depends on where the
   * geometry comes from. Notably,
   *
   * - analytic spacetimes supply MetricData (see analytic-residual.h), 
   *
   * while the real grid solver will supply an interpolation-backed source. 
   * Only the physical-geometry fields of the returned struct are read. 
   * The coordinate-transform fields (T, coordinateGradient, coordinateHessian)
   * are filled by BuildSurface itself.
   */
  using GeometrySource = std::function<
    InterpolatedGeometryData(const Vector3&)>;

  /**
   * Assemble the interpolated surface r = h(rho, sigma) for a coefficient
   * vector on the basis collocation grid, taking the physical geometry from @p
   * geometryAt. For each collocation point this runs the shared pipeline:
   *
   *   - Synthesise (harmonics) for h and its angular derivatives,
   *   - ComputeAnalyticCoordinateGeometry (interpolation) for T, X^u_i, X^u_ij,
   *   - geometryAt for gamma_ij, partial_k gamma_ij, K_ij, trK,
   *   - PopulateSurfaceShapeGeometry (surface-geometry) for s_i and partial_i s_j. 
   *
   * The returned surface has one point per basis collocation point, in
   * basis order, compatible with the spectral AssembleGlobalMatrix conventions.
   */
  InterpolatedSurface BuildSurface(
    const SpectralBasis&       basis,
    const std::vector<double>& coefficients,
    const GeometrySource&      geometryAt,
    const ReferencePoint&      referencePoint
  );

  /**
   * Return the spectral residual 
   *
   * R(a) = project(Theta) 
   *
   * for a geometry source ready to hand to SolveSpectralNewton (BuildSurface 
   * followed by spectral AssembleGlobalMatrix). 
   * The basis is captured by reference and must outlive the returned residual; the
   * geometry source and reference point are captured by value.
   */
  SpectralResidual MakeResidual(
    const SpectralBasis&  basis,
    const GeometrySource& geometryAt,
    const ReferencePoint& referencePoint
  );

} // namespace exahype2::ccz4::ahf

#endif // APPARENT_HORIZON_FINDER_SURFACE_RESIDUAL_H_
