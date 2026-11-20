#ifndef APPARENT_HORIZON_FINDER_INTERPOLATED_SURFACE_RESIDUAL_H_
#define APPARENT_HORIZON_FINDER_INTERPOLATED_SURFACE_RESIDUAL_H_

#include "ahf-data.h"
#include "harmonics.h"
#include "interpolation.h"
#include "spectral-solver.h"

#include <vector>

namespace exahype2::ccz4::ahf {

/**
 * Build a fully populated surface for one spectral coefficient vector using a
 * persistent Cartesian interpolation context.
 *
 * Point p retains the SpectralBasis ordering throughout synthesis, Cartesian
 * interpolation, and surface-shape geometry construction.
 */
InterpolatedSurface BuildInterpolatedSurface(
  const SpectralBasis& basis,
  const std::vector<double>& coefficients,
  const CartesianPointInterpolator& interpolator,
  const ReferencePoint& referencePoint
);

namespace detail {

/**
 * Internal Newton adapter returning R(a) = project(Theta(a)) using batch
 * Cartesian interpolation.
 *
 * The basis and interpolator are captured by reference and must outlive the
 * returned callback. The interpolator retains and reuses its spatial index;
 * neither it nor its underlying GridData is modified by residual evaluation.
 */
SpectralResidual MakeInterpolatedSurfaceResidual(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  const ReferencePoint& referencePoint
);

}  // namespace detail

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_INTERPOLATED_SURFACE_RESIDUAL_H_
