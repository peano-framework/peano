#ifndef APPARENT_HORIZON_FINDER_ITERATION_H_
#define APPARENT_HORIZON_FINDER_ITERATION_H_

#include "ahf-data.h"
#include "harmonics.h"
#include "interpolation.h"
#include "spectral-solver.h"

#include <vector>

namespace exahype2::ccz4::ahf {

struct AHGuess;

/**
 * Solve one interpolation-backed apparent-horizon PDE from a spherical
 * initial guess.
 *
 * This is the public orchestration entry point for the currently implemented
 * direct Cartesian-to-surface backend. The supplied interpolator and its
 * spatial index remain alive and are reused throughout the Newton solve.
 */
SpectralNewtonResult SolvePDEforAH(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  const ReferencePoint& referencePoint,
  double initialRadius,
  const SpectralNewtonOptions& options = {}
);

/** Solve from an explicit star-shaped spectral surface (e.g. a prior result). */
SpectralNewtonResult SolvePDEforAH(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  const ReferencePoint& referencePoint,
  const std::vector<double>& initialCoefficients,
  const SpectralNewtonOptions& options = {}
);

/**
 * Determine whether a common-horizon search should be triggered for a binary-black-hole setup.
 */
void CheckCommonHorizonOfBBH(const AHGuess& initialAHGuess,
                             const ReferencePoint& referencePoint,
                             const std::vector<double>& masses,
                             bool& enableCommonHorizon,
                             ReferencePoint& commonReferencePoint,
                             double& commonMass,
                             AHGuess& commonAHGuesses);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_ITERATION_H_
