#ifndef APPARENT_HORIZON_FINDER_INITIAL_GUESS_H_
#define APPARENT_HORIZON_FINDER_INITIAL_GUESS_H_

#include "harmonics.h"

#include <vector>

namespace exahype2::ccz4::ahf {

/**
 * Construct spectral coefficients for a constant-radius spherical initial
 * guess by projecting h(p) = radius on the basis collocation grid.
 *
 * @throws std::invalid_argument if radius is non-finite or not strictly
 * positive, or if projection does not return one coefficient per mode.
 */
std::vector<double> MakeSphericalInitialCoefficients(
  const SpectralBasis& basis,
  double radius
);

/**
 * Project positive radii in canonical SpectralBasis collocation order into an
 * arbitrary star-shaped spectral initial surface.
 */
std::vector<double> MakeDistortedInitialCoefficients(
  const SpectralBasis& basis,
  const std::vector<double>& radii
);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_INITIAL_GUESS_H_
