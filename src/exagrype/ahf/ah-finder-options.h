#ifndef APPARENT_HORIZON_FINDER_AH_FINDER_OPTIONS_H_
#define APPARENT_HORIZON_FINDER_AH_FINDER_OPTIONS_H_

#include "ahf-data.h"
#include "interpolation.h"
#include "spectral-solver.h"

namespace exahype2::ccz4::ahf {

/**
 * Compact configuration for the future user-facing apparent-horizon façade.
 *
 * The structure intentionally reuses the existing interpolation and Newton
 * option types. It does not construct numerical objects or start a solve.
 */
struct AHFinderOptions {
  ReferencePoint referencePoint{0.0, 0.0, 0.0};
  double initialRadius = 1.0;
  int lMax = 2;
  CartesianToSurfaceOptions interpolation{};
  SpectralNewtonOptions solver{};
};

/** Validate configuration that can be checked before grid data are supplied. */
void ValidateAHFinderOptions(const AHFinderOptions& options);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_AH_FINDER_OPTIONS_H_
