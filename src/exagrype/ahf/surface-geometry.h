#ifndef APPARENT_HORIZON_FINDER_SURFACE_GEOMETRY_H_
#define APPARENT_HORIZON_FINDER_SURFACE_GEOMETRY_H_

#include "ahf-data.h"
#include "harmonics.h"

namespace exahype2::ccz4::ahf {

/** Angular derivatives of h at one surface point. */
struct SurfaceShapeDerivatives {
  double hRho = 0.0;
  double hSigma = 0.0;
  double hRhoRho = 0.0;
  double hRhoSigma = 0.0;
  double hSigmaSigma = 0.0;
};

/**
 * Populate the non-unit normal covector s_i and partial_i s_j.
 *
 * Cartesian indices i,j use x,y,z order. geometry.T is T[i][j]. The
 * coordinate arrays are [u][i] and [u][i][j], where u=0 is rho and u=1 is
 * sigma. surfaceNormalDerivative is stored as [i][j] = partial_i s_j.
 * Existing Cartesian interpolation data are read but never recomputed.
 *
 * @throws std::invalid_argument if radius, derivatives, or analytic coordinate
 * geometry are invalid, or if the resulting surface geometry is non-finite.
 */
void PopulateSurfaceShapeGeometry(
  InterpolatedSurfacePoint& point,
  const SurfaceShapeDerivatives& derivatives
);

/** Populate every surface point from the corresponding SynthesizedField row. */
void PopulateSurfaceShapeGeometry(
  InterpolatedSurface& surface,
  const SynthesizedField& field
);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_SURFACE_GEOMETRY_H_
