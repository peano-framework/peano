#ifndef APPARENT_HORIZON_FINDER_AH_FINDER_RESULT_H_
#define APPARENT_HORIZON_FINDER_AH_FINDER_RESULT_H_

#include "ahf-data.h"
#include "harmonics.h"
#include "interpolation.h"
#include "spectral-solver.h"

#include <limits>

namespace exahype2::ccz4::ahf {

/**
 * User-level result of one apparent-horizon solve.
 *
 * horizon uses the numerical SpectralBasis collocation ordering. The retained
 * lMax and final coefficients are sufficient for a later writer to construct
 * a separate, denser visualisation grid without rerunning Newton.
 *
 * The radii are coordinate radii about referencePoint. meanRadius is the plain
 * average of h over the collocation points, which do not sample the sphere
 * evenly; meanCoordinateRadius is the solid-angle mean a_00 / sqrt(4 pi). They
 * agree for a coordinate sphere. area, arealRadius and irreducibleMass use the
 * physical metric interpolated from the grid onto the surface.
 *
 * maxAbsExpansion is taken at the collocation points, where Newton zeroes
 * only the projection of Theta. maxAbsExpansionOverResolved is taken on a
 * grid with twice as many points in each direction, and relativeExpansionError
 * is that value times arealRadius / 2, i.e. relative to Theta = 2 / R of a
 * round sphere of the same area. It stays small on a resolved horizon and
 * becomes of order one on a surface that only aliasing makes look like one.
 */
struct AHFinderResult {
  SpectralNewtonResult solver;
  ReferencePoint referencePoint{0.0, 0.0, 0.0};
  int lMax = 0;
  AHGuess horizon;

  double minRadius = std::numeric_limits<double>::quiet_NaN();
  double meanRadius = std::numeric_limits<double>::quiet_NaN();
  double maxRadius = std::numeric_limits<double>::quiet_NaN();
  double meanCoordinateRadius = std::numeric_limits<double>::quiet_NaN();
  double area = std::numeric_limits<double>::quiet_NaN();
  double arealRadius = std::numeric_limits<double>::quiet_NaN();
  double irreducibleMass = std::numeric_limits<double>::quiet_NaN();
  double spectralResidualInfinityNorm =
    std::numeric_limits<double>::quiet_NaN();
  double maxAbsExpansion = std::numeric_limits<double>::quiet_NaN();
  double maxAbsExpansionOverResolved = std::numeric_limits<double>::quiet_NaN();
  double relativeExpansionError = std::numeric_limits<double>::quiet_NaN();

  bool surfaceAvailable = false;
};

/**
 * Reconstruct the final numerical surface and its diagnostics without running
 * Newton again. Reconstruction failure is represented by surfaceAvailable ==
 * false; the complete solver result and reference point are still preserved.
 */
AHFinderResult BuildAHFinderResult(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  const ReferencePoint& referencePoint,
  const SpectralNewtonResult& solverResult
);

/**
 * Coordinate radius of the surface in the direction (rho, sigma) about
 * result.referencePoint, from the retained coefficients. Throws
 * std::invalid_argument if their count does not match result.lMax.
 */
double SurfaceRadius(const AHFinderResult& result, double rho, double sigma);

/**
 * SurfaceRadius in the direction from result.referencePoint towards
 * destination. Throws std::invalid_argument if the two points coincide.
 */
double SurfaceRadiusTowards(
  const AHFinderResult& result,
  const ReferencePoint& destination
);

/**
 * True if the surface encloses point, i.e. point is closer to
 * result.referencePoint than the surface is in that direction. The reference
 * point itself is always enclosed.
 */
bool SurfaceEncloses(const AHFinderResult& result, const ReferencePoint& point);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_AH_FINDER_RESULT_H_
