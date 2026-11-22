#ifndef APPARENT_HORIZON_FINDER_INTERPOLATION_H_
#define APPARENT_HORIZON_FINDER_INTERPOLATION_H_

#include "ahf-data.h"

#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace exahype2::ccz4::ahf {

/**
 * Convert one AH point (r, longitude rho, latitude sigma) into its global
 * Cartesian target coordinate relative to @p referencePoint.
 *
 * @throws std::invalid_argument for non-finite inputs or non-positive radius.
 */
GridCoordinate SurfacePointToCartesian(
  const AHGuessPoint& surfacePoint,
  const ReferencePoint& referencePoint
);

struct AnalyticCoordinateGeometry {
  Tensor3 T{};
  /** [u][i], with u=0 for rho, u=1 for sigma, and i=x,y,z. */
  std::array<Vector3, 2> gradient{};
  /** [u][i][j] = partial_i partial_j u, with u=0 rho and u=1 sigma. */
  std::array<Tensor3, 2> hessian{};
};

/** Compute T_ij = r^2 delta_ij - x_i x_j from relative coordinates. */
Tensor3 ComputeRadialProjectionTensor(const GridCoordinate& relativeCoordinate);

/**
 * Compute the appendix's analytic rho/sigma coordinate geometry. Global input
 * is translated by @p referencePoint before evaluating every formula.
 * Longitude is undefined near |cos(sigma)|=0, so poles are rejected rather
 * than assigned an artificial treatment.
 */
AnalyticCoordinateGeometry ComputeAnalyticCoordinateGeometry(
  const GridCoordinate& globalCoordinate,
  const ReferencePoint& referencePoint
);

/**
 * Return the indices of the @p neighbourCount nearest Cartesian cell centres.
 * Results are ordered by increasing squared Euclidean distance, with cell index
 * as the deterministic tie-breaker. This brute-force baseline is deliberately
 * isolated behind this API so its implementation can later use a spatial index.
 *
 * @throws std::invalid_argument for an empty grid, an invalid neighbour count,
 * or a non-finite target coordinate.
 */
std::vector<std::size_t> FindNearestGridCellsBruteForce(
  const GridData& gridData,
  const GridCoordinate& target,
  std::size_t neighbourCount
);

/** Immutable k-d tree; concurrent nearest-neighbour queries are read-only. */
class CartesianSpatialIndex {
public:
  explicit CartesianSpatialIndex(const GridData& gridData);
  std::vector<std::size_t> findNearest(
    const GridCoordinate& target,
    std::size_t neighbourCount
  ) const;
  /** Return all cells with Euclidean distance strictly less than radius. */
  std::vector<std::size_t> findWithinRadius(
    const GridCoordinate& target,
    double radius
  ) const;
  std::size_t size() const;

private:
  struct Implementation;
  std::shared_ptr<const Implementation> _implementation;
};

/** Query a previously built index without rebuilding it. */
std::vector<std::size_t> FindNearestGridCells(
  const CartesianSpatialIndex& spatialIndex,
  const GridCoordinate& target,
  std::size_t neighbourCount
);

/** Query every indexed cell strictly inside @p radius, sorted by distance/index. */
std::vector<std::size_t> FindGridCellsWithinRadius(
  const CartesianSpatialIndex& spatialIndex,
  const GridCoordinate& target,
  double radius
);

/**
 * Compatibility interface for one-off queries. Repeated queries should build
 * CartesianSpatialIndex once and use the overload above.
 */
std::vector<std::size_t> FindNearestGridCells(
  const GridData& gridData,
  const GridCoordinate& target,
  std::size_t neighbourCount
);

struct LinearPolynomialFit {
  double value = 0.0;
  Vector3 derivative{};
  double lengthScale = 0.0;
};

struct PolynomialExponent {
  std::size_t x = 0;
  std::size_t y = 0;
  std::size_t z = 0;

  bool operator==(const PolynomialExponent&) const = default;
};

struct PolynomialBasis {
  static constexpr std::size_t NoTerm = std::numeric_limits<std::size_t>::max();
  std::vector<PolynomialExponent> terms;
  std::size_t constant = 0;
  std::size_t linearX = NoTerm;
  std::size_t linearY = NoTerm;
  std::size_t linearZ = NoTerm;
};

std::size_t PolynomialBasisSize(std::size_t degree);
PolynomialBasis GeneratePolynomialBasis(std::size_t degree);

enum class LinearReconstructionMethod {
  Unweighted,
  WeightedMovingLeastSquares,
  RadiusWeightedWendlandC2
};

struct LinearReconstructionOptions {
  LinearReconstructionMethod method = LinearReconstructionMethod::Unweighted;
  /** Zero uses every supplied sample; otherwise use the first requested samples. */
  std::size_t numberOfNeighbours = 0;
  /** Supported reconstruction degree, currently 0 through 3. */
  std::size_t polynomialDegree = 1;
  /** Gaussian support alpha in exp(-(distance/(alpha*ell))^2). */
  double weightSupport = 1.0;
  /** Physical compact-support radius for RadiusWeightedWendlandC2. */
  double supportRadius = 0.0;
  /**
   * Multiplier applied to epsilon*n*largest-column-norm for rank detection.
   * Controls how strictly the QR solver decides that the neighbour geometry
   * is rank deficient.
   */
  double rankToleranceMultiplier = 64.0;
};

/** Evaluate the configured unweighted, Gaussian, or compact-support row weight. */
double LinearReconstructionWeight(
  double distance,
  double lengthScale,
  const LinearReconstructionOptions& options
);

/** Wendland C2 compact kernel: (1-q)^4(4q+1) for 0<=q<1, else zero. */
double WendlandC2Weight(double q);

/**
 * Scalar total-degree polynomial least-squares reconstruction using a
 * column-pivoted QR solve. The length scale ell is the maximum Euclidean
 * distance of any sample from the target, except for
 * RadiusWeightedWendlandC2 where ell is the configured support radius. This
 * bounds every scaled sample radius by one and is insensitive to sample
 * ordering.
 *
 * The configured degree p requires at least (p+1)(p+2)(p+3)/6 samples; using
 * more gives the intended overdetermined fit. Degrees 0 through 3 are
 * currently supported by the reconstruction kernel.
 *
 * @throws std::invalid_argument for malformed/non-finite input, too few
 * samples, an unsupported degree, or a rank-deficient sample distribution.
 */
LinearPolynomialFit FitLinearPolynomial(
  std::span<const GridCoordinate> sampleCoordinates,
  std::span<const double> sampleValues,
  const GridCoordinate& target,
  const LinearReconstructionOptions& options = {}
);

struct CompactLinearFit {
  std::array<double, NumberOfCompactFields> values{};
  std::array<Vector3, NumberOfCompactFields> gradients{};
  std::size_t numberOfNeighbours = 0;
  std::size_t polynomialDegree = 1;
  std::size_t rank = 0;
  double lengthScale = 0.0;
  /** Approximate max/min absolute diagonal ratio of the pivoted R factor. */
  double conditionEstimate = 1.0;
};

/**
 * Reconstruct all fourteen compact fields with one shared polynomial design
 * matrix and one column-pivoted QR factorisation. Result fields retain compact
 * order, including K at 12 and direct phi at 13.
 */
CompactLinearFit FitCompactLinearPolynomial(
  const GridData& gridData,
  std::span<const std::size_t> neighbourIndices,
  const GridCoordinate& target,
  const LinearReconstructionOptions& options = {}
);

struct CartesianToSurfaceOptions {
  CartesianToSurfaceMethod method = CartesianToSurfaceMethod::WeightedPolynomial;
  std::size_t numberOfNeighbours = 8;
  std::size_t polynomialDegree = 1;
  double weightSupport = 1.0;
  /** Used only by RadiusWeightedPolynomial; must be finite and positive. */
  double supportRadius = 0.0;
  double rankToleranceMultiplier = 64.0;
  bool allowOrderReduction = false;
  /** Allowed distance outside the grid AABB as a fraction of its diagonal. */
  double domainRelativeTolerance = 0.05;
};

/** Physical fields reconstructed at one arbitrary Cartesian target. */
struct InterpolatedCartesianPoint {
  GridCoordinate target{};
  InterpolatedGeometryData geometry{};
  InterpolationDiagnostics diagnostics{};
};

/**
 * Reusable Cartesian reconstruction context. The GridData supplied to the
 * constructor must outlive this object. Its immutable spatial index is built
 * once and reused by every interpolatePoints() call, including calls made by
 * successive nonlinear AH iterations.
 */
class CartesianPointInterpolator {
public:
  explicit CartesianPointInterpolator(
    const GridData& gridData,
    const CartesianToSurfaceOptions& options = {}
  );

  /**
   * True when @p target lies inside the sampled Cartesian domain, widened by
   * the configured domainRelativeTolerance. This is the same test
   * interpolatePoints applies per point, exposed so a caller can reject an
   * unusable reference point or initial radius up front instead of discovering
   * it through non-finite residuals.
   */
  bool coversTarget(const GridCoordinate& target) const;

  /** Corner-to-corner extent of the sampled domain, for diagnostics. */
  GridCoordinate domainMinimum() const { return _domainMinimum; }
  GridCoordinate domainMaximum() const { return _domainMaximum; }

  /** Reconstruct physical fields in exactly the order of @p targets. */
  void interpolatePoints(
    std::span<const GridCoordinate> targets,
    std::vector<InterpolatedCartesianPoint>& output
  ) const;

private:
  const GridData* _gridData;
  CartesianToSurfaceOptions _options;
  CartesianSpatialIndex _spatialIndex;
  GridCoordinate _domainMinimum{};
  GridCoordinate _domainMaximum{};
  double _domainDiagonal = 0.0;
};

/** Surface adapter using an already-built, persistent Cartesian interpolator. */
void InterpolateCartesianToSurface(
  const CartesianPointInterpolator& interpolator,
  const AHGuess& ahGuess,
  const ReferencePoint& referencePoint,
  InterpolatedSurface& interpolatedSurface
);

/**
 * Interpolate compact Cartesian fields directly to every AH surface point.
 * Polynomial modes populate physical metric derivatives in [k][i][j] order,
 * where k is the Cartesian differentiation direction. Surface-shape and
 * angular derivatives are intentionally not computed.
 * This compatibility overload builds a temporary CartesianPointInterpolator;
 * iterative callers should use the overload above to retain its spatial index.
 */
void InterpolateCartesianToSurface(
  const GridData& gridData,
  const AHGuess& ahGuess,
  const ReferencePoint& referencePoint,
  InterpolatedSurface& interpolatedSurface,
  const CartesianToSurfaceOptions& options = {}
);

/**
 * Sample each AH surface point from its nearest Cartesian cell centre and
 * reconstruct the physical spatial metric and extrinsic curvature.
 *
 * This low-order baseline does not compute metric derivatives, coordinate
 * derivatives, or surface normals. Their storage is zero-initialised and
 * spatialMetricDerivativeAvailable is false for every output point.
 */
void InterpolateToSurfaceNearestNeighbour(
  const GridData& gridData,
  const AHGuess& ahGuess,
  const ReferencePoint& referencePoint,
  InterpolatedSurface& interpolatedSurface
);

/**
 * Interpolate the Cartesian grid data to a shell-like subdomain around the current AH guess.
 */
void InterpolateFromCartesianToShell(const GridData& gridData,
                                     const AHGuess& ahGuess,
                                     const ReferencePoint& referencePoint,
                                     InterpolatedShell& interpolatedShell,
                                     std::span<const AngularSample> angularSamples = {});

/**
 * Interpolate the shell data to the apparent-horizon surface defined by the current AH guess.
 */
void InterpolateFromShellToSurface(const InterpolatedShell& interpolatedShell,
                                   const AHGuess& ahGuess,
                                   const ReferencePoint& referencePoint,
                                   InterpolatedSurface& interpolatedSurface);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_INTERPOLATION_H_
