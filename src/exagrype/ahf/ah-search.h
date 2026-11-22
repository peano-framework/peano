#ifndef APPARENT_HORIZON_FINDER_AH_SEARCH_H_
#define APPARENT_HORIZON_FINDER_AH_SEARCH_H_

#include "ah-finder-result.h"
#include "interpolation.h"
#include "spectral-solver.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf {

/** One explicit, ordered initial-surface/reference-point pair. */
struct AHSearchAttempt {
  ReferencePoint referencePoint{0.0, 0.0, 0.0};
  double initialRadius = 1.0;
  /**
   * Optional explicit star-shaped spectral surface. When non-empty this takes
   * precedence over initialRadius and must match the configured lMax.
   */
  std::vector<double> initialCoefficients;
};

/** Continue from a prior reconstructable result using its centre and surface. */
AHSearchAttempt MakePreviousSurfaceAttempt(const AHFinderResult& previous);

/** Shared numerical configuration and an arbitrary ordered attempt list. */
struct AHSearchOptions {
  std::vector<AHSearchAttempt> attempts;
  int lMax = 2;
  CartesianToSurfaceOptions interpolation{};
  SpectralNewtonOptions solver{};
};

/** Complete diagnostics from one executed attempt. */
struct AHSearchAttemptResult {
  ReferencePoint referencePoint{0.0, 0.0, 0.0};
  double initialRadius = 1.0;
  std::vector<double> initialCoefficients;
  AHFinderResult result;
  /**
   * True only when Newton converged, final reconstruction succeeded and the
   * relative expansion error is within solver.maxRelativeExpansionError.
   */
  bool succeeded = false;
  /** Why a converged, reconstructed surface was not accepted; else empty. */
  std::string rejection;
};

/** Aggregate ordered-search outcome. */
struct AHSearchResult {
  static constexpr std::size_t NoSuccessfulAttempt =
    std::numeric_limits<std::size_t>::max();

  bool converged = false;
  std::size_t successfulAttempt = NoSuccessfulAttempt;
  std::optional<AHFinderResult> horizon;
  std::vector<AHSearchAttemptResult> attempts;
};

/** Reject an empty list or any structurally invalid shared/attempt option. */
void ValidateAHSearchOptions(const AHSearchOptions& options);

/**
 * Try explicit attempt pairs in vector order and stop at the first accepted
 * horizon. One persistent Cartesian interpolator is reused.
 *
 * Newton zeroes only the projection of Theta at the collocation points, so a
 * converged, reconstructable surface is accepted only if its
 * AHFinderResult::relativeExpansionError (Theta on a grid twice as fine) is
 * within SpectralNewtonOptions::maxRelativeExpansionError. From a starting
 * sphere that is too large, Newton can converge to a surface that aliasing
 * makes look like a horizon, and the next attempt may still find the real
 * one, so a rejection does not end the search.
 *
 * \dot
 * digraph search {
 *   rankdir=LR;
 *   node [shape=box];
 *   attempt [label="next attempt"];
 *   newton [label="Newton on the\ncollocation points"];
 *   rebuild [label="reconstruct the surface,\nTheta on a finer grid"];
 *   accepted [label="accepted", shape=ellipse];
 *   none [label="no horizon", shape=ellipse];
 *   attempt -> newton;
 *   newton -> rebuild [label="converged"];
 *   newton -> attempt [label="failed"];
 *   rebuild -> accepted [label="error within bound"];
 *   rebuild -> attempt [label="error above bound,\nor not reconstructable"];
 *   attempt -> none [label="none left"];
 * }
 * \enddot
 */
AHSearchResult FindApparentHorizon(
  const GridData& gridData,
  const AHSearchOptions& options
);

/** Physical role assigned by the user to one independently searched surface. */
enum class AHHorizonKind {
  Individual,
  Common
};

/**
 * One physical horizon and its ordered alternative guesses. The target ID is
 * the stable association between attempts and the intended black hole/common
 * horizon; no geometric post-hoc matching is performed.
 */
struct AHHorizonTarget {
  std::string id;
  AHHorizonKind kind = AHHorizonKind::Individual;
  std::vector<AHSearchAttempt> attempts;
  /** Optional per-target override of SpectralNewtonOptions::massScale. */
  std::optional<double> massScale;
  /**
   * Points the surface must not enclose, typically the other black holes'
   * positions. From a starting sphere that is too large, an individual search
   * can converge to the common horizon, which is a genuine horizon and passes
   * every other check; an attempt whose surface encloses one of these points
   * is rejected and the next attempt runs.
   */
  std::vector<ReferencePoint> excludedPoints;
};

/** Shared numerical configuration for any number of physical targets. */
struct AHMultiHorizonOptions {
  std::vector<AHHorizonTarget> targets;
  int lMax = 2;
  CartesianToSurfaceOptions interpolation{};
  SpectralNewtonOptions solver{};
};

/** Ordered attempts and selected result for one physical target. */
struct AHHorizonTargetResult {
  std::string id;
  AHHorizonKind kind = AHHorizonKind::Individual;
  bool converged = false;
  std::size_t successfulAttempt = AHSearchResult::NoSuccessfulAttempt;
  std::optional<AHFinderResult> horizon;
  std::vector<AHSearchAttemptResult> attempts;
  std::optional<double> massScale;
};

/** Aggregate result without collapsing partial success into one boolean. */
struct AHMultiHorizonResult {
  bool anyHorizonFound = false;
  bool allTargetsFound = false;
  std::vector<AHHorizonTargetResult> targets;
};

void ValidateAHMultiHorizonOptions(const AHMultiHorizonOptions& options);

/**
 * Search every physical target in vector order. The first successful attempt
 * stops retries for that target only; remaining targets are still searched.
 * One spectral basis and one persistent Cartesian interpolator are shared.
 */
AHMultiHorizonResult FindApparentHorizons(
  const GridData& gridData,
  const AHMultiHorizonOptions& options
);

/**
 * Execute a validated multi-target search using caller-owned numerical
 * objects. This is the orchestration primitive used when several search
 * phases must share one persistent Cartesian spatial index.
 */
AHMultiHorizonResult FindApparentHorizons(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  const AHMultiHorizonOptions& options
);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_AH_SEARCH_H_
