#ifndef APPARENT_HORIZON_FINDER_COMMON_HORIZON_CANDIDATES_H_
#define APPARENT_HORIZON_FINDER_COMMON_HORIZON_CANDIDATES_H_

#include "ah-finder-result.h"
#include "ah-search.h"

#include <optional>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf {

/** User identity and retry configuration for one physical black hole. */
struct AHPhysicalBlackHole {
  std::string id;
  double mass = 1.0;
  std::vector<AHSearchAttempt> attempts;
  /** Physical puncture/location estimate, distinct from numerical AH origins. */
  std::optional<ReferencePoint> physicalPosition;
};

/** A successfully reconstructed individual horizon associated with its ID. */
struct AHIndividualHorizonObservation {
  std::string id;
  double mass = 1.0;
  AHFinderResult horizon;
};

/** Directional separation diagnostic for one unordered pair. */
struct AHHorizonPairGap {
  std::string firstId;
  std::string secondId;
  double centreDistance = 0.0;
  double firstFacingRadius = 0.0;
  double secondFacingRadius = 0.0;
  double surfaceGap = 0.0;
  bool proximityTriggered = false;
};

struct AHCommonHorizonCandidateOptions {
  /**
   * A pair triggers when surfaceGap <= searchMargin + searchMarginPerMass *
   * (m_i + m_j), with the masses the black holes were given. Both margins are
   * in coordinate-length units.
   *
   * A common horizon forms while the individual horizons are still apart: on
   * equal-mass Brill-Lindquist data it exists up to a separation of 0.766 M,
   * where the gap between the individual horizons is 0.414 M (M the total
   * bare mass). The default per-mass margin covers that with about 20%
   * headroom; with unequal masses the gaps measured were smaller. A pair that
   * triggers without a common horizon costs one failed search. The margins
   * were calibrated on time-symmetric data at t = 0; evolved data uses other
   * coordinates, where the gap at formation has not been measured.
   */
  double searchMargin = 0.0;
  double searchMarginPerMass = 0.5;
  bool includeTriggeredPairs = true;
  bool includeConnectedComponents = true;
  /** Always add the group containing every observed black hole. */
  bool unconditionalAllBlackHoles = false;
  /** Always add these ID subsets, independently of proximity. */
  std::vector<std::vector<std::string>> requestedSubsets;
};

/** Why a canonical group survived deterministic deduplication. */
struct AHCommonHorizonCandidateReason {
  bool triggeredPair = false;
  bool connectedComponent = false;
  bool explicitlyRequested = false;
  bool unconditionalAllBlackHoles = false;
};

/** Geometry and initial spherical guess for one prospective common AH solve. */
struct AHCommonHorizonCandidate {
  /** Sorted stable IDs; also the canonical deduplication key. */
  std::vector<std::string> memberIds;
  AHCommonHorizonCandidateReason reason;
  ReferencePoint referencePoint{0.0, 0.0, 0.0};
  double totalMass = 0.0;
  double enclosingInitialRadius = 0.0;
};

struct AHCommonHorizonCandidatePlan {
  std::vector<AHHorizonPairGap> pairGaps;
  std::vector<AHCommonHorizonCandidate> candidates;
};

void ValidateAHPhysicalBlackHoles(
  const std::vector<AHPhysicalBlackHole>& blackHoles
);

/**
 * Convert user identities into the existing individual-target representation.
 * With excludeOtherBlackHoles, each target excludes the other black holes'
 * physical positions, so that an individual search cannot settle on a surface
 * around another black hole, such as the common horizon.
 */
std::vector<AHHorizonTarget> MakeIndividualHorizonTargets(
  const std::vector<AHPhysicalBlackHole>& blackHoles,
  bool excludeOtherBlackHoles = true
);

void ValidateAHCommonHorizonCandidateOptions(
  const AHCommonHorizonCandidateOptions& options
);

/**
 * Plan prospective common searches from already successful individual AHs.
 * Proximity only creates a candidate; it is not evidence that a common AH
 * exists, and this function deliberately performs no Newton solve.
 */
AHCommonHorizonCandidatePlan PlanCommonHorizonCandidates(
  const std::vector<AHIndividualHorizonObservation>& observations,
  const AHCommonHorizonCandidateOptions& options = {}
);

}  // namespace exahype2::ccz4::ahf

#endif
