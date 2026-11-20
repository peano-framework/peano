#ifndef APPARENT_HORIZON_FINDER_COMMON_HORIZON_SEARCH_H_
#define APPARENT_HORIZON_FINDER_COMMON_HORIZON_SEARCH_H_

#include "common-horizon-candidates.h"

#include <limits>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf {

struct AHCommonHorizonTargetOptions {
  double initialRadiusInflation = 1.0;
  /** Ordered factors applied after initialRadiusInflation. */
  std::vector<double> retryRadiusMultipliers{1.0, 1.5, 2.0};
};

std::string MakeCommonHorizonId(const std::vector<std::string>& memberIds);

AHHorizonTarget MakeCommonHorizonTarget(
  const AHCommonHorizonCandidate& candidate,
  const AHCommonHorizonTargetOptions& options = {}
);

struct AHCommonHorizonValidationOptions {
  double spectralResidualTolerance = 1.0e-8;
  /**
   * Absolute bound on AHFinderResult::maxAbsExpansion (collocation points).
   * Off by default: Theta has units of inverse length, and on grid data it
   * reaches 1e-4 to 1e-2 even on correct horizons.
   */
  double pointwiseExpansionTolerance = std::numeric_limits<double>::infinity();
  /**
   * Bound on AHFinderResult::relativeExpansionError (over-resolved, relative
   * to a round sphere). Physical analytic horizons with lMax >= 4 stayed below
   * 0.13, aliased surfaces above 0.55; see SpectralNewtonOptions::
   * maxRelativeExpansionError, which the search applies first.
   */
  double relativeExpansionTolerance = 0.3;
  double enclosureTolerance = 1.0e-8;
  int enclosureRhoPoints = 32;
  int enclosureSigmaPoints = 17;
  /** Reject surfaces with maxRadius/minRadius above this bound. */
  double maximumRadiusRatio = 10.0;
};

enum class AHCommonHorizonValidationStatus {
  FullyValidated,
  ProvisionallyValidated,
  Rejected
};

enum class AHEnclosureValidationStatus {
  Passed,
  Failed,
  Unavailable
};

struct AHCommonHorizonValidation {
  AHCommonHorizonValidationStatus status =
    AHCommonHorizonValidationStatus::Rejected;
  AHEnclosureValidationStatus memberHorizonEnclosure =
    AHEnclosureValidationStatus::Unavailable;
  AHEnclosureValidationStatus physicalPositionEnclosure =
    AHEnclosureValidationStatus::Unavailable;
  bool numericallyValid = false;
  /** Backward-compatible: true only for FullyValidated. */
  bool valid = false;
  bool newtonConverged = false;
  bool surfaceAvailable = false;
  bool radiiValid = false;
  bool coefficientsFinite = false;
  bool spectralResidualValid = false;
  bool pointwiseExpansionValid = false;
  bool relativeExpansionValid = false;
  bool enclosesMembers = false;
  bool surfaceRegular = false;
  double minimumEnclosureClearance =
    std::numeric_limits<double>::quiet_NaN();
  double minimumPhysicalPositionClearance =
    std::numeric_limits<double>::quiet_NaN();
  std::vector<std::string> reasons;
};

/** Numerical/geometric checks only; this is not proof of outermost status. */
AHCommonHorizonValidation ValidateCommonHorizon(
  const AHFinderResult& common,
  const std::vector<AHIndividualHorizonObservation>& members,
  const AHCommonHorizonValidationOptions& options = {}
);

/**
 * Validate with potentially incomplete member surfaces and independently
 * available physical positions. expectedMemberCount defines completeness.
 */
AHCommonHorizonValidation ValidateCommonHorizon(
  const AHFinderResult& common,
  const std::vector<AHIndividualHorizonObservation>& reconstructedMembers,
  const std::vector<ReferencePoint>& physicalPositions,
  std::size_t expectedMemberCount,
  const AHCommonHorizonValidationOptions& options = {}
);

struct AHCommonHorizonSearchOptions {
  std::vector<AHPhysicalBlackHole> blackHoles;
  AHCommonHorizonCandidateOptions candidates{};
  AHCommonHorizonTargetOptions commonTargets{};
  AHCommonHorizonValidationOptions validation{};
  /**
   * Reject an individual surface that encloses another black hole's
   * physicalPosition; see MakeIndividualHorizonTargets.
   */
  bool individualsExcludeOtherBlackHoles = true;
  int lMax = 2;
  CartesianToSurfaceOptions interpolation{};
  SpectralNewtonOptions solver{};
};

struct AHCommonHorizonCatalogueEntry {
  std::string id;
  std::vector<std::string> memberIds;
  AHCommonHorizonCandidateReason reason;
  AHCommonHorizonCandidate candidate;
  AHHorizonTargetResult search;
  AHCommonHorizonValidation validation;
};

struct AHHorizonCatalogue {
  AHMultiHorizonResult individuals;
  AHCommonHorizonCandidatePlan candidatePlan;
  std::vector<AHCommonHorizonCatalogueEntry> commonHorizons;
};

/**
 * Unified N>=1 workflow. For N=1 only the individual multi-attempt search is
 * executed; for N>=2 common candidates are additionally planned and solved.
 * One basis and one Cartesian interpolator are shared by all phases.
 */
AHHorizonCatalogue FindApparentHorizons(
  const GridData& gridData,
  const AHCommonHorizonSearchOptions& options
);

/** Backward-compatible name for the unified N>=1 workflow. */
AHHorizonCatalogue FindCommonApparentHorizons(
  const GridData& gridData,
  const AHCommonHorizonSearchOptions& options
);

}  // namespace exahype2::ccz4::ahf

#endif
