#include "common-horizon-search.h"

#include "harmonics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <unordered_map>

namespace exahype2::ccz4::ahf {
namespace {

double Distance(const ReferencePoint& a, const ReferencePoint& b) {
  return std::hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

std::vector<std::string> Canonical(std::vector<std::string> ids) {
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

void MergeReason(AHCommonHorizonCandidateReason& a,
                 const AHCommonHorizonCandidateReason& b) {
  a.triggeredPair |= b.triggeredPair;
  a.connectedComponent |= b.connectedComponent;
  a.explicitlyRequested |= b.explicitlyRequested;
  a.unconditionalAllBlackHoles |= b.unconditionalAllBlackHoles;
}

double InitialExtent(const AHPhysicalBlackHole& blackHole) {
  const AHSearchAttempt& attempt = blackHole.attempts.front();
  if (attempt.initialCoefficients.empty()) return attempt.initialRadius;
  const int lmax = static_cast<int>(std::sqrt(attempt.initialCoefficients.size())) - 1;
  if (static_cast<std::size_t>((lmax + 1) * (lmax + 1)) !=
      attempt.initialCoefficients.size()) {
    throw std::invalid_argument("Explicit initial coefficients do not form a spectral basis");
  }
  const auto field = Synthesize(BuildSpectralBasis(lmax), attempt.initialCoefficients);
  return *std::max_element(field.h.begin(), field.h.end());
}

AHCommonHorizonCandidate PhysicalCandidate(
  const std::vector<std::string>& requested,
  const std::unordered_map<std::string, const AHPhysicalBlackHole*>& physical,
  AHCommonHorizonCandidateReason reason
) {
  AHCommonHorizonCandidate candidate;
  candidate.memberIds = Canonical(requested);
  candidate.reason = reason;
  for (const auto& id : candidate.memberIds) {
    const auto found = physical.find(id);
    if (found == physical.end()) {
      throw std::invalid_argument("Requested common group contains unknown ID " + id);
    }
    const auto& hole = *found->second;
    if (!hole.physicalPosition.has_value()) {
      throw std::invalid_argument(
        "Cannot construct fallback common candidate " +
        MakeCommonHorizonId(Canonical(requested)) +
        ": physicalPosition is unavailable for " + id);
    }
    candidate.totalMass += hole.mass;
    candidate.referencePoint.x += hole.mass * hole.physicalPosition->x;
    candidate.referencePoint.y += hole.mass * hole.physicalPosition->y;
    candidate.referencePoint.z += hole.mass * hole.physicalPosition->z;
  }
  candidate.referencePoint.x /= candidate.totalMass;
  candidate.referencePoint.y /= candidate.totalMass;
  candidate.referencePoint.z /= candidate.totalMass;
  for (const auto& id : candidate.memberIds) {
    const auto& hole = *physical.at(id);
    candidate.enclosingInitialRadius = std::max(
      candidate.enclosingInitialRadius,
      Distance(candidate.referencePoint, *hole.physicalPosition) +
        InitialExtent(hole));
  }
  return candidate;
}

void ValidateTargetOptions(const AHCommonHorizonTargetOptions& options) {
  if (!std::isfinite(options.initialRadiusInflation) ||
      options.initialRadiusInflation <= 0.0 ||
      options.retryRadiusMultipliers.empty()) {
    throw std::invalid_argument("Common target inflation and retries must be positive");
  }
  for (double factor : options.retryRadiusMultipliers) {
    if (!std::isfinite(factor) || factor <= 0.0) {
      throw std::invalid_argument("Common target retry multipliers must be finite and positive");
    }
  }
}

void ValidateValidationOptions(const AHCommonHorizonValidationOptions& o) {
  if (!std::isfinite(o.spectralResidualTolerance) || o.spectralResidualTolerance < 0.0 ||
      std::isnan(o.pointwiseExpansionTolerance) ||
      o.pointwiseExpansionTolerance < 0.0 ||
      std::isnan(o.relativeExpansionTolerance) ||
      o.relativeExpansionTolerance <= 0.0 ||
      !std::isfinite(o.enclosureTolerance) || o.enclosureTolerance < 0.0 ||
      o.enclosureRhoPoints < 3 || o.enclosureSigmaPoints < 3 ||
      !std::isfinite(o.maximumRadiusRatio) || o.maximumRadiusRatio < 1.0) {
    throw std::invalid_argument("Invalid common-horizon validation options");
  }
}

}  // namespace

std::string MakeCommonHorizonId(const std::vector<std::string>& memberIds) {
  const auto ids = Canonical(memberIds);
  if (ids.size() < 2) throw std::invalid_argument("A common horizon needs at least two members");
  std::string result = "common[";
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) result += "+";
    result += ids[i];
  }
  return result + "]";
}

AHHorizonTarget MakeCommonHorizonTarget(
  const AHCommonHorizonCandidate& candidate,
  const AHCommonHorizonTargetOptions& options
) {
  ValidateTargetOptions(options);
  if (!std::isfinite(candidate.totalMass) || candidate.totalMass <= 0.0 ||
      !std::isfinite(candidate.enclosingInitialRadius) ||
      candidate.enclosingInitialRadius <= 0.0) {
    throw std::invalid_argument("Common candidate mass/radius is invalid");
  }
  AHHorizonTarget target;
  target.id = MakeCommonHorizonId(candidate.memberIds);
  target.kind = AHHorizonKind::Common;
  target.massScale = candidate.totalMass;
  for (double factor : options.retryRadiusMultipliers) {
    target.attempts.push_back({
      candidate.referencePoint,
      candidate.enclosingInitialRadius * options.initialRadiusInflation * factor,
      {}
    });
  }
  return target;
}

AHCommonHorizonValidation ValidateCommonHorizon(
  const AHFinderResult& common,
  const std::vector<AHIndividualHorizonObservation>& members,
  const AHCommonHorizonValidationOptions& options
) {
  return ValidateCommonHorizon(
    common, members, {}, members.size(), options);
}

AHCommonHorizonValidation ValidateCommonHorizon(
  const AHFinderResult& common,
  const std::vector<AHIndividualHorizonObservation>& members,
  const std::vector<ReferencePoint>& physicalPositions,
  std::size_t expectedMemberCount,
  const AHCommonHorizonValidationOptions& options
) {
  ValidateValidationOptions(options);
  if (expectedMemberCount < 2 || members.size() > expectedMemberCount ||
      physicalPositions.size() > expectedMemberCount) {
    throw std::invalid_argument(
      "ValidateCommonHorizon: inconsistent member counts");
  }
  AHCommonHorizonValidation v;
  v.newtonConverged = common.solver.converged;
  if (!v.newtonConverged) v.reasons.push_back("Newton did not converge");
  v.surfaceAvailable = common.surfaceAvailable;
  if (!v.surfaceAvailable) v.reasons.push_back("surface reconstruction is unavailable");
  v.radiiValid = common.surfaceAvailable && std::isfinite(common.minRadius) &&
    std::isfinite(common.meanRadius) && std::isfinite(common.maxRadius) &&
    common.minRadius > 0.0 && common.meanRadius > 0.0 && common.maxRadius > 0.0;
  if (!v.radiiValid) v.reasons.push_back("surface radii are non-finite or non-positive");
  v.coefficientsFinite = !common.solver.coefficients.empty() &&
    std::all_of(common.solver.coefficients.begin(), common.solver.coefficients.end(),
                [](double x) { return std::isfinite(x); });
  if (!v.coefficientsFinite) v.reasons.push_back("spectral coefficients are unavailable or non-finite");
  v.spectralResidualValid = std::isfinite(common.spectralResidualInfinityNorm) &&
    common.spectralResidualInfinityNorm <= options.spectralResidualTolerance;
  if (!v.spectralResidualValid) v.reasons.push_back("spectral residual exceeds tolerance");
  v.pointwiseExpansionValid = std::isfinite(common.maxAbsExpansion) &&
    common.maxAbsExpansion <= options.pointwiseExpansionTolerance;
  if (!v.pointwiseExpansionValid) v.reasons.push_back("pointwise expansion exceeds tolerance");
  v.relativeExpansionValid = std::isfinite(common.relativeExpansionError) &&
    common.relativeExpansionError <= options.relativeExpansionTolerance;
  if (!v.relativeExpansionValid) {
    v.reasons.push_back("relative expansion error exceeds tolerance");
  }
  v.surfaceRegular = v.radiiValid && v.coefficientsFinite &&
    common.maxRadius / common.minRadius <= options.maximumRadiusRatio;
  if (v.surfaceRegular) {
    for (int is = 0; is < options.enclosureSigmaPoints; ++is) {
      const double sigma = -std::numbers::pi / 2.0 + std::numbers::pi * is /
        static_cast<double>(options.enclosureSigmaPoints - 1);
      for (int ir = 0; ir < options.enclosureRhoPoints; ++ir) {
        const double rho = 2.0 * std::numbers::pi * ir /
          static_cast<double>(options.enclosureRhoPoints);
        const double radius = SurfaceRadius(common, rho, sigma);
        if (!std::isfinite(radius) || radius <= 0.0) v.surfaceRegular = false;
      }
    }
  }
  if (!v.surfaceRegular) v.reasons.push_back("surface regularity check failed");

  v.numericallyValid = v.newtonConverged && v.surfaceAvailable &&
    v.radiiValid && v.coefficientsFinite && v.spectralResidualValid &&
    v.pointwiseExpansionValid && v.relativeExpansionValid && v.surfaceRegular;

  const bool canEvaluateEnclosure = v.radiiValid && v.coefficientsFinite;
  bool availableMembersEnclosed = true;
  if (!members.empty()) {
    v.minimumEnclosureClearance = std::numeric_limits<double>::infinity();
  }
  if (canEvaluateEnclosure) {
    for (const auto& member : members) {
      if (!member.horizon.surfaceAvailable || member.horizon.solver.coefficients.empty()) {
        availableMembersEnclosed = false;
        break;
      }
      for (int is = 0; is < options.enclosureSigmaPoints; ++is) {
        const double sigma = -std::numbers::pi / 2.0 + std::numbers::pi * is /
          static_cast<double>(options.enclosureSigmaPoints - 1);
        for (int ir = 0; ir < options.enclosureRhoPoints; ++ir) {
          const double rho = 2.0 * std::numbers::pi * ir /
            static_cast<double>(options.enclosureRhoPoints);
          const double memberRadius = SurfaceRadius(member.horizon, rho, sigma);
          const double c = std::cos(sigma);
          const ReferencePoint point{
            member.horizon.referencePoint.x + memberRadius * c * std::cos(rho),
            member.horizon.referencePoint.y + memberRadius * c * std::sin(rho),
            member.horizon.referencePoint.z + memberRadius * std::sin(sigma)};
          const double dx = point.x - common.referencePoint.x;
          const double dy = point.y - common.referencePoint.y;
          const double dz = point.z - common.referencePoint.z;
          const double radialDistance = std::hypot(dx, dy, dz);
          const double commonRho = std::atan2(dy, dx);
          const double commonSigma = radialDistance > 0.0
            ? std::asin(std::clamp(dz / radialDistance, -1.0, 1.0)) : 0.0;
          const double clearance =
            SurfaceRadius(common, commonRho, commonSigma) - radialDistance;
          v.minimumEnclosureClearance = std::min(v.minimumEnclosureClearance, clearance);
          if (!std::isfinite(clearance) || clearance < -options.enclosureTolerance) {
            availableMembersEnclosed = false;
          }
        }
      }
    }
  }
  if (!canEvaluateEnclosure) {
    v.memberHorizonEnclosure = AHEnclosureValidationStatus::Unavailable;
  } else if (!availableMembersEnclosed) {
    v.memberHorizonEnclosure = AHEnclosureValidationStatus::Failed;
    v.reasons.push_back(
      "common surface fails enclosure of an available member horizon");
  } else if (members.size() == expectedMemberCount) {
    v.memberHorizonEnclosure = AHEnclosureValidationStatus::Passed;
  } else {
    v.memberHorizonEnclosure = AHEnclosureValidationStatus::Unavailable;
    v.reasons.push_back(
      "complete member-horizon enclosure is unavailable");
  }
  v.enclosesMembers =
    v.memberHorizonEnclosure == AHEnclosureValidationStatus::Passed;

  bool availablePositionsEnclosed = true;
  if (!physicalPositions.empty()) {
    v.minimumPhysicalPositionClearance =
      std::numeric_limits<double>::infinity();
  }
  if (canEvaluateEnclosure) {
    for (const ReferencePoint& point : physicalPositions) {
      const double dx = point.x - common.referencePoint.x;
      const double dy = point.y - common.referencePoint.y;
      const double dz = point.z - common.referencePoint.z;
      const double radialDistance = std::hypot(dx, dy, dz);
      const double rho = std::atan2(dy, dx);
      const double sigma = radialDistance > 0.0
        ? std::asin(std::clamp(dz / radialDistance, -1.0, 1.0)) : 0.0;
      const double clearance =
        SurfaceRadius(common, rho, sigma) - radialDistance;
      v.minimumPhysicalPositionClearance = std::min(
        v.minimumPhysicalPositionClearance, clearance);
      if (!std::isfinite(clearance) || clearance < -options.enclosureTolerance) {
        availablePositionsEnclosed = false;
      }
    }
  }
  if (!canEvaluateEnclosure) {
    v.physicalPositionEnclosure = AHEnclosureValidationStatus::Unavailable;
  } else if (!availablePositionsEnclosed) {
    v.physicalPositionEnclosure = AHEnclosureValidationStatus::Failed;
    v.reasons.push_back(
      "common surface fails enclosure of an available physical position");
  } else if (physicalPositions.size() == expectedMemberCount) {
    v.physicalPositionEnclosure = AHEnclosureValidationStatus::Passed;
  } else {
    v.physicalPositionEnclosure = AHEnclosureValidationStatus::Unavailable;
    if (!physicalPositions.empty()) {
      v.reasons.push_back("physical-position enclosure is incomplete");
    }
  }

  const bool enclosureFailed =
    v.memberHorizonEnclosure == AHEnclosureValidationStatus::Failed ||
    v.physicalPositionEnclosure == AHEnclosureValidationStatus::Failed;
  if (!v.numericallyValid || enclosureFailed) {
    v.status = AHCommonHorizonValidationStatus::Rejected;
  } else if (v.memberHorizonEnclosure == AHEnclosureValidationStatus::Passed) {
    v.status = AHCommonHorizonValidationStatus::FullyValidated;
  } else {
    v.status = AHCommonHorizonValidationStatus::ProvisionallyValidated;
  }
  v.valid = v.status == AHCommonHorizonValidationStatus::FullyValidated;
  return v;
}

AHHorizonCatalogue FindApparentHorizons(
  const GridData& gridData,
  const AHCommonHorizonSearchOptions& options
) {
  ValidateAHPhysicalBlackHoles(options.blackHoles);
  ValidateAHCommonHorizonCandidateOptions(options.candidates);
  ValidateTargetOptions(options.commonTargets);
  ValidateValidationOptions(options.validation);

  const SpectralBasis basis = BuildSpectralBasis(options.lMax);
  const CartesianPointInterpolator interpolator(gridData, options.interpolation);
  const AHMultiHorizonOptions individualOptions{
    MakeIndividualHorizonTargets(
      options.blackHoles, options.individualsExcludeOtherBlackHoles),
    options.lMax,
    options.interpolation, options.solver};
  AHHorizonCatalogue catalogue;
  catalogue.individuals = FindApparentHorizons(basis, interpolator, individualOptions);

  std::unordered_map<std::string, const AHPhysicalBlackHole*> physical;
  for (const auto& hole : options.blackHoles) physical.emplace(hole.id, &hole);
  std::vector<AHIndividualHorizonObservation> observations;
  for (const auto& target : catalogue.individuals.targets) {
    if (target.converged && target.horizon.has_value()) {
      observations.push_back({target.id, physical.at(target.id)->mass, *target.horizon});
    }
  }
  std::unordered_map<std::string, const AHIndividualHorizonObservation*> observed;
  for (const auto& observation : observations) observed.emplace(observation.id, &observation);

  AHCommonHorizonCandidateOptions geometricOptions = options.candidates;
  geometricOptions.requestedSubsets.clear();
  for (const auto& subset : options.candidates.requestedSubsets) {
    const bool complete = std::all_of(
      subset.begin(), subset.end(),
      [&](const std::string& id) { return observed.contains(id); });
    if (complete) geometricOptions.requestedSubsets.push_back(subset);
  }
  geometricOptions.unconditionalAllBlackHoles =
    options.candidates.unconditionalAllBlackHoles &&
    observations.size() == options.blackHoles.size();
  if (!observations.empty()) {
    catalogue.candidatePlan = PlanCommonHorizonCandidates(observations, geometricOptions);
  }

  std::map<std::string, AHCommonHorizonCandidate> candidates;
  for (auto candidate : catalogue.candidatePlan.candidates) {
    candidates.emplace(MakeCommonHorizonId(candidate.memberIds), std::move(candidate));
  }
  auto addFallback = [&](const std::vector<std::string>& ids,
                         AHCommonHorizonCandidateReason reason) {
    const std::string key = MakeCommonHorizonId(Canonical(ids));
    const auto existing = candidates.find(key);
    if (existing != candidates.end()) {
      MergeReason(existing->second.reason, reason);
      return;
    }
    candidates.emplace(key, PhysicalCandidate(ids, physical, reason));
  };
  for (const auto& subset : options.candidates.requestedSubsets) {
    addFallback(subset, {.explicitlyRequested = true});
  }
  if (options.candidates.unconditionalAllBlackHoles && options.blackHoles.size() >= 2) {
    std::vector<std::string> all;
    for (const auto& hole : options.blackHoles) all.push_back(hole.id);
    addFallback(all, {.unconditionalAllBlackHoles = true});
  }
  catalogue.candidatePlan.candidates.clear();
  std::vector<AHHorizonTarget> commonTargets;
  for (auto& [id, candidate] : candidates) {
    (void)id;
    catalogue.candidatePlan.candidates.push_back(candidate);
    commonTargets.push_back(MakeCommonHorizonTarget(candidate, options.commonTargets));
  }
  if (commonTargets.empty()) return catalogue;

  const AHMultiHorizonOptions commonOptions{
    commonTargets, options.lMax, options.interpolation, options.solver};
  AHMultiHorizonResult commonSearch =
    FindApparentHorizons(basis, interpolator, commonOptions);
  for (std::size_t i = 0; i < commonSearch.targets.size(); ++i) {
    const auto& candidate = catalogue.candidatePlan.candidates[i];
    auto search = std::move(commonSearch.targets[i]);
    std::vector<AHIndividualHorizonObservation> members;
    std::vector<ReferencePoint> physicalPositions;
    for (const auto& id : candidate.memberIds) {
      if (observed.contains(id)) members.push_back(*observed.at(id));
      if (physical.at(id)->physicalPosition.has_value()) {
        physicalPositions.push_back(*physical.at(id)->physicalPosition);
      }
    }
    AHCommonHorizonValidation validation;
    if (search.horizon.has_value()) {
      validation = ValidateCommonHorizon(
        *search.horizon, members, physicalPositions,
        candidate.memberIds.size(), options.validation);
    } else {
      validation.reasons.push_back(
        "common-horizon solve did not produce a reconstructed surface");
    }
    catalogue.commonHorizons.push_back({
      search.id, candidate.memberIds, candidate.reason, candidate,
      std::move(search), std::move(validation)});
  }
  return catalogue;
}

AHHorizonCatalogue FindCommonApparentHorizons(
  const GridData& gridData,
  const AHCommonHorizonSearchOptions& options
) {
  return FindApparentHorizons(gridData, options);
}

}  // namespace exahype2::ccz4::ahf
