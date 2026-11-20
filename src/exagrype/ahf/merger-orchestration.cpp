#include "merger-orchestration.h"

#include "harmonics.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace exahype2::ccz4::ahf {
namespace {

double Distance(const ReferencePoint& a, const ReferencePoint& b) {
  return std::sqrt(
    (a.x - b.x) * (a.x - b.x) +
    (a.y - b.y) * (a.y - b.y) +
    (a.z - b.z) * (a.z - b.z));
}

double RadiusInDirection(const AHSearchAttempt& attempt,
                         const SpectralBasis& basis,
                         const ReferencePoint& destination) {
  if (attempt.initialCoefficients.empty()) return attempt.initialRadius;
  const double dx = destination.x - attempt.referencePoint.x;
  const double dy = destination.y - attempt.referencePoint.y;
  const double dz = destination.z - attempt.referencePoint.z;
  const double length = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (!(length > 0.0)) {
    throw std::invalid_argument(
      "BuildBinaryMergerPlan: component reference points must differ");
  }
  double rho = std::atan2(dy, dx);
  if (rho < 0.0) rho += 2.0 * std::numbers::pi;
  const double sigma = std::asin(std::clamp(dz / length, -1.0, 1.0));
  double radius = 0.0;
  const auto modes = EnumerateModes(basis.lmax);
  for (std::size_t j = 0; j < modes.size(); ++j) {
    radius += attempt.initialCoefficients[j] *
      EvalModeValue(modes[j].first, modes[j].second, rho, sigma);
  }
  if (!std::isfinite(radius) || radius <= 0.0) {
    throw std::invalid_argument(
      "BuildBinaryMergerPlan: facing radius is non-finite or non-positive");
  }
  return radius;
}

double MaximumRadius(const AHSearchAttempt& attempt,
                     const SpectralBasis& basis) {
  if (attempt.initialCoefficients.empty()) return attempt.initialRadius;
  const SynthesizedField field = Synthesize(basis, attempt.initialCoefficients);
  return *std::max_element(field.h.begin(), field.h.end());
}

}  // namespace

void ValidateAHBinaryMergerOptions(const AHBinaryMergerOptions& options) {
  if (options.commonHorizonId.empty()) {
    throw std::invalid_argument(
      "ValidateAHBinaryMergerOptions: commonHorizonId must not be empty");
  }
  std::unordered_set<std::string> ids{options.commonHorizonId};
  for (const AHBinaryComponent& component : options.components) {
    if (component.target.kind != AHHorizonKind::Individual) {
      throw std::invalid_argument(
        "ValidateAHBinaryMergerOptions: components must be individual horizons");
    }
    if (!std::isfinite(component.mass) || component.mass <= 0.0) {
      throw std::invalid_argument(
        "ValidateAHBinaryMergerOptions: component masses must be finite and positive");
    }
    if (!ids.insert(component.target.id).second) {
      throw std::invalid_argument(
        "ValidateAHBinaryMergerOptions: horizon IDs must be unique");
    }
  }
  if (options.commonRadiusFactors.empty()) {
    throw std::invalid_argument(
      "ValidateAHBinaryMergerOptions: commonRadiusFactors must not be empty");
  }
  if (!std::isfinite(options.searchMarginPerMass) ||
      options.searchMarginPerMass < 0.0) {
    throw std::invalid_argument(
      "ValidateAHBinaryMergerOptions: searchMarginPerMass must be finite and "
      "non-negative");
  }
  for (double factor : options.commonRadiusFactors) {
    if (!std::isfinite(factor) || factor <= 0.0) {
      throw std::invalid_argument(
        "ValidateAHBinaryMergerOptions: common radius factors must be finite and positive");
    }
  }
  ValidateAHMultiHorizonOptions({
    .targets = {options.components[0].target, options.components[1].target},
    .lMax = options.lMax,
    .interpolation = options.interpolation,
    .solver = options.solver
  });
}

AHBinaryMergerPlan BuildBinaryMergerPlan(
  const AHBinaryMergerOptions& options
) {
  ValidateAHBinaryMergerOptions(options);
  const SpectralBasis basis = BuildSpectralBasis(options.lMax);
  const AHSearchAttempt& first = options.components[0].target.attempts.front();
  const AHSearchAttempt& second = options.components[1].target.attempts.front();

  AHBinaryMergerPlan plan;
  plan.separation = Distance(first.referencePoint, second.referencePoint);
  if (!(plan.separation > 0.0)) {
    throw std::invalid_argument(
      "BuildBinaryMergerPlan: component reference points must differ");
  }
  plan.facingRadii = {
    RadiusInDirection(first, basis, second.referencePoint),
    RadiusInDirection(second, basis, first.referencePoint)
  };
  plan.surfaceGap =
    plan.separation - plan.facingRadii[0] - plan.facingRadii[1];
  plan.commonMass = options.components[0].mass + options.components[1].mass;
  plan.commonSearchEnabled =
    plan.surfaceGap <= options.searchMarginPerMass * plan.commonMass;
  plan.commonReferencePoint = {
    (options.components[0].mass * first.referencePoint.x +
     options.components[1].mass * second.referencePoint.x) / plan.commonMass,
    (options.components[0].mass * first.referencePoint.y +
     options.components[1].mass * second.referencePoint.y) / plan.commonMass,
    (options.components[0].mass * first.referencePoint.z +
     options.components[1].mass * second.referencePoint.z) / plan.commonMass
  };
  plan.enclosingInitialRadius = std::max(
    Distance(plan.commonReferencePoint, first.referencePoint) +
      MaximumRadius(first, basis),
    Distance(plan.commonReferencePoint, second.referencePoint) +
      MaximumRadius(second, basis));

  plan.search = {
    .targets = {},
    .lMax = options.lMax,
    .interpolation = options.interpolation,
    .solver = options.solver
  };
  if (plan.commonSearchEnabled) {
    AHHorizonTarget common{
      .id = options.commonHorizonId,
      .kind = AHHorizonKind::Common,
      .attempts = {},
      .massScale = plan.commonMass
    };
    for (double factor : options.commonRadiusFactors) {
      common.attempts.push_back({
        .referencePoint = plan.commonReferencePoint,
        .initialRadius = factor * plan.enclosingInitialRadius
      });
    }
    plan.search.targets.push_back(std::move(common));
  }
  for (std::size_t i = 0; i < options.components.size(); ++i) {
    AHHorizonTarget target = options.components[i].target;
    target.massScale = options.components[i].mass;
    // Unless told otherwise, a component must not enclose the other one.
    if (target.excludedPoints.empty()) {
      target.excludedPoints.push_back(
        options.components[1 - i].target.attempts.front().referencePoint);
    }
    plan.search.targets.push_back(std::move(target));
  }
  return plan;
}

std::optional<AHCommonHorizonValidation> ValidateBinaryMergerCommonHorizon(
  const AHBinaryMergerOptions& options,
  const AHBinaryMergerPlan& plan,
  const AHMultiHorizonResult& horizons
) {
  if (!plan.commonSearchEnabled) return std::nullopt;
  const AHHorizonTargetResult* common = nullptr;
  std::vector<AHIndividualHorizonObservation> members;
  std::vector<ReferencePoint> physicalPositions;
  for (const AHBinaryComponent& component : options.components) {
    physicalPositions.push_back(
      component.target.attempts.front().referencePoint);
  }
  for (const AHHorizonTargetResult& target : horizons.targets) {
    if (target.id == options.commonHorizonId) {
      common = &target;
      continue;
    }
    for (const AHBinaryComponent& component : options.components) {
      if (target.id == component.target.id && target.converged &&
          target.horizon.has_value()) {
        members.push_back({target.id, component.mass, *target.horizon});
      }
    }
  }
  AHCommonHorizonValidation validation;
  if (common == nullptr || !common->horizon.has_value()) {
    validation.reasons.push_back(
      "common-horizon solve did not produce a reconstructed surface");
    return validation;
  }
  return ValidateCommonHorizon(
    *common->horizon, members, physicalPositions, 2, options.validation);
}

AHBinaryMergerPlan CheckCommonHorizonOfBBH(
  const AHBinaryMergerOptions& options
) {
  return BuildBinaryMergerPlan(options);
}

AHBinaryMergerResult FindBinaryMergerHorizons(
  const GridData& gridData,
  const AHBinaryMergerOptions& options
) {
  AHBinaryMergerResult result;
  result.plan = BuildBinaryMergerPlan(options);
  result.horizons = FindApparentHorizons(gridData, result.plan.search);
  result.commonValidation =
    ValidateBinaryMergerCommonHorizon(options, result.plan, result.horizons);
  return result;
}

}  // namespace exahype2::ccz4::ahf
