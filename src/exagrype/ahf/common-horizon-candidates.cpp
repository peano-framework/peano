#include "common-horizon-candidates.h"

#include "harmonics.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace exahype2::ccz4::ahf {
namespace {

double Distance(const ReferencePoint& a, const ReferencePoint& b) {
  return std::sqrt(
    (a.x - b.x) * (a.x - b.x) +
    (a.y - b.y) * (a.y - b.y) +
    (a.z - b.z) * (a.z - b.z));
}

double RadiusInDirection(const AHFinderResult& horizon,
                         const ReferencePoint& destination) {
  if (!(Distance(horizon.referencePoint, destination) > 0.0)) {
    throw std::invalid_argument(
      "PlanCommonHorizonCandidates: distinct horizons have coincident reference points");
  }
  const double radius = SurfaceRadiusTowards(horizon, destination);
  if (!std::isfinite(radius) || radius <= 0.0) {
    throw std::invalid_argument(
      "PlanCommonHorizonCandidates: directional radius is invalid");
  }
  return radius;
}

std::vector<std::string> CanonicalMembers(std::vector<std::string> members) {
  std::sort(members.begin(), members.end());
  members.erase(std::unique(members.begin(), members.end()), members.end());
  return members;
}

std::string GroupKey(const std::vector<std::string>& members) {
  std::string key;
  for (const std::string& member : members) {
    key += std::to_string(member.size()) + ":" + member + ";";
  }
  return key;
}

struct DisjointSet {
  explicit DisjointSet(std::size_t size): parent(size) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  std::size_t Find(std::size_t value) {
    if (parent[value] != value) parent[value] = Find(parent[value]);
    return parent[value];
  }
  void Join(std::size_t a, std::size_t b) {
    a = Find(a);
    b = Find(b);
    if (a != b) parent[b] = a;
  }
  std::vector<std::size_t> parent;
};

void MergeReason(AHCommonHorizonCandidateReason& target,
                 const AHCommonHorizonCandidateReason& source) {
  target.triggeredPair = target.triggeredPair || source.triggeredPair;
  target.connectedComponent =
    target.connectedComponent || source.connectedComponent;
  target.explicitlyRequested =
    target.explicitlyRequested || source.explicitlyRequested;
  target.unconditionalAllBlackHoles =
    target.unconditionalAllBlackHoles || source.unconditionalAllBlackHoles;
}

}  // namespace

void ValidateAHPhysicalBlackHoles(
  const std::vector<AHPhysicalBlackHole>& blackHoles
) {
  if (blackHoles.empty()) {
    throw std::invalid_argument(
      "ValidateAHPhysicalBlackHoles: at least one black hole is required");
  }
  std::unordered_set<std::string> ids;
  for (const AHPhysicalBlackHole& blackHole : blackHoles) {
    if (blackHole.id.empty() || !ids.insert(blackHole.id).second) {
      throw std::invalid_argument(
        "ValidateAHPhysicalBlackHoles: IDs must be non-empty and unique");
    }
    if (!std::isfinite(blackHole.mass) || blackHole.mass <= 0.0) {
      throw std::invalid_argument(
        "ValidateAHPhysicalBlackHoles: masses must be finite and positive");
    }
    if (blackHole.physicalPosition.has_value() &&
        (!std::isfinite(blackHole.physicalPosition->x) ||
         !std::isfinite(blackHole.physicalPosition->y) ||
         !std::isfinite(blackHole.physicalPosition->z))) {
      throw std::invalid_argument(
        "ValidateAHPhysicalBlackHoles: physical positions must be finite");
    }
    if (blackHole.attempts.empty()) {
      throw std::invalid_argument(
        "ValidateAHPhysicalBlackHoles: each black hole needs an initial attempt");
    }
  }
}

std::vector<AHHorizonTarget> MakeIndividualHorizonTargets(
  const std::vector<AHPhysicalBlackHole>& blackHoles,
  bool excludeOtherBlackHoles
) {
  ValidateAHPhysicalBlackHoles(blackHoles);
  std::vector<AHHorizonTarget> targets;
  targets.reserve(blackHoles.size());
  for (const AHPhysicalBlackHole& blackHole : blackHoles) {
    std::vector<ReferencePoint> others;
    for (const AHPhysicalBlackHole& other : blackHoles) {
      if (excludeOtherBlackHoles && &other != &blackHole &&
          other.physicalPosition.has_value()) {
        others.push_back(*other.physicalPosition);
      }
    }
    targets.push_back({
      .id = blackHole.id,
      .kind = AHHorizonKind::Individual,
      .attempts = blackHole.attempts,
      .massScale = blackHole.mass,
      .excludedPoints = std::move(others)
    });
  }
  return targets;
}

void ValidateAHCommonHorizonCandidateOptions(
  const AHCommonHorizonCandidateOptions& options
) {
  if (!std::isfinite(options.searchMargin) || options.searchMargin < 0.0) {
    throw std::invalid_argument(
      "ValidateAHCommonHorizonCandidateOptions: searchMargin must be finite and non-negative");
  }
  if (!std::isfinite(options.searchMarginPerMass) ||
      options.searchMarginPerMass < 0.0) {
    throw std::invalid_argument(
      "ValidateAHCommonHorizonCandidateOptions: searchMarginPerMass must be "
      "finite and non-negative");
  }
  for (const auto& subset : options.requestedSubsets) {
    if (CanonicalMembers(subset).size() < 2) {
      throw std::invalid_argument(
        "ValidateAHCommonHorizonCandidateOptions: requested subsets need at least two distinct IDs");
    }
  }
}

AHCommonHorizonCandidatePlan PlanCommonHorizonCandidates(
  const std::vector<AHIndividualHorizonObservation>& observations,
  const AHCommonHorizonCandidateOptions& options
) {
  ValidateAHCommonHorizonCandidateOptions(options);
  if (observations.empty()) {
    throw std::invalid_argument(
      "PlanCommonHorizonCandidates: at least one observation is required");
  }

  std::unordered_map<std::string, std::size_t> indexById;
  for (std::size_t i = 0; i < observations.size(); ++i) {
    const auto& observation = observations[i];
    if (observation.id.empty() || !indexById.emplace(observation.id, i).second) {
      throw std::invalid_argument(
        "PlanCommonHorizonCandidates: observation IDs must be non-empty and unique");
    }
    if (!std::isfinite(observation.mass) || observation.mass <= 0.0) {
      throw std::invalid_argument(
        "PlanCommonHorizonCandidates: masses must be finite and positive");
    }
    if (!observation.horizon.solver.converged ||
        !observation.horizon.surfaceAvailable) {
      throw std::invalid_argument(
        "PlanCommonHorizonCandidates: every observation must be a converged, reconstructed individual horizon");
    }
    if (!std::isfinite(observation.horizon.maxRadius) ||
        observation.horizon.maxRadius <= 0.0) {
      throw std::invalid_argument(
        "PlanCommonHorizonCandidates: maximum radius must be finite and positive");
    }
  }

  AHCommonHorizonCandidatePlan plan;
  DisjointSet components(observations.size());
  std::map<std::string, AHCommonHorizonCandidate> candidates;
  auto addCandidate = [&](std::vector<std::string> members,
                          AHCommonHorizonCandidateReason reason) {
    members = CanonicalMembers(std::move(members));
    if (members.size() < 2) return;
    const std::string key = GroupKey(members);
    auto [iterator, inserted] = candidates.try_emplace(key);
    if (inserted) iterator->second.memberIds = std::move(members);
    MergeReason(iterator->second.reason, reason);
  };

  for (std::size_t i = 0; i < observations.size(); ++i) {
    for (std::size_t j = i + 1; j < observations.size(); ++j) {
      const auto& first = observations[i];
      const auto& second = observations[j];
      AHHorizonPairGap pair;
      pair.firstId = first.id;
      pair.secondId = second.id;
      pair.centreDistance = Distance(
        first.horizon.referencePoint, second.horizon.referencePoint);
      if (!(pair.centreDistance > 0.0)) {
        throw std::invalid_argument(
          "PlanCommonHorizonCandidates: distinct horizons have coincident reference points");
      }
      pair.firstFacingRadius = RadiusInDirection(
        first.horizon, second.horizon.referencePoint);
      pair.secondFacingRadius = RadiusInDirection(
        second.horizon, first.horizon.referencePoint);
      pair.surfaceGap = pair.centreDistance - pair.firstFacingRadius -
        pair.secondFacingRadius;
      pair.proximityTriggered = pair.surfaceGap <= options.searchMargin +
        options.searchMarginPerMass * (first.mass + second.mass);
      plan.pairGaps.push_back(pair);
      if (pair.proximityTriggered) {
        components.Join(i, j);
        if (options.includeTriggeredPairs) {
          addCandidate({first.id, second.id}, {.triggeredPair = true});
        }
      }
    }
  }

  if (options.includeConnectedComponents) {
    std::map<std::size_t, std::vector<std::string>> groups;
    for (std::size_t i = 0; i < observations.size(); ++i) {
      groups[components.Find(i)].push_back(observations[i].id);
    }
    for (auto& [root, members] : groups) {
      (void)root;
      if (members.size() >= 2) {
        addCandidate(std::move(members), {.connectedComponent = true});
      }
    }
  }

  for (const auto& requested : options.requestedSubsets) {
    for (const std::string& id : requested) {
      if (!indexById.contains(id)) {
        throw std::invalid_argument(
          "PlanCommonHorizonCandidates: requested subset contains unknown ID " + id);
      }
    }
    addCandidate(requested, {.explicitlyRequested = true});
  }
  if (options.unconditionalAllBlackHoles && observations.size() >= 2) {
    std::vector<std::string> all;
    for (const auto& observation : observations) all.push_back(observation.id);
    addCandidate(std::move(all), {.unconditionalAllBlackHoles = true});
  }

  for (auto& [key, candidate] : candidates) {
    (void)key;
    for (const std::string& id : candidate.memberIds) {
      const auto& observation = observations[indexById.at(id)];
      candidate.totalMass += observation.mass;
      candidate.referencePoint.x +=
        observation.mass * observation.horizon.referencePoint.x;
      candidate.referencePoint.y +=
        observation.mass * observation.horizon.referencePoint.y;
      candidate.referencePoint.z +=
        observation.mass * observation.horizon.referencePoint.z;
    }
    candidate.referencePoint.x /= candidate.totalMass;
    candidate.referencePoint.y /= candidate.totalMass;
    candidate.referencePoint.z /= candidate.totalMass;
    for (const std::string& id : candidate.memberIds) {
      const auto& observation = observations[indexById.at(id)];
      candidate.enclosingInitialRadius = std::max(
        candidate.enclosingInitialRadius,
        Distance(candidate.referencePoint, observation.horizon.referencePoint) +
          observation.horizon.maxRadius);
    }
    plan.candidates.push_back(std::move(candidate));
  }
  return plan;
}

}  // namespace exahype2::ccz4::ahf
