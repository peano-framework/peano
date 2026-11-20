#include "../require-asserts.h"

#include "../../apparent-horizon-finder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf::tests {
namespace {

AHIndividualHorizonObservation Sphere(
  std::string id, double mass, ReferencePoint centre, double radius
) {
  const SpectralBasis basis = BuildSpectralBasis(2);
  AHFinderResult horizon;
  horizon.solver.converged = true;
  horizon.solver.coefficients = MakeSphericalInitialCoefficients(basis, radius);
  horizon.referencePoint = centre;
  horizon.lMax = 2;
  horizon.minRadius = radius;
  horizon.meanRadius = radius;
  horizon.maxRadius = radius;
  horizon.surfaceAvailable = true;
  return {std::move(id), mass, std::move(horizon)};
}

const AHCommonHorizonCandidate& FindCandidate(
  const AHCommonHorizonCandidatePlan& plan,
  const std::vector<std::string>& members
) {
  for (const auto& candidate : plan.candidates) {
    if (candidate.memberIds == members) return candidate;
  }
  throw std::runtime_error("candidate not found");
}

void TestPhysicalConfigurationConversion() {
  const std::vector<AHPhysicalBlackHole> blackHoles{
    {.id = "A", .mass = 1.0,
     .attempts = {{{0.0, 0.0, 0.0}, 1.0}}},
    {.id = "B", .mass = 2.0,
     .attempts = {{{2.0, 0.0, 0.0}, 0.5}}}
  };
  const auto targets = MakeIndividualHorizonTargets(blackHoles);
  assert(targets.size() == 2);
  assert(targets[0].id == "A" && targets[1].id == "B");
  assert(targets[0].kind == AHHorizonKind::Individual);
  assert(*targets[1].massScale == 2.0);
  // Without physical positions there is nothing to exclude.
  assert(targets[0].excludedPoints.empty());
  assert(targets[1].excludedPoints.empty());

  // Each individual target excludes the other black holes' positions, unless
  // told not to.
  auto placed = blackHoles;
  placed[0].physicalPosition = ReferencePoint{0.0, 0.0, 0.0};
  placed[1].physicalPosition = ReferencePoint{2.0, 0.0, 0.0};
  const auto excluding = MakeIndividualHorizonTargets(placed);
  assert(excluding[0].excludedPoints.size() == 1);
  assert(excluding[0].excludedPoints[0].x == 2.0);
  assert(excluding[1].excludedPoints.size() == 1);
  assert(excluding[1].excludedPoints[0].x == 0.0);
  const auto including = MakeIndividualHorizonTargets(placed, false);
  assert(including[0].excludedPoints.empty());
  assert(including[1].excludedPoints.empty());

  bool threw = false;
  auto invalid = blackHoles;
  invalid[1].id = "A";
  try { (void)MakeIndividualHorizonTargets(invalid); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
}

void TestMarginBeforeContact() {
  const std::vector<AHIndividualHorizonObservation> observations{
    Sphere("A", 1.0, {0.0, 0.0, 0.0}, 1.0),
    Sphere("B", 1.0, {2.2, 0.0, 0.0}, 1.0)
  };
  const auto withoutMargin = PlanCommonHorizonCandidates(
    observations, {.searchMarginPerMass = 0.0});
  assert(withoutMargin.pairGaps.size() == 1);
  assert(std::abs(withoutMargin.pairGaps[0].surfaceGap - 0.2) < 1.0e-12);
  assert(!withoutMargin.pairGaps[0].proximityTriggered);
  assert(withoutMargin.candidates.empty());

  const auto withMargin = PlanCommonHorizonCandidates(
    observations, {.searchMargin = 0.25, .searchMarginPerMass = 0.0});
  assert(withMargin.pairGaps[0].proximityTriggered);
  assert(withMargin.candidates.size() == 1); // pair == component, deduplicated
  assert(withMargin.candidates[0].memberIds ==
         std::vector<std::string>({"A", "B"}));
  assert(withMargin.candidates[0].reason.triggeredPair);
  assert(withMargin.candidates[0].reason.connectedComponent);
}

// By default a pair triggers when its gap is at most 0.5 (m_i + m_j): a
// common horizon forms while the individual ones are still apart.
void TestMarginScalesWithMass() {
  const auto pair = [](double mass) {
    return std::vector<AHIndividualHorizonObservation>{
      Sphere("A", mass, {0.0, 0.0, 0.0}, 1.0),
      Sphere("B", mass, {2.2, 0.0, 0.0}, 1.0)};
  };
  // Gap 0.2 in both cases; the default margin is 0.25 and 0.15.
  const auto heavy = PlanCommonHorizonCandidates(pair(0.25));
  assert(heavy.pairGaps[0].proximityTriggered);
  assert(heavy.candidates.size() == 1);
  const auto light = PlanCommonHorizonCandidates(pair(0.15));
  assert(!light.pairGaps[0].proximityTriggered);
  assert(light.candidates.empty());
  // The absolute margin adds to the per-mass one.
  const auto both =
    PlanCommonHorizonCandidates(pair(0.15), {.searchMargin = 0.06});
  assert(both.pairGaps[0].proximityTriggered);
}

void TestSingleObservationNeverCreatesCommonCandidate() {
  const auto plan = PlanCommonHorizonCandidates(
    {Sphere("single", 1.0, {0.0, 0.0, 0.0}, 1.0)},
    {.searchMargin = 100.0,
     .includeTriggeredPairs = true,
     .includeConnectedComponents = true,
     .unconditionalAllBlackHoles = true});
  assert(plan.pairGaps.empty());
  assert(plan.candidates.empty());
}

void TestPairsComponentsExplicitAndUnconditional() {
  const std::vector<AHIndividualHorizonObservation> observations{
    Sphere("C", 3.0, {3.0, 0.0, 0.0}, 1.0),
    Sphere("A", 1.0, {0.0, 0.0, 0.0}, 1.0),
    Sphere("B", 2.0, {1.5, 0.0, 0.0}, 1.0),
    Sphere("D", 4.0, {10.0, 0.0, 0.0}, 0.5)
  };
  const auto plan = PlanCommonHorizonCandidates(
    observations,
    {.searchMargin = 0.0,
     .searchMarginPerMass = 0.0,
     .includeTriggeredPairs = true,
     .includeConnectedComponents = true,
     .unconditionalAllBlackHoles = true,
     .requestedSubsets = {{"D", "A"}, {"B", "A"}}}
  );

  // Triggered chain A-B-C gives AB, BC and component ABC. Explicit AD and
  // unconditional ABCD are added. Explicit BA merges with triggered AB.
  assert(plan.candidates.size() == 5);
  const auto& ab = FindCandidate(plan, {"A", "B"});
  assert(ab.reason.triggeredPair && ab.reason.explicitlyRequested);
  const auto& abc = FindCandidate(plan, {"A", "B", "C"});
  assert(abc.reason.connectedComponent);
  assert(std::abs(abc.totalMass - 6.0) < 1.0e-14);
  assert(std::abs(abc.referencePoint.x - 2.0) < 1.0e-14);
  assert(std::abs(abc.enclosingInitialRadius - 3.0) < 1.0e-14);
  const auto& all = FindCandidate(plan, {"A", "B", "C", "D"});
  assert(all.reason.unconditionalAllBlackHoles);

  for (std::size_t i = 1; i < plan.candidates.size(); ++i) {
    assert(plan.candidates[i - 1].memberIds < plan.candidates[i].memberIds);
  }
}

void TestDirectionalDistortedSurface() {
  const SpectralBasis basis = BuildSpectralBasis(2);
  AHIndividualHorizonObservation first =
    Sphere("A", 1.0, {0.0, 0.0, 0.0}, 1.0);
  std::vector<double> radii(basis.nPoints);
  for (std::size_t p = 0; p < radii.size(); ++p) {
    radii[p] = 1.0 + 0.2 * std::cos(basis.rho[p]) *
      std::cos(basis.sigma[p]);
  }
  first.horizon.solver.coefficients =
    MakeDistortedInitialCoefficients(basis, radii);
  first.horizon.maxRadius = *std::max_element(radii.begin(), radii.end());
  const auto plan = PlanCommonHorizonCandidates({
    first, Sphere("B", 1.0, {2.1, 0.0, 0.0}, 1.0)
  }, {.searchMarginPerMass = 0.0});
  assert(plan.pairGaps[0].firstFacingRadius > 1.19);
  assert(plan.pairGaps[0].firstFacingRadius < 1.21);
  assert(plan.pairGaps[0].proximityTriggered);
}

double Length(double x, double y, double z) { return std::sqrt(x * x + y * y + z * z); }

// Off-axis, unequal masses: distances, facing radii and the mass-weighted
// centre use all three coordinates. The other tests place the holes on x.
void TestOffAxisUnequalMasses() {
  const ReferencePoint a{0.1, 0.3, -0.2};
  const ReferencePoint b{-0.2, -0.2, 0.3};
  const double dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  const double distance = Length(dx, dy, dz);

  // A bulges towards B: r = 0.3 + 0.05 (n . d/|d|), so its radius facing B is 0.35.
  const SpectralBasis basis = BuildSpectralBasis(2);
  AHIndividualHorizonObservation first = Sphere("A", 1.0, a, 0.3);
  std::vector<double> radii(basis.nPoints);
  for (std::size_t p = 0; p < radii.size(); ++p) {
    const double c = std::cos(basis.sigma[p]);
    const double n[3] = {c * std::cos(basis.rho[p]), c * std::sin(basis.rho[p]),
                         std::sin(basis.sigma[p])};
    radii[p] = 0.3 + 0.05 * (n[0] * dx + n[1] * dy + n[2] * dz) / distance;
  }
  first.horizon.solver.coefficients = MakeDistortedInitialCoefficients(basis, radii);
  first.horizon.maxRadius = 0.35;

  const auto plan = PlanCommonHorizonCandidates(
    {first, Sphere("B", 3.0, b, 0.3)},
    {.searchMargin = 0.2, .searchMarginPerMass = 0.0});
  assert(plan.pairGaps.size() == 1);
  assert(std::abs(plan.pairGaps[0].centreDistance - distance) < 1.0e-14);
  assert(std::abs(plan.pairGaps[0].firstFacingRadius - 0.35) < 1.0e-12);
  assert(std::abs(plan.pairGaps[0].secondFacingRadius - 0.3) < 1.0e-12);
  assert(std::abs(plan.pairGaps[0].surfaceGap - (distance - 0.65)) < 1.0e-12);
  assert(plan.pairGaps[0].proximityTriggered);

  const auto& candidate = FindCandidate(plan, {"A", "B"});
  const ReferencePoint centre{(a.x + 3.0 * b.x) / 4.0, (a.y + 3.0 * b.y) / 4.0,
                              (a.z + 3.0 * b.z) / 4.0};
  assert(std::abs(candidate.totalMass - 4.0) < 1.0e-14);
  assert(std::abs(candidate.referencePoint.x - centre.x) < 1.0e-14);
  assert(std::abs(candidate.referencePoint.y - centre.y) < 1.0e-14);
  assert(std::abs(candidate.referencePoint.z - centre.z) < 1.0e-14);
  const double enclosing = std::max(
    Length(centre.x - a.x, centre.y - a.y, centre.z - a.z) + 0.35,
    Length(centre.x - b.x, centre.y - b.y, centre.z - b.z) + 0.3);
  assert(std::abs(candidate.enclosingInitialRadius - enclosing) < 1.0e-14);
}

void TestInvalidObservationsAndRequests() {
  auto observations = std::vector<AHIndividualHorizonObservation>{
    Sphere("A", 1.0, {0.0, 0.0, 0.0}, 1.0),
    Sphere("B", 1.0, {3.0, 0.0, 0.0}, 1.0)
  };
  observations[0].horizon.solver.converged = false;
  bool threw = false;
  try { (void)PlanCommonHorizonCandidates(observations); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);

  observations[0].horizon.solver.converged = true;
  threw = false;
  try {
    (void)PlanCommonHorizonCandidates(
      observations, {.requestedSubsets = {{"A", "unknown"}}});
  } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);

  for (double margin : {-0.1, std::numeric_limits<double>::quiet_NaN()}) {
    threw = false;
    try {
      (void)PlanCommonHorizonCandidates(
        observations, {.searchMarginPerMass = margin});
    } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }
}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests

int main() {
  using namespace exahype2::ccz4::ahf::tests;
  TestPhysicalConfigurationConversion();
  TestMarginBeforeContact();
  TestMarginScalesWithMass();
  TestSingleObservationNeverCreatesCommonCandidate();
  TestPairsComponentsExplicitAndUnconditional();
  TestDirectionalDistortedSurface();
  TestOffAxisUnequalMasses();
  TestInvalidObservationsAndRequests();
}
