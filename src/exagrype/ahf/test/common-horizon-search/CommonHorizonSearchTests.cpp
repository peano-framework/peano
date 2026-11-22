#include "../require-asserts.h"

#include "../../apparent-horizon-finder.h"
#include "../brill-lindquist-grid.h"
#include "../temporary-directory.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace exahype2::ccz4::ahf::tests {
namespace {

AHFinderResult Sphere(ReferencePoint centre, double radius) {
  const SpectralBasis basis = BuildSpectralBasis(2);
  AHFinderResult result;
  result.solver.converged = true;
  result.solver.coefficients = MakeSphericalInitialCoefficients(basis, radius);
  result.referencePoint = centre;
  result.lMax = 2;
  result.minRadius = result.meanRadius = result.maxRadius = radius;
  result.spectralResidualInfinityNorm = 1.0e-12;
  result.maxAbsExpansion = 2.0e-12;
  result.maxAbsExpansionOverResolved = 3.0e-12;
  result.relativeExpansionError = 3.0e-12 * radius / 2.0;
  result.surfaceAvailable = true;
  return result;
}

AHIndividualHorizonObservation Observation(
  std::string id, double mass, ReferencePoint centre, double radius) {
  return {std::move(id), mass, Sphere(centre, radius)};
}

GridData MakeGrid() {
  GridData grid;
  for (int ix = -2; ix <= 2; ++ix) for (int iy = -2; iy <= 2; ++iy)
    for (int iz = -2; iz <= 2; ++iz) {
      grid.coordinates.push_back({double(ix), double(iy), double(iz)});
      const std::array<double, NumberOfCompactFields> row{
        1,0,0,1,0,1, 1,0,0,1,0,1, 3,1};
      grid.values.insert(grid.values.end(), row.begin(), row.end());
    }
  return grid;
}

CartesianToSurfaceOptions Interpolation() {
  return {.method = CartesianToSurfaceMethod::UnweightedPolynomial,
          .numberOfNeighbours = 64, .polynomialDegree = 1,
          .allowOrderReduction = false};
}

void TestCandidateConversion() {
  AHCommonHorizonCandidate candidate;
  candidate.memberIds = {"C", "A", "B"};
  candidate.reason = {.triggeredPair = true, .explicitlyRequested = true};
  candidate.referencePoint = {1, 2, 3};
  candidate.totalMass = 6;
  candidate.enclosingInitialRadius = 4;
  const auto target = MakeCommonHorizonTarget(
    candidate, {.initialRadiusInflation = 1.25,
                .retryRadiusMultipliers = {1, 1.2, 2}});
  assert(target.id == "common[A+B+C]");
  assert(target.kind == AHHorizonKind::Common);
  assert(*target.massScale == 6);
  assert(target.attempts.size() == 3);
  assert(target.attempts[0].initialRadius == 5);
  assert(target.attempts[1].initialRadius == 6);
  assert(target.attempts[2].initialRadius == 10);
  for (const auto& attempt : target.attempts) {
    assert(attempt.referencePoint.x == 1 && attempt.referencePoint.y == 2);
  }
}

void TestValidationAndEnclosure() {
  const auto a = Observation("A", 1, {-0.5, 0, 0}, 0.4);
  const auto b = Observation("B", 1, { 0.5, 0, 0}, 0.4);
  AHFinderResult common = Sphere({0, 0, 0}, 1.1);
  auto validation = ValidateCommonHorizon(common, {a, b});
  assert(validation.valid);
  assert(validation.numericallyValid);
  assert(validation.status ==
         AHCommonHorizonValidationStatus::FullyValidated);
  assert(validation.memberHorizonEnclosure ==
         AHEnclosureValidationStatus::Passed);
  assert(validation.enclosesMembers);
  assert(validation.minimumEnclosureClearance > 0.19);

  common = Sphere({0, 0, 0}, 0.7);
  validation = ValidateCommonHorizon(common, {a, b});
  assert(!validation.valid && !validation.enclosesMembers);
  assert(validation.status == AHCommonHorizonValidationStatus::Rejected);

  // The absolute bound on Theta at the collocation points is off unless set;
  // grid data reaches 1e-2 on correct horizons.
  common = Sphere({0, 0, 0}, 1.1);
  common.maxAbsExpansion = 1.0e-2;
  validation = ValidateCommonHorizon(common, {a, b});
  assert(validation.valid && validation.pointwiseExpansionValid);
  validation = ValidateCommonHorizon(
    common, {a, b}, {.pointwiseExpansionTolerance = 1.0e-3});
  assert(!validation.valid && !validation.pointwiseExpansionValid);

  // An aliased surface: Theta of order that of a sphere between the points.
  common = Sphere({0, 0, 0}, 1.1);
  common.relativeExpansionError = 0.55;
  validation = ValidateCommonHorizon(common, {a, b});
  assert(!validation.valid && !validation.relativeExpansionValid);
  assert(std::find(validation.reasons.begin(), validation.reasons.end(),
                   "relative expansion error exceeds tolerance") !=
         validation.reasons.end());
  common.relativeExpansionError = 0.25;
  assert(ValidateCommonHorizon(common, {a, b}).valid);
  common.relativeExpansionError = std::numeric_limits<double>::quiet_NaN();
  assert(!ValidateCommonHorizon(common, {a, b}).relativeExpansionValid);

  // The absolute bound may be infinite (off); neither bound may be NaN, and
  // the relative one must be positive.
  common = Sphere({0, 0, 0}, 1.1);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const AHCommonHorizonValidationOptions& invalid :
       {AHCommonHorizonValidationOptions{.pointwiseExpansionTolerance = nan},
        AHCommonHorizonValidationOptions{.relativeExpansionTolerance = 0.0},
        AHCommonHorizonValidationOptions{.relativeExpansionTolerance = nan}}) {
    bool threw = false;
    try { (void)ValidateCommonHorizon(common, {a, b}, invalid); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }

  common = Sphere({0, 0, 0}, 1.1);
  common.solver.coefficients[0] = std::numeric_limits<double>::infinity();
  validation = ValidateCommonHorizon(common, {a, b});
  assert(!validation.valid && !validation.coefficientsFinite);
}

void TestProvisionalAndPhysicalEnclosure() {
  const auto a = Observation("A", 1, {-0.5, 0, 0}, 0.4);
  AHFinderResult common = Sphere({0, 0, 0}, 1.1);
  auto validation = ValidateCommonHorizon(
    common, {a}, {{-0.5, 0, 0}, {0.5, 0, 0}}, 2);
  assert(validation.numericallyValid);
  assert(!validation.valid);
  assert(validation.status ==
         AHCommonHorizonValidationStatus::ProvisionallyValidated);
  assert(validation.memberHorizonEnclosure ==
         AHEnclosureValidationStatus::Unavailable);
  assert(validation.physicalPositionEnclosure ==
         AHEnclosureValidationStatus::Passed);

  validation = ValidateCommonHorizon(
    common, {a}, {{-0.5, 0, 0}, {2.0, 0, 0}}, 2);
  assert(validation.numericallyValid);
  assert(validation.status == AHCommonHorizonValidationStatus::Rejected);
  assert(validation.physicalPositionEnclosure ==
         AHEnclosureValidationStatus::Failed);
}

// Enclosure off the x axis: member surfaces and physical positions are
// measured in all three coordinates, and in the direction the common surface
// actually has there.
void TestOffAxisEnclosure() {
  // Members on +y and -z; the one reaching 0.95 from the origin sets the
  // minimum clearance, the other reaches 0.9. Each in turn is the limiting one.
  const auto farY = Observation("A", 1, {0, 0.65, 0}, 0.3);
  const auto nearZ = Observation("B", 1, {0, 0, -0.6}, 0.3);
  const auto nearY = Observation("A", 1, {0, 0.6, 0}, 0.3);
  const auto farZ = Observation("B", 1, {0, 0, -0.65}, 0.3);
  auto validation = ValidateCommonHorizon(Sphere({0, 0, 0}, 1.0), {farY, nearZ});
  assert(validation.memberHorizonEnclosure == AHEnclosureValidationStatus::Passed);
  assert(std::abs(validation.minimumEnclosureClearance - 0.05) < 1.0e-12);
  validation = ValidateCommonHorizon(Sphere({0, 0, 0}, 1.0), {nearY, farZ});
  assert(validation.memberHorizonEnclosure == AHEnclosureValidationStatus::Passed);
  assert(std::abs(validation.minimumEnclosureClearance - 0.05) < 1.0e-12);
  validation = ValidateCommonHorizon(Sphere({0, 0, 0}, 0.9), {farY, nearZ});
  assert(validation.memberHorizonEnclosure == AHEnclosureValidationStatus::Failed);
  assert(std::abs(validation.minimumEnclosureClearance + 0.05) < 1.0e-12);

  // A common surface stretched along y, r = 0.9 + 0.1 n_y: 1.0 towards +y,
  // 0.9 towards -z.
  const SpectralBasis basis = BuildSpectralBasis(2);
  std::vector<double> radii(basis.nPoints);
  for (std::size_t p = 0; p < radii.size(); ++p) {
    radii[p] = 0.9 + 0.1 * std::cos(basis.sigma[p]) * std::sin(basis.rho[p]);
  }
  AHFinderResult stretched = Sphere({0, 0, 0}, 0.9);
  stretched.solver.coefficients = MakeDistortedInitialCoefficients(basis, radii);
  stretched.minRadius = 0.8;
  stretched.maxRadius = 1.0;
  // Each check pairs the position under test with one that has more room
  // (clearance 0.4 or 0.5), so the minimum clearance is the tested one's.
  validation = ValidateCommonHorizon(stretched, {}, {{0, 0.95, 0}, {0, 0, -0.5}}, 2);
  assert(validation.physicalPositionEnclosure == AHEnclosureValidationStatus::Passed);
  assert(std::abs(validation.minimumPhysicalPositionClearance - 0.05) < 1.0e-12);
  validation = ValidateCommonHorizon(stretched, {}, {{0, 0, -0.8}, {0, 0.5, 0}}, 2);
  assert(validation.physicalPositionEnclosure == AHEnclosureValidationStatus::Passed);
  assert(std::abs(validation.minimumPhysicalPositionClearance - 0.1) < 1.0e-12);
  validation = ValidateCommonHorizon(stretched, {}, {{0, -0.85, 0}, {0, 0, -0.5}}, 2);
  assert(validation.physicalPositionEnclosure == AHEnclosureValidationStatus::Failed);
  assert(std::abs(validation.minimumPhysicalPositionClearance + 0.05) < 1.0e-12);
}

// A fallback candidate built from physical positions off the x axis, with
// unequal masses, is centred at the mass-weighted position.
void TestOffAxisFallbackCandidate() {
  const ReferencePoint failed{-0.1, 0.2, -0.15};
  const ReferencePoint found{0.1, -0.05, 0.1};
  AHCommonHorizonSearchOptions options;
  options.blackHoles = {
    {.id = "failed", .mass = 1, .attempts = {{{100, 0, 0}, 1.0}}, .physicalPosition = failed},
    {.id = "found", .mass = 2, .attempts = {{found, 1.3}}, .physicalPosition = found}
  };
  options.candidates.requestedSubsets = {{"found", "failed"}};
  // On this grid every sphere of radius 2 is a horizon, so the individual
  // horizon encloses the other physical position by construction.
  options.individualsExcludeOtherBlackHoles = false;
  options.commonTargets.retryRadiusMultipliers = {1};
  options.lMax = 0;
  options.interpolation = Interpolation();
  const auto result = FindCommonApparentHorizons(MakeGrid(), options);
  assert(result.candidatePlan.candidates.size() == 1);
  const auto& candidate = result.candidatePlan.candidates[0];
  const ReferencePoint centre{(failed.x + 2 * found.x) / 3, (failed.y + 2 * found.y) / 3,
                              (failed.z + 2 * found.z) / 3};
  assert(std::abs(candidate.referencePoint.x - centre.x) < 1.0e-14);
  assert(std::abs(candidate.referencePoint.y - centre.y) < 1.0e-14);
  assert(std::abs(candidate.referencePoint.z - centre.z) < 1.0e-14);
  const double enclosing = std::max(
    std::hypot(centre.x - failed.x, centre.y - failed.y, centre.z - failed.z) + 1.0,
    std::hypot(centre.x - found.x, centre.y - found.y, centre.z - found.z) + 1.3);
  assert(std::abs(candidate.enclosingInitialRadius - enclosing) < 1.0e-14);
}

AHCommonHorizonSearchOptions OneHoleOptions(ReferencePoint referencePoint) {
  AHCommonHorizonSearchOptions options;
  options.blackHoles = {{
    .id = "single", .mass = 1.0,
    .attempts = {{referencePoint, 1.3}},
    .physicalPosition = ReferencePoint{0, 0, 0}
  }};
  options.lMax = 0;
  options.interpolation = Interpolation();
  // N=1 must remain individual-only even when this global option is enabled.
  options.candidates.unconditionalAllBlackHoles = true;
  return options;
}

void TestZeroAndOneBlackHole() {
  AHCommonHorizonSearchOptions empty;
  bool threw = false;
  try { (void)FindApparentHorizons(MakeGrid(), empty); }
  catch (const std::invalid_argument& error) {
    threw = std::string(error.what()).find("at least one black hole") !=
      std::string::npos;
  }
  assert(threw);

  const auto success = FindApparentHorizons(
    MakeGrid(), OneHoleOptions({0, 0, 0}));
  assert(success.individuals.targets.size() == 1);
  assert(success.individuals.targets[0].converged);
  assert(success.candidatePlan.pairGaps.empty());
  assert(success.candidatePlan.candidates.empty());
  assert(success.commonHorizons.empty());

  const auto failure = FindApparentHorizons(
    MakeGrid(), OneHoleOptions({100, 0, 0}));
  assert(failure.individuals.targets.size() == 1);
  assert(!failure.individuals.targets[0].converged);
  assert(!failure.individuals.targets[0].attempts.empty());
  assert(failure.candidatePlan.candidates.empty());
  assert(failure.commonHorizons.empty());

  const TemporaryDirectory temporary("ahf-common-horizon-search-tests");
  const auto& directory = temporary.path();
  const std::string successPrefix =
    (directory / "ahf-single-success").string();
  const std::string successSummary = successPrefix + "-summary.txt";
  const auto writtenSuccess = WriteAHToFile(
    MakeGrid(), OneHoleOptions({0, 0, 0}),
    {.summaryFilename = successSummary,
     .horizonFilenamePrefix = successPrefix,
     .visualization = {8, 5}});
  assert(writtenSuccess.individuals.targets.size() == 1);
  assert(writtenSuccess.commonHorizons.empty());
  assert(std::filesystem::exists(successSummary));
  assert(std::filesystem::exists(
    successPrefix + "-individual-0-single.txt"));
  assert(std::filesystem::exists(
    successPrefix + "-individual-0-single.vtp"));

  const std::string failurePrefix =
    (directory / "ahf-single-failure").string();
  const std::string failureSummary = failurePrefix + "-summary.txt";
  const auto writtenFailure = WriteAHToFile(
    MakeGrid(), OneHoleOptions({100, 0, 0}),
    {.summaryFilename = failureSummary,
     .horizonFilenamePrefix = failurePrefix,
     .visualization = {8, 5}});
  assert(writtenFailure.individuals.targets.size() == 1);
  assert(!writtenFailure.individuals.targets[0].converged);
  assert(writtenFailure.commonHorizons.empty());
  assert(std::filesystem::exists(failureSummary));
  assert(std::filesystem::exists(
    failurePrefix + "-individual-0-single.txt"));
  assert(!std::filesystem::exists(
    failurePrefix + "-individual-0-single.vtp"));

  for (const auto& path : {
         successSummary,
         successPrefix + "-individual-0-single.txt",
         successPrefix + "-individual-0-single.vtp",
         failureSummary,
         failurePrefix + "-individual-0-single.txt"}) {
    std::filesystem::remove(path);
  }
}

void TestExplicitFallbackAfterIndividualFailure() {
  AHCommonHorizonSearchOptions options;
  options.blackHoles = {
    {.id = "failed", .mass = 1,
     .attempts = {{{100, 0, 0}, 1.0}},
     .physicalPosition = ReferencePoint{-0.1, 0, 0}},
    {.id = "found", .mass = 2,
     .attempts = {{{0.1, 0, 0}, 1.3}},
     .physicalPosition = ReferencePoint{0.1, 0, 0}}
  };
  options.candidates.requestedSubsets = {{"found", "failed"}};
  // On this grid every sphere of radius 2 is a horizon, so the individual
  // horizon encloses the other physical position by construction.
  options.individualsExcludeOtherBlackHoles = false;
  options.commonTargets.retryRadiusMultipliers = {1};
  options.lMax = 0;
  options.interpolation = Interpolation();
  options.validation.spectralResidualTolerance = 1.0;
  options.validation.pointwiseExpansionTolerance = 1.0;
  options.validation.enclosureTolerance = 0.5;
  // By default an individual surface that encloses another black hole's
  // physical position is rejected.
  auto excluding = options;
  excluding.individualsExcludeOtherBlackHoles =
    AHCommonHorizonSearchOptions{}.individualsExcludeOtherBlackHoles;
  const auto rejected = FindCommonApparentHorizons(MakeGrid(), excluding);
  const auto& found = rejected.individuals.targets[1];
  assert(!found.converged);
  assert(found.attempts[0].result.solver.converged);
  assert(found.attempts[0].rejection.find("encloses") != std::string::npos);

  const auto result = FindCommonApparentHorizons(MakeGrid(), options);
  assert(result.individuals.targets.size() == 2);
  assert(!result.individuals.targets[0].converged);
  assert(result.individuals.targets[1].converged);
  assert(result.commonHorizons.size() == 1);
  assert(result.commonHorizons[0].id == "common[failed+found]");
  assert(result.commonHorizons[0].reason.explicitlyRequested);
  assert(result.commonHorizons[0].search.converged);
  assert(result.commonHorizons[0].validation.numericallyValid);
  assert(result.commonHorizons[0].validation.status ==
         AHCommonHorizonValidationStatus::ProvisionallyValidated);
  assert(!result.commonHorizons[0].validation.valid);
  assert(result.commonHorizons[0].validation.physicalPositionEnclosure ==
         AHEnclosureValidationStatus::Passed);
  assert(!result.commonHorizons[0].search.attempts.empty());
  assert(std::abs(result.candidatePlan.candidates[0].referencePoint.x -
                  (0.1 / 3.0)) < 1.0e-14);

  const TemporaryDirectory temporary("ahf-common-horizon-search-tests");
  const auto& directory = temporary.path();
  const std::string prefix = (directory / "ahf-provisional").string();
  const std::string summary = prefix + "-summary.txt";
  (void)WriteAHToFile(
    MakeGrid(), options,
    {.summaryFilename = summary, .horizonFilenamePrefix = prefix,
     .visualization = {8, 5}});
  const std::string commonBase =
    prefix + "-common-0-common_failed_found_";
  assert(AHCatalogueFilename({.horizonFilenamePrefix = prefix}, "common", 0,
                             result.commonHorizons[0].id, ".txt") ==
         commonBase + ".txt");
  assert(std::filesystem::exists(commonBase + ".txt"));
  assert(!std::filesystem::exists(commonBase + ".vtp"));
  std::ifstream summaryInput(summary);
  std::ostringstream summaryText;
  summaryText << summaryInput.rdbuf();
  assert(summaryText.str().find(
    "validation.status = provisionally-validated") != std::string::npos);
  std::filesystem::remove(summary);
  std::filesystem::remove(commonBase + ".txt");
  std::filesystem::remove(prefix + "-individual-0-failed.txt");
  std::filesystem::remove(prefix + "-individual-1-found.txt");
  std::filesystem::remove(prefix + "-individual-1-found.vtp");
}

void TestFallbackRequiresPhysicalPositions() {
  AHCommonHorizonSearchOptions options;
  options.blackHoles = {
    {.id = "failed", .mass = 1,
     .attempts = {{{100, 0, 0}, 1.0}}},
    {.id = "found", .mass = 1,
     .attempts = {{{0.1, 0, 0}, 1.3}},
     .physicalPosition = ReferencePoint{0.1, 0, 0}}
  };
  options.candidates.requestedSubsets = {{"failed", "found"}};
  options.lMax = 0;
  options.interpolation = Interpolation();
  bool threw = false;
  try { (void)FindCommonApparentHorizons(MakeGrid(), options); }
  catch (const std::invalid_argument& error) {
    threw = std::string(error.what()).find("physicalPosition is unavailable") !=
      std::string::npos;
  }
  assert(threw);
}

void TestTwoIndividualsProduceOneCommonSearch() {
  AHCommonHorizonSearchOptions options;
  options.blackHoles = {
    {.id = "left", .mass = 1,
     .attempts = {{{-0.1, 0, 0}, 1.3}}},
    {.id = "right", .mass = 1,
     .attempts = {{{0.1, 0, 0}, 1.3}}}
  };
  options.candidates.searchMargin = 3.0;
  options.commonTargets.retryRadiusMultipliers = {1};
  options.lMax = 0;
  options.interpolation = Interpolation();
  options.validation.spectralResidualTolerance = 1.0;
  options.validation.pointwiseExpansionTolerance = 1.0;
  options.validation.enclosureTolerance = 0.5;
  const auto result = FindCommonApparentHorizons(MakeGrid(), options);
  assert(result.individuals.allTargetsFound);
  assert(result.commonHorizons.size() == 1);
  assert(result.commonHorizons[0].id == "common[left+right]");
  assert(result.commonHorizons[0].search.converged);
  assert(result.commonHorizons[0].validation.valid);

  const TemporaryDirectory temporary("ahf-common-horizon-search-tests");
  const auto& directory = temporary.path();
  const std::string prefix = (directory / "ahf-catalogue-horizon").string();
  const std::string summary = (directory / "ahf-catalogue-summary.txt").string();
  const auto written = WriteAHToFile(
    MakeGrid(), options,
    {.summaryFilename = summary, .horizonFilenamePrefix = prefix,
     .visualization = {8, 5}});
  assert(written.commonHorizons[0].validation.valid);
  assert(std::filesystem::exists(summary));
  assert(std::filesystem::exists(prefix + "-individual-0-left.txt"));
  assert(std::filesystem::exists(prefix + "-individual-1-right.vtp"));
  assert(std::filesystem::exists(prefix + "-common-0-common_left_right_.txt"));
  assert(std::filesystem::exists(prefix + "-common-0-common_left_right_.vtp"));
  std::filesystem::remove(summary);
  for (const auto& suffix : {
         "-individual-0-left.txt", "-individual-0-left.vtp",
         "-individual-1-right.txt", "-individual-1-right.vtp",
         "-common-0-common_left_right_.txt", "-common-0-common_left_right_.vtp"}) {
    std::filesystem::remove(prefix + suffix);
  }

  auto rejectedOptions = options;
  rejectedOptions.blackHoles[0].physicalPosition = ReferencePoint{-100, 0, 0};
  rejectedOptions.blackHoles[1].physicalPosition = ReferencePoint{100, 0, 0};
  const std::string rejectedPrefix =
    (directory / "ahf-rejected-horizon").string();
  const std::string rejectedSummary =
    (directory / "ahf-rejected-summary.txt").string();
  const auto rejected = WriteAHToFile(
    MakeGrid(), rejectedOptions,
    {.summaryFilename = rejectedSummary,
     .horizonFilenamePrefix = rejectedPrefix,
     .visualization = {8, 5}});
  assert(rejected.commonHorizons[0].validation.status ==
         AHCommonHorizonValidationStatus::Rejected);
  const std::string rejectedCommon =
    rejectedPrefix + "-common-0-common_left_right_";
  assert(std::filesystem::exists(rejectedCommon + ".txt"));
  assert(!std::filesystem::exists(rejectedCommon + ".vtp"));
  std::filesystem::remove(rejectedSummary);
  for (const auto& suffix : {
         "-individual-0-left.txt", "-individual-0-left.vtp",
         "-individual-1-right.txt", "-individual-1-right.vtp",
         "-common-0-common_left_right_.txt"}) {
    std::filesystem::remove(rejectedPrefix + suffix);
  }
}

void TestThreeHoleDeterminismAndSummary() {
  AHHorizonCatalogue catalogue;
  catalogue.individuals.targets = {
    {.id = "A", .kind = AHHorizonKind::Individual, .converged = false},
    {.id = "B", .kind = AHHorizonKind::Individual, .converged = true}
  };
  catalogue.individuals.targets[0].attempts = {
    {.rejection = "individual rejection"}};
  AHCommonHorizonCatalogueEntry ab;
  ab.id = "common[A+B]";
  ab.memberIds = {"A", "B"};
  ab.reason.triggeredPair = true;
  ab.validation.reasons = {"Newton did not converge"};
  ab.search.attempts = {{.rejection = "common rejection"}};
  AHCommonHorizonCatalogueEntry abc;
  abc.id = "common[A+B+C]";
  abc.memberIds = {"A", "B", "C"};
  abc.reason.connectedComponent = true;
  catalogue.commonHorizons = {ab, abc};
  const TemporaryDirectory temporary("ahf-common-horizon-search-tests");
  const auto path = temporary.path() /
    "ahf-common-catalogue-test.txt";
  WriteAHHorizonCatalogueSummary(catalogue, path.string());
  std::ifstream input(path);
  std::ostringstream text;
  text << input.rdbuf();
  assert(text.str().find("common[0].id = \"common[A+B]\"") != std::string::npos);
  assert(text.str().find("Newton did not converge") != std::string::npos);
  assert(text.str().find("individuals[0].attempts[0].rejection = "
                         "\"individual rejection\"") != std::string::npos);
  assert(text.str().find("common[0].attempts[0].rejection = "
                         "\"common rejection\"") != std::string::npos);
  assert(text.str().find("common[0].validation.relativeExpansionValid = "
                         "false") != std::string::npos);
  std::filesystem::remove(path);
}

// Two punctures on the x axis at +/- separation / 2, sampled as ExaGRyPE
// would (TwoPunctures without momentum or spin is Brill-Lindquist), and the
// catalogue options for them: individual attempts from 0.35 m about each
// puncture.
AHCommonHorizonSearchOptions BinaryOptions(
  double mPlus, double mMinus, double separation, int lMax) {
  const ReferencePoint plus{separation / 2.0, 0.0, 0.0};
  const ReferencePoint minus{-separation / 2.0, 0.0, 0.0};
  AHCommonHorizonSearchOptions options;
  options.blackHoles = {
    {.id = "plus", .mass = mPlus, .attempts = {{plus, 0.35 * mPlus}},
     .physicalPosition = plus},
    {.id = "minus", .mass = mMinus, .attempts = {{minus, 0.35 * mMinus}},
     .physicalPosition = minus}};
  options.lMax = lMax;
  options.interpolation = {
    .method = CartesianToSurfaceMethod::RadiusWeightedPolynomial,
    .polynomialDegree = 3,
    .supportRadius = 0.15};
  return options;
}

// Equal masses 1/2 at separation 1/2: with default options the common search
// is planned (the individual horizons are 0.205 apart, within 0.5 times the
// total mass), and the validation accepts the common horizon found on grid
// data, whose irreducible mass matches the analytic Brill-Lindquist value,
// 0.998248 (lMax 16).
void TestBrillLindquistCommonHorizonValidates() {
  const BrillLindquist binary{
    {{0.5, {0.25, 0.0, 0.0}}, {0.5, {-0.25, 0.0, 0.0}}}};
  const GridData grid = MakeBrillLindquistGrid(binary, 0.9, 36);
  const auto catalogue =
    FindApparentHorizons(grid, BinaryOptions(0.5, 0.5, 0.5, 4));
  assert(catalogue.individuals.allTargetsFound);
  assert(catalogue.commonHorizons.size() == 1);
  const auto& common = catalogue.commonHorizons[0];
  assert(common.reason.triggeredPair);
  assert(!common.reason.unconditionalAllBlackHoles);
  assert(common.search.converged && common.search.horizon.has_value());
  assert(common.validation.valid);
  assert(common.validation.relativeExpansionValid);
  assert(common.search.horizon->relativeExpansionError < 0.05);
  assert(std::abs(common.search.horizon->irreducibleMass / 0.998248 - 1.0) <
         2.0e-4);
}

// Masses 0.6 and 0.3 at separation 0.4. Started from a sphere of radius 0.36
// about the heavier puncture, Newton converges at the collocation points to
// a surface with Mirr above the ADM mass of 0.9, which is not a horizon. The
// search rejects it on its expansion between the points, and the next
// attempt finds the individual horizon (Mirr 0.824914 analytically).
void TestAliasedSurfaceIsRejected() {
  const BrillLindquist binary{
    {{0.6, {0.2, 0.0, 0.0}}, {0.3, {-0.2, 0.0, 0.0}}}};
  const GridData grid = MakeBrillLindquistGrid(binary, 0.9, 36);
  const AHCommonHorizonSearchOptions binaryOptions =
    BinaryOptions(0.6, 0.3, 0.4, 4);
  AHSearchOptions options;
  options.attempts = {{{0.2, 0.0, 0.0}, 0.36}, {{0.2, 0.0, 0.0}, 0.21}};
  options.lMax = binaryOptions.lMax;
  options.interpolation = binaryOptions.interpolation;
  const AHSearchResult search = FindApparentHorizon(grid, options);
  assert(search.attempts.size() == 2);
  const auto& aliased = search.attempts[0];
  assert(aliased.result.solver.converged && aliased.result.surfaceAvailable);
  assert(!aliased.succeeded);
  assert(aliased.result.relativeExpansionError > 0.5);
  assert(aliased.result.irreducibleMass > 0.9);
  assert(!aliased.rejection.empty());
  assert(search.converged && search.successfulAttempt == 1);
  assert(search.horizon->relativeExpansionError < 0.05);
  assert(std::abs(search.horizon->irreducibleMass / 0.824914 - 1.0) < 1.0e-3);
}


}  // namespace
}  // namespace exahype2::ccz4::ahf::tests

int main() {
  using namespace exahype2::ccz4::ahf::tests;
  TestCandidateConversion();
  TestValidationAndEnclosure();
  TestProvisionalAndPhysicalEnclosure();
  TestOffAxisEnclosure();
  TestZeroAndOneBlackHole();
  TestExplicitFallbackAfterIndividualFailure();
  TestOffAxisFallbackCandidate();
  TestFallbackRequiresPhysicalPositions();
  TestTwoIndividualsProduceOneCommonSearch();
  TestThreeHoleDeterminismAndSummary();
  TestBrillLindquistCommonHorizonValidates();
  TestAliasedSurfaceIsRejected();
}
