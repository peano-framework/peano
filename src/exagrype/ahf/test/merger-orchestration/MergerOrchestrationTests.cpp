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

namespace exahype2::ccz4::ahf::tests {
namespace {

GridData MakeGrid() {
  GridData grid;
  for (int ix = -2; ix <= 2; ++ix) {
    for (int iy = -2; iy <= 2; ++iy) {
      for (int iz = -2; iz <= 2; ++iz) {
        grid.coordinates.push_back({double(ix), double(iy), double(iz)});
        const std::array<double, NumberOfCompactFields> row{
          1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
          1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
          3.0, 1.0
        };
        grid.values.insert(grid.values.end(), row.begin(), row.end());
      }
    }
  }
  return grid;
}

AHBinaryMergerOptions Options(double x, double radius = 0.2) {
  AHBinaryMergerOptions options;
  options.components = {{
    {{.id = "plus", .attempts = {{{x, 0.0, 0.0}, radius}}}, 1.0},
    {{.id = "minus", .attempts = {{{-x, 0.0, 0.0}, radius}}}, 3.0}
  }};
  options.lMax = 2;
  options.interpolation = {
    .method = CartesianToSurfaceMethod::UnweightedPolynomial,
    .numberOfNeighbours = 64,
    .polynomialDegree = 1,
    .allowOrderReduction = false
  };
  return options;
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

// The common search is planned when the initial surfaces' gap is at most
// searchMarginPerMass (0.5) times the total mass, here 4.
void TestSeparatedPlan() {
  const AHBinaryMergerPlan plan = BuildBinaryMergerPlan(Options(2.0));
  assert(!plan.commonSearchEnabled);
  assert(std::abs(plan.separation - 4.0) < 1.0e-14);
  assert(std::abs(plan.facingRadii[0] - 0.2) < 1.0e-14);
  assert(std::abs(plan.surfaceGap - 3.6) < 1.0e-14);
  assert(plan.search.targets.size() == 2);
  assert(plan.search.targets[0].id == "plus");
  assert(plan.search.targets[1].id == "minus");
  assert(*plan.search.targets[0].massScale == 1.0);
  assert(*plan.search.targets[1].massScale == 3.0);
  // Each component excludes the other's first reference point.
  assert(plan.search.targets[0].excludedPoints.size() == 1);
  assert(plan.search.targets[0].excludedPoints[0].x == -2.0);
  assert(plan.search.targets[1].excludedPoints[0].x == 2.0);

  // A gap of 1.6 is within the margin of 2; without the margin only overlap
  // triggers.
  AHBinaryMergerOptions options = Options(1.0);
  assert(BuildBinaryMergerPlan(options).commonSearchEnabled);
  options.searchMarginPerMass = 0.0;
  const AHBinaryMergerPlan contactOnly = BuildBinaryMergerPlan(options);
  assert(!contactOnly.commonSearchEnabled);
  assert(std::abs(contactOnly.surfaceGap - 1.6) < 1.0e-14);
}

void TestOverlappingPlan() {
  AHBinaryMergerOptions options = Options(0.125);
  options.commonRadiusFactors = {1.0, 1.5};
  const AHBinaryMergerPlan plan = CheckCommonHorizonOfBBH(options);
  assert(plan.commonSearchEnabled);
  assert(std::abs(plan.separation - 0.25) < 1.0e-14);
  assert(std::abs(plan.facingRadii[0] + plan.facingRadii[1] - 0.4) < 1.0e-14);
  assert(std::abs(plan.commonMass - 4.0) < 1.0e-14);
  assert(std::abs(plan.commonReferencePoint.x + 0.0625) < 1.0e-14);
  assert(std::abs(plan.enclosingInitialRadius - 0.3875) < 1.0e-14);
  assert(plan.search.targets.size() == 3);
  assert(plan.search.targets[0].id == "common");
  assert(plan.search.targets[0].kind == AHHorizonKind::Common);
  assert(*plan.search.targets[0].massScale == 4.0);
  assert(plan.search.targets[0].attempts.size() == 2);
  assert(plan.search.targets[1].id == "plus");
  assert(plan.search.targets[2].id == "minus");
}

void TestDistortedFacingRadius() {
  AHBinaryMergerOptions options = Options(0.5, 0.1);
  const SpectralBasis basis = BuildSpectralBasis(options.lMax);
  std::vector<double> radii(basis.nPoints);
  for (std::size_t p = 0; p < radii.size(); ++p) {
    radii[p] = 0.45 + 0.05 * std::cos(basis.rho[p]) *
      std::cos(basis.sigma[p]);
  }
  options.components[0].target.attempts[0].initialCoefficients =
    MakeDistortedInitialCoefficients(basis, radii);
  options.components[1].target.attempts[0].initialRadius = 0.65;
  const AHBinaryMergerPlan plan = BuildBinaryMergerPlan(options);
  assert(plan.commonSearchEnabled);
  assert(plan.facingRadii[0] > 0.39 && plan.facingRadii[0] < 0.41);
  assert(std::abs(plan.facingRadii[1] - 0.65) < 1.0e-14);
}

double Length(double x, double y, double z) { return std::sqrt(x * x + y * y + z * z); }

// Off-axis, unequal masses: the separation, the facing radius of a distorted
// surface and the mass-weighted centre use all three coordinates.
void TestOffAxisPlan() {
  const ReferencePoint a{0.1, 0.3, -0.2};
  const ReferencePoint b{-0.2, -0.2, 0.3};
  const double dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  const double distance = Length(dx, dy, dz);

  AHBinaryMergerOptions options = Options(1.0);
  options.components[0].target.attempts = {{a, 0.3}};
  options.components[1].target.attempts = {{b, 0.3}};
  // The first surface bulges towards the second: r = 0.3 + 0.05 (n . d/|d|).
  const SpectralBasis basis = BuildSpectralBasis(options.lMax);
  std::vector<double> radii(basis.nPoints);
  for (std::size_t p = 0; p < radii.size(); ++p) {
    const double c = std::cos(basis.sigma[p]);
    const double n[3] = {c * std::cos(basis.rho[p]), c * std::sin(basis.rho[p]),
                         std::sin(basis.sigma[p])};
    radii[p] = 0.3 + 0.05 * (n[0] * dx + n[1] * dy + n[2] * dz) / distance;
  }
  options.components[0].target.attempts[0].initialCoefficients =
    MakeDistortedInitialCoefficients(basis, radii);

  const AHBinaryMergerPlan plan = BuildBinaryMergerPlan(options);
  assert(std::abs(plan.separation - distance) < 1.0e-14);
  assert(std::abs(plan.facingRadii[0] - 0.35) < 1.0e-12);
  assert(std::abs(plan.facingRadii[1] - 0.3) < 1.0e-14);
  assert(std::abs(plan.commonMass - 4.0) < 1.0e-14);
  const ReferencePoint centre{(a.x + 3.0 * b.x) / 4.0, (a.y + 3.0 * b.y) / 4.0,
                              (a.z + 3.0 * b.z) / 4.0};
  assert(std::abs(plan.commonReferencePoint.x - centre.x) < 1.0e-14);
  assert(std::abs(plan.commonReferencePoint.y - centre.y) < 1.0e-14);
  assert(std::abs(plan.commonReferencePoint.z - centre.z) < 1.0e-14);
  const SynthesizedField first =
    Synthesize(basis, options.components[0].target.attempts[0].initialCoefficients);
  const double firstMaximum = *std::max_element(first.h.begin(), first.h.end());
  const double enclosing = std::max(
    Length(centre.x - a.x, centre.y - a.y, centre.z - a.z) + firstMaximum,
    Length(centre.x - b.x, centre.y - b.y, centre.z - b.z) + 0.3);
  assert(std::abs(plan.enclosingInitialRadius - enclosing) < 1.0e-14);
}

void TestValidation() {
  AHBinaryMergerOptions invalid = Options(1.0);
  invalid.components[0].mass = 0.0;
  bool threw = false;
  try { (void)BuildBinaryMergerPlan(invalid); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);

  invalid = Options(1.0);
  invalid.components[1].target.id = "plus";
  threw = false;
  try { (void)BuildBinaryMergerPlan(invalid); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);

  invalid = Options(1.0);
  invalid.searchMarginPerMass = -0.1;
  threw = false;
  try { (void)BuildBinaryMergerPlan(invalid); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
}

void TestExecutionAndOutput() {
  AHBinaryMergerOptions options = Options(0.1, 1.1);
  options.components[1].mass = 1.0;
  options.commonRadiusFactors = {1.0};
  options.solver.maxIterations = 0;
  const TemporaryDirectory temporary("ahf-merger-orchestration-tests");
  const auto& directory = temporary.path();
  const auto summary = directory / "ahf-merger-summary.txt";
  const auto prefix = directory / "ahf-merger";
  std::filesystem::remove(summary);

  const AHBinaryMergerResult result = WriteAHToFile(
    MakeGrid(), options,
    {.summaryFilename = summary.string(),
     .horizonFilenamePrefix = prefix.string(),
     .visualization = {9, 6}});
  assert(result.plan.commonSearchEnabled);
  assert(result.horizons.targets.size() == 3);
  assert(result.horizons.targets[0].id == "common");
  assert(!result.horizons.allTargetsFound);
  assert(!result.horizons.anyHorizonFound);
  const std::string summaryText = ReadText(summary);
  assert(summaryText.find("merger.commonSearchEnabled = true") !=
         std::string::npos);
  assert(summaryText.find("merger.commonMass = 2") != std::string::npos);
  assert(summaryText.find("targets.count = 3") != std::string::npos);
  assert(!std::filesystem::exists(prefix.string() + "-0-common.vtp"));
  // Without a reconstructed common surface the validation rejects it.
  assert(result.commonValidation.has_value());
  assert(result.commonValidation->status ==
         AHCommonHorizonValidationStatus::Rejected);
  assert(summaryText.find("merger.surfaceGap = ") != std::string::npos);
  assert(summaryText.find("merger.commonValidation.status = rejected") !=
         std::string::npos);
  assert(summaryText.find("did not produce a reconstructed surface") !=
         std::string::npos);
  std::filesystem::remove(summary);
}

// Equal masses 1/2 at separation 1/2 on Brill-Lindquist data, as ExaGRyPE
// would produce it. Started from radius 0.375, the plus component converges
// to the common horizon, which encloses the other component and is
// rejected; its next attempt finds its own horizon. The common horizon is
// planned, found and validated, with the analytic irreducible mass 0.998248.
void TestBrillLindquistMerger() {
  const BrillLindquist binary{
    {{0.5, {0.25, 0.0, 0.0}}, {0.5, {-0.25, 0.0, 0.0}}}};
  const GridData grid = MakeBrillLindquistGrid(binary, 0.9, 36);
  AHBinaryMergerOptions options;
  options.components = {{
    {{.id = "plus",
      .attempts = {{{0.25, 0.0, 0.0}, 0.375}, {{0.25, 0.0, 0.0}, 0.175}}}, 0.5},
    {{.id = "minus", .attempts = {{{-0.25, 0.0, 0.0}, 0.175}}}, 0.5}
  }};
  options.commonRadiusFactors = {1.0};
  options.lMax = 4;
  options.interpolation = {
    .method = CartesianToSurfaceMethod::RadiusWeightedPolynomial,
    .polynomialDegree = 3,
    .supportRadius = 0.15};
  const AHBinaryMergerResult result = FindBinaryMergerHorizons(grid, options);
  assert(result.plan.commonSearchEnabled);
  assert(result.horizons.targets.size() == 3);
  const auto& plus = result.horizons.targets[1];
  assert(plus.id == "plus" && plus.attempts.size() == 2);
  assert(plus.attempts[0].result.solver.converged);
  assert(!plus.attempts[0].succeeded);
  assert(plus.attempts[0].rejection.find("encloses") != std::string::npos);
  assert(plus.converged && plus.successfulAttempt == 1);
  assert(std::abs(plus.horizon->irreducibleMass / 0.749987 - 1.0) < 5.0e-3);
  const auto& common = result.horizons.targets[0];
  assert(common.converged);
  assert(result.commonValidation.has_value() && result.commonValidation->valid);
  assert(std::abs(common.horizon->irreducibleMass / 0.998248 - 1.0) < 2.0e-4);
}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests

int main() {
  using namespace exahype2::ccz4::ahf::tests;
  TestSeparatedPlan();
  TestOverlappingPlan();
  TestDistortedFacingRadius();
  TestOffAxisPlan();
  TestValidation();
  TestExecutionAndOutput();
  TestBrillLindquistMerger();
}
