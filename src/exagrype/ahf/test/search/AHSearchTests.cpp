#include "../require-asserts.h"

#include "../../apparent-horizon-finder.h"
#include "../brill-lindquist-grid.h"
#include "../temporary-directory.h"

#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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

AHSearchOptions SearchOptions(std::vector<AHSearchAttempt> attempts) {
  AHSearchOptions options;
  options.attempts = std::move(attempts);
  options.lMax = 0;
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

void TestFirstAttemptStopsSearch() {
  const AHSearchResult result = FindApparentHorizon(
    MakeGrid(),
    SearchOptions({{{0.0, 0.0, 0.0}, 1.3}, {{100.0, 0.0, 0.0}, 1.0}})
  );
  assert(result.converged);
  assert(result.successfulAttempt == 0);
  assert(result.attempts.size() == 1);
  assert(result.horizon.has_value());
  assert(result.horizon->solver.converged);
}

void TestRetryOrderingAndArbitraryCount() {
  std::vector<AHSearchAttempt> attempts;
  for (int i = 0; i < 5; ++i) {
    attempts.push_back({{100.0 + i, 0.0, 0.0}, 1.0});
  }
  attempts.push_back({{0.0, 0.0, 0.0}, 1.3});
  const AHSearchResult result = FindApparentHorizon(
    MakeGrid(), SearchOptions(attempts));
  assert(result.converged);
  assert(result.successfulAttempt == 5);
  assert(result.attempts.size() == 6);
  for (std::size_t i = 0; i < 5; ++i) {
    assert(result.attempts[i].referencePoint.x == 100.0 + double(i));
    assert(!result.attempts[i].succeeded);
    assert(!result.attempts[i].result.solver.message.empty());
  }
  assert(result.attempts[5].succeeded);
}

void TestAllAttemptsFail() {
  const AHSearchResult result = FindApparentHorizon(
    MakeGrid(),
    SearchOptions({{{100.0, 0.0, 0.0}, 1.0},
                   {{101.0, 0.0, 0.0}, 1.1},
                   {{102.0, 0.0, 0.0}, 1.2}})
  );
  assert(!result.converged);
  assert(result.successfulAttempt == AHSearchResult::NoSuccessfulAttempt);
  assert(!result.horizon.has_value());
  assert(result.attempts.size() == 3);
  for (const AHSearchAttemptResult& attempt : result.attempts) {
    assert(!attempt.succeeded);
    assert(!attempt.result.solver.message.empty());
  }
}

// Newton zeroes only the projection of Theta at the collocation points. A
// converged surface whose Theta in between exceeds maxRelativeExpansionError
// is rejected with a reason, and the search moves on to the next attempt.
// On a single puncture of mass 1 (horizon: the sphere of radius 1/2 about
// it), lMax 2 represents the horizon exactly about the puncture but not about
// an off-centre point, where the relative error is larger. A bound between
// the two errors rejects the off-centre attempt.
void TestConvergedSurfaceRejectedAboveExpansionBound() {
  const GridData grid =
    MakeBrillLindquistGrid(BrillLindquist::single(1.0), 1.0, 20);
  AHSearchOptions options;
  options.lMax = 2;
  options.interpolation = {
    .method = CartesianToSurfaceMethod::RadiusWeightedPolynomial,
    .polynomialDegree = 3,
    .supportRadius = 0.3};
  const std::array<ReferencePoint, 2> centres{
    {{0.0, 0.0, 0.0}, {0.2, 0.0, 0.0}}};
  std::array<double, 2> error{};
  for (std::size_t i = 0; i < 2; ++i) {
    options.attempts = {{centres[i], 0.7}};
    const AHSearchResult alone = FindApparentHorizon(grid, options);
    assert(alone.converged && alone.attempts[0].rejection.empty());
    error[i] = alone.horizon->relativeExpansionError;
    assert(error[i] < options.solver.maxRelativeExpansionError);
  }
  assert(error[1] > 2.0 * error[0]);

  options.attempts = {{centres[1], 0.7}, {centres[0], 0.7}};
  options.solver.maxRelativeExpansionError = 0.5 * (error[0] + error[1]);
  const AHSearchResult result = FindApparentHorizon(grid, options);
  assert(result.converged);
  assert(result.successfulAttempt == 1);
  assert(result.attempts.size() == 2);
  const AHSearchAttemptResult& rejected = result.attempts[0];
  assert(rejected.result.solver.converged && rejected.result.surfaceAvailable);
  assert(!rejected.succeeded);
  assert(rejected.result.relativeExpansionError == error[1]);
  assert(rejected.rejection.find("relative expansion error") !=
         std::string::npos);
  assert(result.attempts[1].succeeded && result.attempts[1].rejection.empty());
  assert(result.horizon->relativeExpansionError == error[0]);
}

void TestValidation() {
  bool threw = false;
  try { (void)FindApparentHorizon(MakeGrid(), SearchOptions({})); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);

  threw = false;
  try {
    (void)FindApparentHorizon(
      MakeGrid(), SearchOptions({{{0.0, 0.0, 0.0}, 0.0}}));
  } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);

  for (double bound : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
    auto options = SearchOptions({{{0.0, 0.0, 0.0}, 1.3}});
    options.solver.maxRelativeExpansionError = bound;
    threw = false;
    try { (void)FindApparentHorizon(MakeGrid(), options); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }
}

void TestSearchOutput() {
  const GridData grid = MakeGrid();
  const TemporaryDirectory temporary("ahf-search-tests");
  const auto& directory = temporary.path();
  const auto summary = directory / "ahf-search-summary.txt";
  const auto surface = directory / "ahf-search-surface.vtp";
  std::filesystem::remove(summary);
  std::filesystem::remove(surface);

  AHSearchResult failed = WriteAHToFile(
    grid,
    SearchOptions({{{100.0, 0.0, 0.0}, 1.0},
                   {{101.0, 0.0, 0.0}, 1.1}}),
    {.summaryFilename = summary.string(), .surfaceFilename = surface.string()}
  );
  assert(!failed.converged);
  assert(std::filesystem::is_regular_file(summary));
  assert(!std::filesystem::exists(surface));
  const std::string failedText = ReadText(summary);
  assert(failedText.find("attempts.count = 2") != std::string::npos);
  assert(failedText.find("attempts[0].referencePoint = 100 0 0") !=
         std::string::npos);
  assert(failedText.find("attempts[1].referencePoint = 101 0 0") !=
         std::string::npos);

  std::filesystem::remove(summary);
  AHSearchResult success = WriteAHToFile(
    grid,
    SearchOptions({{{100.0, 0.0, 0.0}, 1.0},
                   {{0.0, 0.0, 0.0}, 1.3}}),
    {.summaryFilename = summary.string(),
     .surfaceFilename = surface.string(),
     .visualization = {9, 6}}
  );
  assert(success.converged);
  assert(success.successfulAttempt == 1);
  assert(std::filesystem::is_regular_file(summary));
  assert(std::filesystem::is_regular_file(surface));
  const std::string successText = ReadText(summary);
  assert(successText.find("AHSearchSummaryVersion = 1") != std::string::npos);
  assert(successText.find("successfulAttempt = 1") != std::string::npos);
  assert(successText.find("attempts.count = 2") != std::string::npos);
  assert(successText.find("attempts[0].succeeded = false") != std::string::npos);
  assert(successText.find("attempts[1].succeeded = true") != std::string::npos);
  assert(successText.find("horizon.meanRadius = ") != std::string::npos);
  assert(successText.find("horizon.meanCoordinateRadius = ") != std::string::npos);
  assert(successText.find("horizon.area = ") != std::string::npos);
  assert(successText.find("horizon.arealRadius = ") != std::string::npos);
  assert(successText.find("horizon.irreducibleMass = ") != std::string::npos);
  assert(successText.find("horizon.relativeExpansionError = ") !=
         std::string::npos);
  assert(successText.find("attempts[1].rejection = \"\"") != std::string::npos);
  // The grid is flat with K_ij = 2 delta_ij and K = 3, so Theta = 2/r - 1:
  // the horizon is a sphere of radius 2, of mass 1.
  assert(std::abs(success.horizon->irreducibleMass - 1.0) < 1.0e-6);
  assert(ReadText(surface).find("NumberOfPoints=\"38\"") != std::string::npos);
  std::filesystem::remove(summary);
  std::filesystem::remove(surface);
}

AHMultiHorizonOptions MultiOptions(std::vector<AHHorizonTarget> targets) {
  AHMultiHorizonOptions options;
  options.targets = std::move(targets);
  options.lMax = 0;
  options.interpolation = {
    .method = CartesianToSurfaceMethod::UnweightedPolynomial,
    .numberOfNeighbours = 64,
    .polynomialDegree = 1,
    .allowOrderReduction = false
  };
  return options;
}

void TestPhysicalTargetGrouping() {
  const AHMultiHorizonResult result = FindApparentHorizons(
    MakeGrid(),
    MultiOptions({
      {
        .id = "BH-plus",
        .kind = AHHorizonKind::Individual,
        .attempts = {{{100.0, 0.0, 0.0}, 1.0},
                     {{0.25, 0.0, 0.0}, 1.3},
                     {{0.0, 0.0, 0.0}, 1.3}}
      },
      {
        .id = "BH-minus",
        .kind = AHHorizonKind::Individual,
        .attempts = {{{-0.25, 0.0, 0.0}, 1.3}}
      }
    })
  );
  assert(result.anyHorizonFound);
  assert(result.allTargetsFound);
  assert(result.targets.size() == 2);
  assert(result.targets[0].id == "BH-plus");
  assert(result.targets[0].successfulAttempt == 1);
  assert(result.targets[0].attempts.size() == 2); // third retry was not run
  assert(!result.targets[0].attempts[0].succeeded);
  assert(result.targets[0].attempts[1].succeeded);
  assert(result.targets[1].id == "BH-minus");
  assert(result.targets[1].converged); // first target success did not stop it
  assert(result.targets[1].attempts.size() == 1);
}

void TestPhysicalTargetPartialSuccess() {
  const AHMultiHorizonResult result = FindApparentHorizons(
    MakeGrid(),
    MultiOptions({
      {.id = "missing", .attempts = {{{100.0, 0.0, 0.0}, 1.0}}},
      {.id = "common", .kind = AHHorizonKind::Common,
       .attempts = {{{0.0, 0.0, 0.0}, 1.3}}}
    })
  );
  assert(result.anyHorizonFound);
  assert(!result.allTargetsFound);
  assert(!result.targets[0].converged);
  assert(!result.targets[0].horizon.has_value());
  assert(result.targets[1].converged);
  assert(result.targets[1].horizon.has_value());
  assert(result.targets[1].kind == AHHorizonKind::Common);
}

// A target's massScale reaches its solve: the effective tolerance becomes
// tolerance / massScale for that target only.
void TestTargetMassScaleReachesSolver() {
  AHMultiHorizonOptions options = MultiOptions({
    {.id = "shared", .attempts = {{{0.0, 0.0, 0.0}, 1.3}}},
    {.id = "heavy", .attempts = {{{0.0, 0.0, 0.0}, 1.3}}, .massScale = 1.0e8}
  });
  options.solver.tolerance = 1.0e-2;
  const AHMultiHorizonResult result = FindApparentHorizons(MakeGrid(), options);
  assert(result.allTargetsFound);
  const SpectralNewtonResult& shared = result.targets[0].horizon->solver;
  const SpectralNewtonResult& heavy = result.targets[1].horizon->solver;
  assert(shared.residualHistory.back() < 1.0e-2);
  assert(shared.residualHistory.back() >= 1.0e-10);
  assert(heavy.residualHistory.back() < 1.0e-10);
  assert(heavy.iterations > shared.iterations);
}

void TestPhysicalTargetValidation() {
  bool threw = false;
  try { (void)FindApparentHorizons(MakeGrid(), MultiOptions({})); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);

  threw = false;
  try {
    (void)FindApparentHorizons(
      MakeGrid(),
      MultiOptions({{.id = "same", .attempts = {{{0, 0, 0}, 1.0}}},
                    {.id = "same", .attempts = {{{0, 0, 0}, 1.0}}}}));
  } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
}

void TestSpectralAndPreviousSurfaceAttempts() {
  const SpectralBasis basis = BuildSpectralBasis(0);
  const std::vector<double> coefficients =
    MakeSphericalInitialCoefficients(basis, 1.3);
  AHSearchOptions options = SearchOptions({{
    .referencePoint = {0.0, 0.0, 0.0},
    .initialRadius = 0.0,
    .initialCoefficients = coefficients
  }});
  const AHSearchResult first = FindApparentHorizon(MakeGrid(), options);
  assert(first.converged);
  assert(first.attempts.size() == 1);
  assert(first.attempts[0].initialCoefficients == coefficients);

  const AHSearchAttempt continuation =
    MakePreviousSurfaceAttempt(*first.horizon);
  assert(continuation.referencePoint.x == first.horizon->referencePoint.x);
  assert(continuation.initialCoefficients == first.horizon->solver.coefficients);
  const AHSearchResult second = FindApparentHorizon(
    MakeGrid(), SearchOptions({continuation}));
  assert(second.converged);

  options.attempts[0].initialCoefficients.push_back(0.0);
  bool rejected = false;
  try { (void)FindApparentHorizon(MakeGrid(), options); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);

  AHSearchOptions distortedOptions = SearchOptions({});
  distortedOptions.lMax = 2;
  const SpectralBasis distortedBasis = BuildSpectralBasis(2);
  std::vector<double> radii(distortedBasis.nPoints);
  for (std::size_t p = 0; p < radii.size(); ++p) {
    radii[p] = 1.3 + 0.05 * EvalMode(
      2, 0, distortedBasis.rho[p], distortedBasis.sigma[p]).Y;
  }
  distortedOptions.attempts = {{
    .referencePoint = {0.0, 0.0, 0.0},
    .initialCoefficients =
      MakeDistortedInitialCoefficients(distortedBasis, radii)
  }};
  const AHSearchResult distorted =
    FindApparentHorizon(MakeGrid(), distortedOptions);
  assert(distorted.converged);
  assert(distorted.horizon->solver.coefficients.size() ==
         distortedBasis.nModes);
}

void TestPhysicalTargetOutput() {
  const GridData grid = MakeGrid();
  const TemporaryDirectory temporary("ahf-search-tests");
  const auto& directory = temporary.path();
  const auto aggregate = directory / "ahf-multi-horizon-summary.txt";
  const auto prefix = directory / "ahf-multi-horizon";
  const auto missingSummary = directory / "ahf-multi-horizon-0-missing.txt";
  const auto missingSurface = directory / "ahf-multi-horizon-0-missing.vtp";
  const auto plusSummary = directory / "ahf-multi-horizon-1-BH_plus.txt";
  const auto plusSurface = directory / "ahf-multi-horizon-1-BH_plus.vtp";
  const auto commonSummary = directory / "ahf-multi-horizon-2-common.txt";
  const auto commonSurface = directory / "ahf-multi-horizon-2-common.vtp";
  for (const auto& path : {aggregate, missingSummary, missingSurface,
                           plusSummary, plusSurface, commonSummary,
                           commonSurface}) {
    std::filesystem::remove(path);
  }

  const AHMultiHorizonResult result = WriteAHToFile(
    grid,
    MultiOptions({
      {.id = "missing", .attempts = {{{100.0, 0.0, 0.0}, 1.0}}},
      {.id = "BH plus", .attempts = {{{0.25, 0.0, 0.0}, 1.3}}},
      {.id = "common", .kind = AHHorizonKind::Common,
       .attempts = {{{0.0, 0.0, 0.0}, 1.3}}}
    }),
    {.summaryFilename = aggregate.string(),
     .horizonFilenamePrefix = prefix.string(),
     .visualization = {9, 6}}
  );

  assert(result.anyHorizonFound);
  assert(!result.allTargetsFound);
  assert(std::filesystem::is_regular_file(aggregate));
  assert(!std::filesystem::exists(missingSummary));
  assert(!std::filesystem::exists(missingSurface));
  assert(std::filesystem::is_regular_file(plusSummary));
  assert(std::filesystem::is_regular_file(plusSurface));
  assert(std::filesystem::is_regular_file(commonSummary));
  assert(std::filesystem::is_regular_file(commonSurface));

  const std::string aggregateText = ReadText(aggregate);
  assert(aggregateText.find("AHMultiHorizonSummaryVersion = 1") !=
         std::string::npos);
  assert(aggregateText.find("anyHorizonFound = true") != std::string::npos);
  assert(aggregateText.find("allTargetsFound = false") != std::string::npos);
  assert(aggregateText.find("targets.count = 3") != std::string::npos);
  assert(aggregateText.find("targets[0].id = \"missing\"") !=
         std::string::npos);
  assert(aggregateText.find("targets[0].solver.message") == std::string::npos);
  assert(aggregateText.find("targets[0].attempts[0].solver.message") !=
         std::string::npos);
  assert(aggregateText.find("targets[1].id = \"BH plus\"") !=
         std::string::npos);
  assert(aggregateText.find("targets[2].kind = common") != std::string::npos);
  assert(aggregateText.find("targets[1].horizon.meanCoordinateRadius = ") !=
         std::string::npos);
  assert(aggregateText.find("targets[1].horizon.area = ") != std::string::npos);
  assert(aggregateText.find("targets[1].horizon.arealRadius = ") !=
         std::string::npos);
  assert(aggregateText.find("targets[1].horizon.irreducibleMass = ") !=
         std::string::npos);
  assert(aggregateText.find("targets[1].horizon.relativeExpansionError = ") !=
         std::string::npos);
  assert(aggregateText.find("targets[1].attempts[0].rejection = ") !=
         std::string::npos);
  assert(std::abs(result.targets[1].horizon->irreducibleMass - 1.0) < 1.0e-6);
  assert(ReadText(plusSurface).find("NumberOfPoints=\"38\"") !=
         std::string::npos);

  for (const auto& path : {aggregate, plusSummary, plusSurface, commonSummary,
                           commonSurface}) {
    std::filesystem::remove(path);
  }
}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests

int main() {
  exahype2::ccz4::ahf::tests::TestFirstAttemptStopsSearch();
  exahype2::ccz4::ahf::tests::TestRetryOrderingAndArbitraryCount();
  exahype2::ccz4::ahf::tests::TestAllAttemptsFail();
  exahype2::ccz4::ahf::tests::TestConvergedSurfaceRejectedAboveExpansionBound();
  exahype2::ccz4::ahf::tests::TestValidation();
  exahype2::ccz4::ahf::tests::TestSearchOutput();
  exahype2::ccz4::ahf::tests::TestPhysicalTargetGrouping();
  exahype2::ccz4::ahf::tests::TestPhysicalTargetPartialSuccess();
  exahype2::ccz4::ahf::tests::TestTargetMassScaleReachesSolver();
  exahype2::ccz4::ahf::tests::TestPhysicalTargetValidation();
  exahype2::ccz4::ahf::tests::TestSpectralAndPreviousSurfaceAttempts();
  exahype2::ccz4::ahf::tests::TestPhysicalTargetOutput();
}
