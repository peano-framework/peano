#include "../require-asserts.h"

#include "../../ahf-postprocess-options.h"
#include "../brill-lindquist-grid.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf::tests::postprocess {
namespace {

void ExpectRejected(const std::vector<std::string>& arguments,
                    const std::string& expectedMessagePart) {
  bool rejected = false;
  try {
    (void)ParsePostprocessArguments(arguments);
  } catch (const std::invalid_argument& error) {
    rejected = true;
    assert(std::string(error.what()).find(expectedMessagePart) != std::string::npos);
  }
  assert(rejected);
}

void TestDefaults() {
  const PostprocessOptions options =
    ParsePostprocessArguments({"--input", "snapshot.vtu", "--radius", "1"});
  assert(options.inputFile == "snapshot.vtu");
  assert(options.outputPrefix == "apparent-horizon");
  assert(options.mapping == PostprocessMapping::Names);
  assert(options.centre.x == 0.0 && options.centre.y == 0.0 && options.centre.z == 0.0);
  assert(options.initialRadii == std::vector<double>{1.0});
  assert(options.lMax == 4);
  assert(!options.supportRadius.has_value());
  assert(options.tolerance == 1.0e-8);
  assert(options.punctures.empty());
  assert((options.radiiPerMass == std::vector<double>{0.35, 0.5, 0.25}));
  assert(!options.help);

  std::cout << "ahf-postprocess needs only --input and --radius" << std::endl;
}

void TestEveryOption() {
  const PostprocessOptions options = ParsePostprocessArguments({
    "--input", "pieces/snapshot.pvtu", "--output", "run/horizon.vtp", "--mapping", "legacy",
    "--centre", "1,-2,0.5", "--radius", "1.0,0.7", "--radius", "0.5", "--lmax", "6",
    "--support-radius", "0.2", "--tolerance", "1e-9"});
  assert(options.inputFile == "pieces/snapshot.pvtu");
  assert(options.outputPrefix == "run/horizon");
  assert(options.mapping == PostprocessMapping::LegacyCCZ4);
  assert(options.centre.x == 1.0 && options.centre.y == -2.0 && options.centre.z == 0.5);
  assert((options.initialRadii == std::vector<double>{1.0, 0.7, 0.5}));
  assert(options.lMax == 6);
  assert(options.supportRadius == 0.2);
  assert(options.tolerance == 1.0e-9);

  assert(ParsePostprocessArguments({"--input", "a.vtu", "--radius", "1", "--output", "b.txt"})
           .outputPrefix == "b");
  assert(ParsePostprocessArguments({"--input", "a.vtu", "--radius", "1", "--center", "0,0,3"})
           .centre.z == 3.0);
  assert(ParsePostprocessArguments({"--input", "a.vtu", "--radius", "1", "--mapping", "exagrype"})
           .mapping == PostprocessMapping::ExaGRyPE);

  std::cout << "ahf-postprocess parses every option, with radii repeated or comma separated"
            << std::endl;
}

void TestHelp() {
  assert(ParsePostprocessArguments({"--help"}).help);
  assert(ParsePostprocessArguments({"-h"}).help);
  assert(PostprocessUsage("ahf-postprocess").find("--support-radius") != std::string::npos);
  assert(PostprocessUsage("ahf-postprocess").find("--puncture") !=
         std::string::npos);

  std::cout << "ahf-postprocess --help needs no other option" << std::endl;
}

void TestInvalidArguments() {
  ExpectRejected({"--radius", "1"}, "--input is required");
  ExpectRejected({"--input", "a.vtu"}, "--radius or --puncture is required");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--verbose"},
                 "--verbose is not an option");
  ExpectRejected({"--input", "a.vtu", "--radius"}, "--radius needs a value");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--lmax", "x"},
                 "--lmax expects a positive integer, not 'x'");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--lmax", "4.5"},
                 "--lmax expects a positive integer");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--lmax", "0"},
                 "--lmax expects a positive integer");
  ExpectRejected({"--input", "a.vtu", "--radius", "1,"}, "--radius expects a number, not ''");
  ExpectRejected({"--input", "a.vtu", "--radius", "-1"}, "--radius expects a positive number");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--support-radius", "0"},
                 "--support-radius expects a positive number");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--tolerance", "nan"},
                 "--tolerance expects a number");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--centre", "1,2"},
                 "--centre expects x,y,z");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--centre", "1,2,z"},
                 "--centre expects a number, not 'z'");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--mapping", "auto"},
                 "--mapping expects names, exagrype or legacy");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--output", ""},
                 "--output must not be empty");

  std::cout << "ahf-postprocess rejects unknown options and malformed values" << std::endl;
}

void TestSourceMapping() {
  assert(!PostprocessSourceMapping(PostprocessMapping::Names).has_value());
  assert(PostprocessSourceMapping(PostprocessMapping::ExaGRyPE)->traceK == 23);
  assert(PostprocessSourceMapping(PostprocessMapping::LegacyCCZ4)->traceK == 53);

  std::cout << "--mapping selects reading by name or one of the presets" << std::endl;
}

GridData GridAt(const std::vector<GridCoordinate>& coordinates) {
  GridData grid;
  grid.coordinates = coordinates;
  grid.values.assign(coordinates.size() * GridData::numberOfVariables, 0.0);
  return grid;
}

void TestFinestGridSpacing() {
  // A coarse row with spacing 0.3 next to a refined row with spacing 0.1, as
  // on a Peano grid with 3:1 refinement; both lie on the fine lattice.
  const GridData twoLevels = GridAt({
    {0.15, 0.15, 0.15}, {0.45, 0.15, 0.15}, {0.75, 0.15, 0.15},
    {0.95, 0.05, 0.05}, {1.05, 0.05, 0.05}, {1.15, 0.05, 0.05}});
  assert(std::abs(FinestGridSpacing(twoLevels) - 0.1) < 1.0e-12);

  // Cells stacked along z only: x and y have one value each.
  const GridData column = GridAt({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.25}, {0.0, 0.0, 0.5}});
  assert(std::abs(FinestGridSpacing(column) - 0.25) < 1.0e-12);

  // Centres that differ only by rounding count as the same plane. render.py
  // stores points in single precision, so centres on one plane of a refined
  // grid differ by about one float ulp.
  const GridData rounded = GridAt({{0.1, 0.0, 0.0}, {0.1 + 1.0e-15, 0.5, 0.0}, {0.6, 1.0, 0.0}});
  assert(std::abs(FinestGridSpacing(rounded) - 0.5) < 1.0e-12);

  const double fine = 4.0 / 162.0;
  std::vector<GridCoordinate> singlePrecision;
  for (int i = 0; i < 4; ++i) {
    const double x = -1.0 + (i + 0.5) * fine;
    singlePrecision.push_back({static_cast<float>(x), 0.25, 0.25});
    singlePrecision.push_back({std::nextafter(static_cast<float>(x), 2.0f), 0.25, 0.25});
  }
  assert(std::abs(FinestGridSpacing(GridAt(singlePrecision)) - fine) < 1.0e-6);

  bool rejected = false;
  try {
    (void)FinestGridSpacing(GridAt({{1.0, 2.0, 3.0}}));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);

  std::cout << "FinestGridSpacing finds the cell size of the finest level" << std::endl;
}

void TestSearchOptions() {
  const PostprocessOptions options = ParsePostprocessArguments({
    "--input", "a.vtu", "--centre", "1,2,3", "--radius", "1.0,0.5", "--lmax", "6",
    "--tolerance", "1e-7"});
  const AHSearchOptions search = MakePostprocessSearchOptions(options, 0.25);
  assert(search.attempts.size() == 2);
  for (const AHSearchAttempt& attempt : search.attempts) {
    assert(attempt.referencePoint.x == 1.0);
    assert(attempt.referencePoint.y == 2.0);
    assert(attempt.referencePoint.z == 3.0);
    assert(attempt.initialCoefficients.empty());
  }
  assert(search.attempts[0].initialRadius == 1.0);
  assert(search.attempts[1].initialRadius == 0.5);
  assert(search.lMax == 6);
  assert(search.interpolation.method == CartesianToSurfaceMethod::RadiusWeightedPolynomial);
  assert(search.interpolation.polynomialDegree == 3);
  assert(search.interpolation.supportRadius == 0.25);
  assert(!search.interpolation.allowOrderReduction);
  assert(search.solver.tolerance == 1.0e-7);
  ValidateAHSearchOptions(search);

  std::cout << "Each initial radius becomes one search attempt around the centre" << std::endl;
}

void TestPunctures() {
  const PostprocessOptions options = ParsePostprocessArguments({
    "--input", "a.vtu", "--puncture", "0.25,0,0:0.5",
    "--puncture", "-0.2,0.1,0:0.3"});
  assert(options.initialRadii.empty());
  assert(options.punctures.size() == 2);
  assert(options.punctures[0].position.x == 0.25);
  assert(options.punctures[0].position.y == 0.0);
  assert(options.punctures[0].mass == 0.5);
  assert(options.punctures[1].position.x == -0.2);
  assert(options.punctures[1].position.y == 0.1);
  assert(options.punctures[1].mass == 0.3);
  assert((options.radiiPerMass == std::vector<double>{0.35, 0.5, 0.25}));
  assert(options.lMax == 8);

  // The first --radius-per-mass replaces the defaults, later ones add to it.
  assert((ParsePostprocessArguments({"--input", "a.vtu",
                                     "--puncture", "0,0,0:1",
                                     "--radius-per-mass", "0.4,0.3",
                                     "--radius-per-mass", "0.2"})
            .radiiPerMass == std::vector<double>{0.4, 0.3, 0.2}));

  ExpectRejected({"--input", "a.vtu", "--puncture", "0,0,0"}, "--puncture expects x,y,z:m");
  ExpectRejected({"--input", "a.vtu", "--puncture", "0,0:1"}, "--puncture expects x,y,z:m");
  ExpectRejected({"--input", "a.vtu", "--puncture", "0,0,0:1:2"},
                 "--puncture expects x,y,z:m");
  ExpectRejected({"--input", "a.vtu", "--puncture", "0,0,0:0"},
                 "--puncture expects a positive number");
  ExpectRejected({"--input", "a.vtu", "--puncture", "0,y,0:1"},
                 "--puncture expects a number, not 'y'");
  ExpectRejected({"--input", "a.vtu", "--puncture", "0,0,0:1", "--radius", "1"},
                 "with --puncture use --radius-per-mass");
  ExpectRejected({"--input", "a.vtu", "--puncture", "0,0,0:1", "--centre", "0,0,0"},
                 "--puncture gives the positions");
  ExpectRejected({"--input", "a.vtu", "--radius", "1", "--radius-per-mass", "0.3"},
                 "--radius-per-mass needs --puncture");
  ExpectRejected({"--input", "a.vtu", "--puncture", "0,0,0:1", "--radius-per-mass", "0"},
                 "--radius-per-mass expects a positive number");

  std::cout << "--puncture gives each black hole's position and mass, instead of --radius"
            << std::endl;
}

void TestCatalogueOptions() {
  const PostprocessOptions options = ParsePostprocessArguments({
    "--input", "a.vtu", "--puncture", "0.25,0,0:0.5",
    "--puncture", "-0.2,0.1,0:0.25", "--radius-per-mass", "0.4,0.2",
    "--lmax", "6", "--tolerance", "1e-7"});
  const AHCommonHorizonSearchOptions catalogue =
    MakePostprocessCatalogueOptions(options, 0.25);
  assert(catalogue.blackHoles.size() == 2);
  for (std::size_t i = 0; i < 2; ++i) {
    const AHPhysicalBlackHole& blackHole = catalogue.blackHoles[i];
    const PostprocessPuncture& puncture = options.punctures[i];
    assert(blackHole.id == "bh" + std::to_string(i));
    assert(blackHole.mass == puncture.mass);
    assert(blackHole.physicalPosition.has_value());
    assert(blackHole.physicalPosition->x == puncture.position.x);
    assert(blackHole.physicalPosition->y == puncture.position.y);
    assert(blackHole.attempts.size() == 2);
    for (const AHSearchAttempt& attempt : blackHole.attempts) {
      assert(attempt.referencePoint.x == puncture.position.x);
      assert(attempt.referencePoint.y == puncture.position.y);
      assert(attempt.initialCoefficients.empty());
    }
    assert(blackHole.attempts[0].initialRadius == 0.4 * puncture.mass);
    assert(blackHole.attempts[1].initialRadius == 0.2 * puncture.mass);
  }
  assert(catalogue.lMax == 6);
  assert(catalogue.interpolation.method ==
         CartesianToSurfaceMethod::RadiusWeightedPolynomial);
  assert(catalogue.interpolation.polynomialDegree == 3);
  assert(catalogue.interpolation.supportRadius == 0.25);
  assert(catalogue.solver.tolerance == 1.0e-7);
  // The common search stops at 1e-7 / 0.75; the residual bound of the
  // validation, 1e-7 / 0.25, is above that.
  assert(catalogue.validation.spectralResidualTolerance == 1.0e-7 / 0.25);
  assert(catalogue.individualsExcludeOtherBlackHoles);
  assert(catalogue.candidates.searchMarginPerMass == 0.5);
  assert(!catalogue.candidates.unconditionalAllBlackHoles);
  ValidateAHPhysicalBlackHoles(catalogue.blackHoles);

  std::cout << "Each puncture becomes a black hole searched from multiples of its mass"
            << std::endl;
}

// What ahf-postprocess --puncture 0.25,0,0:0.5 --puncture -0.25,0,0:0.5 runs,
// on Brill-Lindquist data sampled as TwoPunctures gives it for punctures
// without momentum or spin (spacing 0.05). With the defaults, both individual
// horizons and the common horizon are found from the first attempt, and the
// common horizon validates. Analytic irreducible masses (lMax 16): 0.749987
// for each individual horizon, 0.998248 for the common one; this grid gives
// them to 6e-4 and 1e-5. The test runs at lMax 4, not the program's default
// of 8 with punctures, which takes about ten times longer in a debug build.
void TestBrillLindquistPunctures() {
  const BrillLindquist binary{
    {{0.5, {0.25, 0.0, 0.0}}, {0.5, {-0.25, 0.0, 0.0}}}};
  const GridData grid = MakeBrillLindquistGrid(binary, 0.9, 36);
  const PostprocessOptions options = ParsePostprocessArguments({
    "--input", "a.vtu", "--puncture", "0.25,0,0:0.5",
    "--puncture", "-0.25,0,0:0.5", "--lmax", "4"});
  const double supportRadius = 3.0 * FinestGridSpacing(grid);
  const AHHorizonCatalogue catalogue = FindApparentHorizons(
    grid, MakePostprocessCatalogueOptions(options, supportRadius));

  assert(catalogue.individuals.allTargetsFound);
  for (const AHHorizonTargetResult& target : catalogue.individuals.targets) {
    assert(target.successfulAttempt == 0);
    assert(std::abs(target.horizon->irreducibleMass / 0.749987 - 1.0) <
           2.0e-3);
  }
  assert(catalogue.commonHorizons.size() == 1);
  const AHCommonHorizonCatalogueEntry& common = catalogue.commonHorizons[0];
  assert(common.memberIds == (std::vector<std::string>{"bh0", "bh1"}));
  assert(common.reason.triggeredPair);
  assert(common.search.successfulAttempt == 0);
  assert(common.validation.status ==
         AHCommonHorizonValidationStatus::FullyValidated);
  assert(std::abs(common.search.horizon->irreducibleMass / 0.998248 - 1.0) <
         2.0e-4);

  std::cout << "On Brill-Lindquist data, the --puncture defaults find and validate all three "
               "horizons"
            << std::endl;
}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests::postprocess

int main() {
  using namespace exahype2::ccz4::ahf::tests::postprocess;

  std::cout << "Apparent Horizon Finder postprocess option tests begin..." << std::endl;

  TestDefaults();
  TestEveryOption();
  TestHelp();
  TestInvalidArguments();
  TestSourceMapping();
  TestFinestGridSpacing();
  TestSearchOptions();
  TestPunctures();
  TestCatalogueOptions();
  TestBrillLindquistPunctures();

  std::cout << "All postprocess option tests completed." << std::endl;
  return 0;
}
