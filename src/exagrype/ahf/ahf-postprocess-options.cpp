#include "ahf-postprocess-options.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace exahype2::ccz4::ahf {

namespace {

/** Remove a trailing @p extension from @p path, if it has one. */
std::string withoutExtension(const std::string& path, const std::string& extension) {
  if (path.size() > extension.size()
      && path.compare(path.size() - extension.size(), extension.size(), extension) == 0) {
    return path.substr(0, path.size() - extension.size());
  }
  return path;
}

std::vector<std::string> splitAtCommas(const std::string& text) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, ',')) {
    parts.push_back(part);
  }
  if (!text.empty() && text.back() == ',') {
    parts.push_back("");
  }
  return parts;
}

/** A finite number that takes up all of @p text. */
double parseNumber(const std::string& option, const std::string& text) {
  std::size_t used = 0;
  double value = std::numeric_limits<double>::quiet_NaN();
  try {
    value = std::stod(text, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used == 0 || used != text.size() || !std::isfinite(value)) {
    throw std::invalid_argument(option + " expects a number, not '" + text + "'");
  }
  return value;
}

double parsePositiveNumber(const std::string& option, const std::string& text) {
  const double value = parseNumber(option, text);
  if (value <= 0.0) {
    throw std::invalid_argument(option + " expects a positive number, not '" + text + "'");
  }
  return value;
}

int parsePositiveInteger(const std::string& option, const std::string& text) {
  std::size_t used = 0;
  int value = 0;
  try {
    value = std::stoi(text, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used == 0 || used != text.size() || value <= 0) {
    throw std::invalid_argument(option + " expects a positive integer, not '" + text + "'");
  }
  return value;
}

PostprocessMapping parseMapping(const std::string& text) {
  if (text == "names") {
    return PostprocessMapping::Names;
  }
  if (text == "exagrype") {
    return PostprocessMapping::ExaGRyPE;
  }
  if (text == "legacy") {
    return PostprocessMapping::LegacyCCZ4;
  }
  throw std::invalid_argument(
    "--mapping expects names, exagrype or legacy, not '" + text + "'");
}

ReferencePoint parseCentre(const std::string& option, const std::string& text) {
  const auto parts = splitAtCommas(text);
  if (parts.size() != 3) {
    throw std::invalid_argument(option + " expects x,y,z, not '" + text + "'");
  }
  return {parseNumber(option, parts[0]), parseNumber(option, parts[1]),
          parseNumber(option, parts[2])};
}

PostprocessPuncture parsePuncture(const std::string& option,
                                  const std::string& text) {
  const std::size_t colon = text.find(':');
  if (colon == std::string::npos
      || text.find(':', colon + 1) != std::string::npos
      || splitAtCommas(text.substr(0, colon)).size() != 3) {
    throw std::invalid_argument(
      option + " expects x,y,z:m, not '" + text + "'");
  }
  return {.position = parseCentre(option, text.substr(0, colon)),
          .mass = parsePositiveNumber(option, text.substr(colon + 1))};
}

/** The smallest gap between distinct values of @p values, sorted in place. */
double smallestGap(std::vector<double>& values) {
  std::sort(values.begin(), values.end());
  const double scale = std::max({values.back() - values.front(), std::abs(values.front()),
                                 std::abs(values.back())});
  // Centres on the same plane differ by rounding: render.py stores the points
  // in single precision, about 1e-7 relative. A real cell size is many orders
  // of magnitude above this threshold.
  const double sameValue = 1.0e-5 * scale;
  double gap = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i < values.size(); ++i) {
    const double difference = values[i] - values[i - 1];
    if (difference > sameValue) {
      gap = std::min(gap, difference);
    }
  }
  return gap;
}

}  // namespace

PostprocessOptions ParsePostprocessArguments(const std::vector<std::string>& arguments) {
  PostprocessOptions options;
  const std::vector<std::string> optionsWithValue{
    "--input", "--output", "--mapping", "--centre", "--center", "--radius",
    "--puncture", "--radius-per-mass", "--lmax", "--support-radius",
    "--tolerance"};
  bool centreGiven = false;
  bool radiiPerMassGiven = false;
  bool lMaxGiven = false;
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    const std::string& argument = arguments[i];
    if (argument == "--help" || argument == "-h") {
      options.help = true;
      continue;
    }
    if (std::find(optionsWithValue.begin(), optionsWithValue.end(), argument)
        == optionsWithValue.end()) {
      throw std::invalid_argument(argument + " is not an option that ahf-postprocess knows");
    }
    if (i + 1 >= arguments.size()) {
      throw std::invalid_argument(argument + " needs a value");
    }
    const std::string& value = arguments[++i];
    if (argument == "--input") {
      options.inputFile = value;
    } else if (argument == "--output") {
      options.outputPrefix = withoutExtension(withoutExtension(value, ".vtp"), ".txt");
    } else if (argument == "--mapping") {
      options.mapping = parseMapping(value);
    } else if (argument == "--centre" || argument == "--center") {
      options.centre = parseCentre(argument, value);
      centreGiven = true;
    } else if (argument == "--radius") {
      for (const std::string& radius : splitAtCommas(value)) {
        options.initialRadii.push_back(parsePositiveNumber(argument, radius));
      }
    } else if (argument == "--puncture") {
      options.punctures.push_back(parsePuncture(argument, value));
    } else if (argument == "--radius-per-mass") {
      if (!radiiPerMassGiven) {
        options.radiiPerMass.clear();
        radiiPerMassGiven = true;
      }
      for (const std::string& factor : splitAtCommas(value)) {
        options.radiiPerMass.push_back(parsePositiveNumber(argument, factor));
      }
    } else if (argument == "--lmax") {
      options.lMax = parsePositiveInteger(argument, value);
      lMaxGiven = true;
    } else if (argument == "--support-radius") {
      options.supportRadius = parsePositiveNumber(argument, value);
    } else if (argument == "--tolerance") {
      options.tolerance = parsePositiveNumber(argument, value);
    }
  }

  if (options.help) {
    return options;
  }
  if (options.inputFile.empty()) {
    throw std::invalid_argument("--input is required");
  }
  if (!options.punctures.empty()) {
    if (!options.initialRadii.empty()) {
      throw std::invalid_argument(
        "--radius is for a single search; with --puncture use "
        "--radius-per-mass");
    }
    if (centreGiven) {
      throw std::invalid_argument(
        "--centre is for a single search; --puncture gives the positions");
    }
    if (!lMaxGiven) {
      options.lMax = PostprocessOptions::PunctureLMax;
    }
  } else if (radiiPerMassGiven) {
    throw std::invalid_argument("--radius-per-mass needs --puncture");
  } else if (options.initialRadii.empty()) {
    throw std::invalid_argument("--radius or --puncture is required");
  }
  if (options.outputPrefix.empty()) {
    throw std::invalid_argument("--output must not be empty");
  }
  return options;
}

std::string PostprocessUsage(const std::string& executable) {
  const std::string indent(executable.size() + 7, ' ');
  return "Usage: " + executable + " --input <snapshot> --radius <r>[,<r>...]\n"
    + indent + "[--centre <x,y,z>] [options]\n"
    "       " + executable + " --input <snapshot> --puncture <x,y,z:m>...\n"
    + indent + "[--radius-per-mass <f>[,<f>...]] [options]\n"
    "Options: [--output <prefix>] [--mapping names|exagrype|legacy] [--lmax <n>]\n"
    "         [--support-radius <h>] [--tolerance <t>]\n"
    "\n"
    "Finds apparent horizons in a VTK snapshot (convert a Peano patch file with\n"
    "src/tarch/plotter/api/render.py --filter-fine-grid).\n"
    "\n"
    "With --radius it searches for one horizon: starting from a sphere of each\n"
    "initial radius around the centre in turn, it stops at the first search that\n"
    "converges.\n"
    "\n"
    "With --puncture, given once per black hole with m its mass, it searches for\n"
    "each puncture's own horizon, then for a common horizon around each pair whose\n"
    "horizons are at most 0.5 (m1 + m2) apart, and checks each common horizon\n"
    "found. An individual search rejects a surface that encloses another puncture.\n"
    "\n"
    "  --input           snapshot to read (.vtu, or a .pvtu series)\n"
    "  --radius          initial radii, tried in order; repeat or separate by commas\n"
    "  --centre          centre of the initial spheres; default 0,0,0\n"
    "  --puncture        position and mass of a black hole; repeat for each\n"
    "  --radius-per-mass initial radii of each puncture's search, in units of its\n"
    "                    mass, tried in order; default 0.35,0.5,0.25\n"
    "  --output          one search writes <prefix>.txt (summary) and <prefix>.vtp\n"
    "                    (surface); with --puncture, <prefix>.txt is the catalogue\n"
    "                    and each horizon gets <prefix>-<kind>-<i>-<id>.txt and .vtp;\n"
    "                    default apparent-horizon\n"
    "  --mapping         where the fields are: by component name (names, default),\n"
    "                    exagrype.api layout (exagrype), legacy ccz4.py (legacy)\n"
    "  --lmax            highest spherical-harmonic degree; default 4, or 8 with\n"
    "                    --puncture\n"
    "  --support-radius  interpolation support; default three times the finest\n"
    "                    cell spacing\n"
    "  --tolerance       convergence threshold on the spectral residual; default 1e-8\n"
    "\n"
    "Exit status: 0 if a horizon was found (with --puncture, every puncture's own\n"
    "horizon; common horizons are reported but do not count), 1 if not, 2 on an\n"
    "error.\n";
}

std::optional<SourceVariableMapping> PostprocessSourceMapping(PostprocessMapping mapping) {
  switch (mapping) {
    case PostprocessMapping::ExaGRyPE:
      return ExaGRyPESourceMapping();
    case PostprocessMapping::LegacyCCZ4:
      return LegacyCCZ4SourceMapping();
    case PostprocessMapping::Names:
      break;
  }
  return std::nullopt;
}

double FinestGridSpacing(const GridData& gridData) {
  double spacing = std::numeric_limits<double>::infinity();
  std::vector<double> values(gridData.coordinates.size());
  for (double GridCoordinate::*axis : {&GridCoordinate::x, &GridCoordinate::y, &GridCoordinate::z}) {
    if (values.size() < 2) {
      break;
    }
    std::transform(gridData.coordinates.begin(), gridData.coordinates.end(), values.begin(),
                   [axis](const GridCoordinate& coordinate) { return coordinate.*axis; });
    spacing = std::min(spacing, smallestGap(values));
  }
  if (!std::isfinite(spacing)) {
    throw std::invalid_argument(
      "FinestGridSpacing: no axis has two distinct cell-centre coordinates");
  }
  return spacing;
}

AHSearchOptions MakePostprocessSearchOptions(const PostprocessOptions& options,
                                             double supportRadius) {
  AHSearchOptions search;
  for (const double radius : options.initialRadii) {
    search.attempts.push_back({.referencePoint = options.centre, .initialRadius = radius});
  }
  search.lMax = options.lMax;
  search.interpolation = {
    .method = CartesianToSurfaceMethod::RadiusWeightedPolynomial,
    .polynomialDegree = 3,
    .supportRadius = supportRadius,
    .rankToleranceMultiplier = 64.0,
    .allowOrderReduction = false
  };
  search.solver.tolerance = options.tolerance;
  return search;
}

AHCommonHorizonSearchOptions MakePostprocessCatalogueOptions(
  const PostprocessOptions& options, double supportRadius) {
  const AHSearchOptions single =
    MakePostprocessSearchOptions(options, supportRadius);
  AHCommonHorizonSearchOptions catalogue;
  double smallestMass = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < options.punctures.size(); ++i) {
    const PostprocessPuncture& puncture = options.punctures[i];
    AHPhysicalBlackHole blackHole{.id = "bh" + std::to_string(i),
                                  .mass = puncture.mass,
                                  .physicalPosition = puncture.position};
    for (const double factor : options.radiiPerMass) {
      blackHole.attempts.push_back({.referencePoint = puncture.position,
                                    .initialRadius = factor * puncture.mass});
    }
    catalogue.blackHoles.push_back(std::move(blackHole));
    smallestMass = std::min(smallestMass, puncture.mass);
  }
  catalogue.lMax = single.lMax;
  catalogue.interpolation = single.interpolation;
  catalogue.solver = single.solver;
  if (!options.punctures.empty()) {
    catalogue.validation.spectralResidualTolerance =
      options.tolerance / smallestMass;
  }
  return catalogue;
}

}  // namespace exahype2::ccz4::ahf
