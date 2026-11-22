#include "ahf-postprocess-options.h"
#include "file-input.h"
#include "output.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace exahype2::ccz4::ahf;

namespace {

constexpr int HorizonNotFound = 1;
constexpr int Error = 2;

void PrintAttempts(const std::vector<AHSearchAttemptResult>& attempts) {
  for (std::size_t attempt = 0; attempt < attempts.size(); ++attempt) {
    const AHSearchAttemptResult& result = attempts[attempt];
    std::cout << "Attempt " << attempt << ": initial radius " << result.initialRadius << ", "
              << (result.result.solver.converged ? "converged" : "failed")
              << " after " << result.result.solver.iterations << " iterations";
    if (!result.rejection.empty()) {
      std::cout << ", rejected: " << result.rejection;
    } else if (!result.succeeded && !result.result.solver.message.empty()) {
      std::cout << " (" << result.result.solver.message << ")";
    }
    std::cout << "\n";
  }
}

void PrintCentre(const ReferencePoint& point) {
  std::cout << "(" << point.x << ", " << point.y << ", " << point.z << ")";
}

void PrintHorizon(const AHFinderResult& horizon) {
  std::cout << "  coordinate radius min/mean/max: " << horizon.minRadius << " / "
            << horizon.meanRadius << " / " << horizon.maxRadius << "\n"
            << "  solid-angle mean radius:        " << horizon.meanCoordinateRadius << "\n"
            << "  area:                           " << horizon.area << "\n"
            << "  areal radius:                   " << horizon.arealRadius << "\n"
            << "  irreducible mass:               " << horizon.irreducibleMass << "\n"
            << "  spectral residual (max norm):   " << horizon.spectralResidualInfinityNorm << "\n"
            << "  max |expansion| on the surface: " << horizon.maxAbsExpansion << "\n"
            << "  relative expansion error:       " << horizon.relativeExpansionError << "\n";
}

const char* ValidationStatus(AHCommonHorizonValidationStatus status) {
  switch (status) {
    case AHCommonHorizonValidationStatus::FullyValidated:
      return "fully validated";
    case AHCommonHorizonValidationStatus::ProvisionallyValidated:
      return "provisionally validated";
    case AHCommonHorizonValidationStatus::Rejected:
      break;
  }
  return "rejected";
}

int FindOneHorizon(const GridData& grid, const PostprocessOptions& options,
                   double supportRadius) {
  const AHOutputOptions output{
    .summaryFilename = options.outputPrefix + ".txt",
    .surfaceFilename = options.outputPrefix + ".vtp"
  };
  const AHSearchResult search = WriteAHToFile(
    grid, MakePostprocessSearchOptions(options, supportRadius), output);

  PrintAttempts(search.attempts);
  if (!search.converged || !search.horizon) {
    std::cout << "No apparent horizon found. Summary: "
              << output.summaryFilename << "\n";
    return HorizonNotFound;
  }
  std::cout << "Apparent horizon found around ";
  PrintCentre(search.horizon->referencePoint);
  std::cout << "\n";
  PrintHorizon(*search.horizon);
  std::cout << "Summary: " << output.summaryFilename << "\n"
            << "Surface: " << output.surfaceFilename << "\n";
  return EXIT_SUCCESS;
}

/**
 * Every puncture's horizon, then the common horizons planned from them. Only
 * the individual horizons decide the exit status: a common search also runs
 * for pairs a little beyond the separation where a common horizon exists, and
 * finding nothing there is the expected outcome.
 */
int FindHorizonsOfPunctures(const GridData& grid,
                            const PostprocessOptions& options,
                            double supportRadius) {
  const AHCommonHorizonSearchOptions search =
    MakePostprocessCatalogueOptions(options, supportRadius);
  const AHMultiHorizonOutputOptions output{
    .summaryFilename = options.outputPrefix + ".txt",
    .horizonFilenamePrefix = options.outputPrefix
  };
  const AHHorizonCatalogue catalogue = WriteAHToFile(grid, search, output);

  for (std::size_t i = 0; i < catalogue.individuals.targets.size(); ++i) {
    const AHHorizonTargetResult& target = catalogue.individuals.targets[i];
    std::cout << "Puncture " << target.id << " at ";
    PrintCentre(options.punctures[i].position);
    std::cout << ", mass " << options.punctures[i].mass << "\n";
    PrintAttempts(target.attempts);
    if (!target.converged || !target.horizon) {
      std::cout << "No individual horizon found for " << target.id << "\n";
      continue;
    }
    std::cout << "Individual horizon of " << target.id << "\n";
    PrintHorizon(*target.horizon);
    if (target.horizon->surfaceAvailable) {
      std::cout << "  surface: " << AHCatalogueFilename(
        output, "individual", i, target.id, ".vtp") << "\n";
    }
  }

  for (const AHHorizonPairGap& pair : catalogue.candidatePlan.pairGaps) {
    std::cout << "Pair " << pair.firstId << "-" << pair.secondId
              << ": individual horizons " << pair.surfaceGap << " apart, "
              << (pair.proximityTriggered ? "close enough" : "too far apart")
              << " for a common search\n";
  }

  for (std::size_t i = 0; i < catalogue.commonHorizons.size(); ++i) {
    const AHCommonHorizonCatalogueEntry& entry = catalogue.commonHorizons[i];
    std::cout << "Common horizon " << entry.id << "\n";
    PrintAttempts(entry.search.attempts);
    if (!entry.search.converged || !entry.search.horizon) {
      std::cout << "No common horizon found for " << entry.id << "\n";
      continue;
    }
    std::cout << "Common horizon " << entry.id << ": "
              << ValidationStatus(entry.validation.status) << "\n";
    for (const std::string& reason : entry.validation.reasons) {
      std::cout << "  " << reason << "\n";
    }
    PrintHorizon(*entry.search.horizon);
    if (entry.validation.status ==
        AHCommonHorizonValidationStatus::FullyValidated) {
      std::cout << "  surface: " << AHCatalogueFilename(
        output, "common", i, entry.id, ".vtp") << "\n";
    }
  }

  std::cout << "Summary: " << output.summaryFilename << "\n";
  return catalogue.individuals.allTargetsFound ? EXIT_SUCCESS : HorizonNotFound;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string executable = std::filesystem::path(argv[0]).filename().string();
  PostprocessOptions options;
  try {
    options = ParsePostprocessArguments(std::vector<std::string>(argv + 1, argv + argc));
  } catch (const std::invalid_argument& error) {
    std::cerr << executable << ": " << error.what() << "\n\n" << PostprocessUsage(executable);
    return Error;
  }
  if (options.help) {
    std::cout << PostprocessUsage(executable);
    return EXIT_SUCCESS;
  }

  try {
    GridData grid;
    const auto mapping = PostprocessSourceMapping(options.mapping);
    if (mapping) {
      ReadGridFromFile(options.inputFile, grid, *mapping);
    } else {
      ReadGridFromFile(options.inputFile, grid);
    }
    std::cout << "Read " << grid.numberOfPoints() << " cells from " << options.inputFile << "\n";

    double supportRadius = 0.0;
    if (options.supportRadius) {
      supportRadius = *options.supportRadius;
      std::cout << "Interpolation support radius " << supportRadius << "\n";
    } else {
      const double spacing = FinestGridSpacing(grid);
      supportRadius = 3.0 * spacing;
      std::cout << "Interpolation support radius " << supportRadius
                << " (three times the finest cell spacing, " << spacing << ")\n";
    }

    return options.punctures.empty()
      ? FindOneHorizon(grid, options, supportRadius)
      : FindHorizonsOfPunctures(grid, options, supportRadius);
  } catch (const std::exception& error) {
    std::cerr << executable << ": " << error.what() << "\n";
    return Error;
  }
}
