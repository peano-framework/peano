#include "../require-asserts.h"

#include "../../apparent-horizon-finder.h"
#include "../../file-input.h"

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace exahype2::ccz4::ahf::tests::integration {
namespace {

const char* StatusName(AHCommonHorizonValidationStatus status) {
  switch (status) {
    case AHCommonHorizonValidationStatus::FullyValidated: return "FullyValidated";
    case AHCommonHorizonValidationStatus::ProvisionallyValidated: return "ProvisionallyValidated";
    case AHCommonHorizonValidationStatus::Rejected: return "Rejected";
  }
  return "unknown";
}

const char* EnclosureName(AHEnclosureValidationStatus status) {
  switch (status) {
    case AHEnclosureValidationStatus::Passed: return "Passed";
    case AHEnclosureValidationStatus::Failed: return "Failed";
    case AHEnclosureValidationStatus::Unavailable: return "Unavailable";
  }
  return "unknown";
}

struct DatasetConfiguration {
  std::string filename;
  std::string outputName;
  double separation;
  double punctureAdmMass;
  double supportRadius;
};

std::string DatasetPath(const std::string& filename) {
  const std::string source = __FILE__;
  const std::string directory = source.substr(0, source.find_last_of("/\\"));
  return directory + "/../../ccz4Results/" + filename;
}

// The GitLab-era snapshots in ccz4Results/ come from the legacy ccz4.py, which
// stores traceK at 53 and phi at 54; current exagrype.api output does not.
SourceVariableMapping SourceMapping() {
  return LegacyCCZ4SourceMapping();
}

AHCommonHorizonSearchOptions CatalogueOptions(
  const DatasetConfiguration& dataset
) {
  const ReferencePoint plus{dataset.separation / 2.0, 0.0, 0.0};
  const ReferencePoint minus{-dataset.separation / 2.0, 0.0, 0.0};
  AHCommonHorizonSearchOptions options;
  options.blackHoles = {
    {.id = "individual-plus", .mass = dataset.punctureAdmMass,
     .attempts = {{plus, 0.2}, {plus, 0.12}, {plus, 0.3}},
     .physicalPosition = plus},
    {.id = "individual-minus", .mass = dataset.punctureAdmMass,
     .attempts = {{minus, 0.2}, {minus, 0.12}, {minus, 0.3}},
     .physicalPosition = minus}
  };
  // Exercise the common solver independently of the directional-gap trigger.
  options.candidates.unconditionalAllBlackHoles = true;
  options.commonTargets = {
    .initialRadiusInflation = 1.0,
    .retryRadiusMultipliers = {1.0, 1.5, 2.0}
  };
  options.validation = {
    .spectralResidualTolerance = 1.0e-8,
    .pointwiseExpansionTolerance = 1.0e-4,
    .enclosureTolerance = 1.0e-8,
    .enclosureRhoPoints = 64,
    .enclosureSigmaPoints = 33,
    .maximumRadiusRatio = 10.0
  };
  options.lMax = 6;
  options.interpolation = {
    .method = CartesianToSurfaceMethod::RadiusWeightedPolynomial,
    .polynomialDegree = 3,
    .supportRadius = dataset.supportRadius,
    .rankToleranceMultiplier = 64.0,
    .allowOrderReduction = false
  };
  options.solver = {
    .tolerance = 1.0e-8,
    .massScale = 1.0,
    .maxIterations = 40,
    .finiteDifferenceStep = 1.0e-6,
    .maxLineSearchSteps = 20,
    .maxRadiusFactor = 6.0
  };
  return options;
}

AHMultiHorizonOutputOptions DatasetOutput(const std::string& name) {
  return {.summaryFilename = name + "-summary.txt",
          .horizonFilenamePrefix = name,
          .visualization = {32, 17}};
}

bool KeepTestOutputs() {
  const char* value = std::getenv("AHF_KEEP_TEST_OUTPUTS");
  return value != nullptr && std::string(value) == "1";
}

std::string SafeId(std::string id) {
  for (char& character : id) {
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '-' && character != '_' && character != '.') {
      character = '_';
    }
  }
  return id;
}

std::string IndividualBase(const AHMultiHorizonOutputOptions& output,
                           std::size_t index, const std::string& id) {
  return output.horizonFilenamePrefix + "-individual-" +
    std::to_string(index) + "-" + SafeId(id);
}

std::string CommonBase(const AHMultiHorizonOutputOptions& output,
                       std::size_t index, const std::string& id) {
  return output.horizonFilenamePrefix + "-common-" +
    std::to_string(index) + "-" + SafeId(id);
}

void RemoveOutput(const AHMultiHorizonOutputOptions& output,
                  const AHHorizonCatalogue& catalogue) {
  if (KeepTestOutputs()) {
    std::cout << "[AHF] Keeping N-hole catalogue outputs in "
              << std::filesystem::current_path() << "\n";
    return;
  }
  std::filesystem::remove(output.summaryFilename);
  for (std::size_t i = 0; i < catalogue.individuals.targets.size(); ++i) {
    const std::string base = IndividualBase(
      output, i, catalogue.individuals.targets[i].id);
    std::filesystem::remove(base + ".txt");
    std::filesystem::remove(base + ".vtp");
  }
  for (std::size_t i = 0; i < catalogue.commonHorizons.size(); ++i) {
    const std::string base = CommonBase(
      output, i, catalogue.commonHorizons[i].id);
    std::filesystem::remove(base + ".txt");
    std::filesystem::remove(base + ".vtp");
  }
}

void ReportCatalogue(const DatasetConfiguration& dataset,
                     const AHHorizonCatalogue& catalogue) {
  std::cout << "[AHF] D=" << dataset.separation << " individual outcomes:\n";
  for (const auto& target : catalogue.individuals.targets) {
    std::cout << "  " << target.id << ": converged=" << target.converged
              << " attempts=" << target.attempts.size();
    if (target.horizon.has_value()) {
      std::cout << " meanRadius=" << target.horizon->meanRadius
                << " residualInf=" << target.horizon->spectralResidualInfinityNorm
                << " maxAbsExpansion=" << target.horizon->maxAbsExpansion;
    }
    std::cout << '\n';
  }
  for (const auto& entry : catalogue.commonHorizons) {
    std::cout << "[AHF] " << entry.id
              << ": converged=" << entry.search.converged
              << " status=" << StatusName(entry.validation.status)
              << " numerical=" << entry.validation.numericallyValid
              << " memberEnclosure="
              << EnclosureName(entry.validation.memberHorizonEnclosure)
              << " physicalEnclosure="
              << EnclosureName(entry.validation.physicalPositionEnclosure);
    if (entry.search.horizon.has_value()) {
      const auto& h = *entry.search.horizon;
      std::cout << " radii=" << h.minRadius << '/' << h.meanRadius << '/'
                << h.maxRadius << " residualInf="
                << h.spectralResidualInfinityNorm << " maxAbsExpansion="
                << h.maxAbsExpansion << " memberClearance="
                << entry.validation.minimumEnclosureClearance
                << " physicalClearance="
                << entry.validation.minimumPhysicalPositionClearance;
    }
    std::cout << '\n';
  }
}

const AHCommonHorizonCatalogueEntry& OnlyCommon(
  const AHHorizonCatalogue& catalogue
) {
  assert(catalogue.commonHorizons.size() == 1);
  const auto& common = catalogue.commonHorizons.front();
  assert(common.id == "common[individual-minus+individual-plus]");
  assert(common.memberIds ==
         std::vector<std::string>({"individual-minus", "individual-plus"}));
  assert(common.reason.unconditionalAllBlackHoles);
  return common;
}

AHHorizonCatalogue RunDataset(const DatasetConfiguration& dataset,
                              AHMultiHorizonOutputOptions& output) {
  std::cout << "[AHF] Loading " << dataset.filename << "...\n";
  GridData grid;
  ReadGridFromFile(DatasetPath(dataset.filename), grid, SourceMapping());
  assert(grid.numberOfPoints() == 373248);
  output = DatasetOutput(dataset.outputName);
  AHHorizonCatalogue catalogue = WriteAHToFile(
    grid, CatalogueOptions(dataset), output);
  ReportCatalogue(dataset, catalogue);
  assert(std::filesystem::is_regular_file(output.summaryFilename));
  assert(catalogue.individuals.targets.size() == 2);
  for (std::size_t i = 0; i < catalogue.individuals.targets.size(); ++i) {
    const auto& target = catalogue.individuals.targets[i];
    assert(!target.attempts.empty());
    assert(std::filesystem::is_regular_file(
      IndividualBase(output, i, target.id) + ".txt"));
  }
  assert(std::filesystem::is_regular_file(
    CommonBase(output, 0, OnlyCommon(catalogue).id) + ".txt"));
  return catalogue;
}

void TestSeparatedCatalogue() {
  // parameters.json: t=0, par_b=1, punctures (+/-1,0,0), bare masses
  // 0.46477, inactive target masses 0.5, reported puncture ADM masses 0.525373.
  const DatasetConfiguration dataset{
    "two-punctures-D-2-t0.vtu", "ahf-two-punctures-separated",
    2.0, 0.525373, 0.2};
  AHMultiHorizonOutputOptions output;
  const AHHorizonCatalogue catalogue = RunDataset(dataset, output);
  assert(catalogue.individuals.allTargetsFound);
  for (const auto& target : catalogue.individuals.targets) {
    assert(target.converged && target.horizon.has_value());
    assert(target.horizon->surfaceAvailable);
    assert(target.horizon->meanRadius > 0.19);
    assert(target.horizon->meanRadius < 0.22);
  }
  const auto& common = OnlyCommon(catalogue);
  assert(catalogue.candidatePlan.pairGaps.size() == 1);
  assert(!catalogue.candidatePlan.pairGaps[0].proximityTriggered);
  assert(!common.reason.triggeredPair);
  assert(!common.reason.connectedComponent);
  // Record the unconditional search outcome without treating failure as proof
  // that a common horizon is physically absent.
  const std::string vtp = CommonBase(output, 0, common.id) + ".vtp";
  if (common.validation.status ==
      AHCommonHorizonValidationStatus::FullyValidated) {
    assert(std::filesystem::is_regular_file(vtp));
  } else {
    assert(!std::filesystem::exists(vtp));
  }
  RemoveOutput(output, catalogue);
}

void TestCloseCatalogue() {
  // parameters.json: t=0, par_b=0.125, punctures (+/-0.125,0,0), bare masses
  // 0.46477, inactive target masses 0.5, reported puncture ADM masses 0.897452.
  const DatasetConfiguration dataset{
    "two-punctures-D-0.25-t0.vtu", "ahf-two-punctures-close",
    0.25, 0.897452, 0.15};
  AHMultiHorizonOutputOptions output;
  const AHHorizonCatalogue catalogue = RunDataset(dataset, output);
  const auto& common = OnlyCommon(catalogue);
  assert(catalogue.candidatePlan.pairGaps.size() == 1);
  assert(catalogue.candidatePlan.pairGaps[0].proximityTriggered);
  assert(common.reason.triggeredPair);
  assert(common.reason.connectedComponent);
  assert(common.search.converged && common.search.horizon.has_value());
  assert(common.validation.numericallyValid);
  assert(common.validation.status ==
           AHCommonHorizonValidationStatus::FullyValidated ||
         common.validation.status ==
           AHCommonHorizonValidationStatus::ProvisionallyValidated);
  assert(common.validation.physicalPositionEnclosure ==
         AHEnclosureValidationStatus::Passed);

  const auto& horizon = *common.search.horizon;
  assert(horizon.surfaceAvailable);
  assert(std::isfinite(horizon.spectralResidualInfinityNorm));
  assert(horizon.spectralResidualInfinityNorm <= 1.0e-8);
  assert(std::isfinite(horizon.maxAbsExpansion));
  assert(horizon.maxAbsExpansion <= 1.0e-4);
  assert(horizon.minRadius > 0.0);
  assert(horizon.minRadius <= horizon.meanRadius);
  assert(horizon.meanRadius <= horizon.maxRadius);
  assert(std::abs(horizon.meanRadius - 0.46142331149) < 2.0e-3);
  assert(std::isfinite(common.validation.minimumPhysicalPositionClearance));
  assert(common.validation.minimumPhysicalPositionClearance >= -1.0e-8);

  if (catalogue.individuals.allTargetsFound) {
    assert(common.validation.status ==
           AHCommonHorizonValidationStatus::FullyValidated);
    assert(common.validation.memberHorizonEnclosure ==
           AHEnclosureValidationStatus::Passed);
    assert(std::isfinite(common.validation.minimumEnclosureClearance));
  } else {
    assert(common.validation.status ==
           AHCommonHorizonValidationStatus::ProvisionallyValidated);
    assert(common.validation.memberHorizonEnclosure ==
           AHEnclosureValidationStatus::Unavailable);
  }

  const std::string vtp = CommonBase(output, 0, common.id) + ".vtp";
  if (common.validation.status ==
      AHCommonHorizonValidationStatus::FullyValidated) {
    assert(std::filesystem::is_regular_file(vtp));
  } else {
    assert(!std::filesystem::exists(vtp));
  }
  RemoveOutput(output, catalogue);
}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests::integration

int main() {
  using namespace exahype2::ccz4::ahf::tests::integration;
  TestSeparatedCatalogue();
  TestCloseCatalogue();
}
