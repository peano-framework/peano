#include "../require-asserts.h"

#include "../../apparent-horizon-finder.h"
#include "../../file-input.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace exahype2::ccz4::ahf::tests::integration {
namespace {

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

std::string DatasetPath() {
  const std::string source = __FILE__;
  const std::string directory = source.substr(0, source.find_last_of("/\\"));
  return directory + "/../../ccz4Results/sample_t=0.vtu";
}

// The GitLab-era snapshots in ccz4Results/ come from the legacy ccz4.py, which
// stores traceK at 53 and phi at 54; current exagrype.api output does not.
SourceVariableMapping SourceMapping() {
  return LegacyCCZ4SourceMapping();
}

void TestSampleT0UserWorkflow() {
  const auto totalStart = Clock::now();

  std::cout << "[AHF] Preparing AH finder and output options..."
            << std::endl;
  const auto setupStart = Clock::now();
  AHSearchOptions searchOptions;
  searchOptions.attempts = {
    {{8.0, 3.0, 1.0}, 200.0}, // deliberately diverged initial guess
    {{10.0, 0.0, 0.0}, 0.2}, // deliberately outside the sampled domain
    {{0.0, 0.0, 0.0}, 0.9}, // expected to succeed
    {{0.0, 0.0, 0.0}, 0.7}, // expected to succeed, but will be skipped because the previous attempt already succeeded
    {{0.0, 0.0, 0.0}, 0.5} // expected to succeed, but will be skipped because the previous attempt already succeeded
  };
  searchOptions.lMax = 2;
  searchOptions.interpolation = {
    .method = CartesianToSurfaceMethod::RadiusWeightedPolynomial,
    .polynomialDegree = 3,
    .supportRadius = 0.1,
    .rankToleranceMultiplier = 64.0,
    .allowOrderReduction = false
  };
  searchOptions.solver = {
    .tolerance = 1.0e-8,
    .massScale = 1.0,
    .maxIterations = 12,
    .finiteDifferenceStep = 1.0e-6,
    .maxLineSearchSteps = 20,
    .maxRadiusFactor = 6.0
  };
  const std::filesystem::path outputDirectory =
    std::filesystem::current_path();
  const auto summaryPath = outputDirectory / "sample-t0-ah.txt";
  const auto surfacePath = outputDirectory / "sample-t0-ah.vtp";
  const AHOutputOptions outputOptions{
    .summaryFilename = summaryPath.string(),
    .surfaceFilename = surfacePath.string(),
    .visualization = {32, 17}
  };
  const auto setupFinish = Clock::now();
  std::cout << "[AHF] Option setup took "
            << Seconds(setupStart, setupFinish) << " s." << std::endl;

  std::cout << "[AHF] Loading Cartesian data from sample_t=0.vtu..."
            << std::endl;
  const auto loadStart = Clock::now();
  GridData grid;
  ReadGridFromFile(DatasetPath(), grid, SourceMapping());
  const auto loadFinish = Clock::now();
  std::cout << "[AHF] Loaded " << grid.numberOfPoints()
            << " Cartesian cells in " << Seconds(loadStart, loadFinish)
            << " s." << std::endl;

  std::cout << "[AHF] Running the AH finder and writing its outputs..."
            << std::endl;
  std::cout << "[AHF] Numerical summary: " << summaryPath << std::endl;
  std::cout << "[AHF] ParaView surface: " << surfacePath << std::endl;
  const auto solveStart = Clock::now();
  const AHSearchResult search = WriteAHToFile(
    grid, searchOptions, outputOptions);
  const auto solveFinish = Clock::now();
  assert(search.converged);
  assert(search.successfulAttempt == 2);
  assert(search.attempts.size() == 3);
  assert(!search.attempts[0].succeeded);
  assert(!search.attempts[1].succeeded);
  assert(search.attempts[2].succeeded);
  assert(search.horizon.has_value());
  const AHFinderResult& horizon = *search.horizon;
  std::cout << "[AHF] Iteration converged after " << horizon.solver.iterations
            << " Newton iterations; reconstruction and output took "
            << Seconds(solveStart, solveFinish) << " s." << std::endl;

  constexpr double expectedRadius = 0.5;
  constexpr double radiusTolerance = 1.0e-2;

  std::cout << "[AHF] Finished: mean AH radius = " << horizon.meanRadius
            << ", expected = " << expectedRadius << std::endl;

  std::cout << "[AHF] Total wall-clock time: "
            << Seconds(totalStart, solveFinish) << " s."
            << std::endl;

  assert(horizon.solver.converged);
  assert(horizon.surfaceAvailable);
  assert(horizon.referencePoint.x == searchOptions.attempts[2].referencePoint.x);
  assert(horizon.referencePoint.y == searchOptions.attempts[2].referencePoint.y);
  assert(horizon.referencePoint.z == searchOptions.attempts[2].referencePoint.z);
  assert(std::isfinite(horizon.minRadius));
  assert(std::isfinite(horizon.meanRadius));
  assert(std::isfinite(horizon.maxRadius));
  assert(horizon.minRadius <= horizon.meanRadius);
  assert(horizon.meanRadius <= horizon.maxRadius);
  assert(std::isfinite(horizon.maxAbsExpansion));
  assert(std::isfinite(horizon.spectralResidualInfinityNorm));
  assert(!horizon.solver.coefficients.empty());
  assert(std::abs(horizon.meanRadius - expectedRadius) < radiusTolerance);

  assert(std::filesystem::is_regular_file(summaryPath));
  assert(std::filesystem::file_size(summaryPath) > 0);
  assert(std::filesystem::is_regular_file(surfacePath));
  assert(std::filesystem::file_size(surfacePath) > 0);

  std::ifstream summaryInput(summaryPath);
  std::ostringstream summaryText;
  summaryText << summaryInput.rdbuf();
  assert(summaryText.str().find("converged = true") != std::string::npos);
  assert(summaryText.str().find("successfulAttempt = 2") != std::string::npos);
  assert(summaryText.str().find("attempts[0].succeeded = false") !=
         std::string::npos);
  assert(summaryText.str().find("attempts[1].succeeded = false") !=
         std::string::npos);
  assert(summaryText.str().find("attempts[2].succeeded = true") !=
         std::string::npos);
  assert(summaryText.str().find("horizon.meanRadius = ") != std::string::npos);

  std::ifstream surfaceInput(surfacePath);
  std::ostringstream surfaceText;
  surfaceText << surfaceInput.rdbuf();
  assert(surfaceText.str().find("NumberOfPoints=\"") != std::string::npos);
  assert(surfaceText.str().find("NumberOfPolys=\"") != std::string::npos);
  assert(surfaceText.str().find("nan") == std::string::npos);
  assert(surfaceText.str().find("inf") == std::string::npos);

}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests::integration

int main() {
  exahype2::ccz4::ahf::tests::integration::TestSampleT0UserWorkflow();
}
