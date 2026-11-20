#include "../require-asserts.h"

#include "../../apparent-horizon-finder.h"
#include "../temporary-directory.h"

#include <cassert>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace exahype2::ccz4::ahf::tests {
namespace {

AHFinderResult SphereResult(double radius, int lMax = 2, bool converged = true) {
  const SpectralBasis basis = BuildSpectralBasis(lMax);
  AHFinderResult result;
  result.solver.coefficients = SphereCoefficients(basis, radius);
  result.solver.converged = converged;
  result.solver.iterations = 3;
  result.solver.residualHistory = {1.0, 0.25, 1.0e-12};
  result.solver.message = converged ? "converged" : "line search failed";
  result.referencePoint = {0.25, -0.5, 0.75};
  result.lMax = lMax;
  result.minRadius = radius;
  result.meanRadius = radius;
  result.maxRadius = radius;
  result.meanCoordinateRadius = radius;
  result.area = 4.0 * std::numbers::pi * radius * radius;
  result.arealRadius = radius;
  result.irreducibleMass = 0.5 * radius;
  result.spectralResidualInfinityNorm = 1.0e-12;
  result.maxAbsExpansion = 2.0e-12;
  result.maxAbsExpansionOverResolved = 0.25;
  result.relativeExpansionError = 0.25 * radius / 2.0;
  result.surfaceAvailable = true;
  return result;
}

GridData MakeConstantHorizonGrid() {
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

AHFinderOptions FinderOptions(int maxIterations = 40) {
  AHFinderOptions options;
  options.referencePoint = {0.0, 0.0, 0.0};
  options.initialRadius = 1.3;
  options.lMax = 0;
  options.interpolation = {
    .method = CartesianToSurfaceMethod::UnweightedPolynomial,
    .numberOfNeighbours = 64,
    .polynomialDegree = 1,
    .allowOrderReduction = false
  };
  options.solver.maxIterations = maxIterations;
  return options;
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

void TestSummary() {
  const TemporaryDirectory temporary("ahf-output-tests");
  const auto& directory = temporary.path();
  const auto first = directory / "ahf-output-summary-1.txt";
  const auto second = directory / "ahf-output-summary-2.txt";
  const AHFinderResult result = SphereResult(2.5);
  WriteAHSummary(result, first.string());
  WriteAHSummary(result, second.string());
  const std::string text = ReadText(first);
  assert(text == ReadText(second));
  assert(text.find("solver.converged = true") != std::string::npos);
  assert(text.find("referencePoint = 0.25 -0.5 0.75") != std::string::npos);
  assert(text.find("meanRadius = 2.5") != std::string::npos);
  assert(text.find("\nmeanCoordinateRadius = 2.5\n") != std::string::npos);
  assert(text.find("\narea = 78.5") != std::string::npos);
  assert(text.find("\narealRadius = 2.5\n") != std::string::npos);
  assert(text.find("\nirreducibleMass = 1.25\n") != std::string::npos);
  assert(text.find("\nmaxAbsExpansionOverResolved = 0.25\n") != std::string::npos);
  assert(text.find("\nrelativeExpansionError = 0.3125\n") != std::string::npos);
  assert(text.find("residualHistory[2]") != std::string::npos);
  assert(text.find("coefficients[8]") != std::string::npos);

  AHFinderResult failure = result;
  failure.solver.converged = false;
  failure.surfaceAvailable = false;
  failure.solver.message = "controlled failure";
  WriteAHSummary(failure, second.string());
  const std::string failureText = ReadText(second);
  assert(failureText.find("solver.converged = false") != std::string::npos);
  assert(failureText.find("surfaceAvailable = false") != std::string::npos);
  assert(failureText.find("minRadius = unavailable") != std::string::npos);
  assert(failureText.find("\narea = unavailable\n") != std::string::npos);
  assert(failureText.find("\nirreducibleMass = unavailable\n") != std::string::npos);
  assert(failureText.find("\nmaxAbsExpansionOverResolved = unavailable\n") != std::string::npos);
  assert(failureText.find("\nrelativeExpansionError = unavailable\n") != std::string::npos);
  assert(failureText.find("controlled failure") != std::string::npos);
  std::filesystem::remove(first);
  std::filesystem::remove(second);
}

void TestDenseSphereGeometry() {
  constexpr int nRho = 12;
  constexpr int nSigma = 7;
  const AHFinderResult result = SphereResult(2.5, 1);
  const AHVisualizationSurface surface =
    BuildAHVisualizationSurface(result, {nRho, nSigma});
  assert(surface.points.size() == 2 + (nSigma - 2) * nRho);
  assert(surface.triangles.size() == 2 * (nSigma - 2) * nRho);
  assert(surface.radius.size() == surface.points.size());
  for (std::size_t i = 0; i < surface.points.size(); ++i) {
    assert(std::abs(surface.radius[i] - 2.5) < 1.0e-12);
    const auto& p = surface.points[i];
    assert(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]));
    const double distance = std::sqrt(
      std::pow(p[0] - result.referencePoint.x, 2) +
      std::pow(p[1] - result.referencePoint.y, 2) +
      std::pow(p[2] - result.referencePoint.z, 2));
    assert(std::abs(distance - 2.5) < 1.0e-12);
  }

  std::map<std::pair<std::size_t, std::size_t>, int> edgeUses;
  for (const auto& triangle : surface.triangles) {
    for (std::size_t vertex : triangle) assert(vertex < surface.points.size());
    assert(triangle[0] != triangle[1] && triangle[1] != triangle[2] &&
           triangle[0] != triangle[2]);
    for (int edge = 0; edge < 3; ++edge) {
      std::size_t a = triangle[edge];
      std::size_t b = triangle[(edge + 1) % 3];
      if (a > b) std::swap(a, b);
      ++edgeUses[{a, b}];
    }
  }
  for (const auto& [edge, count] : edgeUses) {
    (void)edge;
    assert(count == 2); // closed surface, including the periodic rho seam
  }

  const auto denser = BuildAHVisualizationSurface(result, {19, 9});
  assert(denser.points.size() != surface.points.size());
  assert(result.lMax == 1); // visualization resolution did not alter the solve

  bool threw = false;
  try { (void)BuildAHVisualizationSurface(result, {2, 7}); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
}

void TestVTP() {
  const TemporaryDirectory temporary("ahf-output-tests");
  const auto path = temporary.path() / "ahf-output-sphere.vtp";
  const AHFinderResult result = SphereResult(1.75);
  WriteAHSurfaceToVTP(result, path.string(), {10, 6});
  assert(std::filesystem::is_regular_file(path));
  const std::string xml = ReadText(path);
  assert(xml.find("<VTKFile type=\"PolyData\"") != std::string::npos);
  assert(xml.find("NumberOfPoints=\"42\"") != std::string::npos);
  assert(xml.find("NumberOfPolys=\"80\"") != std::string::npos);
  assert(xml.find("Name=\"radius\"") != std::string::npos);
  assert(xml.find("Name=\"connectivity\"") != std::string::npos);
  assert(xml.find("Name=\"offsets\"") != std::string::npos);

  AHFinderResult nonconverged = result;
  nonconverged.solver.converged = false;
  bool threw = false;
  try { WriteAHSurfaceToVTP(nonconverged, path.string()); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
  std::filesystem::remove(path);
}

void TestFacade() {
  const GridData grid = MakeConstantHorizonGrid();
  const TemporaryDirectory temporary("ahf-output-tests");
  const auto& directory = temporary.path();
  const auto summary = directory / "ahf-facade-summary.txt";
  const auto surface = directory / "ahf-facade-surface.vtp";
  std::filesystem::remove(summary);
  std::filesystem::remove(surface);

  const AHOutputOptions output{
    .summaryFilename = summary.string(),
    .surfaceFilename = surface.string(),
    .visualization = {11, 8}
  };
  const AHFinderResult result = WriteAHToFile(grid, FinderOptions(), output);
  assert(result.solver.converged);
  assert(result.surfaceAvailable);
  assert(result.solver.coefficients.size() == 1);
  assert(std::filesystem::is_regular_file(summary));
  assert(std::filesystem::is_regular_file(surface));
  const std::string summaryText = ReadText(summary);
  const std::string surfaceText = ReadText(surface);
  assert(summaryText.find("solver.converged = true") != std::string::npos);
  // The facade grid is flat (phi = 1) with K_ij = 2 delta_ij and K = 3, so
  // Theta = 2/r - 1 and the horizon is the sphere of radius 2: mass 1.
  assert(std::abs(result.arealRadius - 2.0) < 1.0e-6);
  assert(std::abs(result.irreducibleMass - 1.0) < 1.0e-6);
  assert(summaryText.find("\nirreducibleMass = ") != std::string::npos);
  // 2 poles + 6 rings * 11 rho points; 2 * 6 * 11 triangles.
  assert(surfaceText.find("NumberOfPoints=\"68\"") != std::string::npos);
  assert(surfaceText.find("NumberOfPolys=\"132\"") != std::string::npos);
  std::filesystem::remove(summary);
  std::filesystem::remove(surface);
}

void TestFacadeNonConvergence() {
  const GridData grid = MakeConstantHorizonGrid();
  const TemporaryDirectory temporary("ahf-output-tests");
  const auto& directory = temporary.path();
  const auto summary = directory / "ahf-facade-failure.txt";
  const auto surface = directory / "ahf-facade-failure.vtp";
  std::filesystem::remove(summary);
  std::filesystem::remove(surface);

  const AHFinderResult result = WriteAHToFile(
    grid,
    FinderOptions(0),
    {.summaryFilename = summary.string(), .surfaceFilename = surface.string()}
  );
  assert(!result.solver.converged);
  assert(std::filesystem::is_regular_file(summary));
  assert(ReadText(summary).find("solver.converged = false") != std::string::npos);
  assert(ReadText(summary).find("max_iter reached") != std::string::npos);
  assert(!std::filesystem::exists(surface));
  std::filesystem::remove(summary);
}

void TestFacadeValidationAndIOFailures() {
  const GridData grid = MakeConstantHorizonGrid();
  bool threw = false;
  AHFinderOptions invalidFinder = FinderOptions();
  invalidFinder.initialRadius = 0.0;
  try { (void)WriteAHToFile(grid, invalidFinder); }
  catch (const std::invalid_argument&) { threw = true; }
  assert(threw);

  threw = false;
  try {
    (void)WriteAHToFile(
      grid, FinderOptions(),
      {.summaryFilename = "", .surfaceFilename = "surface.vtp"});
  } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);

  threw = false;
  try {
    (void)WriteAHToFile(
      grid, FinderOptions(),
      {.summaryFilename = "summary.txt",
       .surfaceFilename = "surface.vtp",
       .visualization = {2, 8}});
  } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);

  threw = false;
  try {
    (void)WriteAHToFile(
      grid, FinderOptions(),
      {.summaryFilename = "/definitely-missing-ahf-directory/summary.txt",
       .surfaceFilename = "/definitely-missing-ahf-directory/surface.vtp"});
  } catch (const std::runtime_error&) { threw = true; }
  assert(threw);
}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests

int main() {
  exahype2::ccz4::ahf::tests::TestSummary();
  exahype2::ccz4::ahf::tests::TestDenseSphereGeometry();
  exahype2::ccz4::ahf::tests::TestVTP();
  exahype2::ccz4::ahf::tests::TestFacade();
  exahype2::ccz4::ahf::tests::TestFacadeNonConvergence();
  exahype2::ccz4::ahf::tests::TestFacadeValidationAndIOFailures();
}
