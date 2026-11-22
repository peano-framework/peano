#include "../require-asserts.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <string>

#include "../..//apparent-horizon-finder.h"

namespace exahype2::ccz4::ahf::tests::interpolation {

static_assert(requires(const GridData& gridData,
                       const AHGuess& ahGuess,
                       const ReferencePoint& referencePoint,
                       InterpolatedShell& interpolatedShell,
                       const std::vector<AngularSample>& angularSamples) {
  InterpolateFromCartesianToShell(
    gridData,
    ahGuess,
    referencePoint,
    interpolatedShell
  );
  InterpolateFromCartesianToShell(
    gridData,
    ahGuess,
    referencePoint,
    interpolatedShell,
    angularSamples
  );
});

bool approximatelyEqual(double actual, double expected) {
  constexpr double AbsoluteTolerance = 1.0e-14;
  constexpr double RelativeTolerance = 1.0e-13;
  return std::abs(actual - expected)
      <= AbsoluteTolerance
       + RelativeTolerance * std::max(std::abs(actual), std::abs(expected));
}

void assertCoordinate(const GridCoordinate& actual,
                      double expectedX,
                      double expectedY,
                      double expectedZ) {
  assert(approximatelyEqual(actual.x, expectedX));
  assert(approximatelyEqual(actual.y, expectedY));
  assert(approximatelyEqual(actual.z, expectedZ));
}

void expectSurfaceCoordinateRejected(const AHGuessPoint& surfacePoint,
                                     const ReferencePoint& referencePoint,
                                     const std::string& expectedMessage) {
  bool rejected = false;
  try {
    (void)SurfacePointToCartesian(surfacePoint, referencePoint);
  } catch (const std::invalid_argument& error) {
    rejected = true;
    assert(std::string(error.what()).find(expectedMessage) != std::string::npos);
  }
  assert(rejected);
}

void testSurfacePointToCartesian() {
  constexpr double Pi = std::numbers::pi_v<double>;
  constexpr ReferencePoint Origin{0.0, 0.0, 0.0};

  assertCoordinate(SurfacePointToCartesian({2.0, 0.0, 0.0}, Origin), 2.0, 0.0, 0.0);
  assertCoordinate(SurfacePointToCartesian({2.0, Pi / 2.0, 0.0}, Origin), 0.0, 2.0, 0.0);
  assertCoordinate(SurfacePointToCartesian({2.0, Pi, 0.0}, Origin), -2.0, 0.0, 0.0);

  const double sqrtThree = std::sqrt(3.0);
  assertCoordinate(
    SurfacePointToCartesian({2.0, 0.0, Pi / 6.0}, Origin),
    sqrtThree, 0.0, 1.0
  );
  assertCoordinate(
    SurfacePointToCartesian({2.0, 0.0, -Pi / 6.0}, Origin),
    sqrtThree, 0.0, -1.0
  );

  constexpr ReferencePoint Shift{1.5, -2.0, 0.25};
  assertCoordinate(
    SurfacePointToCartesian({2.0, Pi / 2.0, 0.0}, Shift),
    1.5, 0.0, 0.25
  );

  const GridCoordinate periodicBase =
    SurfacePointToCartesian({1.7, 0.37, -0.21}, Shift);
  const GridCoordinate periodicShift =
    SurfacePointToCartesian({1.7, 0.37 + 2.0 * Pi, -0.21}, Shift);
  assertCoordinate(periodicShift, periodicBase.x, periodicBase.y, periodicBase.z);

  std::cout << "Surface-point spherical coordinates convert to Cartesian targets"
            << std::endl;
}

void testInvalidSurfacePointToCartesianInputs() {
  constexpr ReferencePoint Origin{0.0, 0.0, 0.0};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  expectSurfaceCoordinateRejected({0.0, 0.0, 0.0}, Origin, "strictly positive");
  expectSurfaceCoordinateRejected({-1.0, 0.0, 0.0}, Origin, "strictly positive");
  expectSurfaceCoordinateRejected({nan, 0.0, 0.0}, Origin, "must be finite");
  expectSurfaceCoordinateRejected({infinity, 0.0, 0.0}, Origin, "must be finite");
  expectSurfaceCoordinateRejected({1.0, nan, 0.0}, Origin, "must be finite");
  expectSurfaceCoordinateRejected({1.0, 0.0, infinity}, Origin, "must be finite");
  expectSurfaceCoordinateRejected({1.0, 0.0, 0.0}, {nan, 0.0, 0.0}, "must be finite");
  expectSurfaceCoordinateRejected({1.0, 0.0, 0.0}, {0.0, infinity, 0.0}, "must be finite");
}

void testAnalyticCoordinateGeometry() {
  const Tensor3 expectedX{{
    {{0.0, 0.0, 0.0}}, {{0.0, 4.0, 0.0}}, {{0.0, 0.0, 4.0}}
  }};
  const Tensor3 expectedY{{
    {{4.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 4.0}}
  }};
  const Tensor3 expectedZ{{
    {{4.0, 0.0, 0.0}}, {{0.0, 4.0, 0.0}}, {{0.0, 0.0, 0.0}}
  }};
  assert(ComputeRadialProjectionTensor({2.0, 0.0, 0.0}) == expectedX);
  assert(ComputeRadialProjectionTensor({0.0, 2.0, 0.0}) == expectedY);
  assert(ComputeRadialProjectionTensor({0.0, 0.0, 2.0}) == expectedZ);

  const Tensor3 expectedGeneral{{
    {{13.0, -2.0, -3.0}}, {{-2.0, 10.0, -6.0}}, {{-3.0, -6.0, 5.0}}
  }};
  const Tensor3 general = ComputeRadialProjectionTensor({1.0, 2.0, 3.0});
  assert(general == expectedGeneral);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      assert(general[i][j] == general[j][i]);
    }
  }

  const AnalyticCoordinateGeometry originGeometry =
    ComputeAnalyticCoordinateGeometry({1.0, 2.0, 3.0}, {0.0, 0.0, 0.0});
  const AnalyticCoordinateGeometry shiftedGeometry =
    ComputeAnalyticCoordinateGeometry({5.0, 1.0, 5.0}, {4.0, -1.0, 2.0});
  assert(originGeometry.T == shiftedGeometry.T);
  assert(originGeometry.gradient == shiftedGeometry.gradient);
  assert(originGeometry.hessian == shiftedGeometry.hessian);
}

void testAnalyticCoordinateDerivativesAgainstFiniteDifferences() {
  constexpr ReferencePoint Reference{1.5, -0.75, 0.4};
  const GridCoordinate global =
    SurfacePointToCartesian({2.0, 0.6, 0.35}, Reference);
  const AnalyticCoordinateGeometry analytic =
    ComputeAnalyticCoordinateGeometry(global, Reference);
  constexpr double Step = 1.0e-5;

  const auto angles = [&Reference](const GridCoordinate& point) {
    const double x = point.x - Reference.x;
    const double y = point.y - Reference.y;
    const double z = point.z - Reference.z;
    return std::array<double, 2>{
      std::atan2(y, x), std::atan2(z, std::hypot(x, y))
    };
  };
  for (std::size_t direction = 0; direction < 3; ++direction) {
    GridCoordinate plus = global;
    GridCoordinate minus = global;
    if (direction == 0) {
      plus.x += Step;
      minus.x -= Step;
    } else if (direction == 1) {
      plus.y += Step;
      minus.y -= Step;
    } else {
      plus.z += Step;
      minus.z -= Step;
    }
    const auto plusAngles = angles(plus);
    const auto minusAngles = angles(minus);
    for (std::size_t u = 0; u < 2; ++u) {
      const double finiteDifference =
        (plusAngles[u] - minusAngles[u]) / (2.0 * Step);
      assert(std::abs(analytic.gradient[u][direction] - finiteDifference) < 1.0e-9);
    }

    const AnalyticCoordinateGeometry plusGeometry =
      ComputeAnalyticCoordinateGeometry(plus, Reference);
    const AnalyticCoordinateGeometry minusGeometry =
      ComputeAnalyticCoordinateGeometry(minus, Reference);
    for (std::size_t u = 0; u < 2; ++u) {
      for (std::size_t component = 0; component < 3; ++component) {
        const double finiteDifference =
          (plusGeometry.gradient[u][component]
           - minusGeometry.gradient[u][component]) / (2.0 * Step);
        assert(std::abs(
          analytic.hessian[u][component][direction] - finiteDifference) < 1.0e-8);
      }
    }
  }

  bool poleRejected = false;
  try {
    (void)ComputeAnalyticCoordinateGeometry(
      {Reference.x + 1.0e-14, Reference.y, Reference.z + 1.0}, Reference);
  } catch (const std::domain_error& error) {
    poleRejected = std::string(error.what()).find("cos(sigma)=0") != std::string::npos;
  }
  assert(poleRejected);

  std::cout << "Analytic angular coordinate geometry matches Cartesian finite differences"
            << std::endl;
}

void expectNeighbourSearchRejected(
  const GridData& gridData,
  const GridCoordinate& target,
  std::size_t neighbourCount,
  const std::string& expectedMessage
) {
  bool rejected = false;
  try {
    (void)FindNearestGridCells(gridData, target, neighbourCount);
  } catch (const std::invalid_argument& error) {
    rejected = true;
    assert(std::string(error.what()).find(expectedMessage) != std::string::npos);
  }
  assert(rejected);
}

void testFindNearestGridCells() {
  GridData gridData;
  gridData.coordinates = {
    {0.0, 0.0, 0.0},  // index 0
    {2.0, 0.0, 0.0},  // index 1
    {1.0, 0.0, 0.0},  // index 2
    {0.0, 2.0, 0.0},  // index 3
    {0.0, 0.0, 3.0}   // index 4
  };

  // Exact coincidence is the unique nearest point.
  assert(FindNearestGridCells(gridData, {1.0, 0.0, 0.0}, 1)
         == std::vector<std::size_t>{2});

  // Distances from this target are 0.04, 0.64, 3.24, 4.04 and 9.04.
  assert(FindNearestGridCells(gridData, {0.2, 0.0, 0.0}, 4)
         == (std::vector<std::size_t>{0, 2, 1, 3}));

  // Index 2 is nearest; indices 0, 1 and 3 tie, so cell index orders the tie.
  assert(FindNearestGridCells(gridData, {1.0, 1.0, 0.0}, 4)
         == (std::vector<std::size_t>{2, 0, 1, 3}));

  std::cout << "Brute-force neighbour search returns sorted deterministic cell indices"
            << std::endl;
}

void testRadiusQueryAndWendlandKernel() {
  GridData grid;
  grid.coordinates = {
    {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {1.0, 0.0, 0.0},
    {-0.5, 0.0, 0.0}, {0.0, 0.75, 0.0}
  };
  const CartesianSpatialIndex index(grid);
  assert(FindGridCellsWithinRadius(index, {0.0, 0.0, 0.0}, 1.0)
         == (std::vector<std::size_t>{0, 1, 3, 4}));
  assert(FindGridCellsWithinRadius(index, {0.51, 0.0, 0.0}, 0.5)
         == (std::vector<std::size_t>{1, 2}));
  bool invalidRadiusRejected = false;
  try {
    (void)FindGridCellsWithinRadius(index, {0.0, 0.0, 0.0}, 0.0);
  } catch (const std::invalid_argument&) {
    invalidRadiusRejected = true;
  }
  assert(invalidRadiusRejected);

  assert(WendlandC2Weight(0.0) == 1.0);
  assert(WendlandC2Weight(1.0) == 0.0);
  assert(WendlandC2Weight(2.0) == 0.0);
  double previous = 1.0;
  for (int sample = 1; sample < 100; ++sample) {
    const double weight = WendlandC2Weight(sample / 100.0);
    assert(std::isfinite(weight));
    assert(weight >= 0.0);
    assert(weight <= previous);
    previous = weight;
  }
  assert(WendlandC2Weight(1.0 - 1.0e-6) < 1.0e-20);
}

void testAcceleratedNeighbourSearchMatchesBruteForce() {
  std::mt19937_64 generator(4815162342ULL);
  std::uniform_real_distribution<double> coordinate(-5.0, 5.0);
  GridData randomGrid;
  randomGrid.coordinates.resize(2000);
  for (GridCoordinate& point : randomGrid.coordinates) {
    point = {coordinate(generator), coordinate(generator), coordinate(generator)};
  }
  const CartesianSpatialIndex randomIndex(randomGrid);
  for (std::size_t query = 0; query < 30; ++query) {
    const GridCoordinate target{
      coordinate(generator), coordinate(generator), coordinate(generator)};
    for (std::size_t k : {
           std::size_t{1}, std::size_t{2}, std::size_t{5},
           std::size_t{16}, std::size_t{64}}) {
      assert(FindNearestGridCells(randomIndex, target, k)
             == FindNearestGridCellsBruteForce(randomGrid, target, k));
    }
  }

  GridData structured;
  for (int z = -2; z <= 2; ++z) {
    for (int y = -2; y <= 2; ++y) {
      for (int x = -2; x <= 2; ++x) {
        structured.coordinates.push_back(
          {static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)});
      }
    }
  }
  const CartesianSpatialIndex structuredIndex(structured);
  for (const GridCoordinate target : {
         GridCoordinate{0.0, 0.0, 0.0},
         GridCoordinate{0.5, 0.5, 0.5},
         GridCoordinate{1.0, -1.0, 2.0}}) {
    for (std::size_t k : {
           std::size_t{1}, std::size_t{4}, std::size_t{12},
           structured.coordinates.size()}) {
      assert(FindNearestGridCells(structuredIndex, target, k)
             == FindNearestGridCellsBruteForce(structured, target, k));
    }
  }

  // The origin is an exact hit; the six axis neighbours then tie by cell index.
  const auto exact = FindNearestGridCells(structuredIndex, {0.0, 0.0, 0.0}, 7);
  const auto exactReference =
    FindNearestGridCellsBruteForce(structured, {0.0, 0.0, 0.0}, 7);
  assert(exact == exactReference);
  assert(structured.coordinates[exact.front()].x == 0.0);
  assert(structured.coordinates[exact.front()].y == 0.0);
  assert(structured.coordinates[exact.front()].z == 0.0);

  std::cout << "K-d tree matches brute force for random, structured, exact, and tied queries"
            << std::endl;
}

void reportNeighbourSearchPerformance() {
  std::mt19937_64 generator(8675309ULL);
  std::uniform_real_distribution<double> coordinate(-10.0, 10.0);
  GridData gridData;
  gridData.coordinates.resize(20000);
  for (GridCoordinate& point : gridData.coordinates) {
    point = {coordinate(generator), coordinate(generator), coordinate(generator)};
  }
  std::vector<GridCoordinate> targets(50);
  for (GridCoordinate& target : targets) {
    target = {coordinate(generator), coordinate(generator), coordinate(generator)};
  }

  const auto buildStart = std::chrono::steady_clock::now();
  const CartesianSpatialIndex spatialIndex(gridData);
  const auto buildEnd = std::chrono::steady_clock::now();
  const auto acceleratedStart = std::chrono::steady_clock::now();
  for (const GridCoordinate& target : targets) {
    (void)FindNearestGridCells(spatialIndex, target, 8);
  }
  const auto acceleratedEnd = std::chrono::steady_clock::now();
  const auto bruteStart = std::chrono::steady_clock::now();
  for (const GridCoordinate& target : targets) {
    (void)FindNearestGridCellsBruteForce(gridData, target, 8);
  }
  const auto bruteEnd = std::chrono::steady_clock::now();

  const auto microseconds = [](auto start, auto end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  };
  std::cout << "Neighbour-search timing (informational only): build="
            << microseconds(buildStart, buildEnd) << " us, k-d queries="
            << microseconds(acceleratedStart, acceleratedEnd) << " us, brute queries="
            << microseconds(bruteStart, bruteEnd) << " us" << std::endl;
}

void testInvalidNearestGridCellInputs() {
  GridData empty;
  GridData twoCells;
  twoCells.coordinates = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  expectNeighbourSearchRejected(empty, {0.0, 0.0, 0.0}, 1, "empty");
  expectNeighbourSearchRejected(twoCells, {0.0, 0.0, 0.0}, 0, "greater than zero");
  expectNeighbourSearchRejected(twoCells, {0.0, 0.0, 0.0}, 3, "only 2 cells");
  expectNeighbourSearchRejected(twoCells, {nan, 0.0, 0.0}, 1, "must be finite");
  expectNeighbourSearchRejected(twoCells, {0.0, infinity, 0.0}, 1, "must be finite");
  expectNeighbourSearchRejected(twoCells, {0.0, 0.0, -infinity}, 1, "must be finite");
}

std::vector<double> evaluateLinearField(
  const std::vector<GridCoordinate>& coordinates,
  double constant,
  const Vector3& derivative
) {
  std::vector<double> values;
  values.reserve(coordinates.size());
  for (const GridCoordinate& coordinate : coordinates) {
    values.push_back(
      constant + derivative[0] * coordinate.x
      + derivative[1] * coordinate.y + derivative[2] * coordinate.z
    );
  }
  return values;
}

void assertLinearFit(
  const std::vector<GridCoordinate>& coordinates,
  double constant,
  const Vector3& derivative,
  const GridCoordinate& target
) {
  const std::vector<double> values =
    evaluateLinearField(coordinates, constant, derivative);
  const LinearPolynomialFit fit = FitLinearPolynomial(coordinates, values, target);
  const double expectedValue =
    constant + derivative[0] * target.x
    + derivative[1] * target.y + derivative[2] * target.z;
  assert(approximatelyEqual(fit.value, expectedValue));
  for (std::size_t direction = 0; direction < 3; ++direction) {
    assert(approximatelyEqual(fit.derivative[direction], derivative[direction]));
  }
  assert(std::isfinite(fit.lengthScale));
  assert(fit.lengthScale > 0.0);
}

void testFitLinearPolynomialExactReproduction() {
  const std::vector<GridCoordinate> regular{
    {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0},
    {-1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0},
    {1.0, 1.0, 1.0}, {2.0, -0.5, 0.75}
  };
  const GridCoordinate origin{0.0, 0.0, 0.0};

  assertLinearFit(regular, 7.0, {0.0, 0.0, 0.0}, origin);
  assertLinearFit(regular, 2.0, {3.0, -4.0, 5.0}, origin);
  assertLinearFit(regular, -1.0, {6.0, 0.0, 0.0}, origin);
  assertLinearFit(regular, -1.0, {0.0, 6.0, 0.0}, origin);
  assertLinearFit(regular, -1.0, {0.0, 0.0, 6.0}, origin);
  assertLinearFit(regular, 2.0, {3.0, -4.0, 5.0}, {4.0, -3.0, 2.0});

  const std::vector<GridCoordinate> irregular{
    {-2.1, 0.3, 1.7}, {0.2, -1.4, 0.1}, {3.0, 2.2, -0.8},
    {-0.4, 1.1, -2.5}, {1.7, -0.6, 2.9}, {-3.2, -2.0, 0.4},
    {0.9, 3.1, 1.2}, {2.5, 0.4, -3.3}
  };
  assertLinearFit(irregular, 2.0, {3.0, -4.0, 5.0}, {0.7, -0.2, 1.3});

  std::cout << "Degree-1 QR reconstruction exactly reproduces analytic linear fields"
            << std::endl;
}

void expectLinearFitRejected(
  const std::vector<GridCoordinate>& coordinates,
  const std::vector<double>& values,
  const std::string& expectedMessage
) {
  bool rejected = false;
  try {
    (void)FitLinearPolynomial(coordinates, values, {0.0, 0.0, 0.0});
  } catch (const std::invalid_argument& error) {
    rejected = true;
    assert(std::string(error.what()).find(expectedMessage) != std::string::npos);
  }
  assert(rejected);
}

void testFitLinearPolynomialInvalidStencils() {
  const std::vector<GridCoordinate> tooFew{
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}
  };
  expectLinearFitRejected(tooFew, {1.0, 2.0, 3.0}, "at least four");

  const std::vector<GridCoordinate> coplanar{
    {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0},
    {-1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {0.25, -0.5, 0.0}
  };
  expectLinearFitRejected(
    coplanar,
    evaluateLinearField(coplanar, 2.0, {3.0, -4.0, 5.0}),
    "rank deficient"
  );
}

double evaluatePolynomial(
  const PolynomialBasis& basis,
  std::span<const double> coefficients,
  const GridCoordinate& coordinate
) {
  double value = 0.0;
  for (std::size_t term = 0; term < basis.terms.size(); ++term) {
    const PolynomialExponent& exponent = basis.terms[term];
    value += coefficients[term]
      * std::pow(coordinate.x, static_cast<int>(exponent.x))
      * std::pow(coordinate.y, static_cast<int>(exponent.y))
      * std::pow(coordinate.z, static_cast<int>(exponent.z));
  }
  return value;
}

Vector3 evaluatePolynomialGradient(
  const PolynomialBasis& basis,
  std::span<const double> coefficients,
  const GridCoordinate& coordinate
) {
  Vector3 gradient{};
  for (std::size_t term = 0; term < basis.terms.size(); ++term) {
    const std::array<std::size_t, 3> exponent{
      basis.terms[term].x, basis.terms[term].y, basis.terms[term].z};
    const std::array<double, 3> position{coordinate.x, coordinate.y, coordinate.z};
    for (std::size_t direction = 0; direction < 3; ++direction) {
      if (exponent[direction] == 0) continue;
      double derivative = coefficients[term] * static_cast<double>(exponent[direction]);
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::size_t power = exponent[axis] - (axis == direction ? 1 : 0);
        derivative *= std::pow(position[axis], static_cast<int>(power));
      }
      gradient[direction] += derivative;
    }
  }
  return gradient;
}

void testConfigurablePolynomialBasisAndReproduction() {
  const std::array<std::size_t, 6> expectedSizes{1, 4, 10, 20, 35, 56};
  for (std::size_t degree = 0; degree < expectedSizes.size(); ++degree) {
    const PolynomialBasis basis = GeneratePolynomialBasis(degree);
    assert(PolynomialBasisSize(degree) == expectedSizes[degree]);
    assert(basis.terms.size() == expectedSizes[degree]);
    assert((basis.terms.front() == PolynomialExponent{0, 0, 0}));
    for (std::size_t first = 0; first < basis.terms.size(); ++first) {
      const auto& exponent = basis.terms[first];
      assert(exponent.x + exponent.y + exponent.z <= degree);
      for (std::size_t second = first + 1; second < basis.terms.size(); ++second) {
        assert(!(exponent == basis.terms[second]));
      }
    }
    if (degree == 0) {
      assert(basis.linearX == PolynomialBasis::NoTerm);
      assert(basis.linearY == PolynomialBasis::NoTerm);
      assert(basis.linearZ == PolynomialBasis::NoTerm);
    } else {
      assert((basis.terms[basis.linearX] == PolynomialExponent{1, 0, 0}));
      assert((basis.terms[basis.linearY] == PolynomialExponent{0, 1, 0}));
      assert((basis.terms[basis.linearZ] == PolynomialExponent{0, 0, 1}));
    }
  }

  std::vector<GridCoordinate> samples;
  for (double x : {-1.5, -0.5, 0.5, 1.5}) {
    for (double y : {-1.4, -0.3, 0.7, 1.6}) {
      for (double z : {-1.3, -0.2, 0.8, 1.7}) {
        samples.push_back({x, y, z});
      }
    }
  }
  const GridCoordinate target{0.2, -0.3, 0.4};
  for (std::size_t degree = 0; degree <= 3; ++degree) {
    const PolynomialBasis basis = GeneratePolynomialBasis(degree);
    std::vector<double> coefficients(basis.terms.size());
    for (std::size_t term = 0; term < coefficients.size(); ++term) {
      coefficients[term] = (term % 2 == 0 ? 1.0 : -1.0)
                         * 0.07 * static_cast<double>(term + 1);
    }
    std::vector<double> values;
    values.reserve(samples.size());
    for (const GridCoordinate& sample : samples) {
      values.push_back(evaluatePolynomial(basis, coefficients, sample));
    }
    const LinearPolynomialFit fit = FitLinearPolynomial(
      samples,
      values,
      target,
      {.method = LinearReconstructionMethod::Unweighted,
       .polynomialDegree = degree}
    );
    const double expectedValue = evaluatePolynomial(basis, coefficients, target);
    const Vector3 expectedGradient =
      evaluatePolynomialGradient(basis, coefficients, target);
    assert(std::abs(fit.value - expectedValue) < 1.0e-9);
    for (std::size_t direction = 0; direction < 3; ++direction) {
      assert(std::abs(fit.derivative[direction] - expectedGradient[direction]) < 1.0e-8);
    }
  }

  const PolynomialBasis quadraticBasis = GeneratePolynomialBasis(2);
  std::vector<double> quadraticCoefficients(quadraticBasis.terms.size(), 1.0);
  std::vector<double> insufficientValues;
  for (std::size_t sample = 0; sample < 9; ++sample) {
    insufficientValues.push_back(
      evaluatePolynomial(quadraticBasis, quadraticCoefficients, samples[sample]));
  }
  bool insufficientRejected = false;
  try {
    (void)FitLinearPolynomial(
      std::span<const GridCoordinate>(samples).first(9),
      insufficientValues,
      target,
      {.polynomialDegree = 2}
    );
  } catch (const std::invalid_argument& error) {
    insufficientRejected = std::string(error.what()).find("at least 10") != std::string::npos;
  }
  assert(insufficientRejected);

  // The shared fourteen-right-hand-side path uses the same quadratic basis.
  GridData compactGrid;
  compactGrid.coordinates = samples;
  compactGrid.resize(samples.size());
  std::vector<std::size_t> compactIndices(samples.size());
  std::iota(compactIndices.begin(), compactIndices.end(), std::size_t{0});
  std::array<std::vector<double>, NumberOfCompactFields> compactCoefficients;
  for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
    compactCoefficients[field].resize(quadraticBasis.terms.size());
    for (std::size_t term = 0; term < quadraticBasis.terms.size(); ++term) {
      compactCoefficients[field][term] = 0.01 * static_cast<double>(field + 1)
        * static_cast<double>(term + 1) * (term % 2 == 0 ? 1.0 : -1.0);
    }
  }
  for (std::size_t cell = 0; cell < samples.size(); ++cell) {
    for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
      compactGrid.values[cell * GridData::numberOfVariables + field] =
        evaluatePolynomial(quadraticBasis, compactCoefficients[field], samples[cell]);
    }
  }
  const CompactLinearFit compactFit = FitCompactLinearPolynomial(
    compactGrid,
    compactIndices,
    target,
    {.polynomialDegree = 2}
  );
  assert(compactFit.polynomialDegree == 2);
  assert(compactFit.rank == PolynomialBasisSize(2));
  for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
    assert(std::abs(compactFit.values[field]
                    - evaluatePolynomial(
                      quadraticBasis, compactCoefficients[field], target)) < 1.0e-9);
    const Vector3 expected = evaluatePolynomialGradient(
      quadraticBasis, compactCoefficients[field], target);
    for (std::size_t direction = 0; direction < 3; ++direction) {
      assert(std::abs(compactFit.gradients[field][direction] - expected[direction]) < 1.0e-8);
    }
  }

  std::cout << "Programmatic total-degree bases reproduce polynomials through degree three"
            << std::endl;
}

struct ConvergenceErrors {
  double meshWidth = 0.0;
  double valueLInfinity = 0.0;
  double gradientLInfinity = 0.0;
};

double smoothConvergenceField(const GridCoordinate& point) {
  return std::sin(0.7 * point.x) + std::exp(0.3 * point.y)
       + 0.2 * point.x * point.z + 0.1 * point.y * point.z * point.z;
}

Vector3 smoothConvergenceGradient(const GridCoordinate& point) {
  return {
    0.7 * std::cos(0.7 * point.x) + 0.2 * point.z,
    0.3 * std::exp(0.3 * point.y) + 0.1 * point.z * point.z,
    0.2 * point.x + 0.2 * point.y * point.z
  };
}

ConvergenceErrors measurePolynomialConvergenceLevel(
  std::size_t cellsPerDirection,
  std::size_t degree,
  std::span<const GridCoordinate> targets
) {
  const double meshWidth = 2.0 / static_cast<double>(cellsPerDirection);
  std::vector<GridCoordinate> coordinates;
  std::vector<double> values;
  coordinates.reserve(cellsPerDirection * cellsPerDirection * cellsPerDirection);
  values.reserve(coordinates.capacity());
  for (std::size_t ix = 0; ix < cellsPerDirection; ++ix) {
    for (std::size_t iy = 0; iy < cellsPerDirection; ++iy) {
      for (std::size_t iz = 0; iz < cellsPerDirection; ++iz) {
        const GridCoordinate cellCentre{
          -1.0 + (static_cast<double>(ix) + 0.5) * meshWidth,
          -1.0 + (static_cast<double>(iy) + 0.5) * meshWidth,
          -1.0 + (static_cast<double>(iz) + 0.5) * meshWidth
        };
        coordinates.push_back(cellCentre);
        values.push_back(smoothConvergenceField(cellCentre));
      }
    }
  }

  GridData searchGrid;
  searchGrid.coordinates = coordinates;
  const std::size_t stencilSize = 3 * PolynomialBasisSize(degree);
  ConvergenceErrors errors{.meshWidth = meshWidth};
  for (const GridCoordinate& target : targets) {
    // Use a local overdetermined stencil; targets are fixed grid-face/edge
    // locations and therefore never coincide with these cell centres.
    const std::vector<std::size_t> neighbours =
      FindNearestGridCellsBruteForce(searchGrid, target, stencilSize);
    std::vector<GridCoordinate> stencilCoordinates;
    std::vector<double> stencilValues;
    stencilCoordinates.reserve(stencilSize);
    stencilValues.reserve(stencilSize);
    for (std::size_t cell : neighbours) {
      stencilCoordinates.push_back(coordinates[cell]);
      stencilValues.push_back(values[cell]);
    }
    const LinearPolynomialFit fit = FitLinearPolynomial(
      stencilCoordinates,
      stencilValues,
      target,
      {.method = LinearReconstructionMethod::WeightedMovingLeastSquares,
       .polynomialDegree = degree,
       .weightSupport = 1.0}
    );
    errors.valueLInfinity = std::max(
      errors.valueLInfinity,
      std::abs(fit.value - smoothConvergenceField(target)));
    const Vector3 exactGradient = smoothConvergenceGradient(target);
    for (std::size_t direction = 0; direction < 3; ++direction) {
      errors.gradientLInfinity = std::max(
        errors.gradientLInfinity,
        std::abs(fit.derivative[direction] - exactGradient[direction]));
    }
  }
  return errors;
}

double observedOrder(double coarseError, double fineError,
                     double coarseWidth, double fineWidth) {
  return std::log(coarseError / fineError) / std::log(coarseWidth / fineWidth);
}

void testPolynomialInterpolationConvergence() {
  const std::array<GridCoordinate, 4> targets{{
    {0.0, 0.0, 0.0},
    {0.5, -0.5, 0.25},
    {-0.5, 0.25, -0.25},
    {0.25, 0.5, -0.5}
  }};
  const std::array<std::size_t, 3> refinements{8, 16, 32};

  // For smooth fields, degree-p value interpolation nominally approaches
  // order p+1, while its first derivatives nominally approach order p.
  // Only the two finest levels are asserted to avoid coarse-grid transients.
  for (std::size_t degree = 1; degree <= 3; ++degree) {
    std::array<ConvergenceErrors, 3> errors{};
    for (std::size_t level = 0; level < refinements.size(); ++level) {
      errors[level] = measurePolynomialConvergenceLevel(
        refinements[level], degree, targets);
    }
    const double valueOrder = observedOrder(
      errors[1].valueLInfinity, errors[2].valueLInfinity,
      errors[1].meshWidth, errors[2].meshWidth);
    const double gradientOrder = observedOrder(
      errors[1].gradientLInfinity, errors[2].gradientLInfinity,
      errors[1].meshWidth, errors[2].meshWidth);
    std::cout << "Degree " << degree << " convergence: value L-inf="
              << errors[2].valueLInfinity << " order=" << valueOrder
              << ", gradient L-inf=" << errors[2].gradientLInfinity
              << " order=" << gradientOrder << std::endl;
    assert(errors[2].valueLInfinity < errors[1].valueLInfinity);
    assert(errors[2].gradientLInfinity < errors[1].gradientLInfinity);
    assert(valueOrder > static_cast<double>(degree) + 0.25);
    assert(gradientOrder > static_cast<double>(degree) - 0.35);
  }
}

void testWeightedLinearPolynomialReconstruction() {
  const std::vector<GridCoordinate> irregular{
    {-2.1, 0.3, 1.7}, {0.2, -1.4, 0.1}, {3.0, 2.2, -0.8},
    {-0.4, 1.1, -2.5}, {1.7, -0.6, 2.9}, {-3.2, -2.0, 0.4},
    {0.9, 3.1, 1.2}, {2.5, 0.4, -3.3}
  };
  const GridCoordinate target{0.7, -0.2, 1.3};
  const std::vector<double> values =
    evaluateLinearField(irregular, 2.0, {3.0, -4.0, 5.0});
  const LinearReconstructionOptions weighted{
    .method = LinearReconstructionMethod::WeightedMovingLeastSquares,
    .weightSupport = 0.75
  };
  const LinearPolynomialFit weightedFit =
    FitLinearPolynomial(irregular, values, target, weighted);
  const LinearPolynomialFit unweightedFit =
    FitLinearPolynomial(irregular, values, target);

  assert(approximatelyEqual(weightedFit.value, 11.4));
  assert(approximatelyEqual(weightedFit.derivative[0], 3.0));
  assert(approximatelyEqual(weightedFit.derivative[1], -4.0));
  assert(approximatelyEqual(weightedFit.derivative[2], 5.0));
  assert(approximatelyEqual(weightedFit.value, unweightedFit.value));
  for (std::size_t direction = 0; direction < 3; ++direction) {
    assert(approximatelyEqual(
      weightedFit.derivative[direction], unweightedFit.derivative[direction]));
  }

  // A coincident sample has distance zero and therefore exactly unit weight.
  const std::vector<GridCoordinate> coincident{
    target, {1.2, -0.2, 1.3}, {0.7, 0.3, 1.3}, {0.7, -0.2, 1.8},
    {0.1, -0.8, 0.7}
  };
  assert(LinearReconstructionWeight(0.0, 1.0, weighted) == 1.0);
  assertLinearFit(coincident, 2.0, {3.0, -4.0, 5.0}, target);
  const auto coincidentValues =
    evaluateLinearField(coincident, 2.0, {3.0, -4.0, 5.0});
  const LinearPolynomialFit coincidentFit =
    FitLinearPolynomial(coincident, coincidentValues, target, weighted);
  assert(approximatelyEqual(coincidentFit.value, 11.4));
  assert(approximatelyEqual(coincidentFit.derivative[0], 3.0));
  assert(approximatelyEqual(coincidentFit.derivative[1], -4.0));
  assert(approximatelyEqual(coincidentFit.derivative[2], 5.0));
}

void testWeightedFitFavoursNearbySamples() {
  const std::vector<GridCoordinate> coordinates{
    {0.0, 0.0, 0.0}, {0.1, 0.0, 0.0},
    {0.0, 0.1, 0.0}, {0.0, 0.0, 0.1},
    {-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0},
    {0.0, 0.0, -1.0}, {1.0, 1.0, 1.0}
  };
  const std::vector<double> noisyValues{0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0};
  const LinearPolynomialFit unweighted =
    FitLinearPolynomial(coordinates, noisyValues, {0.0, 0.0, 0.0});
  const LinearPolynomialFit weighted = FitLinearPolynomial(
    coordinates,
    noisyValues,
    {0.0, 0.0, 0.0},
    {.method = LinearReconstructionMethod::WeightedMovingLeastSquares,
     .weightSupport = 0.5}
  );
  assert(std::abs(weighted.value) < std::abs(unweighted.value));
}

void testWeightedLinearPolynomialFailures() {
  const std::vector<GridCoordinate> fullRank{
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}
  };
  const std::vector<double> values{1.0, 2.0, 3.0, 4.0};
  for (double support : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try {
      (void)FitLinearPolynomial(
        fullRank,
        values,
        {0.0, 0.0, 0.0},
        {.method = LinearReconstructionMethod::WeightedMovingLeastSquares,
         .weightSupport = support}
      );
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }

  const std::vector<GridCoordinate> coplanar{
    {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0},
    {-1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 0.0, 0.0}
  };
  bool rankRejected = false;
  try {
    (void)FitLinearPolynomial(
      coplanar,
      evaluateLinearField(coplanar, 2.0, {3.0, -4.0, 5.0}),
      {0.0, 0.0, 0.0},
      {.method = LinearReconstructionMethod::WeightedMovingLeastSquares}
    );
  } catch (const std::invalid_argument& error) {
    rankRejected = std::string(error.what()).find("rank deficient") != std::string::npos;
  }
  assert(rankRejected);

  std::cout << "Weighted MLS handles locality, coincidence, and invalid stencils"
            << std::endl;
}

void testFitCompactLinearPolynomial() {
  const GridCoordinate target{0.7, -0.2, 1.3};
  GridData gridData;
  gridData.coordinates = {
    {-2.1, 0.3, 1.7}, {0.2, -1.4, 0.1}, {3.0, 2.2, -0.8},
    {-0.4, 1.1, -2.5}, {1.7, -0.6, 2.9}, {-3.2, -2.0, 0.4},
    {0.9, 3.1, 1.2}, {2.5, 0.4, -3.3}
  };
  gridData.resize(gridData.coordinates.size());

  std::array<double, NumberOfCompactFields> constants{};
  std::array<Vector3, NumberOfCompactFields> derivatives{};
  for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
    constants[field] = 100.0 + 7.0 * static_cast<double>(field);
    derivatives[field] = {
      1.0 + static_cast<double>(field),
      -2.0 - 2.0 * static_cast<double>(field),
      3.0 + 0.5 * static_cast<double>(field)
    };
  }
  for (std::size_t cell = 0; cell < gridData.coordinates.size(); ++cell) {
    const GridCoordinate& coordinate = gridData.coordinates[cell];
    for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
      gridData.values[cell * GridData::numberOfVariables + field] =
        constants[field] + derivatives[field][0] * coordinate.x
        + derivatives[field][1] * coordinate.y
        + derivatives[field][2] * coordinate.z;
    }
  }

  const std::vector<std::size_t> neighbours{7, 0, 5, 2, 4, 1, 6, 3};
  const CompactLinearFit fit = FitCompactLinearPolynomial(
    gridData,
    neighbours,
    target,
    {.method = LinearReconstructionMethod::WeightedMovingLeastSquares,
     .weightSupport = 0.75}
  );
  const CompactLinearFit unweightedFit =
    FitCompactLinearPolynomial(gridData, neighbours, target);

  assert(fit.numberOfNeighbours == neighbours.size());
  assert(fit.polynomialDegree == 1);
  assert(fit.rank == 4);
  assert(fit.lengthScale > 0.0);
  for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
    const double expectedValue =
      constants[field] + derivatives[field][0] * target.x
      + derivatives[field][1] * target.y
      + derivatives[field][2] * target.z;
    assert(approximatelyEqual(fit.values[field], expectedValue));
    assert(approximatelyEqual(fit.values[field], unweightedFit.values[field]));
    for (std::size_t direction = 0; direction < 3; ++direction) {
      assert(approximatelyEqual(
        fit.gradients[field][direction], derivatives[field][direction]));
      assert(approximatelyEqual(
        fit.gradients[field][direction], unweightedFit.gradients[field][direction]));
    }
  }

  // Named compact indices must still identify K, direct phi, and tensor order.
  assert(approximatelyEqual(
    fit.values[static_cast<std::size_t>(CompactField::TraceK)],
    constants[12] + derivatives[12][0] * target.x
      + derivatives[12][1] * target.y + derivatives[12][2] * target.z));
  assert(approximatelyEqual(
    fit.values[static_cast<std::size_t>(CompactField::Phi)],
    constants[13] + derivatives[13][0] * target.x
      + derivatives[13][1] * target.y + derivatives[13][2] * target.z));
  for (std::size_t component = 0;
       component < NumberOfSymmetricTensorComponents;
       ++component) {
    assert(ConformalMetricCompactFields[component]
           == static_cast<CompactField>(component));
    assert(ConformalExtrinsicCurvatureCompactFields[component]
           == static_cast<CompactField>(component + 6));
  }

  std::cout << "One QR factorisation reconstructs all fourteen compact fields"
            << std::endl;
}

void setCompactCell(
  GridData& gridData,
  std::size_t cellIndex,
  const std::array<double, NumberOfCompactFields>& values
) {
  std::copy(
    values.begin(),
    values.end(),
    gridData.values.begin() + cellIndex * GridData::numberOfVariables
  );
}

void assertTensor(const Tensor3& actual, const Tensor3& expected) {
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      assert(approximatelyEqual(actual[i][j], expected[i][j]));
    }
  }
}

void assertReconstructedGeometryEqual(
  const InterpolatedGeometryData& actual,
  const InterpolatedGeometryData& expected
) {
  assertTensor(actual.spatialMetric, expected.spatialMetric);
  assertTensor(actual.extrinsicCurvature, expected.extrinsicCurvature);
  assert(approximatelyEqual(actual.traceK, expected.traceK));
  assert(actual.spatialMetricDerivativeAvailable
         == expected.spatialMetricDerivativeAvailable);
  for (std::size_t k = 0; k < 3; ++k) {
    assertTensor(
      actual.spatialMetricDerivative[k], expected.spatialMetricDerivative[k]);
  }
}

void testInterpolateToSurfaceNearestNeighbour() {
  constexpr double Pi = std::numbers::pi_v<double>;
  constexpr ReferencePoint Reference{10.0, -4.0, 2.0};
  GridData gridData;
  gridData.resize(3);
  gridData.coordinates = {
    {11.0, -4.0, 2.0},  // target for first AH point
    {10.0, -3.0, 2.0},  // target for second AH point
    {30.0, 30.0, 30.0}  // deliberately irrelevant cell
  };
  setCompactCell(gridData, 0, {
    4.0, 2.0, 6.0, 8.0, 10.0, 12.0,
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    3.0, 2.0
  });
  setCompactCell(gridData, 1, {
    9.0, 3.0, 6.0, 12.0, 15.0, 18.0,
    2.0, 4.0, 6.0, 8.0, 10.0, 12.0,
    6.0, 3.0
  });
  setCompactCell(gridData, 2, {
    101.0, 102.0, 103.0, 104.0, 105.0, 106.0,
    107.0, 108.0, 109.0, 110.0, 111.0, 112.0,
    113.0, 4.0
  });

  const AHGuess ahGuess{.points = {
    {.r = 1.0, .rho = 0.0, .sigma = 0.0},
    {.r = 1.0, .rho = Pi / 2.0, .sigma = 0.0}
  }};
  InterpolatedSurface surface;
  InterpolateToSurfaceNearestNeighbour(gridData, ahGuess, Reference, surface);

  assert(surface.matches(ahGuess));
  assert(surface.points.size() == 2);

  const Tensor3 expectedMetric0{{
    {{1.0, 0.5, 1.5}}, {{0.5, 2.0, 2.5}}, {{1.5, 2.5, 3.0}}
  }};
  const Tensor3 expectedCurvature0{{
    {{1.25, 1.0, 2.25}}, {{1.0, 3.0, 3.75}}, {{2.25, 3.75, 4.5}}
  }};
  const Tensor3 expectedMetric1{{
    {{1.0, 1.0 / 3.0, 2.0 / 3.0}},
    {{1.0 / 3.0, 4.0 / 3.0, 5.0 / 3.0}},
    {{2.0 / 3.0, 5.0 / 3.0, 2.0}}
  }};
  const Tensor3 expectedCurvature1{{
    {{20.0 / 9.0, 10.0 / 9.0, 2.0}},
    {{10.0 / 9.0, 32.0 / 9.0, 40.0 / 9.0}},
    {{2.0, 40.0 / 9.0, 16.0 / 3.0}}
  }};

  assertTensor(surface.points[0].geometry.spatialMetric, expectedMetric0);
  assertTensor(surface.points[0].geometry.extrinsicCurvature, expectedCurvature0);
  assert(surface.points[0].geometry.traceK == 3.0);
  assertTensor(surface.points[1].geometry.spatialMetric, expectedMetric1);
  assertTensor(surface.points[1].geometry.extrinsicCurvature, expectedCurvature1);
  assert(surface.points[1].geometry.traceK == 6.0);

  const Tensor3Derivative zeroDerivative{};
  for (const InterpolatedSurfacePoint& point : surface.points) {
    assert(point.geometry.spatialMetricDerivative == zeroDerivative);
    assert(!point.geometry.spatialMetricDerivativeAvailable);
    assert(point.surfaceNormal == Vector3{});
    assert(point.surfaceNormalDerivative == Tensor3{});
  }

  std::cout << "Nearest-neighbour sampling populates surface physical geometry in AH order"
            << std::endl;
}

void testInterpolateCartesianToSurfaceDegree1() {
  constexpr double Pi = std::numbers::pi_v<double>;
  constexpr ReferencePoint Reference{1.0, -2.0, 0.5};
  GridData gridData;
  gridData.coordinates = {
    {-1.0, -4.0, -1.0}, {3.0, -4.0, -1.0},
    {-1.0, 0.0, -1.0}, {3.0, 0.0, -1.0},
    {-1.0, -4.0, 2.0}, {3.0, -4.0, 2.0},
    {-1.0, 0.0, 2.0}, {3.0, 0.0, 2.0}
  };
  gridData.resize(gridData.coordinates.size());

  std::array<double, NumberOfCompactFields> constants{};
  std::array<Vector3, NumberOfCompactFields> gradients{};
  constants = {
    4.0, 0.2, 0.3, 5.0, 0.4, 6.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    3.0, 2.0
  };
  for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
    const double scale = static_cast<double>(field + 1);
    gradients[field] = {0.01 * scale, -0.005 * scale, 0.002 * scale};
  }
  gradients[12] = {0.2, 0.1, -0.1};
  gradients[13] = {0.05, -0.04, 0.03};

  for (std::size_t cell = 0; cell < gridData.coordinates.size(); ++cell) {
    const GridCoordinate& coordinate = gridData.coordinates[cell];
    for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
      gridData.values[cell * GridData::numberOfVariables + field] =
        constants[field] + gradients[field][0] * coordinate.x
        + gradients[field][1] * coordinate.y
        + gradients[field][2] * coordinate.z;
    }
  }

  const AHGuess ahGuess{.points = {
    {.r = 0.5, .rho = 0.0, .sigma = 0.0},
    {.r = 0.5, .rho = Pi / 2.0, .sigma = 0.0},
    {.r = 0.5, .rho = 0.0, .sigma = Pi / 6.0}
  }};
  InterpolatedSurface mlsSurface;
  InterpolateCartesianToSurface(
    gridData,
    ahGuess,
    Reference,
    mlsSurface,
    {.method = CartesianToSurfaceMethod::WeightedPolynomial,
     .numberOfNeighbours = 8,
     .weightSupport = 0.8}
  );
  assert(mlsSurface.matches(ahGuess));

  constexpr std::array<std::array<std::size_t, 3>, 3> tensorComponent{{
    {{0, 1, 2}}, {{1, 3, 4}}, {{2, 4, 5}}
  }};
  for (std::size_t pointIndex = 0; pointIndex < ahGuess.points.size(); ++pointIndex) {
    const GridCoordinate target =
      SurfacePointToCartesian(ahGuess.points[pointIndex], Reference);
    std::array<double, NumberOfCompactFields> targetValues{};
    for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
      targetValues[field] = constants[field] + gradients[field][0] * target.x
        + gradients[field][1] * target.y + gradients[field][2] * target.z;
    }
    const double phi = targetValues[13];
    const double inversePhiSquared = 1.0 / (phi * phi);
    const double inversePhiCubed = inversePhiSquared / phi;
    const auto& geometry = mlsSurface.points[pointIndex].geometry;
    assert(geometry.spatialMetricDerivativeAvailable);
    assert(approximatelyEqual(geometry.traceK, targetValues[12]));
    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = 0; j < 3; ++j) {
        const std::size_t component = tensorComponent[i][j];
        const double gammaTilde = targetValues[component];
        const double ATilde = targetValues[component + 6];
        assert(approximatelyEqual(
          geometry.spatialMetric[i][j], gammaTilde * inversePhiSquared));
        assert(approximatelyEqual(
          geometry.extrinsicCurvature[i][j],
          (ATilde + targetValues[12] * gammaTilde / 3.0) * inversePhiSquared));
        for (std::size_t k = 0; k < 3; ++k) {
          const double expectedDerivative =
            gradients[component][k] * inversePhiSquared
            - 2.0 * gammaTilde * gradients[13][k] * inversePhiCubed;
          assert(approximatelyEqual(
            geometry.spatialMetricDerivative[k][i][j], expectedDerivative));
        }
      }
    }
  }

  InterpolatedSurface nearestSurface;
  InterpolateCartesianToSurface(
    gridData,
    ahGuess,
    Reference,
    nearestSurface,
    {.method = CartesianToSurfaceMethod::NearestNeighbour, .numberOfNeighbours = 1}
  );
  assert(!nearestSurface.points[0].geometry.spatialMetricDerivativeAvailable);
  assert(!approximatelyEqual(
    nearestSurface.points[0].geometry.spatialMetric[0][0],
    mlsSurface.points[0].geometry.spatialMetric[0][0]));

  std::cout << "Degree-1 MLS reconstructs physical surface geometry and Cartesian derivatives"
            << std::endl;
}

void testReusableCartesianPointInterpolator() {
  constexpr double Pi = std::numbers::pi_v<double>;
  constexpr ReferencePoint Reference{2.0, -1.0, 0.5};
  const AHGuess guess{.points = {
    {.r = 1.0, .rho = 0.0, .sigma = 0.0},
    {.r = 1.0, .rho = Pi / 2.0, .sigma = 0.0},
    {.r = 1.0, .rho = Pi, .sigma = Pi / 6.0}
  }};

  GridData gridData;
  gridData.resize(guess.points.size());
  std::vector<GridCoordinate> targets;
  for (const AHGuessPoint& point : guess.points) {
    targets.push_back(SurfacePointToCartesian(point, Reference));
  }
  gridData.coordinates = targets;
  for (std::size_t cell = 0; cell < gridData.coordinates.size(); ++cell) {
    const double offset = static_cast<double>(10 * cell);
    setCompactCell(gridData, cell, {
      4.0 + offset, 0.2 + offset, 0.3 + offset,
      5.0 + offset, 0.4 + offset, 6.0 + offset,
      0.1 + offset, 0.2 + offset, 0.3 + offset,
      0.4 + offset, 0.5 + offset, 0.6 + offset,
      3.0 + offset, 2.0 + static_cast<double>(cell)
    });
  }

  const CartesianPointInterpolator interpolator(
    gridData,
    {.method = CartesianToSurfaceMethod::NearestNeighbour,
     .numberOfNeighbours = 1}
  );
  std::vector<InterpolatedCartesianPoint> generic;
  interpolator.interpolatePoints(targets, generic);
  InterpolatedSurface surface;
  InterpolateCartesianToSurface(interpolator, guess, Reference, surface);
  assert(generic.size() == targets.size());
  assert(surface.points.size() == targets.size());
  for (std::size_t point = 0; point < targets.size(); ++point) {
    assertCoordinate(
      generic[point].target, targets[point].x, targets[point].y, targets[point].z);
    assert(approximatelyEqual(generic[point].geometry.traceK, 3.0 + 10.0 * point));
    assertReconstructedGeometryEqual(
      generic[point].geometry, surface.points[point].geometry);
  }

  // Reuse the same immutable index for another ordering and again for the
  // original ordering, as happens across nonlinear iterations.
  std::vector<GridCoordinate> reversedTargets(targets.rbegin(), targets.rend());
  std::vector<InterpolatedCartesianPoint> reversed;
  interpolator.interpolatePoints(reversedTargets, reversed);
  for (std::size_t point = 0; point < reversed.size(); ++point) {
    assertCoordinate(
      reversed[point].target,
      reversedTargets[point].x,
      reversedTargets[point].y,
      reversedTargets[point].z);
    assert(approximatelyEqual(
      reversed[point].geometry.traceK,
      3.0 + 10.0 * static_cast<double>(targets.size() - 1 - point)));
  }
  std::vector<InterpolatedCartesianPoint> repeated;
  interpolator.interpolatePoints(targets, repeated);
  for (std::size_t point = 0; point < repeated.size(); ++point) {
    assertReconstructedGeometryEqual(repeated[point].geometry, generic[point].geometry);
  }

  // The compatibility overload remains numerically identical, though callers
  // doing iterations should retain the interpolator above.
  InterpolatedSurface compatibilitySurface;
  InterpolateCartesianToSurface(
    gridData,
    guess,
    Reference,
    compatibilitySurface,
    {.method = CartesianToSurfaceMethod::NearestNeighbour,
     .numberOfNeighbours = 1}
  );
  for (std::size_t point = 0; point < surface.points.size(); ++point) {
    assertReconstructedGeometryEqual(
      compatibilitySurface.points[point].geometry, surface.points[point].geometry);
  }

  std::cout << "Persistent Cartesian interpolation preserves target order and surface results"
            << std::endl;
}

void testInterpolateCartesianToSurfaceContextualFailures() {
  GridData gridData;
  gridData.resize(4);
  gridData.coordinates = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}
  };
  for (std::size_t cell = 0; cell < gridData.coordinates.size(); ++cell) {
    for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
      gridData.values[cell * GridData::numberOfVariables + field] = 1.0;
    }
    gridData.values[cell * GridData::numberOfVariables + 13] = 2.0;
  }

  bool rankFailureHasContext = false;
  try {
    InterpolatedSurface surface;
    InterpolateCartesianToSurface(
      gridData,
      {.points = {{.r = 1.0, .rho = 0.0, .sigma = 0.0}}},
      {0.0, 0.0, 0.0},
      surface,
      {.method = CartesianToSurfaceMethod::UnweightedPolynomial,
       .numberOfNeighbours = 4}
    );
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    rankFailureHasContext = message.find("surface point 0") != std::string::npos
                         && message.find("target (1") != std::string::npos;
  }
  assert(rankFailureHasContext);

  bool invalidTargetHasContext = false;
  try {
    InterpolatedSurface surface;
    InterpolateCartesianToSurface(
      gridData,
      {.points = {
        {.r = 1.0, .rho = 0.0, .sigma = 0.0},
        {.r = std::numeric_limits<double>::quiet_NaN(), .rho = 0.0, .sigma = 0.0}
      }},
      {0.0, 0.0, 0.0},
      surface,
      {.method = CartesianToSurfaceMethod::NearestNeighbour,
       .numberOfNeighbours = 1}
    );
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    invalidTargetHasContext = message.find("surface point 1") != std::string::npos
                           && message.find("target (") != std::string::npos;
  }
  assert(invalidTargetHasContext);
}

// The domain test must hold on every face of the box and combine the axes:
// before this test it was only exercised on one side of one axis.
void testDomainCoverageOnEveryFace() {
  GridData gridData;
  gridData.resize(8);
  const GridCoordinate low{0.0, 0.0, 0.0};
  const GridCoordinate high{1.0, 2.0, 3.0};
  for (std::size_t corner = 0; corner < 8; ++corner) {
    gridData.coordinates[corner] = {
      (corner & 1) ? high.x : low.x,
      (corner & 2) ? high.y : low.y,
      (corner & 4) ? high.z : low.z};
    setCompactCell(gridData, corner, {1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 1.0});
  }
  const CartesianPointInterpolator interpolator(
    gridData,
    {.method = CartesianToSurfaceMethod::NearestNeighbour,
     .numberOfNeighbours = 1,
     .domainRelativeTolerance = 0.05});
  assertCoordinate(interpolator.domainMinimum(), low.x, low.y, low.z);
  assertCoordinate(interpolator.domainMaximum(), high.x, high.y, high.z);

  // Allowed distance outside the box: 5% of the diagonal sqrt(14).
  const double allowed = 0.05 * std::sqrt(14.0);
  const GridCoordinate centre{0.5, 1.0, 1.5};
  assert(interpolator.coversTarget(centre));
  for (int axis = 0; axis < 3; ++axis) {
    for (const bool upper : {false, true}) {
      for (const double fraction : {0.5, 1.5}) {
        GridCoordinate target = centre;
        double& coordinate = axis == 0 ? target.x : axis == 1 ? target.y : target.z;
        const double face = axis == 0 ? (upper ? high.x : low.x)
                          : axis == 1 ? (upper ? high.y : low.y)
                                      : (upper ? high.z : low.z);
        coordinate = face + (upper ? 1.0 : -1.0) * fraction * allowed;
        assert(interpolator.coversTarget(target) == (fraction < 1.0));
      }
    }
  }

  // Within the allowance on each axis separately, but not in combination.
  const double offset = 0.8 * allowed;
  assert(interpolator.coversTarget({high.x + offset, centre.y, centre.z}));
  assert(!interpolator.coversTarget({high.x + offset, high.y + offset, centre.z}));
  assert(!interpolator.coversTarget({centre.x, low.y - offset, low.z - offset}));
  assert(!interpolator.coversTarget({low.x - offset, centre.y, high.z + offset}));

  std::cout << "The sampled domain is checked on all six faces, axes combined" << std::endl;
}

GridData makeValidationGrid() {
  GridData gridData;
  gridData.resize(4);
  gridData.coordinates = {
    {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
    {1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}
  };
  for (std::size_t cell = 0; cell < gridData.coordinates.size(); ++cell) {
    const std::array<double, NumberOfCompactFields> values{
      1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 2.0
    };
    setCompactCell(gridData, cell, values);
  }
  return gridData;
}

void testRadiusWeightedPolynomialProductionPath() {
  GridData grid;
  for (int x = -2; x <= 2; ++x) {
    for (int y = -2; y <= 2; ++y) {
      for (int z = -2; z <= 2; ++z) {
        grid.coordinates.push_back({
          static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)
        });
        for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
          grid.values.push_back(
            2.0 + 0.1 * field + 0.3 * x - 0.2 * y + 0.05 * z
          );
        }
      }
    }
  }
  const CartesianSpatialIndex index(grid);
  const LinearReconstructionOptions reconstruction{
    .method = LinearReconstructionMethod::RadiusWeightedWendlandC2,
    .numberOfNeighbours = 0,
    .polynomialDegree = 1,
    .supportRadius = 2.1,
    .rankToleranceMultiplier = 64.0
  };
  for (const GridCoordinate target : std::array<GridCoordinate, 3>{
         GridCoordinate{0.1, 0.2, -0.1},
         GridCoordinate{-0.35, 0.15, 0.25},
         GridCoordinate{0.4, -0.3, 0.2}}) {
    const auto support = FindGridCellsWithinRadius(index, target, 2.1);
    const CompactLinearFit fit = FitCompactLinearPolynomial(
      grid, support, target, reconstruction
    );
    assert(fit.rank == 4);
    assert(std::isfinite(fit.conditionEstimate));
    for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
      const double expected = 2.0 + 0.1 * field
        + 0.3 * target.x - 0.2 * target.y + 0.05 * target.z;
      assert(std::abs(fit.values[field] - expected) < 1.0e-12);
      assert(std::abs(fit.gradients[field][0] - 0.3) < 1.0e-12);
      assert(std::abs(fit.gradients[field][1] + 0.2) < 1.0e-12);
      assert(std::abs(fit.gradients[field][2] - 0.05) < 1.0e-12);
    }
  }
  GridData constantGrid = grid;
  std::fill(constantGrid.values.begin(), constantGrid.values.end(), 3.25);
  const GridCoordinate constantTarget{0.15, -0.1, 0.25};
  const auto constantSupport = FindGridCellsWithinRadius(
    index, constantTarget, 2.1
  );
  const CompactLinearFit constantFit = FitCompactLinearPolynomial(
    constantGrid, constantSupport, constantTarget, reconstruction
  );
  for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
    assert(std::abs(constantFit.values[field] - 3.25) < 1.0e-12);
    for (double gradient : constantFit.gradients[field]) {
      assert(std::abs(gradient) < 1.0e-12);
    }
  }

  const CartesianPointInterpolator interpolator(
    grid,
    {.method = CartesianToSurfaceMethod::RadiusWeightedPolynomial,
     .numberOfNeighbours = 0,
     .polynomialDegree = 1,
     .supportRadius = 2.1,
     .allowOrderReduction = false}
  );
  std::vector<InterpolatedCartesianPoint> output;
  const std::array<GridCoordinate, 1> target{{0.5, 0.0, 0.0}};
  interpolator.interpolatePoints(target, output);
  assert(output.size() == 1);
  assert(output[0].diagnostics.requestedMethod
         == CartesianToSurfaceMethod::RadiusWeightedPolynomial);
  assert(output[0].diagnostics.neighbourCount
         == FindGridCellsWithinRadius(index, target[0], 2.1).size());
  assert(output[0].geometry.spatialMetricDerivativeAvailable);

  GridData boundaryGrid;
  boundaryGrid.coordinates = {
    {-0.2, 0.0, 0.0}, {0.0, 0.2, 0.0}, {0.0, 0.0, 0.2},
    {-0.1, -0.1, -0.1}, {1.0, 0.0, 0.0}
  };
  for (std::size_t cell = 0; cell < boundaryGrid.coordinates.size(); ++cell) {
    for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
      const GridCoordinate& coordinate = boundaryGrid.coordinates[cell];
      const double affine = 1.0 + coordinate.x + 2.0 * coordinate.y
        - 0.5 * coordinate.z;
      boundaryGrid.values.push_back(cell == 4 ? affine + 100.0 : affine);
    }
  }
  const CartesianSpatialIndex boundaryIndex(boundaryGrid);
  const GridCoordinate inside{1.0e-6, 0.0, 0.0};
  const GridCoordinate outside{-1.0e-6, 0.0, 0.0};
  const auto insideSupport = FindGridCellsWithinRadius(boundaryIndex, inside, 1.0);
  const auto outsideSupport = FindGridCellsWithinRadius(boundaryIndex, outside, 1.0);
  assert(std::find(insideSupport.begin(), insideSupport.end(), 4)
         != insideSupport.end());
  assert(std::find(outsideSupport.begin(), outsideSupport.end(), 4)
         == outsideSupport.end());
  const LinearReconstructionOptions boundaryOptions{
    .method = LinearReconstructionMethod::RadiusWeightedWendlandC2,
    .polynomialDegree = 1,
    .supportRadius = 1.0
  };
  const CompactLinearFit insideFit = FitCompactLinearPolynomial(
    boundaryGrid, insideSupport, inside, boundaryOptions
  );
  const CompactLinearFit outsideFit = FitCompactLinearPolynomial(
    boundaryGrid, outsideSupport, outside, boundaryOptions
  );
  assert(std::abs(insideFit.values[0] - outsideFit.values[0]) < 3.0e-6);
  for (std::size_t direction = 0; direction < 3; ++direction) {
    assert(std::abs(insideFit.gradients[0][direction]
                    - outsideFit.gradients[0][direction]) < 1.0e-8);
  }

  bool insufficientRejected = false;
  try {
    CartesianPointInterpolator insufficient(
      grid,
      {.method = CartesianToSurfaceMethod::RadiusWeightedPolynomial,
       .polynomialDegree = 1,
       .supportRadius = 0.01,
       .allowOrderReduction = false}
    );
    std::vector<InterpolatedCartesianPoint> ignored;
    insufficient.interpolatePoints(target, ignored);
  } catch (const std::runtime_error& error) {
    insufficientRejected = std::string(error.what()).find("insufficient cells")
      != std::string::npos;
  }
  assert(insufficientRejected);
}

std::string interpolationFailure(
  const GridData& gridData,
  const CartesianToSurfaceOptions& options
) {
  try {
    InterpolatedSurface surface;
    InterpolateCartesianToSurface(
      gridData,
      {.points = {{.r = 1.0, .rho = 0.0, .sigma = 0.0}}},
      {0.0, 0.0, 0.0},
      surface,
      options
    );
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

void testInterpolationValidationAndDiagnostics() {
  const CartesianToSurfaceOptions degreeOne{
    .method = CartesianToSurfaceMethod::UnweightedPolynomial,
    .numberOfNeighbours = 4
  };

  GridData malformed = makeValidationGrid();
  malformed.values.pop_back();
  assert(interpolationFailure(malformed, degreeOne).find("times fourteen")
         != std::string::npos);

  GridData nonFinite = makeValidationGrid();
  nonFinite.values[2 * GridData::numberOfVariables + 4] =
    std::numeric_limits<double>::quiet_NaN();
  const std::string nonFiniteMessage = interpolationFailure(nonFinite, degreeOne);
  assert(nonFiniteMessage.find("surface point 0") != std::string::npos);
  assert(nonFiniteMessage.find("non-finite compact value") != std::string::npos);

  GridData invalidPhi = makeValidationGrid();
  for (std::size_t cell = 0; cell < invalidPhi.coordinates.size(); ++cell) {
    invalidPhi.values[cell * GridData::numberOfVariables + 13] = -1.0;
  }
  const std::string phiMessage = interpolationFailure(
    invalidPhi,
    {.method = CartesianToSurfaceMethod::NearestNeighbour,
     .numberOfNeighbours = 1}
  );
  assert(phiMessage.find("surface point 0") != std::string::npos);
  assert(phiMessage.find("strictly positive") != std::string::npos);

  GridData insufficient = makeValidationGrid();
  insufficient.coordinates.resize(3);
  insufficient.values.resize(3 * GridData::numberOfVariables);
  const CartesianToSurfaceOptions fallbackDisabled{
    .method = CartesianToSurfaceMethod::WeightedPolynomial,
    .numberOfNeighbours = 3,
    .allowOrderReduction = false
  };
  const std::string insufficientMessage =
    interpolationFailure(insufficient, fallbackDisabled);
  assert(insufficientMessage.find("insufficient neighbours") != std::string::npos);
  assert(insufficientMessage.find("surface point 0") != std::string::npos);

  InterpolatedSurface reducedSurface;
  InterpolateCartesianToSurface(
    insufficient,
    {.points = {{.r = 1.0, .rho = 0.0, .sigma = 0.0}}},
    {0.0, 0.0, 0.0},
    reducedSurface,
    {.method = CartesianToSurfaceMethod::WeightedPolynomial,
     .numberOfNeighbours = 3,
     .allowOrderReduction = true}
  );
  const InterpolationDiagnostics& reduced =
    reducedSurface.points[0].interpolationDiagnostics;
  assert(reduced.requestedMethod == CartesianToSurfaceMethod::WeightedPolynomial);
  assert(reduced.requestedDegree == 1);
  assert(reduced.degreeUsed == 0);
  assert(reduced.neighbourCount == 1);
  assert(reduced.rank == 1);
  assert(reduced.conditionEstimate == 1.0);
  assert(reduced.orderReduced);
  assertCoordinate(reduced.target, 1.0, 0.0, 0.0);
  assert(!reducedSurface.points[0].geometry.spatialMetricDerivativeAvailable);

  GridData rankDeficient = makeValidationGrid();
  rankDeficient.coordinates = {
    {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
    {0.0, 1.0, 0.0}, {2.0, 1.0, 0.0}
  };
  const std::string rankMessage = interpolationFailure(rankDeficient, degreeOne);
  assert(rankMessage.find("rank deficient") != std::string::npos);
  assert(rankMessage.find("surface point 0") != std::string::npos);

  InterpolatedSurface successfulSurface;
  InterpolateCartesianToSurface(
    makeValidationGrid(),
    {.points = {{.r = 1.0, .rho = 0.0, .sigma = 0.0}}},
    {0.0, 0.0, 0.0},
    successfulSurface,
    degreeOne
  );
  const InterpolationDiagnostics& successful =
    successfulSurface.points[0].interpolationDiagnostics;
  assert(successful.requestedMethod == CartesianToSurfaceMethod::UnweightedPolynomial);
  assert(successful.requestedDegree == 1);
  assert(successful.degreeUsed == 1);
  assert(successful.neighbourCount == 4);
  assert(successful.rank == 4);
  assert(std::isfinite(successful.conditionEstimate));
  assert(successful.conditionEstimate >= 1.0);
  assert(!successful.orderReduced);
  assertCoordinate(successful.target, 1.0, 0.0, 0.0);

  std::cout << "Interpolation validation records successful and reduced-order diagnostics"
            << std::endl;
}

void testInterpolateFromCartesianToShell() {
  std::cout << "Placeholder test for InterpolateFromCartesianToShell" << std::endl;
}

void testInterpolateFromShellToSurface() {
  std::cout << "Placeholder test for InterpolateFromShellToSurface" << std::endl;
}

}  // namespace exahype2::ccz4::ahf::tests::interpolation

int main() {
  using namespace exahype2::ccz4::ahf::tests::interpolation;

  std::cout << "Apparent Horizon Finder interpolation tests begin..." << std::endl;

  testSurfacePointToCartesian();
  testInvalidSurfacePointToCartesianInputs();
  testAnalyticCoordinateGeometry();
  testAnalyticCoordinateDerivativesAgainstFiniteDifferences();
  testFindNearestGridCells();
  testRadiusQueryAndWendlandKernel();
  testAcceleratedNeighbourSearchMatchesBruteForce();
  reportNeighbourSearchPerformance();
  testInvalidNearestGridCellInputs();
  testFitLinearPolynomialExactReproduction();
  testFitLinearPolynomialInvalidStencils();
  testConfigurablePolynomialBasisAndReproduction();
  testPolynomialInterpolationConvergence();
  testWeightedLinearPolynomialReconstruction();
  testWeightedFitFavoursNearbySamples();
  testWeightedLinearPolynomialFailures();
  testFitCompactLinearPolynomial();
  testInterpolateToSurfaceNearestNeighbour();
  testInterpolateCartesianToSurfaceDegree1();
  testReusableCartesianPointInterpolator();
  testInterpolateCartesianToSurfaceContextualFailures();
  testDomainCoverageOnEveryFace();
  testInterpolationValidationAndDiagnostics();
  testRadiusWeightedPolynomialProductionPath();
  testInterpolateFromCartesianToShell();
  testInterpolateFromShellToSurface();

  std::cout << "All interpolation tests completed." << std::endl;
  return 0;
}
