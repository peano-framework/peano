#include "../require-asserts.h"

#include "../../ah-finder-options.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace exahype2::ccz4::ahf::tests::options {
namespace {

bool IsRejected(const AHFinderOptions& options) {
  try {
    ValidateAHFinderOptions(options);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

void TestDefaultsAndComposedTypes() {
  const AHFinderOptions options;
  ValidateAHFinderOptions(options);
  assert(options.referencePoint.x == 0.0);
  assert(options.initialRadius == 1.0);
  assert(options.lMax == 2);
  assert(options.interpolation.method
         == CartesianToSurfaceMethod::WeightedPolynomial);
  assert(options.solver.tolerance == SpectralNewtonOptions{}.tolerance);
}

void TestReferencePointAndRadiusValidation() {
  AHFinderOptions options;
  options.referencePoint.y = std::numeric_limits<double>::quiet_NaN();
  assert(IsRejected(options));

  options = {};
  options.referencePoint.z = std::numeric_limits<double>::infinity();
  assert(IsRejected(options));

  for (double radius : {
         0.0,
         -1.0,
         std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::infinity()}) {
    options = {};
    options.initialRadius = radius;
    assert(IsRejected(options));
  }
}

void TestLMaxValidation() {
  AHFinderOptions options;
  options.lMax = -1;
  assert(IsRejected(options));

  options.lMax = 0;
  ValidateAHFinderOptions(options);
}

void TestInterpolationValidation() {
  AHFinderOptions options;

  options.interpolation = {
    .method = CartesianToSurfaceMethod::RadiusWeightedPolynomial,
    .numberOfNeighbours = 0,
    .polynomialDegree = 1,
    .supportRadius = 0.1,
    .allowOrderReduction = false
  };
  ValidateAHFinderOptions(options);

  options.interpolation.supportRadius = 0.0;
  assert(IsRejected(options));

  options.interpolation = {
    .method = CartesianToSurfaceMethod::WeightedPolynomial,
    .numberOfNeighbours = 3,
    .polynomialDegree = 1,
    .weightSupport = 1.0,
    .allowOrderReduction = false
  };
  assert(IsRejected(options));

  options.interpolation.allowOrderReduction = true;
  ValidateAHFinderOptions(options);

  options.interpolation.numberOfNeighbours = 4;
  options.interpolation.allowOrderReduction = false;
  options.interpolation.weightSupport = 0.0;
  assert(IsRejected(options));

  options.interpolation.weightSupport = 1.0;
  options.interpolation.polynomialDegree = 4;
  assert(IsRejected(options));

  options.interpolation = {
    .method = CartesianToSurfaceMethod::NearestNeighbour,
    .numberOfNeighbours = 0
  };
  assert(IsRejected(options));

  options.interpolation.numberOfNeighbours = 1;
  options.interpolation.domainRelativeTolerance =
    std::numeric_limits<double>::quiet_NaN();
  assert(IsRejected(options));
}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests::options

int main() {
  using namespace exahype2::ccz4::ahf::tests::options;
  TestDefaultsAndComposedTypes();
  TestReferencePointAndRadiusValidation();
  TestLMaxValidation();
  TestInterpolationValidation();
  std::cout << "All AHFinderOptions tests completed.\n";
}
