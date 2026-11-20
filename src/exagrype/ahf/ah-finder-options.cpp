#include "ah-finder-options.h"

#include <cmath>
#include <stdexcept>

namespace exahype2::ccz4::ahf {
namespace {

void ValidateInterpolationOptions(const CartesianToSurfaceOptions& options) {
  if (!std::isfinite(options.domainRelativeTolerance)
      || options.domainRelativeTolerance < 0.0) {
    throw std::invalid_argument(
      "ValidateAHFinderOptions: interpolation domainRelativeTolerance must be finite and non-negative"
    );
  }

  switch (options.method) {
  case CartesianToSurfaceMethod::NearestNeighbour:
    if (options.numberOfNeighbours == 0) {
      throw std::invalid_argument(
        "ValidateAHFinderOptions: nearest-neighbour interpolation requires numberOfNeighbours > 0"
      );
    }
    return;

  case CartesianToSurfaceMethod::UnweightedPolynomial:
  case CartesianToSurfaceMethod::WeightedPolynomial:
    if (options.numberOfNeighbours == 0) {
      throw std::invalid_argument(
        "ValidateAHFinderOptions: fixed-count polynomial interpolation requires numberOfNeighbours > 0"
      );
    }
    break;

  case CartesianToSurfaceMethod::RadiusWeightedPolynomial:
    if (!std::isfinite(options.supportRadius)
        || options.supportRadius <= 0.0) {
      throw std::invalid_argument(
        "ValidateAHFinderOptions: radius-weighted interpolation requires a finite positive supportRadius"
      );
    }
    break;

  default:
    throw std::invalid_argument(
      "ValidateAHFinderOptions: interpolation method is not recognised"
    );
  }

  if (options.polynomialDegree > 3) {
    throw std::invalid_argument(
      "ValidateAHFinderOptions: interpolation polynomialDegree must be between 0 and 3"
    );
  }
  if (!std::isfinite(options.rankToleranceMultiplier)
      || options.rankToleranceMultiplier <= 0.0) {
    throw std::invalid_argument(
      "ValidateAHFinderOptions: interpolation rankToleranceMultiplier must be finite and positive"
    );
  }
  if (options.method == CartesianToSurfaceMethod::WeightedPolynomial
      && (!std::isfinite(options.weightSupport)
          || options.weightSupport <= 0.0)) {
    throw std::invalid_argument(
      "ValidateAHFinderOptions: weighted interpolation requires a finite positive weightSupport"
    );
  }
  if (options.method != CartesianToSurfaceMethod::RadiusWeightedPolynomial
      && !options.allowOrderReduction
      && options.numberOfNeighbours
           < PolynomialBasisSize(options.polynomialDegree)) {
    throw std::invalid_argument(
      "ValidateAHFinderOptions: numberOfNeighbours is too small for the requested polynomialDegree"
    );
  }
}

}  // namespace

void ValidateAHFinderOptions(const AHFinderOptions& options) {
  if (!std::isfinite(options.referencePoint.x)
      || !std::isfinite(options.referencePoint.y)
      || !std::isfinite(options.referencePoint.z)) {
    throw std::invalid_argument(
      "ValidateAHFinderOptions: referencePoint must be finite"
    );
  }
  if (!std::isfinite(options.initialRadius) || options.initialRadius <= 0.0) {
    throw std::invalid_argument(
      "ValidateAHFinderOptions: initialRadius must be finite and strictly positive"
    );
  }
  if (options.lMax < 0) {
    throw std::invalid_argument(
      "ValidateAHFinderOptions: lMax must be non-negative"
    );
  }
  if (std::isnan(options.solver.maxRelativeExpansionError)
      || options.solver.maxRelativeExpansionError <= 0.0) {
    throw std::invalid_argument(
      "ValidateAHFinderOptions: solver.maxRelativeExpansionError must be positive"
    );
  }
  ValidateInterpolationOptions(options.interpolation);
}

}  // namespace exahype2::ccz4::ahf
