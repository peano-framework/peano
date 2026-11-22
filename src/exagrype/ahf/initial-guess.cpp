#include "initial-guess.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace exahype2::ccz4::ahf {

std::vector<double> MakeSphericalInitialCoefficients(
  const SpectralBasis& basis,
  double radius
) {
  if (!std::isfinite(radius) || radius <= 0.0) {
    throw std::invalid_argument(
      "MakeSphericalInitialCoefficients: radius must be finite and strictly positive"
    );
  }

  const std::vector<double> field(basis.nPoints, radius);
  std::vector<double> coefficients = Project(basis, field);
  if (coefficients.size() != basis.nModes) {
    throw std::runtime_error(
      "MakeSphericalInitialCoefficients: projected coefficient count does not match basis.nModes"
    );
  }
  return coefficients;
}

std::vector<double> MakeDistortedInitialCoefficients(
  const SpectralBasis& basis,
  const std::vector<double>& radii
) {
  if (radii.size() != basis.nPoints) {
    throw std::invalid_argument(
      "MakeDistortedInitialCoefficients: radius count must equal basis.nPoints");
  }
  for (std::size_t p = 0; p < radii.size(); ++p) {
    if (!std::isfinite(radii[p]) || radii[p] <= 0.0) {
      throw std::invalid_argument(
        "MakeDistortedInitialCoefficients: radius at point " +
        std::to_string(p) + " must be finite and strictly positive");
    }
  }
  std::vector<double> coefficients = Project(basis, radii);
  if (coefficients.size() != basis.nModes) {
    throw std::runtime_error(
      "MakeDistortedInitialCoefficients: Project returned an invalid coefficient count");
  }
  return coefficients;
}

}  // namespace exahype2::ccz4::ahf
