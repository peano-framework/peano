#include "surface-geometry.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace exahype2::ccz4::ahf {
namespace {

void requireFinite(double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string("PopulateSurfaceShapeGeometry: ")
                                + name + " must be finite");
  }
}

}  // namespace

void PopulateSurfaceShapeGeometry(
  InterpolatedSurfacePoint& point,
  const SurfaceShapeDerivatives& derivatives
) {
  requireFinite(point.r, "radius");
  if (point.r <= 0.0) {
    throw std::invalid_argument(
      "PopulateSurfaceShapeGeometry: radius must be strictly positive");
  }
  requireFinite(point.rho, "rho");
  requireFinite(point.sigma, "sigma");

  const std::array<double, 5> shapeDerivatives{
    derivatives.hRho, derivatives.hSigma, derivatives.hRhoRho,
    derivatives.hRhoSigma, derivatives.hSigmaSigma
  };
  for (double derivative : shapeDerivatives) {
    requireFinite(derivative, "surface derivative");
  }

  // Indices: gradient[u][i], hessian[u][i][j], T[i][j], with
  // u = 0 (rho), 1 (sigma), and Cartesian i,j = 0 (x), 1 (y), 2 (z).
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t u = 0; u < 2; ++u) {
      requireFinite(point.geometry.coordinateGradient[u][i], "coordinate gradient");
    }
    for (std::size_t j = 0; j < 3; ++j) {
      requireFinite(point.geometry.T[i][j], "radial projection tensor");
      for (std::size_t u = 0; u < 2; ++u) {
        requireFinite(point.geometry.coordinateHessian[u][i][j],
                      "coordinate Hessian");
      }
    }
  }

  const double cosSigma = std::cos(point.sigma);
  const Vector3 radialUnit{
    cosSigma * std::cos(point.rho),
    cosSigma * std::sin(point.rho),
    std::sin(point.sigma)
  };
  const double inverseRadiusCubed = 1.0 / (point.r * point.r * point.r);

  Vector3 normal{};
  Tensor3 normalDerivative{};
  for (std::size_t i = 0; i < 3; ++i) {
    const double xRho = point.geometry.coordinateGradient[0][i];
    const double xSigma = point.geometry.coordinateGradient[1][i];
    normal[i] = radialUnit[i]
              - xRho * derivatives.hRho
              - xSigma * derivatives.hSigma;
    requireFinite(normal[i], "surface normal");

    for (std::size_t j = 0; j < 3; ++j) {
      const double jRho = point.geometry.coordinateGradient[0][j];
      const double jSigma = point.geometry.coordinateGradient[1][j];
      normalDerivative[i][j] =
        point.geometry.T[i][j] * inverseRadiusCubed
        - point.geometry.coordinateHessian[0][i][j] * derivatives.hRho
        - point.geometry.coordinateHessian[1][i][j] * derivatives.hSigma
        - xRho * jRho * derivatives.hRhoRho
        - xRho * jSigma * derivatives.hRhoSigma
        - xSigma * jRho * derivatives.hRhoSigma
        - xSigma * jSigma * derivatives.hSigmaSigma;
      requireFinite(normalDerivative[i][j], "surface normal derivative");
    }
  }

  point.surfaceNormal = normal;
  point.surfaceNormalDerivative = normalDerivative;
}

void PopulateSurfaceShapeGeometry(
  InterpolatedSurface& surface,
  const SynthesizedField& field
) {
  const std::size_t size = surface.points.size();
  if (field.h.size() != size || field.hRho.size() != size
      || field.hSigma.size() != size || field.hRhoRho.size() != size
      || field.hRhoSigma.size() != size || field.hSigmaSigma.size() != size) {
    throw std::invalid_argument(
      "PopulateSurfaceShapeGeometry: SynthesizedField sizes must match surface");
  }
  for (std::size_t point = 0; point < size; ++point) {
    requireFinite(field.h[point], "synthesized radius");
    if (field.h[point] != surface.points[point].r) {
      throw std::invalid_argument(
        "PopulateSurfaceShapeGeometry: synthesized radius must match surface radius");
    }
    PopulateSurfaceShapeGeometry(surface.points[point], {
      field.hRho[point], field.hSigma[point], field.hRhoRho[point],
      field.hRhoSigma[point], field.hSigmaSigma[point]
    });
  }
}

}  // namespace exahype2::ccz4::ahf
