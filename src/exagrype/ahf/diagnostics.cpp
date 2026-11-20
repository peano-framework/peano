#include "diagnostics.h"

#include "spacetimes.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace exahype2::ccz4::ahf {

  namespace {

    /** gamma_ij u^i v^j for a symmetric metric and two vectors. */
    double contract(const Tensor3& gamma, const Vector3& u, const Vector3& v) {
      double sum = 0.0;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          sum += gamma[i][j] * u[i] * v[j];
        }
      }
      return sum;
    }

    /** The unit radial direction n^i at collocation point p. */
    Vector3 radialDirection(const SpectralBasis& basis, std::size_t p) {
      const double cosSigma = std::cos(basis.sigma[p]);
      return {
        cosSigma * std::cos(basis.rho[p]),
        cosSigma * std::sin(basis.rho[p]),
        std::sin(basis.sigma[p])};
    }

    SynthesizedField synthesizeOnBasisGrid(
      const SpectralBasis&       basis,
      const std::vector<double>& coefficients
    ) {
      SynthesizedField field = Synthesize(basis, coefficients);
      if (field.h.size() != basis.nPoints) {
        throw std::invalid_argument(
          "HorizonArea: synthesized field does not match the basis grid"
        );
      }
      return field;
    }

  } // namespace

  double HorizonArea(
    const SpectralBasis&       basis,
    const std::vector<double>& coefficients,
    std::span<const Tensor3>   spatialMetric
  ) {
    const SynthesizedField field = synthesizeOnBasisGrid(basis, coefficients);
    if (spatialMetric.size() != basis.nPoints) {
      throw std::invalid_argument(
        "HorizonArea: expected the metric at each of the basis' collocation points"
      );
    }

    double area = 0.0;
    for (std::size_t p = 0; p < basis.nPoints; ++p) {
      const double h        = field.h[p];
      const double rho      = basis.rho[p];
      const double sigma    = basis.sigma[p];
      const double cosSigma = std::cos(sigma);
      const double sinSigma = std::sin(sigma);
      const double cosRho   = std::cos(rho);
      const double sinRho   = std::sin(rho);

      const Vector3 normal{cosSigma * cosRho, cosSigma * sinRho, sinSigma};
      const Vector3 dNormalRho{-cosSigma * sinRho, cosSigma * cosRho, 0.0};
      const Vector3
        dNormalSigma{-sinSigma * cosRho, -sinSigma * sinRho, cosSigma};

      Vector3 dxRho{};
      Vector3 dxSigma{};
      for (int i = 0; i < 3; ++i) {
        dxRho[i]   = field.hRho[p] * normal[i] + h * dNormalRho[i];
        dxSigma[i] = field.hSigma[p] * normal[i] + h * dNormalSigma[i];
      }

      const Tensor3& gamma = spatialMetric[p];

      const double qRhoRho     = contract(gamma, dxRho, dxRho);
      const double qSigmaSigma = contract(gamma, dxSigma, dxSigma);
      const double qRhoSigma   = contract(gamma, dxRho, dxSigma);
      const double detq        = qRhoRho * qSigmaSigma - qRhoSigma * qRhoSigma;

      const double jacobian = 1.0 / cosSigma; // dmu drho -> dsigma drho
      area += basis.weight[p] * jacobian * std::sqrt(std::max(detq, 0.0));
    }
    return area;
  }

  template <class Spacetime>
  double HorizonArea(
    const SpectralBasis&       basis,
    const std::vector<double>& coefficients,
    const Spacetime&           spacetime,
    const ReferencePoint&      referencePoint
  ) {
    const SynthesizedField field = synthesizeOnBasisGrid(basis, coefficients);

    std::vector<Tensor3> spatialMetric(basis.nPoints);
    for (std::size_t p = 0; p < basis.nPoints; ++p) {
      const Vector3 normal = radialDirection(basis, p);
      const Vector3 absolute{
        referencePoint.x + field.h[p] * normal[0],
        referencePoint.y + field.h[p] * normal[1],
        referencePoint.z + field.h[p] * normal[2]};
      spatialMetric[p] = MetricData(spacetime, absolute).spatialMetric;
    }
    return HorizonArea(basis, coefficients, spatialMetric);
  }

  double ArealRadius(double area) { return std::sqrt(area / (4.0 * std::numbers::pi)); }

  double IrreducibleMass(double area) {
    return std::sqrt(area / (16.0 * std::numbers::pi));
  }

  double MeanCoordinateRadius(const std::vector<double>& coefficients) {
    if (coefficients.empty()) {
      return 0.0;
    }
    return coefficients[0] / std::sqrt(4.0 * std::numbers::pi);
  }

  std::vector<double> PowerSpectrum(
    const SpectralBasis&       basis,
    const std::vector<double>& coefficients
  ) {
    std::vector<double> power(static_cast<std::size_t>(basis.lmax) + 1, 0.0);
    for (std::size_t j = 0; j < basis.modes.size() && j < coefficients.size();
         ++j) {
      const int l = basis.modes[j].first;
      power[static_cast<std::size_t>(l)] += coefficients[j] * coefficients[j];
    }
    return power;
  }

  // Explicit instantiations for the concrete spacetimes used in the tests
  template double HorizonArea<
    SchwarzschildIsotropic>(const SpectralBasis&, const std::vector<double>&, const SchwarzschildIsotropic&, const ReferencePoint&);
  template double HorizonArea<
    SchwarzschildKerrSchild>(const SpectralBasis&, const std::vector<double>&, const SchwarzschildKerrSchild&, const ReferencePoint&);
  template double HorizonArea<
    KerrKerrSchild>(const SpectralBasis&, const std::vector<double>&, const KerrKerrSchild&, const ReferencePoint&);
  template double HorizonArea<
    BrillLindquist>(const SpectralBasis&, const std::vector<double>&, const BrillLindquist&, const ReferencePoint&);

} // namespace exahype2::ccz4::ahf
