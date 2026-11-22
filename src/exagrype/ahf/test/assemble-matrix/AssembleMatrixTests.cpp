#include "../require-asserts.h"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

#include "../../apparent-horizon-finder.h"

namespace exahype2::ccz4::ahf::tests::assemble_matrix {

namespace {

bool approxEqual(double a, double b, double tol) {
  return std::abs(a - b) <= tol;
}

// Non-unit normal and its gradient for a *coordinate sphere* r = const. With the
// guess radius constant the angular derivatives of h vanish, so (from the
// appendix) s_i reduces to the unit radial direction n_i and
//   partial_i s_j = (delta_ij - n_i n_j) / r.
// These are metric-independent (pure embedding), so they serve both the flat and
// the Schwarzschild tests below.
Vector3 sphereNormal(const Vector3& n) {
  return n;
}

Tensor3 sphereNormalDerivative(const Vector3& n, double r) {
  Tensor3 ds{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      ds[i][j] = ((i == j ? 1.0 : 0.0) - n[i] * n[j]) / r;
    }
  }
  return ds;
}

Tensor3 identityMetric() {
  Tensor3 g{};
  for (int i = 0; i < 3; ++i) {
    g[i][i] = 1.0;
  }
  return g;
}

}  // namespace


// Schwarzschild in isotropic coordinates (gamma_ij = psi^4 delta_ij,
// psi = 1 + M/2r, K_ij = 0) has its horizon at the coordinate radius r = M/2.
// There the expansion must vanish. This additionally exercises A2, B1 and B3 (the
// partial_k gamma terms) that the flat test leaves at zero.
void testExpansionSchwarzschildIsotropicHorizon() {
  const double M = 1.0;
  const double r = 0.5 * M;  // isotropic horizon
  const Vector3 n = {1.0 / std::sqrt(14.0), 2.0 / std::sqrt(14.0), 3.0 / std::sqrt(14.0)};

  const double psi = 1.0 + M / (2.0 * r);
  const double psi4 = psi * psi * psi * psi;

  Tensor3 gamma{};
  for (int i = 0; i < 3; ++i) {
    gamma[i][i] = psi4;
  }

  // partial_k psi = -M x_k / (2 r^3) = -M n_k / (2 r^2);  partial_k psi^4 = 4 psi^3 partial_k psi.
  std::array<Tensor3, 3> dgamma{};
  for (int k = 0; k < 3; ++k) {
    const double dpsi = -M * n[k] / (2.0 * r * r);
    const double dpsi4 = 4.0 * psi * psi * psi * dpsi;
    for (int i = 0; i < 3; ++i) {
      dgamma[k][i][i] = dpsi4;
    }
  }

  const Tensor3 K{};
  const double theta = computeExpansion(
    sphereNormal(n), sphereNormalDerivative(n, r), gamma, dgamma, K, 0.0);

  assert(approxEqual(theta, 0.0, 1.0e-12));

  std::cout << "Schwarzschild isotropic horizon (r = M/2): Theta = " << theta
            << " (~0)" << std::endl;
}

// End-to-end: AssembleGlobalMatrix must fill one residual per surface point equal
// to the expansion there. We build flat-space coordinate-sphere points, so every
// residual must equal 2/r.
void testAssembleGlobalMatrixResiduals() {
  const double r = 1.3;
  const std::array<AngularSample, 3> samples = {
    AngularSample{0.3, 0.4}, AngularSample{1.9, -0.2}, AngularSample{4.5, 1.1}
  };

  InterpolatedSurface surface;
  AHGuess ahGuess;
  for (const AngularSample& sample : samples) {
    const double cs = std::cos(sample.sigma);
    const Vector3 n = {cs * std::cos(sample.rho), cs * std::sin(sample.rho),
                       std::sin(sample.sigma)};

    InterpolatedSurfacePoint point;
    point.r = r;
    point.rho = sample.rho;
    point.sigma = sample.sigma;
    point.surfaceNormal = sphereNormal(n);
    point.surfaceNormalDerivative = sphereNormalDerivative(n, r);
    point.geometry.spatialMetric = identityMetric();
    surface.points.push_back(point);

    ahGuess.points.push_back({r, sample.rho, sample.sigma});
  }

  std::vector<std::vector<double>> matrixCoefficients;
  std::vector<double> residuals;
  AssembleGlobalMatrix(surface, ahGuess, matrixCoefficients, residuals);

  assert(residuals.size() == surface.numberOfPoints());
  for (double residual : residuals) {
    assert(approxEqual(residual, 2.0 / r, 1.0e-12));
  }

  std::cout << "AssembleGlobalMatrix fills one residual (= Theta) per surface point" << std::endl;
}

// The closed-form 3x3 inverse must satisfy gamma gamma^{-1} = I.
void testMetricInverse() {
  const Tensor3 g = {{{4.0, 1.0, 0.5}, {1.0, 3.0, 0.25}, {0.5, 0.25, 2.0}}};
  const Tensor3 gInv = inverse(g);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double product = 0.0;
      for (int k = 0; k < 3; ++k) {
        product += g[i][k] * gInv[k][j];
      }
      assert(approxEqual(product, i == j ? 1.0 : 0.0, 1.0e-12));
    }
  }

  std::cout << "Symmetric 3x3 inverse satisfies gamma gamma^{-1} = I" << std::endl;
}

// Spectral overload: AssembleGlobalMatrix must return the harmonic projection of
// Theta. On a flat coordinate sphere sampled on the basis collocation grid,
// Theta = 2/r everywhere (a constant field), so the only non-zero coefficient is
// the monopole a_00 = (2/r) sqrt(4 pi); all higher modes must vanish.
void testAssembleGlobalMatrixSpectralResidual() {
  const double r = 1.3;
  const SpectralBasis basis = BuildSpectralBasis(4);

  // Build the surface on exactly the basis collocation points (point p <-> mode grid p).
  InterpolatedSurface surface;
  surface.points.resize(basis.nPoints);
  for (std::size_t p = 0; p < basis.nPoints; ++p) {
    const double rho = basis.rho[p];
    const double sigma = basis.sigma[p];
    const double cs = std::cos(sigma);
    const Vector3 n = {cs * std::cos(rho), cs * std::sin(rho), std::sin(sigma)};

    InterpolatedSurfacePoint& point = surface.points[p];
    point.r = r;
    point.rho = rho;
    point.sigma = sigma;
    point.surfaceNormal = sphereNormal(n);
    point.surfaceNormalDerivative = sphereNormalDerivative(n, r);
    point.geometry.spatialMetric = identityMetric();
  }

  std::vector<std::vector<double>> matrixCoefficients;
  std::vector<double> residuals;
  AssembleGlobalMatrix(surface, basis, matrixCoefficients, residuals);

  assert(residuals.size() == basis.nModes);
  assert(approxEqual(residuals[0], (2.0 / r) * std::sqrt(4.0 * std::numbers::pi), 1.0e-11));
  for (std::size_t j = 1; j < basis.nModes; ++j) {
    assert(approxEqual(residuals[j], 0.0, 1.0e-11));
  }

  std::cout << "AssembleGlobalMatrix (spectral) projects Theta onto the harmonic basis" << std::endl;
}

}  // namespace exahype2::ccz4::ahf::tests::assemble_matrix



int main() {
  using namespace exahype2::ccz4::ahf::tests::assemble_matrix;

  std::cout << "Apparent Horizon Finder assemble-matrix tests begin..." << std::endl;

  testMetricInverse();
  testExpansionSchwarzschildIsotropicHorizon();
  testAssembleGlobalMatrixResiduals();
  testAssembleGlobalMatrixSpectralResidual();

  std::cout << "All assemble-matrix tests completed." << std::endl;
  return 0;
}
