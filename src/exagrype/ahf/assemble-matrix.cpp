#include "assemble-matrix.h"

#include <cmath>
#include <stdexcept>

#include "ahf-data.h"
#include "expansion.h"
#include "harmonics.h"

namespace exahype2::ccz4::ahf {

void AssembleGlobalMatrix(const InterpolatedSurface& interpolatedSurface,
                          const AHGuess& ahGuess,
                          std::vector<std::vector<double>>& matrixCoefficients,
                          std::vector<double>& residuals) {
  (void)ahGuess;

  // Compute the Residual = Theta_I: the expansion scalar at each surface point. 
  // The surface point I corresponds to AHGuess::points[I] by construction (InterpolatedSurface::matches).
  // The A/B/C/D calculation for Theta is done by the discretisation-agnostic expansion kernel.
  // s_i and partial_i s_j are already present on each point (filled by the
  // shell-to-surface step).
  const std::size_t numberOfPoints = interpolatedSurface.numberOfPoints();
  residuals.resize(numberOfPoints);
  for (std::size_t i = 0; i < numberOfPoints; ++i) {
    residuals[i] = computeExpansion(interpolatedSurface.points[i]);
  }

  // Global matrix for the Newton update.
  //   FD solver:       (sparse) matrix J[h] with J delta h = -Theta[h] 
  //   Spectral solver: (dense) matrix  J[a] with J delta a = -Theta[a] 
  //
  // TODO(next kernel): assemble the Jacobian. In the finite-difference prototype
  // (ahf/fd_solver.py) this is a coloured sparse finite difference over the angular
  // stencil pattern; it needs the FD angular operators, which are not yet ported.
  // Left empty here so this pass delivers a verified residual in isolation.
  (void)matrixCoefficients;
}

void AssembleGlobalMatrix(const InterpolatedSurface& interpolatedSurface,
                          const SpectralBasis& basis,
                          std::vector<std::vector<double>>& matrixCoefficients,
                          std::vector<double>& residuals) {
  // Spectral residual: evaluate the expansion Theta at each collocation point and
  // project it onto the real spherical-harmonic basis, R = project(Theta). The
  // result lives in coefficient space (one entry per mode), which is what the
  // spectral Newton step solves for.
  //
  // Contract: surface point p corresponds to basis collocation point p (the
  // surface is sampled on the basis's (rho, sigma) grid). We check the counts
  // match; the projection weights come from the basis.
  const std::size_t numberOfPoints = interpolatedSurface.numberOfPoints();
  if (numberOfPoints != basis.nPoints) {
    throw std::invalid_argument(
      "AssembleGlobalMatrix: surface point count does not match the spectral basis");
  }

  std::vector<double> expansion(numberOfPoints);
  for (std::size_t p = 0; p < numberOfPoints; ++p) {
    expansion[p] = computeExpansion(interpolatedSurface.points[p]);
  }

  residuals = Project(basis, expansion);

  // Dense Newton Jacobian J[a] with J delta a = -R[a]. Deferred: it is formed by
  // finite differences over the coefficients a, which re-synthesises h and
  // re-interpolates the surface per perturbation, so it belongs to the iteration
  // loop (SolvePDEforAH), not to this per-surface assembly.
  (void)matrixCoefficients;
}
}  // namespace exahype2::ccz4::ahf
