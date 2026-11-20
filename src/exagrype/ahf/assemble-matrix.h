#ifndef APPARENT_HORIZON_FINDER_ASSEMBLE_MATRIX_H_
#define APPARENT_HORIZON_FINDER_ASSEMBLE_MATRIX_H_

#include <vector>

namespace exahype2::ccz4::ahf {

struct InterpolatedSurface;
struct AHGuess;
struct SpectralBasis;

/**
 * Assemble the global system matrix and residuals from the interpolated surface data.
 *
 * Real-space (finite-difference) form: the residual is the expansion Theta_I at
 * each surface point, so residuals has one entry per surface point.
 */
void AssembleGlobalMatrix(const InterpolatedSurface& interpolatedSurface,
                          const AHGuess& ahGuess,
                          std::vector<std::vector<double>>& matrixCoefficients,
                          std::vector<double>& residuals);

/**
 * Spectral form: the residual is the harmonic projection of the expansion,
 * R = project(Theta) = Y^T (weight .* Theta), so residuals has one entry per
 * spectral mode (basis.nModes). Surface point p must correspond to basis
 * collocation point p (same (rho, sigma) grid); throws std::invalid_argument
 * otherwise.
 */
void AssembleGlobalMatrix(const InterpolatedSurface& interpolatedSurface,
                          const SpectralBasis& basis,
                          std::vector<std::vector<double>>& matrixCoefficients,
                          std::vector<double>& residuals);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_ASSEMBLE_MATRIX_H_
