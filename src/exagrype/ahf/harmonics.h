#ifndef APPARENT_HORIZON_FINDER_HARMONICS_H_
#define APPARENT_HORIZON_FINDER_HARMONICS_H_

#include <cstddef>
#include <utility>
#include <vector>

// Real spherical-harmonic basis and spectral collocation grid for the apparent
// horizon finder. Rdepends Depends on the C++ standard library only (no BLAS).
//
// The horizon shape h(rho, sigma) is expanded in real spherical harmonics as
//
//   h(rho, sigma) = sum_{l=0}^{lmax} sum_{m=-l}^{l} a_lm Y_lm(rho, sigma).
//
// Note: because the notes use sigma as an *elevation* angle, the harmonics are
// evaluated at colatitude theta = pi/2 - sigma and azimuth phi = rho, so that
// cos(theta) = sin(sigma) = mu. 
// The collocation grid uses Gauss-Legendre nodes in
// mu = sin(sigma) and a uniform grid in rho; with nSigma >= lmax+1 and
// nRho >= 2 lmax+1  so that the discrete transform is exact for the band l, m <= lmax, 
// and #coefficients == #equations.

namespace exahype2::ccz4::ahf {

/**
 * Ordered list of (l, m) modes, l = 0..lmax, m = -l..l.
 */
std::vector<std::pair<int, int>> EnumerateModes(int lmax);

/**
 * Gauss-Legendre nodes and weights on [-1, 1] (integrates d mu). Returns a pair
 * {nodes, weights}, each of length n, nodes in ascending order.
 */
std::pair<std::vector<double>, std::vector<double>> GaussLegendre(int n);

/**
 * Y_lm and its first/second rho- and sigma-derivatives at one (rho, sigma).
 */
struct ModeValues {
  double Y = 0.0;
  double Yrho = 0.0;
  double Ysigma = 0.0;
  double Yrhorho = 0.0;
  double Yrhosigma = 0.0;
  double Ysigmasigma = 0.0;
};

/**
 * Evaluate the real Y_lm and its rho/sigma derivatives at (rho, sigma).
 *
 * Uses mu = sin(sigma); the associated-Legendre part is differentiated by
 * recurrence and the sigma-derivatives obtained by the chain rule
 *
 *   d/dsigma     = cos(sigma) d/dmu,
 *   d^2/dsigma^2 = -sin(sigma) d/dmu + cos^2(sigma) d^2/dmu^2.
 *
 * The derivative recurrences divide by cos^2(sigma), so this is finite only for
 * sigma strictly inside (-pi/2, pi/2) (the Gauss-Legendre nodes never touch the poles).
 */
ModeValues EvalMode(int l, int m, double rho, double sigma);

/**
 * The real Y_lm at (rho, sigma), without any derivatives.
 * Introduced for plots, but keeping it as might me useful in the future.
 *
 * Same normalisation, Condon-Shortley phase and cos/sin(m rho) convention as
 * EvalMode, so EvalModeValue(l, m, rho, sigma) == EvalMode(l, m, rho, sigma).Y
 * wherever the latter is finite.
 *
 * This is not a second copy of that formula: EvalMode calls this function for
 * its own Y field, so the value is defined in exactly one place and the two
 * cannot drift apart. What the separate entry point buys is the ability to ask
 * for the value *without* paying for the derivatives -- and at the poles that
 * is not merely an optimisation but the only way to get an answer at all,
 * because EvalMode's derivative recurrences divide by cos^2(sigma) and return
 * non-finite Yrho/Ysigma/... at sigma = +/- pi/2. Reading .
 *
 * The core never needs it: the Gauss-Legendre collocation nodes are open in
 * mu = sin(sigma) and so never land on a pole, which is why EvalMode alone
 * sufficed until something had to evaluate h off that grid. Anything that
 * closes the surface does need it -- synthesising h on a plotting or output
 * mesh spanning the full sigma range, for instance.
 *
 * @param l    degree, l >= 0.
 * @param m    order, -l <= m <= l; m > 0 selects cos(|m| rho), m < 0
 *             sin(|m| rho).
 * @param rho  azimuth.
 * @param sigma  elevation in [-pi/2, pi/2] (not colatitude); the poles are
 *               safe here.
 */
double EvalModeValue(int l, int m, double rho, double sigma);

/**
 * Spectral basis plus collocation grid and precomputed design matrices.
 *
 * Each design matrix is stored row-major with shape (nPoints x nModes); use
 * designMatrixValue() to index it. 
 * Column j corresponds to modes[j], row p to the collocation point (rho[p], sigma[p]).
 */
struct SpectralBasis {
  int lmax = 0;
  int nSigma = 0;
  int nRho = 0;
  std::size_t nModes = 0;
  std::size_t nPoints = 0;
  std::vector<std::pair<int, int>> modes;

  std::vector<double> rho;     // (nPoints,)
  std::vector<double> sigma;   // (nPoints,)
  std::vector<double> weight;  // (nPoints,) quadrature weight, integrates f dmu drho

  std::vector<double> Y;             // (nPoints x nModes) design matrices
  std::vector<double> Yrho;
  std::vector<double> Ysigma;
  std::vector<double> Yrhorho;
  std::vector<double> Yrhosigma;
  std::vector<double> Ysigmasigma;

  double designMatrixValue(const std::vector<double>& matrix,
                           std::size_t point, std::size_t mode) const {
    return matrix[point * nModes + mode];
  }
};

/**
 * Construct a SpectralBasis for the given lmax. With the default node counts
 * (nSigma = lmax+1, nRho = 2 lmax+1) the discrete transform is exact for the
 * l, m <= lmax band. Pass nSigma/nRho <= 0 to use the defaults.
 */
SpectralBasis BuildSpectralBasis(int lmax, int nSigma = -1, int nRho = -1);

/**
 * h and its angular derivatives at every collocation point (each (nPoints,)).
 */
struct SynthesizedField {
  std::vector<double> h;
  std::vector<double> hRho;
  std::vector<double> hSigma;
  std::vector<double> hRhoRho;
  std::vector<double> hRhoSigma;
  std::vector<double> hSigmaSigma;
};

/**
 * Synthesis (coefficients -> point values): h = Y a, hRho = Yrho a, etc.
 * @param coefficients (nModes,) spherical-harmonic coefficient vector.
 */
SynthesizedField Synthesize(const SpectralBasis& basis,
                            const std::vector<double>& coefficients);

/**
 * Analysis (point values -> coefficients): a = Y^T (weight .* field), the
 * discrete spherical-harmonic transform.
 * @param field (nPoints,) sampled at the collocation points.
 */
std::vector<double> Project(const SpectralBasis& basis,
                            const std::vector<double>& field);

/**
 * Coefficient vector for a coordinate sphere h = radius (only a_00 != 0, since
 * Y_00 = 1/sqrt(4 pi) so a_00 = radius sqrt(4 pi)).
 */
std::vector<double> SphereCoefficients(const SpectralBasis& basis, double radius);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_HARMONICS_H_
