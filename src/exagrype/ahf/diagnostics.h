#ifndef APPARENT_HORIZON_FINDER_DIAGNOSTICS_H_
#define APPARENT_HORIZON_FINDER_DIAGNOSTICS_H_

#include "ahf-data.h"
#include "harmonics.h"

#include <span>
#include <vector>

namespace exahype2::ccz4::ahf {

  /**
   * Post-solve horizon diagnostics.
   * The horizon area uses the metric induced on the surface
   *
   * x^i(rho, sigma) = s0^i + h n^i: q_AB = gamma_ij d_A x^i d_B x^j,
   *
   * where A, B in {rho, sigma}. The area is given by
   *
   * Area = integral sqrt(det q) drho dsigma. 
   *
   * The collocation weights integrate dmu drho (mu = sin sigma), 
   * so the sum carries an extra 1/cos(sigma) Jacobian to convert to dsigma drho.
   *
   * spatialMetric[p] is the physical metric gamma_ij at collocation point p of
   * the surface, in basis order, wherever it came from (an analytic spacetime
   * or the interpolated grid data).
   */
  double HorizonArea(
    const SpectralBasis&       basis,
    const std::vector<double>& coefficients,
    std::span<const Tensor3>   spatialMetric
  );

  /** The area on an analytic spacetime, whose metric is evaluated on the surface. */
  template <class Spacetime>
  double HorizonArea(
    const SpectralBasis&       basis,
    const std::vector<double>& coefficients,
    const Spacetime&           spacetime,
    const ReferencePoint&      referencePoint
  );

  /** Areal radius sqrt(area / 4 pi). */
  double ArealRadius(double area);

  /** Irreducible mass sqrt(area / 16 pi). */
  double IrreducibleMass(double area);

  /** Mean coordinate radius a_00 / sqrt(4 pi). */
  double MeanCoordinateRadius(const std::vector<double>& coefficients);

  /**
   * Per-degree coefficient power sum_m a_lm^2 (a spectral-convergence /
   * non-sphericity indicator).
   * Returns a vector of length lmax + 1 indexed by l.
   */
  std::vector<double> PowerSpectrum(
    const SpectralBasis&       basis,
    const std::vector<double>& coefficients
  );

} // namespace exahype2::ccz4::ahf

#endif // APPARENT_HORIZON_FINDER_DIAGNOSTICS_H_
