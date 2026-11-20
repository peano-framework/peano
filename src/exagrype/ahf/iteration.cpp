#include "iteration.h"

#include "initial-guess.h"
#include "interpolated-surface-residual.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace exahype2::ccz4::ahf {

SpectralNewtonResult SolvePDEforAH(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  const ReferencePoint& referencePoint,
  double initialRadius,
  const SpectralNewtonOptions& options
) {
  if (!std::isfinite(initialRadius) || initialRadius <= 0.0) {
    throw std::invalid_argument(
      "SolvePDEforAH: initialRadius must be finite and strictly positive"
    );
  }

  const std::vector<double> initialCoefficients =
    MakeSphericalInitialCoefficients(basis, initialRadius);

  return SolvePDEforAH(
    basis, interpolator, referencePoint, initialCoefficients, options);
}

SpectralNewtonResult SolvePDEforAH(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  const ReferencePoint& referencePoint,
  const std::vector<double>& initialCoefficients,
  const SpectralNewtonOptions& options
) {
  if (initialCoefficients.size() != basis.nModes) {
    throw std::invalid_argument(
      "SolvePDEforAH: initial coefficient count must equal basis.nModes");
  }
  const SynthesizedField field = Synthesize(basis, initialCoefficients);
  for (std::size_t p = 0; p < basis.nPoints; ++p) {
    if (!std::isfinite(field.h[p]) || field.h[p] <= 0.0) {
      throw std::invalid_argument(
        "SolvePDEforAH: synthesized initial radius at point " +
        std::to_string(p) + " must be finite and strictly positive");
    }
  }

  // Reject an initial surface the data cannot cover before spending a Newton
  // solve on it. Without this the interpolator fails at every collocation
  // point, the residual is non-finite from the first evaluation, and the
  // failure surfaces far from its cause.
  {
    for (std::size_t p = 0; p < basis.nPoints; ++p) {
      const AHGuessPoint   guessPoint{field.h[p], basis.rho[p], basis.sigma[p]};
      const GridCoordinate target = SurfacePointToCartesian(
        guessPoint,
        referencePoint
      );
      if (!interpolator.coversTarget(target)) {
        const GridCoordinate low  = interpolator.domainMinimum();
        const GridCoordinate high = interpolator.domainMaximum();
        std::ostringstream   message;
        message
          << "SolvePDEforAH: the initial guess leaves the sampled domain. "
          << "Reference point (" << referencePoint.x << ", " << referencePoint.y
          << ", " << referencePoint.z
          << ") with its initial spectral surface puts collocation point "
          << p << " at (" << target.x << ", "
          << target.y << ", " << target.z << "), outside the sampled bounds ["
          << low.x << ", " << high.x << "] x [" << low.y << ", " << high.y
          << "] x [" << low.z << ", " << high.z
          << "]. Move the reference point towards the data or reduce the "
             "initial surface.";
        // Reported, not thrown: an unreachable initial guess is a statement
        // about the data and the guess, not a bug, and callers that search over
        // candidate reference points must be able to try the next one without
        // unwinding. Contrast initialRadius above, which is a precondition
        // violation and does throw.
        SpectralNewtonResult rejected;
        rejected.coefficients = initialCoefficients;
        rejected.converged    = false;
        rejected.iterations   = 0;
        rejected.message      = message.str();
        return rejected;
      }
    }
  }
  const SpectralResidual residual = detail::MakeInterpolatedSurfaceResidual(
    basis, interpolator, referencePoint
  );
  return SolveSpectralNewton(
    basis, residual, initialCoefficients, options
  );
}

void CheckCommonHorizonOfBBH(const AHGuess& initialAHGuess,
                             const ReferencePoint& referencePoint,
                             const std::vector<double>& masses,
                             bool& enableCommonHorizon,
                             ReferencePoint& commonReferencePoint,
                             double& commonMass,
                             AHGuess& commonAHGuesses) {
  (void)initialAHGuess;
  (void)referencePoint;
  (void)masses;
  (void)enableCommonHorizon;
  (void)commonReferencePoint;
  (void)commonMass;
  (void)commonAHGuesses;
  //used in SolvePDEforAH() for pre-conditioning for binary black hole scenarios. It calculates
  //     the coordinate distance between the two reference points and compares it to the sum of
  //     the apparent horizon sizes ($h_{1,2}^{(0)}$) in the corresponding direction. If the latter
  //     is larger, we tell SolvePDEforAH() to solve AH first using the weight center of the two
  //     black holes as a reference point, and provide it with associated mass and new initial
  //     guess functions.
  // – Input: Initial AH Guess functions ($ \{h_{1,2}^{(0)}(\rho _I, \sigma _I)\}$), Reference
  //      points($\{s_1,s_2\}$), The black hole mass associated with corresponding reference
  //      points($\{M_1, M_2\}$).
  // – Output: Flag to enable common horizon search, new reference point($s_{com}$), mass
  //      scale($M_{com}$), and Initial AH Guess functions ($ \{h_{com}^{(0)}(\rho _I, \sigma _I)\}$)
}

}  // namespace exahype2::ccz4::ahf
