#include "spectral-solver.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numbers>

#include "dense-linear-algebra.h"

// The residual R(a) is injected; everything here is the Newton machinery which
// is discretisation-agnostic.

namespace exahype2::ccz4::ahf {

namespace {

double NormInfinity(const std::vector<double>& v) {
  double norm = 0.0;
  for (double value : v) {
    const double magnitude = std::abs(value);
    // std::max(a, b) is `a < b ? b : a`, and any comparison against NaN is
    // false, so std::max(norm, NaN) would return norm.
    if (std::isnan(magnitude)) {
      return magnitude;
    }
    norm = std::max(norm, magnitude);
  }
  return norm;
}

double NormTwo(const std::vector<double>& v) {
  double sum = 0.0;
  for (double value : v) {
    sum += value * value;
  }
  return std::sqrt(sum);
}

bool AllFinite(const std::vector<double>& v) {
  for (double value : v) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

// Snapshot one state of the march for SpectralNewtonResult::iterationHistory.
SpectralNewtonIterate RecordIterate(
  int                        iteration,
  const std::vector<double>& a,
  const std::vector<double>& R,
  double                     lambda,
  const std::vector<double>& h
) {
  SpectralNewtonIterate record;
  record.iteration            = iteration;
  record.residualNormInfinity = NormInfinity(R);
  record.residualNormTwo      = NormTwo(R);
  record.damping              = lambda;
  record.coefficients         = a;
  record.h                    = h;
  return record;
}

// Newton-Raphson update step: a + lambda * delta a
std::vector<double> Step(const std::vector<double>& a, double lambda,
                         const std::vector<double>& d) {
  std::vector<double> result(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + lambda * d[i];
  }
  return result;
}

}  // namespace

SpectralNewtonResult SolveSpectralNewton(const SpectralBasis& basis,
                                         const SpectralResidual& residual,
                                         const std::vector<double>& initialCoefficients,
                                         const SpectralNewtonOptions& options) {
  const std::size_t n = initialCoefficients.size();
  const double effectiveTolerance =
    options.tolerance / (options.massScale > 0.0 ? options.massScale : 1.0);

  SpectralNewtonResult result;
  std::vector<double> a = initialCoefficients;

  // Runaway cap from the initial mean radius: a_00 = r_mean * sqrt(4 pi).  The
  // guard watches the synthesised h, since a runaway collapses a_00 while blowing
  // up the higher modes.
  const double initialMeanRadius = std::abs(a.empty() ? 0.0 : a[0]) / std::sqrt(4.0 * std::numbers::pi);
  const double radiusCap = initialMeanRadius > 0.0
    ? options.maxRadiusFactor * initialMeanRadius
    : std::numeric_limits<double>::infinity();

  std::vector<double> R = residual(a);
  double residualNorm = NormInfinity(R);
  result.residualHistory.push_back(residualNorm);

  // A non-finite residual from the very first evaluation is a property of the
  // initial guess, not of anything the iteration did: the trial surface could
  // not be evaluated at all (e.g. outside the data domain, unavailable
  // derivatives, invalid geometry).
  // This is reported separately to avoid confusion with other failures, e.g.
  // convergence failure.
  if (!AllFinite(R)) {
  result.coefficients = a;
  result.converged    = false;
  result.iterations   = 0;
  result.message
    = "initial guess produced a non-finite residual (surface not evaluable: "
      "check the reference point, the initial radius, the data domain, and the "
      "variable mapping)";
  return result;
  }

  if (options.recordIterationHistory) {
    result.iterationHistory.push_back(
      RecordIterate(0, a, R, 1.0, Synthesize(basis, a).h)
    );
  }

  for (int iteration = 1; iteration <= options.maxIterations; ++iteration) {
    if (residualNorm < effectiveTolerance) {
      result.coefficients = a;
      result.converged = true;
      result.iterations = iteration - 1;
      result.message = "converged";
      return result;
    }

    // Dense central-difference Jacobian J[p][q] = dR_p / da_q.
    std::vector<std::vector<double>> jacobian(n, std::vector<double>(n, 0.0));
    for (std::size_t q = 0; q < n; ++q) {
      const double step = options.finiteDifferenceStep * std::max(1.0, std::abs(a[q]));
      std::vector<double> aPlus = a;
      std::vector<double> aMinus = a;
      aPlus[q] += step;
      aMinus[q] -= step;
      const std::vector<double> residualPlus = residual(aPlus);
      const std::vector<double> residualMinus = residual(aMinus);
      for (std::size_t p = 0; p < n; ++p) {
        jacobian[p][q] = (residualPlus[p] - residualMinus[p]) / (2.0 * step);
      }
    }

    std::vector<double> delta;
    try {
      std::vector<double> negativeResidual(n);
      for (std::size_t p = 0; p < n; ++p) {
        negativeResidual[p] = -R[p];
      }
      delta = SolveLinearSystem(jacobian, negativeResidual);
    } catch (const std::exception& error) {
      result.coefficients = a;
      result.converged = false;
      result.iterations = iteration;
      result.message = std::string("LU failure: ") + error.what();
      return result;
    }

    // Backtracking line search on ||R||_2.
    double lambda = 1.0;
    const double referenceNorm = NormTwo(R);
    std::vector<double> aTrial;
    std::vector<double> residualTrial;
    bool accepted = false;
    bool                anyTrialFinite = false;
    for (int lineSearchStep = 0; lineSearchStep < options.maxLineSearchSteps; ++lineSearchStep) {
      aTrial = Step(a, lambda, delta);
      residualTrial = residual(aTrial);
      const bool finite = AllFinite(residualTrial);
      anyTrialFinite    = anyTrialFinite || finite;
      if (finite && NormTwo(residualTrial) < referenceNorm) {
        accepted = true;
        break;
      }
      lambda *= 0.5;
    }
    if (!accepted) {
      result.coefficients = a;
      result.converged = false;
      result.iterations = iteration;
      // Two different faults share this exit: every trial surface was
      // unevaluable, or they were evaluable but none reduced the residual.
      result.message
        = anyTrialFinite
            ? "line search failed (no trial step reduced the residual)"
            : "line search failed (every trial surface was non-finite)";
      return result;
    }

    a = aTrial;
    R = residualTrial;

    // Star-shaped / runaway guard on the synthesised radius.
    const SynthesizedField field = Synthesize(basis, a);
    if (options.recordIterationHistory) {
      result.iterationHistory.push_back(
        RecordIterate(iteration, a, R, lambda, field.h)
      );
    }
    const double hMin = *std::min_element(field.h.begin(), field.h.end());
    const double hMax = *std::max_element(field.h.begin(), field.h.end());
    if (hMin <= 0.0 || hMax > radiusCap) {
      result.coefficients = a;
      result.converged = false;
      result.iterations = iteration;
      result.message = "diverged to spurious / non-star-shaped root (h left the valid range)";
      return result;
    }

    residualNorm = NormInfinity(R);
    result.residualHistory.push_back(residualNorm);
  }

  result.coefficients = a;
  result.converged = residualNorm < effectiveTolerance;
  result.iterations = options.maxIterations;
  result.message = result.converged ? "converged" : "max_iter reached";
  return result;
}

}  // namespace exahype2::ccz4::ahf
