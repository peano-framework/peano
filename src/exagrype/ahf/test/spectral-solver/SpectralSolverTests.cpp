#include "../require-asserts.h"

#include "../../apparent-horizon-finder.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf::tests::spectral_solver {

namespace {

bool approxEqual(double a, double b, double tol) {
  return std::abs(a - b) <= tol;
}

}  // namespace

// Dense LU solve: recover the known solution of small systems and reject a
// singular matrix.
void testDenseLinearSolve() {
  // 2x + y = 3 ; x + 3y = 5  ->  x = 0.8, y = 1.4.
  const std::vector<std::vector<double>> A = {{2.0, 1.0}, {1.0, 3.0}};
  const std::vector<double> b = {3.0, 5.0};
  const std::vector<double> x = SolveLinearSystem(A, b);
  assert(approxEqual(x[0], 0.8, 1.0e-12));
  assert(approxEqual(x[1], 1.4, 1.0e-12));

  // A larger system: check the residual A x - b is ~0.
  const std::vector<std::vector<double>> C = {
    {4.0, 1.0, 2.0}, {1.0, 5.0, -1.0}, {2.0, -1.0, 6.0}};
  const std::vector<double> d = {7.0, 5.0, 7.0};
  const std::vector<double> y = SolveLinearSystem(C, d);
  for (std::size_t i = 0; i < 3; ++i) {
    double row = 0.0;
    for (std::size_t j = 0; j < 3; ++j) {
      row += C[i][j] * y[j];
    }
    assert(approxEqual(row, d[i], 1.0e-10));
  }

  // A singular matrix must be rejected.
  bool singularRejected = false;
  try {
    (void)SolveLinearSystem({{1.0, 1.0}, {1.0, 1.0}}, {2.0, 2.0});
  } catch (const std::runtime_error&) {
    singularRejected = true;
  }
  assert(singularRejected);

  std::cout << "Dense LU solves known systems and rejects singular matrices" << std::endl;
}

// Newton converges to a known coordinate sphere. The synthetic residual
// R(a) = project(2/h - 2/R_target) has its unique root at h = R_target everywhere,
// i.e. a_00 = R_target sqrt(4 pi) with all higher modes zero. 
// Starting from a larger sphere exercises the full damped-Newton path (dense Jacobian, LU, line
// search).
void testNewtonConvergesToKnownSphere() {
  const SpectralBasis basis = BuildSpectralBasis(4);
  const double targetRadius = 1.5;

  const SpectralResidual residual =
    [&basis, targetRadius](const std::vector<double>& a) {
      const SynthesizedField field = Synthesize(basis, a);
      std::vector<double> theta(field.h.size());
      for (std::size_t p = 0; p < field.h.size(); ++p) {
        theta[p] = 2.0 / field.h[p] - 2.0 / targetRadius;
      }
      return Project(basis, theta);
    };

  const std::vector<double> initial = SphereCoefficients(basis, 3.0);
  const SpectralNewtonResult result = SolveSpectralNewton(basis, residual, initial);

  assert(result.converged);
  assert(approxEqual(result.coefficients[0], targetRadius * std::sqrt(4.0 * std::numbers::pi), 1.0e-8));
  for (std::size_t j = 1; j < basis.nModes; ++j) {
    assert(approxEqual(result.coefficients[j], 0.0, 1.0e-8));
  }

  std::cout << "Newton converges to the known sphere in " << result.iterations
            << " iterations (final ||R||_inf = " << result.residualHistory.back()
            << ")" << std::endl;
}

// The runaway guard flags divergence. The synthetic residual R(a) = project(-1/h)
// has no finite root: ||R|| -> 0 as h -> infinity, so undamped Newton marches the
// surface outward (h doubles each step) until it exceeds the star-shaped cap.
void testRunawayGuardFlagsDivergence() {
  const SpectralBasis basis = BuildSpectralBasis(3);

  const SpectralResidual residual =
    [&basis](const std::vector<double>& a) {
      const SynthesizedField field = Synthesize(basis, a);
      std::vector<double> theta(field.h.size());
      for (std::size_t p = 0; p < field.h.size(); ++p) {
        theta[p] = -1.0 / field.h[p];
      }
      return Project(basis, theta);
    };

  const std::vector<double> initial = SphereCoefficients(basis, 1.0);
  const SpectralNewtonResult result = SolveSpectralNewton(basis, residual, initial);

  assert(!result.converged);
  assert(result.message.find("spurious") != std::string::npos
      || result.message.find("star") != std::string::npos);

  std::cout << "Runaway guard flags divergence: \"" << result.message << "\"" << std::endl;
}


// A NaN anywhere in the matrix must be rejected, not silently eliminated.
// std::max(a, b) is `a < b ? b : a`, so a NaN pivot compares false against the
// running best and a `best == 0.0` guard would let it through, returning a NaN
// solution that the caller cannot distinguish from a real step.
void testSingularGuardRejectsNaN() {
  const double nan = std::numeric_limits<double>::quiet_NaN();

  bool threw = false;
  try {
    SolveLinearSystem({{nan, 0.0}, {0.0, 1.0}}, {1.0, 1.0});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  // An all-zero column must still be rejected, as before.
  threw = false;
  try {
    SolveLinearSystem({{0.0, 0.0}, {0.0, 1.0}}, {1.0, 1.0});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);

  // A well-conditioned system is unaffected.
  const std::vector<double>
    x = SolveLinearSystem({{2.0, 0.0}, {0.0, 4.0}}, {2.0, 8.0});
  assert(approxEqual(x[0], 1.0, 1.0e-12));
  assert(approxEqual(x[1], 2.0, 1.0e-12));

  std::cout
    << "Singular guard rejects a NaN pivot and an empty column" << std::endl;
}

// An all-NaN residual must not read as converged: NormInfinity has to let NaN
// dominate, or ||R||_inf comes back as 0 and passes the tolerance test.
void testNonFiniteInitialResidualIsNotConvergence() {
  const SpectralBasis    basis = BuildSpectralBasis(2);
  const SpectralResidual residual =
    [&basis](const std::vector<double>&) -> std::vector<double> {
    return std::vector<double>(
      basis.nModes,
      std::numeric_limits<double>::quiet_NaN()
    );
  };

  const SpectralNewtonResult result = SolveSpectralNewton(
    basis,
    residual,
    SphereCoefficients(basis, 1.0)
  );

  assert(!result.converged);
  assert(result.message.find("non-finite residual") != std::string::npos);
  std::cout
    << "An all-NaN initial residual reports \"" << result.message << "\""
    << std::endl;
}


namespace {

/**
 * R_0 = (a_0 - t)^3 and R_j = a_j for j > 0, with t the coefficient of the unit
 * sphere. Newton removes a third of the error in a_0 per step, so the residual
 * falls by 8/27 per iteration: slow enough that the stopping tolerance decides
 * the iteration count.
 */
SpectralResidual cubicResidual(const SpectralBasis& basis) {
  const double target = SphereCoefficients(basis, 1.0)[0];
  return [target](const std::vector<double>& a) {
    std::vector<double> R = a;
    R[0] = std::pow(a[0] - target, 3);
    return R;
  };
}

}  // namespace

// The effective tolerance is tolerance / massScale.
void testMassScaleTightensTolerance() {
  const SpectralBasis basis = BuildSpectralBasis(1);
  const SpectralResidual residual = cubicResidual(basis);
  const std::vector<double> initial = SphereCoefficients(basis, 1.5);

  const SpectralNewtonResult loose =
    SolveSpectralNewton(basis, residual, initial, {.tolerance = 1.0e-2, .massScale = 1.0});
  const SpectralNewtonResult tight =
    SolveSpectralNewton(basis, residual, initial, {.tolerance = 1.0e-2, .massScale = 1.0e4});
  assert(loose.converged && tight.converged);
  assert(loose.residualHistory.back() < 1.0e-2);
  assert(loose.residualHistory.back() > 1.0e-6);
  assert(tight.residualHistory.back() < 1.0e-6);
  assert(tight.iterations > loose.iterations);

  std::cout << "massScale 1e4 tightens the tolerance: " << loose.iterations << " -> "
            << tight.iterations << " iterations" << std::endl;
}

// With recordIterationHistory, entry 0 is the initial guess and entry k the
// iterate accepted in iteration k.
void testIterationHistory() {
  const SpectralBasis basis = BuildSpectralBasis(1);
  const SpectralResidual residual = cubicResidual(basis);
  const std::vector<double> initial = SphereCoefficients(basis, 1.5);

  const SpectralNewtonResult withoutHistory =
    SolveSpectralNewton(basis, residual, initial, {.tolerance = 1.0e-2});
  assert(withoutHistory.iterationHistory.empty());

  const SpectralNewtonResult result = SolveSpectralNewton(
    basis, residual, initial, {.tolerance = 1.0e-2, .recordIterationHistory = true});
  assert(result.converged);
  assert(result.iterations == withoutHistory.iterations);
  assert(result.iterationHistory.size() == static_cast<std::size_t>(result.iterations) + 1);
  assert(result.iterationHistory.front().coefficients == initial);
  assert(result.iterationHistory.back().coefficients == result.coefficients);
  for (std::size_t k = 0; k < result.iterationHistory.size(); ++k) {
    const SpectralNewtonIterate& iterate = result.iterationHistory[k];
    assert(iterate.iteration == static_cast<int>(k));
    assert(iterate.residualNormInfinity == result.residualHistory[k]);
    assert(iterate.h.size() == basis.nPoints);
    assert(k > 0 || iterate.damping == 1.0);
  }

  std::cout << "The iteration history records the initial guess and every accepted step"
            << std::endl;
}

// A singular Jacobian ends the solve as a failure, never as convergence.
void testLUFailureIsNotConvergence() {
  const SpectralBasis basis = BuildSpectralBasis(1);
  const SpectralResidual constant = [&basis](const std::vector<double>&) {
    return std::vector<double>(basis.nModes, 1.0);
  };
  const std::vector<double> initial = SphereCoefficients(basis, 1.0);

  const SpectralNewtonResult result = SolveSpectralNewton(basis, constant, initial);
  assert(!result.converged);
  assert(result.iterations == 1);
  assert(result.message.rfind("LU failure", 0) == 0);
  assert(result.coefficients == initial);

  std::cout << "A singular Jacobian reports \"" << result.message << "\"" << std::endl;
}

// When no trial step of the line search reduces the residual, the solve stops
// and says whether the trials were worse or not evaluable at all.
void testLineSearchFailure() {
  const SpectralBasis basis = BuildSpectralBasis(1);
  std::vector<double> initial = SphereCoefficients(basis, 1.5);
  initial[1] = 0.1;
  const std::vector<double> target = SphereCoefficients(basis, 1.0);

  // Linear, R = a - target, at the initial guess and at the Jacobian's
  // probes, which each move one coefficient. The Newton step moves two, so
  // every trial point lands in the other branch.
  const auto residualWith = [&](double otherwise) -> SpectralResidual {
    return [&, otherwise](const std::vector<double>& a) {
      std::size_t moved = 0;
      for (std::size_t j = 0; j < a.size(); ++j) {
        moved += a[j] != initial[j] ? 1 : 0;
      }
      std::vector<double> R(a.size(), otherwise);
      if (moved <= 1) {
        for (std::size_t j = 0; j < a.size(); ++j) {
          R[j] = a[j] - target[j];
        }
      }
      return R;
    };
  };

  const SpectralNewtonResult worse = SolveSpectralNewton(basis, residualWith(1.0e3), initial);
  assert(!worse.converged);
  assert(worse.iterations == 1);
  assert(worse.message == "line search failed (no trial step reduced the residual)");
  assert(worse.coefficients == initial);

  const SpectralNewtonResult unevaluable = SolveSpectralNewton(
    basis, residualWith(std::numeric_limits<double>::quiet_NaN()), initial);
  assert(!unevaluable.converged);
  assert(unevaluable.iterations == 1);
  assert(unevaluable.message == "line search failed (every trial surface was non-finite)");

  std::cout << "A failed line search stops the solve and names the reason" << std::endl;
}

// MakeResidual returns NaN, not just a large number, for a trial surface it
// cannot evaluate: the line search and the runaway guard rely on it.
void testInvalidTrialSurfaceGivesNaNResidual() {
  const SpectralBasis basis = BuildSpectralBasis(2);
  const SpectralResidual residual =
    MakeAnalyticResidual(basis, SchwarzschildIsotropic{1.0}, {0.0, 0.0, 0.0});

  const std::vector<double> valid = residual(SphereCoefficients(basis, 0.7));
  for (const double entry : valid) {
    assert(std::isfinite(entry));
  }

  const std::vector<double> invalid = residual(SphereCoefficients(basis, -0.5));
  assert(invalid.size() == basis.nModes);
  for (const double entry : invalid) {
    assert(std::isnan(entry));
  }

  std::cout << "A trial surface with negative radius gives an all-NaN residual" << std::endl;
}

}  // namespace exahype2::ccz4::ahf::tests::spectral_solver


int main() {
  using namespace exahype2::ccz4::ahf::tests::spectral_solver;

  std::cout << "Apparent Horizon Finder spectral-solver tests begin..." << std::endl;

  testDenseLinearSolve();
  testNewtonConvergesToKnownSphere();
  testRunawayGuardFlagsDivergence();

  testSingularGuardRejectsNaN();

  testNonFiniteInitialResidualIsNotConvergence();
  testMassScaleTightensTolerance();
  testIterationHistory();
  testLUFailureIsNotConvergence();
  testLineSearchFailure();
  testInvalidTrialSurfaceGivesNaNResidual();


  std::cout << "All spectral-solver tests completed." << std::endl;
  return 0;
}
