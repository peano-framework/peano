#include "../require-asserts.h"

#include "../../initial-guess.h"
#include "../../interpolated-surface-residual.h"
#include "../../iteration.h"

#include <array>
#include <cassert>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace exahype2::ccz4::ahf::tests::solve_pde_for_ah {
namespace {

static_assert(std::is_same_v<
  decltype(SolvePDEforAH(
    std::declval<const SpectralBasis&>(),
    std::declval<const CartesianPointInterpolator&>(),
    std::declval<const ReferencePoint&>(),
    std::declval<double>(),
    std::declval<const SpectralNewtonOptions&>()
  )),
  SpectralNewtonResult
>);

GridData MakeConstantHorizonGrid(double curvature = 1.0) {
  GridData grid;
  for (int ix = -2; ix <= 2; ++ix) {
    for (int iy = -2; iy <= 2; ++iy) {
      for (int iz = -2; iz <= 2; ++iz) {
        grid.coordinates.push_back({
          static_cast<double>(ix),
          static_cast<double>(iy),
          static_cast<double>(iz)
        });
        // gamma_ij = delta_ij and K_ij = curvature delta_ij. Therefore a
        // coordinate sphere has Theta = 2/r - 2 curvature.
        const std::array<double, NumberOfCompactFields> row{
          1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
          curvature, 0.0, 0.0, curvature, 0.0, curvature,
          3.0 * curvature, 1.0
        };
        grid.values.insert(grid.values.end(), row.begin(), row.end());
      }
    }
  }
  return grid;
}

CartesianToSurfaceOptions InterpolationOptions() {
  return {
    .method = CartesianToSurfaceMethod::UnweightedPolynomial,
    .numberOfNeighbours = 64,
    .polynomialDegree = 1,
    .allowOrderReduction = false
  };
}

SpectralNewtonResult SolveDirectly(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  double initialRadius,
  const SpectralNewtonOptions& options
) {
  const std::vector<double> initial =
    MakeSphericalInitialCoefficients(basis, initialRadius);
  assert(initial.size() == basis.nModes);
  const SpectralResidual residual = detail::MakeInterpolatedSurfaceResidual(
    basis, interpolator, {0.0, 0.0, 0.0}
  );
  return SolveSpectralNewton(basis, residual, initial, options);
}

void AssertSameResult(
  const SpectralNewtonResult& actual,
  const SpectralNewtonResult& expected
) {
  assert(actual.coefficients == expected.coefficients);
  assert(actual.converged == expected.converged);
  assert(actual.iterations == expected.iterations);
  assert(actual.residualHistory == expected.residualHistory);
  assert(actual.message == expected.message);
}

void TestSuccessfulOrchestrationAndResultPropagation() {
  const GridData grid = MakeConstantHorizonGrid();
  const CartesianPointInterpolator interpolator(grid, InterpolationOptions());
  const SpectralBasis basis = BuildSpectralBasis(0);
  constexpr double initialRadius = 1.3;

  const SpectralNewtonResult expected = SolveDirectly(
    basis, interpolator, initialRadius, {}
  );
  const SpectralNewtonResult actual = SolvePDEforAH(
    basis, interpolator, {0.0, 0.0, 0.0}, initialRadius
  );

  AssertSameResult(actual, expected);
  assert(actual.converged);
  assert(actual.coefficients.size() == basis.nModes);
  assert(actual.iterations > 0);
  assert(actual.residualHistory.size()
         == static_cast<std::size_t>(actual.iterations + 1));
}

void TestFailureResultPropagation() {
  const GridData grid = MakeConstantHorizonGrid();
  const CartesianPointInterpolator interpolator(grid, InterpolationOptions());
  const SpectralBasis basis = BuildSpectralBasis(0);
  constexpr double initialRadius = 1.3;
  const SpectralNewtonOptions options{.maxIterations = 0};

  const SpectralNewtonResult expected = SolveDirectly(
    basis, interpolator, initialRadius, options
  );
  const SpectralNewtonResult actual = SolvePDEforAH(
    basis, interpolator, {0.0, 0.0, 0.0}, initialRadius, options
  );

  AssertSameResult(actual, expected);
  assert(!actual.converged);
  assert(actual.message == "max_iter reached");
}

void TestExplicitSpectralInitialSurface() {
  const GridData grid = MakeConstantHorizonGrid();
  const CartesianPointInterpolator interpolator(grid, InterpolationOptions());
  const SpectralBasis basis = BuildSpectralBasis(0);
  const std::vector<double> initial =
    MakeSphericalInitialCoefficients(basis, 1.3);
  const SpectralNewtonResult expected = SolveDirectly(
    basis, interpolator, 1.3, {});
  const SpectralNewtonResult actual = SolvePDEforAH(
    basis, interpolator, {0.0, 0.0, 0.0}, initial);
  AssertSameResult(actual, expected);

  bool rejected = false;
  try {
    (void)SolvePDEforAH(
      basis, interpolator, {0.0, 0.0, 0.0}, std::vector<double>{});
  } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}

void TestInvalidLineSearchTrialDoesNotEscape() {
  // The analytic root is r=10, beyond this grid. Starting at r=1 causes a
  // later full Newton trial to leave the interpolation domain. The residual
  // adapter must translate that trial to NaNs so the existing line search can
  // shorten it or return its controlled failure result.
  const GridData grid = MakeConstantHorizonGrid(0.1);
  const CartesianPointInterpolator interpolator(grid, InterpolationOptions());
  const SpectralBasis basis = BuildSpectralBasis(0);

  SpectralNewtonResult result;
  bool exceptionEscaped = false;
  try {
    result = SolvePDEforAH(
      basis, interpolator, {0.0, 0.0, 0.0}, 1.0
    );
  } catch (const std::exception&) {
    exceptionEscaped = true;
  }

  assert(!exceptionEscaped);
  assert(result.coefficients.size() == basis.nModes);
  assert(!result.message.empty());
}

}  // namespace
}  // namespace exahype2::ccz4::ahf::tests::solve_pde_for_ah

int main() {
  using namespace exahype2::ccz4::ahf::tests::solve_pde_for_ah;
  TestSuccessfulOrchestrationAndResultPropagation();
  TestFailureResultPropagation();
  TestExplicitSpectralInitialSurface();
  TestInvalidLineSearchTrialDoesNotEscape();
  std::cout << "All SolvePDEforAH tests completed.\n";
}
