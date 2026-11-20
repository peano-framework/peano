#include "ah-search.h"

#include "ah-finder-options.h"
#include "harmonics.h"
#include "iteration.h"

#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace exahype2::ccz4::ahf {
namespace {

AHFinderOptions SingleAttemptOptions(
  const AHSearchOptions& options,
  const AHSearchAttempt& attempt
) {
  return {
    .referencePoint = attempt.referencePoint,
    .initialRadius = attempt.initialRadius,
    .lMax = options.lMax,
    .interpolation = options.interpolation,
    .solver = options.solver
  };
}

AHSearchResult ExecuteAttempts(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  const std::vector<AHSearchAttempt>& attempts,
  const SpectralNewtonOptions& solverOptions,
  const std::vector<ReferencePoint>& excludedPoints = {}
) {
  AHSearchResult search;
  search.attempts.reserve(attempts.size());
  for (std::size_t index = 0; index < attempts.size(); ++index) {
    const AHSearchAttempt& attempt = attempts[index];
    const SpectralNewtonResult solver = attempt.initialCoefficients.empty()
      ? SolvePDEforAH(
          basis, interpolator, attempt.referencePoint, attempt.initialRadius,
          solverOptions)
      : SolvePDEforAH(
          basis, interpolator, attempt.referencePoint,
          attempt.initialCoefficients, solverOptions);
    AHFinderResult result = BuildAHFinderResult(
      basis, interpolator, attempt.referencePoint, solver);
    const bool reconstructed =
      result.solver.converged && result.surfaceAvailable;
    std::string rejection;
    if (reconstructed && !(result.relativeExpansionError <=
                           solverOptions.maxRelativeExpansionError)) {
      rejection = "relative expansion error " +
        std::to_string(result.relativeExpansionError) + " exceeds " +
        std::to_string(solverOptions.maxRelativeExpansionError) +
        " between the collocation points (aliased surface)";
    }
    for (const ReferencePoint& point : excludedPoints) {
      if (reconstructed && rejection.empty() &&
          SurfaceEncloses(result, point)) {
        rejection = "surface encloses the excluded point (" +
          std::to_string(point.x) + ", " + std::to_string(point.y) + ", " +
          std::to_string(point.z) + "), another black hole";
      }
    }
    const bool succeeded = reconstructed && rejection.empty();
    search.attempts.push_back({
      .referencePoint = attempt.referencePoint,
      .initialRadius = attempt.initialRadius,
      .initialCoefficients = attempt.initialCoefficients,
      .result = result,
      .succeeded = succeeded,
      .rejection = std::move(rejection)
    });
    if (succeeded) {
      search.converged = true;
      search.successfulAttempt = index;
      search.horizon = std::move(result);
      return search;
    }
  }
  return search;
}

}  // namespace

AHSearchAttempt MakePreviousSurfaceAttempt(const AHFinderResult& previous) {
  if (!previous.surfaceAvailable || previous.solver.coefficients.empty()) {
    throw std::invalid_argument(
      "MakePreviousSurfaceAttempt: previous surface is unavailable");
  }
  return {
    .referencePoint = previous.referencePoint,
    .initialRadius = previous.meanRadius,
    .initialCoefficients = previous.solver.coefficients
  };
}

void ValidateAHSearchOptions(const AHSearchOptions& options) {
  if (options.attempts.empty()) {
    throw std::invalid_argument(
      "ValidateAHSearchOptions: at least one search attempt is required");
  }
  const SpectralBasis basis = BuildSpectralBasis(options.lMax);
  for (const AHSearchAttempt& attempt : options.attempts) {
    if (attempt.initialCoefficients.empty()) {
      ValidateAHFinderOptions(SingleAttemptOptions(options, attempt));
      continue;
    }
    AHSearchAttempt validationAttempt = attempt;
    validationAttempt.initialRadius = 1.0;
    ValidateAHFinderOptions(SingleAttemptOptions(options, validationAttempt));
    if (attempt.initialCoefficients.size() != basis.nModes) {
      throw std::invalid_argument(
        "ValidateAHSearchOptions: initial coefficient count does not match lMax");
    }
    const SynthesizedField field = Synthesize(basis, attempt.initialCoefficients);
    for (std::size_t p = 0; p < field.h.size(); ++p) {
      if (!std::isfinite(field.h[p]) || field.h[p] <= 0.0) {
        throw std::invalid_argument(
          "ValidateAHSearchOptions: explicit initial surface has an invalid "
          "radius at point " + std::to_string(p));
      }
    }
  }
}

AHSearchResult FindApparentHorizon(const GridData& gridData,
                                   const AHSearchOptions& options) {
  ValidateAHSearchOptions(options);

  const SpectralBasis basis = BuildSpectralBasis(options.lMax);
  const CartesianPointInterpolator interpolator(gridData, options.interpolation);

  return ExecuteAttempts(basis, interpolator, options.attempts, options.solver);
}

void ValidateAHMultiHorizonOptions(const AHMultiHorizonOptions& options) {
  if (options.targets.empty()) {
    throw std::invalid_argument(
      "ValidateAHMultiHorizonOptions: at least one horizon target is required");
  }
  std::unordered_set<std::string> ids;
  for (const AHHorizonTarget& target : options.targets) {
    if (target.id.empty()) {
      throw std::invalid_argument(
        "ValidateAHMultiHorizonOptions: target IDs must not be empty");
    }
    if (!ids.insert(target.id).second) {
      throw std::invalid_argument(
        "ValidateAHMultiHorizonOptions: target IDs must be unique");
    }
    ValidateAHSearchOptions({
      .attempts = target.attempts,
      .lMax = options.lMax,
      .interpolation = options.interpolation,
      .solver = options.solver
    });
    if (target.massScale.has_value() &&
        (!std::isfinite(*target.massScale) || *target.massScale <= 0.0)) {
      throw std::invalid_argument(
        "ValidateAHMultiHorizonOptions: target massScale must be finite and positive");
    }
    for (const ReferencePoint& point : target.excludedPoints) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z)) {
        throw std::invalid_argument(
          "ValidateAHMultiHorizonOptions: excluded points must be finite");
      }
    }
  }
}

AHMultiHorizonResult FindApparentHorizons(
  const GridData& gridData,
  const AHMultiHorizonOptions& options
) {
  ValidateAHMultiHorizonOptions(options);
  const SpectralBasis basis = BuildSpectralBasis(options.lMax);
  const CartesianPointInterpolator interpolator(gridData, options.interpolation);

  return FindApparentHorizons(basis, interpolator, options);
}

AHMultiHorizonResult FindApparentHorizons(
  const SpectralBasis& basis,
  const CartesianPointInterpolator& interpolator,
  const AHMultiHorizonOptions& options
) {
  ValidateAHMultiHorizonOptions(options);
  if (basis.lmax != options.lMax) {
    throw std::invalid_argument(
      "FindApparentHorizons: supplied basis does not match options.lMax");
  }

  AHMultiHorizonResult result;
  result.allTargetsFound = true;
  result.targets.reserve(options.targets.size());
  for (const AHHorizonTarget& target : options.targets) {
    SpectralNewtonOptions targetSolver = options.solver;
    if (target.massScale.has_value()) {
      targetSolver.massScale = *target.massScale;
    }
    AHSearchResult search = ExecuteAttempts(
      basis, interpolator, target.attempts, targetSolver,
      target.excludedPoints);
    result.anyHorizonFound = result.anyHorizonFound || search.converged;
    result.allTargetsFound = result.allTargetsFound && search.converged;
    result.targets.push_back({
      .id = target.id,
      .kind = target.kind,
      .converged = search.converged,
      .successfulAttempt = search.successfulAttempt,
      .horizon = std::move(search.horizon),
      .attempts = std::move(search.attempts),
      .massScale = target.massScale
    });
  }
  return result;
}

}  // namespace exahype2::ccz4::ahf
