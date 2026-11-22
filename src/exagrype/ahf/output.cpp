#include "output.h"

#include "harmonics.h"
#include "iteration.h"

#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace exahype2::ccz4::ahf {
namespace {

const char* HorizonKindName(AHHorizonKind kind) {
  return kind == AHHorizonKind::Common ? "common" : "individual";
}

const char* CommonValidationStatusName(AHCommonHorizonValidationStatus status) {
  switch (status) {
    case AHCommonHorizonValidationStatus::FullyValidated: return "fully-validated";
    case AHCommonHorizonValidationStatus::ProvisionallyValidated: return "provisionally-validated";
    case AHCommonHorizonValidationStatus::Rejected: return "rejected";
  }
  return "unknown";
}

const char* EnclosureStatusName(AHEnclosureValidationStatus status) {
  switch (status) {
    case AHEnclosureValidationStatus::Passed: return "passed";
    case AHEnclosureValidationStatus::Failed: return "failed";
    case AHEnclosureValidationStatus::Unavailable: return "unavailable";
  }
  return "unknown";
}

std::string SafeFilenameId(const std::string& id) {
  std::string safe = id;
  for (char& character : safe) {
    const unsigned char value = static_cast<unsigned char>(character);
    if (!std::isalnum(value) && character != '-' && character != '_' &&
        character != '.') {
      character = '_';
    }
  }
  return safe;
}

std::string TargetFilename(const AHMultiHorizonOutputOptions& options,
                           std::size_t targetIndex, const std::string& targetId,
                           const char* extension) {
  return options.horizonFilenamePrefix + "-" + std::to_string(targetIndex) +
         "-" + SafeFilenameId(targetId) + extension;
}

AHSearchResult AsSearchResult(const AHHorizonTargetResult& target) {
  return {
    .converged = target.converged,
    .successfulAttempt = target.successfulAttempt,
    .horizon = target.horizon,
    .attempts = target.attempts
  };
}

void ValidateVisualizationOptions(const AHVisualizationOptions& options) {
  if (options.numberOfRhoPoints < 3) {
    throw std::invalid_argument(
      "AH visualization requires at least 3 rho points");
  }
  if (options.numberOfSigmaPoints < 3) {
    throw std::invalid_argument(
      "AH visualization requires at least 3 sigma points including the poles");
  }
}

double EvaluateRadius(const AHFinderResult& result, double rho, double sigma,
                      bool atPole) {
  double radius = 0.0;
  const auto modes = EnumerateModes(result.lMax);
  if (modes.size() != result.solver.coefficients.size()) {
    throw std::invalid_argument(
      "AH visualization coefficient count does not match lMax");
  }
  for (std::size_t j = 0; j < modes.size(); ++j) {
    const auto [l, m] = modes[j];
    double y = 0.0;
    if (atPole) {
      // Only m=0 survives at a pole. Avoid derivative recurrences in EvalMode,
      // which intentionally assume collocation points strictly inside the poles.
      if (m == 0) {
        const double sign = sigma > 0.0 || l % 2 == 0 ? 1.0 : -1.0;
        y = sign * std::sqrt((2.0 * l + 1.0) /
                             (4.0 * std::numbers::pi));
      }
    } else {
      y = EvalMode(l, m, rho, sigma).Y;
    }
    radius += result.solver.coefficients[j] * y;
  }
  if (!std::isfinite(radius) || radius <= 0.0) {
    throw std::runtime_error(
      "AH visualization produced a non-positive or non-finite radius");
  }
  return radius;
}

void AppendPoint(AHVisualizationSurface& surface, const ReferencePoint& origin,
                 double radius, double rho, double sigma) {
  const double cosSigma = std::cos(sigma);
  surface.points.push_back({
    origin.x + radius * cosSigma * std::cos(rho),
    origin.y + radius * cosSigma * std::sin(rho),
    origin.z + radius * std::sin(sigma)
  });
  surface.radius.push_back(radius);
}

std::ofstream OpenOutput(const std::string& filename) {
  std::ofstream stream(filename);
  if (!stream) {
    throw std::runtime_error("Unable to open AH output file: " + filename);
  }
  return stream;
}

}  // namespace

std::string AHCatalogueFilename(const AHMultiHorizonOutputOptions& options,
                                const char* kind, std::size_t index,
                                const std::string& id, const char* extension) {
  return options.horizonFilenamePrefix + "-" + kind + "-" +
    std::to_string(index) + "-" + SafeFilenameId(id) + extension;
}

void ValidateAHOutputOptions(const AHOutputOptions& options) {
  if (options.summaryFilename.empty()) {
    throw std::invalid_argument(
      "ValidateAHOutputOptions: summaryFilename must not be empty");
  }
  if (options.surfaceFilename.empty()) {
    throw std::invalid_argument(
      "ValidateAHOutputOptions: surfaceFilename must not be empty");
  }
  if (options.summaryFilename == options.surfaceFilename) {
    throw std::invalid_argument(
      "ValidateAHOutputOptions: summary and surface filenames must differ");
  }
  ValidateVisualizationOptions(options.visualization);
}

void WriteAHSummary(const AHFinderResult& result, const std::string& filename) {
  std::ofstream output = OpenOutput(filename);
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << std::boolalpha
         << "AHFinderSummaryVersion = 1\n"
         << "solver.converged = " << result.solver.converged << '\n'
         << "surfaceAvailable = " << result.surfaceAvailable << '\n'
         << "solver.iterations = " << result.solver.iterations << '\n'
         << "solver.message = " << std::quoted(result.solver.message) << '\n'
         << "referencePoint = " << result.referencePoint.x << ' '
         << result.referencePoint.y << ' ' << result.referencePoint.z << '\n'
         << "lMax = " << result.lMax << '\n';

  if (result.surfaceAvailable) {
    output << "minRadius = " << result.minRadius << '\n'
           << "meanRadius = " << result.meanRadius << '\n'
           << "maxRadius = " << result.maxRadius << '\n'
           << "meanCoordinateRadius = " << result.meanCoordinateRadius << '\n'
           << "area = " << result.area << '\n'
           << "arealRadius = " << result.arealRadius << '\n'
           << "irreducibleMass = " << result.irreducibleMass << '\n'
           << "spectralResidualInfinityNorm = "
           << result.spectralResidualInfinityNorm << '\n'
           << "maxAbsExpansion = " << result.maxAbsExpansion << '\n'
           << "maxAbsExpansionOverResolved = "
           << result.maxAbsExpansionOverResolved << '\n'
           << "relativeExpansionError = "
           << result.relativeExpansionError << '\n';
  } else {
    output << "minRadius = unavailable\n"
           << "meanRadius = unavailable\n"
           << "maxRadius = unavailable\n"
           << "meanCoordinateRadius = unavailable\n"
           << "area = unavailable\n"
           << "arealRadius = unavailable\n"
           << "irreducibleMass = unavailable\n"
           << "spectralResidualInfinityNorm = unavailable\n"
           << "maxAbsExpansion = unavailable\n"
           << "maxAbsExpansionOverResolved = unavailable\n"
           << "relativeExpansionError = unavailable\n";
  }

  output << "residualHistory.count = " << result.solver.residualHistory.size()
         << '\n';
  for (std::size_t i = 0; i < result.solver.residualHistory.size(); ++i) {
    output << "residualHistory[" << i << "] = "
           << result.solver.residualHistory[i] << '\n';
  }
  output << "coefficients.count = " << result.solver.coefficients.size() << '\n';
  for (std::size_t i = 0; i < result.solver.coefficients.size(); ++i) {
    output << "coefficients[" << i << "] = "
           << result.solver.coefficients[i] << '\n';
  }
  if (!output) {
    throw std::runtime_error("Failed while writing AH summary: " + filename);
  }
}

void WriteAHSearchSummary(const AHSearchResult& result,
                          const std::string& filename) {
  std::ofstream output = OpenOutput(filename);
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << std::boolalpha
         << "AHSearchSummaryVersion = 1\n"
         << "converged = " << result.converged << '\n';
  if (result.converged) {
    output << "successfulAttempt = " << result.successfulAttempt << '\n';
  } else {
    output << "successfulAttempt = unavailable\n";
  }
  output << "attempts.count = " << result.attempts.size() << '\n';
  for (std::size_t i = 0; i < result.attempts.size(); ++i) {
    const AHSearchAttemptResult& attempt = result.attempts[i];
    const std::string prefix = "attempts[" + std::to_string(i) + "].";
    output << prefix << "referencePoint = " << attempt.referencePoint.x << ' '
           << attempt.referencePoint.y << ' ' << attempt.referencePoint.z << '\n'
           << prefix << "initialSurface = "
           << (attempt.initialCoefficients.empty() ? "sphere" : "spectral")
           << '\n'
           << prefix << "initialRadius = ";
    if (attempt.initialCoefficients.empty()) {
      output << attempt.initialRadius << '\n';
    } else {
      output << "unavailable\n";
    }
    output << prefix << "succeeded = " << attempt.succeeded << '\n'
           << prefix << "solver.converged = "
           << attempt.result.solver.converged << '\n'
           << prefix << "surfaceAvailable = "
           << attempt.result.surfaceAvailable << '\n'
           << prefix << "solver.iterations = "
           << attempt.result.solver.iterations << '\n'
           << prefix << "solver.message = "
           << std::quoted(attempt.result.solver.message) << '\n'
           << prefix << "rejection = " << std::quoted(attempt.rejection) << '\n'
           << prefix << "residualHistory.count = "
           << attempt.result.solver.residualHistory.size() << '\n';
    output << prefix << "initialCoefficients.count = "
           << attempt.initialCoefficients.size() << '\n';
    for (std::size_t j = 0; j < attempt.initialCoefficients.size(); ++j) {
      output << prefix << "initialCoefficients[" << j << "] = "
             << attempt.initialCoefficients[j] << '\n';
    }
    for (std::size_t j = 0;
         j < attempt.result.solver.residualHistory.size(); ++j) {
      output << prefix << "residualHistory[" << j << "] = "
             << attempt.result.solver.residualHistory[j] << '\n';
    }
  }
  if (result.converged && result.horizon.has_value()) {
    const AHFinderResult& horizon = *result.horizon;
    output << "horizon.referencePoint = " << horizon.referencePoint.x << ' '
           << horizon.referencePoint.y << ' ' << horizon.referencePoint.z << '\n'
           << "horizon.lMax = " << horizon.lMax << '\n'
           << "horizon.minRadius = " << horizon.minRadius << '\n'
           << "horizon.meanRadius = " << horizon.meanRadius << '\n'
           << "horizon.maxRadius = " << horizon.maxRadius << '\n'
           << "horizon.meanCoordinateRadius = "
           << horizon.meanCoordinateRadius << '\n'
           << "horizon.area = " << horizon.area << '\n'
           << "horizon.arealRadius = " << horizon.arealRadius << '\n'
           << "horizon.irreducibleMass = " << horizon.irreducibleMass << '\n'
           << "horizon.spectralResidualInfinityNorm = "
           << horizon.spectralResidualInfinityNorm << '\n'
           << "horizon.maxAbsExpansion = " << horizon.maxAbsExpansion << '\n'
           << "horizon.maxAbsExpansionOverResolved = "
           << horizon.maxAbsExpansionOverResolved << '\n'
           << "horizon.relativeExpansionError = "
           << horizon.relativeExpansionError << '\n'
           << "horizon.coefficients.count = "
           << horizon.solver.coefficients.size() << '\n';
    for (std::size_t i = 0; i < horizon.solver.coefficients.size(); ++i) {
      output << "horizon.coefficients[" << i << "] = "
             << horizon.solver.coefficients[i] << '\n';
    }
  }
  if (!output) {
    throw std::runtime_error(
      "Failed while writing AH search summary: " + filename);
  }
}

void WriteAHMultiHorizonSummary(const AHMultiHorizonResult& result,
                                const std::string& filename) {
  std::ofstream output = OpenOutput(filename);
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << std::boolalpha
         << "AHMultiHorizonSummaryVersion = 1\n"
         << "anyHorizonFound = " << result.anyHorizonFound << '\n'
         << "allTargetsFound = " << result.allTargetsFound << '\n'
         << "targets.count = " << result.targets.size() << '\n';

  for (std::size_t i = 0; i < result.targets.size(); ++i) {
    const AHHorizonTargetResult& target = result.targets[i];
    const std::string prefix = "targets[" + std::to_string(i) + "].";
    output << prefix << "id = " << std::quoted(target.id) << '\n'
           << prefix << "kind = " << HorizonKindName(target.kind) << '\n'
           << prefix << "converged = " << target.converged << '\n';
    if (target.massScale.has_value()) {
      output << prefix << "massScale = " << *target.massScale << '\n';
    } else {
      output << prefix << "massScale = shared\n";
    }
    if (target.converged) {
      output << prefix << "successfulAttempt = "
             << target.successfulAttempt << '\n';
    } else {
      output << prefix << "successfulAttempt = unavailable\n";
    }
    output << prefix << "attempts.count = " << target.attempts.size() << '\n';
    for (std::size_t j = 0; j < target.attempts.size(); ++j) {
      const AHSearchAttemptResult& attempt = target.attempts[j];
      const std::string attemptPrefix =
        prefix + "attempts[" + std::to_string(j) + "].";
      output << attemptPrefix << "referencePoint = "
             << attempt.referencePoint.x << ' ' << attempt.referencePoint.y
             << ' ' << attempt.referencePoint.z << '\n'
             << attemptPrefix << "initialSurface = "
             << (attempt.initialCoefficients.empty() ? "sphere" : "spectral")
             << '\n'
             << attemptPrefix << "initialRadius = ";
      if (attempt.initialCoefficients.empty()) {
        output << attempt.initialRadius << '\n';
      } else {
        output << "unavailable\n";
      }
      output << attemptPrefix << "succeeded = " << attempt.succeeded << '\n'
             << attemptPrefix << "solver.converged = "
             << attempt.result.solver.converged << '\n'
             << attemptPrefix << "surfaceAvailable = "
             << attempt.result.surfaceAvailable << '\n'
             << attemptPrefix << "solver.iterations = "
             << attempt.result.solver.iterations << '\n'
             << attemptPrefix << "solver.message = "
             << std::quoted(attempt.result.solver.message) << '\n'
             << attemptPrefix << "rejection = "
             << std::quoted(attempt.rejection) << '\n'
             << attemptPrefix << "residualHistory.count = "
             << attempt.result.solver.residualHistory.size() << '\n';
      output << attemptPrefix << "initialCoefficients.count = "
             << attempt.initialCoefficients.size() << '\n';
      for (std::size_t k = 0; k < attempt.initialCoefficients.size(); ++k) {
        output << attemptPrefix << "initialCoefficients[" << k << "] = "
               << attempt.initialCoefficients[k] << '\n';
      }
      for (std::size_t k = 0;
           k < attempt.result.solver.residualHistory.size(); ++k) {
        output << attemptPrefix << "residualHistory[" << k << "] = "
               << attempt.result.solver.residualHistory[k] << '\n';
      }
    }
    if (target.horizon.has_value()) {
      const AHFinderResult& horizon = *target.horizon;
      output << prefix << "horizon.referencePoint = "
             << horizon.referencePoint.x << ' ' << horizon.referencePoint.y
             << ' ' << horizon.referencePoint.z << '\n'
             << prefix << "horizon.minRadius = " << horizon.minRadius << '\n'
             << prefix << "horizon.meanRadius = " << horizon.meanRadius << '\n'
             << prefix << "horizon.maxRadius = " << horizon.maxRadius << '\n'
             << prefix << "horizon.meanCoordinateRadius = "
             << horizon.meanCoordinateRadius << '\n'
             << prefix << "horizon.area = " << horizon.area << '\n'
             << prefix << "horizon.arealRadius = " << horizon.arealRadius << '\n'
             << prefix << "horizon.irreducibleMass = "
             << horizon.irreducibleMass << '\n'
             << prefix << "horizon.spectralResidualInfinityNorm = "
             << horizon.spectralResidualInfinityNorm << '\n'
             << prefix << "horizon.maxAbsExpansion = "
             << horizon.maxAbsExpansion << '\n'
             << prefix << "horizon.maxAbsExpansionOverResolved = "
             << horizon.maxAbsExpansionOverResolved << '\n'
             << prefix << "horizon.relativeExpansionError = "
             << horizon.relativeExpansionError << '\n';
    }
  }
  if (!output) {
    throw std::runtime_error(
      "Failed while writing AH multi-horizon summary: " + filename);
  }
}

AHVisualizationSurface BuildAHVisualizationSurface(
  const AHFinderResult& result, const AHVisualizationOptions& options) {
  ValidateVisualizationOptions(options);
  if (!result.surfaceAvailable) {
    throw std::invalid_argument(
      "Cannot build AH visualization: final surface is unavailable");
  }

  AHVisualizationSurface surface;
  const std::size_t nRho = static_cast<std::size_t>(options.numberOfRhoPoints);
  const std::size_t nSigma =
    static_cast<std::size_t>(options.numberOfSigmaPoints);
  const std::size_t ringCount = nSigma - 2;
  surface.points.reserve(2 + ringCount * nRho);
  surface.radius.reserve(2 + ringCount * nRho);
  surface.triangles.reserve(2 * nRho * ringCount);

  const double southSigma = -std::numbers::pi / 2.0;
  const double northSigma = std::numbers::pi / 2.0;
  AppendPoint(surface, result.referencePoint,
              EvaluateRadius(result, 0.0, southSigma, true), 0.0, southSigma);

  for (std::size_t i = 1; i + 1 < nSigma; ++i) {
    const double sigma = southSigma + std::numbers::pi *
      static_cast<double>(i) / static_cast<double>(nSigma - 1);
    for (std::size_t j = 0; j < nRho; ++j) {
      const double rho = 2.0 * std::numbers::pi * static_cast<double>(j) /
                         static_cast<double>(nRho);
      AppendPoint(surface, result.referencePoint,
                  EvaluateRadius(result, rho, sigma, false), rho, sigma);
    }
  }
  const std::size_t north = surface.points.size();
  AppendPoint(surface, result.referencePoint,
              EvaluateRadius(result, 0.0, northSigma, true), 0.0, northSigma);

  const std::size_t firstRing = 1;
  for (std::size_t j = 0; j < nRho; ++j) {
    const std::size_t next = (j + 1) % nRho;
    surface.triangles.push_back({0, firstRing + next, firstRing + j});
  }
  for (std::size_t i = 0; i + 1 < ringCount; ++i) {
    const std::size_t lower = firstRing + i * nRho;
    const std::size_t upper = lower + nRho;
    for (std::size_t j = 0; j < nRho; ++j) {
      const std::size_t next = (j + 1) % nRho;
      surface.triangles.push_back({lower + j, lower + next, upper + j});
      surface.triangles.push_back({lower + next, upper + next, upper + j});
    }
  }
  const std::size_t lastRing = firstRing + (ringCount - 1) * nRho;
  for (std::size_t j = 0; j < nRho; ++j) {
    const std::size_t next = (j + 1) % nRho;
    surface.triangles.push_back({lastRing + j, lastRing + next, north});
  }
  return surface;
}

void WriteAHSurfaceToVTP(const AHFinderResult& result,
                         const std::string& filename,
                         const AHVisualizationOptions& options) {
  if (!result.solver.converged) {
    throw std::invalid_argument(
      "Cannot write normal AH visualization: solver did not converge");
  }
  const AHVisualizationSurface surface =
    BuildAHVisualizationSurface(result, options);
  std::ofstream output = OpenOutput(filename);
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
         << "  <PolyData>\n"
         << "    <Piece NumberOfPoints=\"" << surface.points.size()
         << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""
         << surface.triangles.size() << "\">\n"
         << "      <PointData Scalars=\"radius\">\n"
         << "        <DataArray type=\"Float64\" Name=\"radius\" format=\"ascii\">\n          ";
  for (double radius : surface.radius) output << radius << ' ';
  output << "\n        </DataArray>\n      </PointData>\n"
         << "      <CellData/>\n      <Points>\n"
         << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
  for (const auto& point : surface.points) {
    output << point[0] << ' ' << point[1] << ' ' << point[2] << ' ';
  }
  output << "\n        </DataArray>\n      </Points>\n      <Polys>\n"
         << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n          ";
  for (const auto& triangle : surface.triangles) {
    output << triangle[0] << ' ' << triangle[1] << ' ' << triangle[2] << ' ';
  }
  output << "\n        </DataArray>\n"
         << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n          ";
  for (std::size_t i = 1; i <= surface.triangles.size(); ++i) output << 3 * i << ' ';
  output << "\n        </DataArray>\n      </Polys>\n"
         << "    </Piece>\n  </PolyData>\n</VTKFile>\n";
  if (!output) {
    throw std::runtime_error("Failed while writing AH VTP surface: " + filename);
  }
}

AHFinderResult WriteAHToFile(const GridData& gridData,
                             const AHFinderOptions& finderOptions,
                             const AHOutputOptions& outputOptions) {
  ValidateAHFinderOptions(finderOptions);
  ValidateAHOutputOptions(outputOptions);

  const SpectralBasis basis = BuildSpectralBasis(finderOptions.lMax);
  const CartesianPointInterpolator interpolator(
    gridData, finderOptions.interpolation);
  const SpectralNewtonResult solverResult = SolvePDEforAH(
    basis,
    interpolator,
    finderOptions.referencePoint,
    finderOptions.initialRadius,
    finderOptions.solver
  );
  AHFinderResult result = BuildAHFinderResult(
    basis, interpolator, finderOptions.referencePoint, solverResult);

  WriteAHSummary(result, outputOptions.summaryFilename);
  if (result.solver.converged && result.surfaceAvailable) {
    WriteAHSurfaceToVTP(
      result, outputOptions.surfaceFilename, outputOptions.visualization);
  }
  return result;
}

AHSearchResult WriteAHToFile(const GridData& gridData,
                             const AHSearchOptions& searchOptions,
                             const AHOutputOptions& outputOptions) {
  ValidateAHSearchOptions(searchOptions);
  ValidateAHOutputOptions(outputOptions);

  AHSearchResult result = FindApparentHorizon(gridData, searchOptions);
  WriteAHSearchSummary(result, outputOptions.summaryFilename);
  if (result.converged && result.horizon.has_value()) {
    WriteAHSurfaceToVTP(
      *result.horizon,
      outputOptions.surfaceFilename,
      outputOptions.visualization
    );
  }
  return result;
}

AHMultiHorizonResult WriteAHToFile(
  const GridData& gridData,
  const AHMultiHorizonOptions& searchOptions,
  const AHMultiHorizonOutputOptions& outputOptions
) {
  ValidateAHMultiHorizonOptions(searchOptions);
  if (outputOptions.summaryFilename.empty()) {
    throw std::invalid_argument(
      "AHMultiHorizonOutputOptions: summaryFilename must not be empty");
  }
  if (outputOptions.horizonFilenamePrefix.empty()) {
    throw std::invalid_argument(
      "AHMultiHorizonOutputOptions: horizonFilenamePrefix must not be empty");
  }
  ValidateVisualizationOptions(outputOptions.visualization);

  AHMultiHorizonResult result = FindApparentHorizons(gridData, searchOptions);
  WriteAHMultiHorizonSummary(result, outputOptions.summaryFilename);
  for (std::size_t i = 0; i < result.targets.size(); ++i) {
    const AHHorizonTargetResult& target = result.targets[i];
    if (!target.converged || !target.horizon.has_value() ||
        !target.horizon->surfaceAvailable) {
      continue;
    }
    WriteAHSummary(
      *target.horizon,
      TargetFilename(outputOptions, i, target.id, ".txt"));
    WriteAHSurfaceToVTP(
      *target.horizon,
      TargetFilename(outputOptions, i, target.id, ".vtp"),
      outputOptions.visualization);
  }
  return result;
}

AHBinaryMergerResult WriteAHToFile(
  const GridData& gridData,
  const AHBinaryMergerOptions& mergerOptions,
  const AHMultiHorizonOutputOptions& outputOptions
) {
  AHBinaryMergerResult result;
  result.plan = BuildBinaryMergerPlan(mergerOptions);
  result.horizons = WriteAHToFile(gridData, result.plan.search, outputOptions);
  result.commonValidation = ValidateBinaryMergerCommonHorizon(
    mergerOptions, result.plan, result.horizons);

  std::ofstream output(outputOptions.summaryFilename, std::ios::app);
  if (!output) {
    throw std::runtime_error(
      "Unable to append merger diagnostics to AH output file: " +
      outputOptions.summaryFilename);
  }
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << std::boolalpha
         << "merger.commonSearchEnabled = "
         << result.plan.commonSearchEnabled << '\n'
         << "merger.separation = " << result.plan.separation << '\n'
         << "merger.facingRadius[0] = " << result.plan.facingRadii[0] << '\n'
         << "merger.facingRadius[1] = " << result.plan.facingRadii[1] << '\n'
         << "merger.surfaceGap = " << result.plan.surfaceGap << '\n'
         << "merger.commonReferencePoint = "
         << result.plan.commonReferencePoint.x << ' '
         << result.plan.commonReferencePoint.y << ' '
         << result.plan.commonReferencePoint.z << '\n'
         << "merger.commonMass = " << result.plan.commonMass << '\n'
         << "merger.enclosingInitialRadius = "
         << result.plan.enclosingInitialRadius << '\n';
  if (result.commonValidation.has_value()) {
    const AHCommonHorizonValidation& validation = *result.commonValidation;
    output << "merger.commonValidation.status = "
           << CommonValidationStatusName(validation.status) << '\n'
           << "merger.commonValidation.reasons.count = "
           << validation.reasons.size() << '\n';
    for (std::size_t i = 0; i < validation.reasons.size(); ++i) {
      output << "merger.commonValidation.reasons[" << i << "] = "
             << std::quoted(validation.reasons[i]) << '\n';
    }
  } else {
    output << "merger.commonValidation.status = unavailable\n";
  }
  if (!output) {
    throw std::runtime_error(
      "Failed while appending AH merger diagnostics: " +
      outputOptions.summaryFilename);
  }
  return result;
}

void WriteAHHorizonCatalogueSummary(const AHHorizonCatalogue& result,
                                    const std::string& filename) {
  std::ofstream output = OpenOutput(filename);
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << std::boolalpha
         << "AHHorizonCatalogueVersion = 1\n"
         << "individuals.count = " << result.individuals.targets.size() << '\n'
         << "common.count = " << result.commonHorizons.size() << '\n';
  for (std::size_t i = 0; i < result.individuals.targets.size(); ++i) {
    const auto& target = result.individuals.targets[i];
    const std::string p = "individuals[" + std::to_string(i) + "].";
    output << p << "id = " << std::quoted(target.id) << '\n'
           << p << "converged = " << target.converged << '\n'
           << p << "attempts.count = " << target.attempts.size() << '\n';
    for (std::size_t j = 0; j < target.attempts.size(); ++j) {
      const auto& attempt = target.attempts[j];
      output << p << "attempts[" << j << "].message = "
             << std::quoted(attempt.result.solver.message) << '\n'
             << p << "attempts[" << j << "].rejection = "
             << std::quoted(attempt.rejection) << '\n'
             << p << "attempts[" << j << "].iterations = "
             << attempt.result.solver.iterations << '\n'
             << p << "attempts[" << j << "].residualHistory.count = "
             << attempt.result.solver.residualHistory.size() << '\n';
      for (std::size_t k = 0; k < attempt.result.solver.residualHistory.size(); ++k) {
        output << p << "attempts[" << j << "].residualHistory[" << k << "] = "
               << attempt.result.solver.residualHistory[k] << '\n';
      }
    }
  }
  for (std::size_t i = 0; i < result.commonHorizons.size(); ++i) {
    const auto& entry = result.commonHorizons[i];
    const std::string p = "common[" + std::to_string(i) + "].";
    output << p << "id = " << std::quoted(entry.id) << '\n'
           << p << "members.count = " << entry.memberIds.size() << '\n';
    for (std::size_t j = 0; j < entry.memberIds.size(); ++j) {
      output << p << "members[" << j << "] = "
             << std::quoted(entry.memberIds[j]) << '\n';
    }
    output << p << "reason.triggeredPair = " << entry.reason.triggeredPair << '\n'
           << p << "reason.connectedComponent = " << entry.reason.connectedComponent << '\n'
           << p << "reason.explicitlyRequested = " << entry.reason.explicitlyRequested << '\n'
           << p << "reason.unconditionalAllBlackHoles = "
           << entry.reason.unconditionalAllBlackHoles << '\n'
           << p << "solver.converged = " << entry.search.converged << '\n'
           << p << "validation.status = "
           << CommonValidationStatusName(entry.validation.status) << '\n'
           << p << "validation.numericallyValid = "
           << entry.validation.numericallyValid << '\n'
           << p << "validation.valid = " << entry.validation.valid << '\n'
           << p << "validation.relativeExpansionValid = "
           << entry.validation.relativeExpansionValid << '\n'
           << p << "validation.memberHorizonEnclosure = "
           << EnclosureStatusName(entry.validation.memberHorizonEnclosure) << '\n'
           << p << "validation.physicalPositionEnclosure = "
           << EnclosureStatusName(entry.validation.physicalPositionEnclosure) << '\n'
           << p << "validation.reasons.count = " << entry.validation.reasons.size() << '\n';
    for (std::size_t j = 0; j < entry.validation.reasons.size(); ++j) {
      output << p << "validation.reasons[" << j << "] = "
             << std::quoted(entry.validation.reasons[j]) << '\n';
    }
    for (std::size_t j = 0; j < entry.search.attempts.size(); ++j) {
      const auto& attempt = entry.search.attempts[j];
      output << p << "attempts[" << j << "].message = "
             << std::quoted(attempt.result.solver.message) << '\n'
             << p << "attempts[" << j << "].rejection = "
             << std::quoted(attempt.rejection) << '\n'
             << p << "attempts[" << j << "].iterations = "
             << attempt.result.solver.iterations << '\n'
             << p << "attempts[" << j << "].residualHistory.count = "
             << attempt.result.solver.residualHistory.size() << '\n';
      for (std::size_t k = 0; k < attempt.result.solver.residualHistory.size(); ++k) {
        output << p << "attempts[" << j << "].residualHistory[" << k << "] = "
               << attempt.result.solver.residualHistory[k] << '\n';
      }
    }
  }
  if (!output) throw std::runtime_error("Failed while writing AH catalogue: " + filename);
}

AHHorizonCatalogue WriteAHToFile(
  const GridData& gridData,
  const AHCommonHorizonSearchOptions& searchOptions,
  const AHMultiHorizonOutputOptions& outputOptions
) {
  if (outputOptions.summaryFilename.empty() ||
      outputOptions.horizonFilenamePrefix.empty()) {
    throw std::invalid_argument("Catalogue output filenames must not be empty");
  }
  ValidateVisualizationOptions(outputOptions.visualization);
  AHHorizonCatalogue result = FindApparentHorizons(gridData, searchOptions);
  WriteAHHorizonCatalogueSummary(result, outputOptions.summaryFilename);
  for (std::size_t i = 0; i < result.individuals.targets.size(); ++i) {
    const auto& target = result.individuals.targets[i];
    WriteAHSearchSummary(AsSearchResult(target), AHCatalogueFilename(
      outputOptions, "individual", i, target.id, ".txt"));
    if (!target.converged || !target.horizon.has_value() ||
        !target.horizon->surfaceAvailable) continue;
    WriteAHSurfaceToVTP(*target.horizon, AHCatalogueFilename(
      outputOptions, "individual", i, target.id, ".vtp"),
      outputOptions.visualization);
  }
  for (std::size_t i = 0; i < result.commonHorizons.size(); ++i) {
    const auto& entry = result.commonHorizons[i];
    WriteAHSearchSummary(AsSearchResult(entry.search), AHCatalogueFilename(
      outputOptions, "common", i, entry.id, ".txt"));
    if (entry.validation.status ==
          AHCommonHorizonValidationStatus::FullyValidated &&
        entry.search.horizon.has_value()) {
      WriteAHSurfaceToVTP(*entry.search.horizon, AHCatalogueFilename(
        outputOptions, "common", i, entry.id, ".vtp"),
        outputOptions.visualization);
    }
  }
  return result;
}

void WriteAHToVTK(const std::string& inputFile, const std::string& outputFile) {
  (void)inputFile;
  (void)outputFile;
  // Helper function to convert the ASKII AH file into a VTK file for visualisation.
  // – Input: ASCII file(s) contain all coordinate origins $\{s_i\}$ with a valid AH, and the 
  //          list of tuples of corresponding $ \{h_i^{(f)}(\rho, \sigma)\}$.
  // – Output: VTK file(s) for the AH surface matching the $ \{s_i, h_i^{(f)}(\rho, \sigma)\}$.
}

}  // namespace exahype2::ccz4::ahf
