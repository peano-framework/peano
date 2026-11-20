#ifndef APPARENT_HORIZON_FINDER_AHF_DATA_H_
#define APPARENT_HORIZON_FINDER_AHF_DATA_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf {

/**
 * Cartesian reference coordinate for one apparent-horizon centre point.
 */
struct ReferencePoint {
  double x;
  double y;
  double z;
};

/**
 * One point on an apparent-horizon guess, expressed in spherical coordinates.
 */
struct AHGuessPoint {
  double r;
  double rho;
  double sigma;
};

/**
 * User-provided angular coordinate at which the shell should be sampled.
 */
struct AngularSample {
  double rho = 0.0;
  double sigma = 0.0;
};

/**
 * One apparent-horizon hypersurface guess.
 *
 * Multiple candidate guesses are represented by std::vector<AHGuess>.
 */
struct AHGuess {
  std::vector<AHGuessPoint> points;
};

/**
 * Cartesian coordinate for one sampled grid point.
 */
struct GridCoordinate {
  double x;
  double y;
  double z;
};

enum class CartesianToSurfaceMethod {
  NearestNeighbour,
  UnweightedPolynomial,
  WeightedPolynomial,
  /** All cells inside supportRadius, using compact Wendland-C2 weights. */
  RadiusWeightedPolynomial,
  /** Backward-compatible names retained for callers of the original degree-1 API. */
  UnweightedDegree1 = UnweightedPolynomial,
  WeightedDegree1 = WeightedPolynomial
};

struct InterpolationDiagnostics {
  CartesianToSurfaceMethod requestedMethod = CartesianToSurfaceMethod::NearestNeighbour;
  std::size_t requestedDegree = 0;
  std::size_t degreeUsed = 0;
  std::size_t neighbourCount = 0;
  GridCoordinate target{};
  std::size_t rank = 0;
  double conditionEstimate = 1.0;
  bool orderReduced = false;
};

/**
 * Fixed column layout of every compact row stored in GridData.
 *
 * This enum is the only field-index contract used after file input. Its values
 * are independent of the component numbers in a source VTU tuple.
 */
enum class CompactField : std::size_t {
  GammaTildeXX = 0,
  GammaTildeXY = 1,
  GammaTildeXZ = 2,
  GammaTildeYY = 3,
  GammaTildeYZ = 4,
  GammaTildeZZ = 5,
  ATildeXX = 6,
  ATildeXY = 7,
  ATildeXZ = 8,
  ATildeYY = 9,
  ATildeYZ = 10,
  ATildeZZ = 11,
  TraceK = 12,
  Phi = 13,
  Count = 14
};

inline constexpr std::size_t NumberOfCompactFields =
  static_cast<std::size_t>(CompactField::Count);

inline constexpr std::array<const char*, NumberOfCompactFields> CompactFieldNames{
  "gammaTilde_xx", "gammaTilde_xy", "gammaTilde_xz",
  "gammaTilde_yy", "gammaTilde_yz", "gammaTilde_zz",
  "ATilde_xx", "ATilde_xy", "ATilde_xz",
  "ATilde_yy", "ATilde_yz", "ATilde_zz",
  "K", "phi"
};

/** Symmetric-tensor component order used by both compact tensors. */
enum class SymmetricTensorComponent : std::size_t {
  XX = 0,
  XY = 1,
  XZ = 2,
  YY = 3,
  YZ = 4,
  ZZ = 5,
  Count = 6
};

inline constexpr std::size_t NumberOfSymmetricTensorComponents =
  static_cast<std::size_t>(SymmetricTensorComponent::Count);

inline constexpr std::array<CompactField, NumberOfSymmetricTensorComponents>
ConformalMetricCompactFields{
  CompactField::GammaTildeXX, CompactField::GammaTildeXY,
  CompactField::GammaTildeXZ, CompactField::GammaTildeYY,
  CompactField::GammaTildeYZ, CompactField::GammaTildeZZ
};

inline constexpr std::array<CompactField, NumberOfSymmetricTensorComponents>
ConformalExtrinsicCurvatureCompactFields{
  CompactField::ATildeXX, CompactField::ATildeXY, CompactField::ATildeXZ,
  CompactField::ATildeYY, CompactField::ATildeYZ, CompactField::ATildeZZ
};

/**
 * Reader-side mapping from compact fields to components of a source tuple.
 * Source indices are used only while constructing a compact GridData row.
 */
struct SourceVariableMapping {
  using Index = std::size_t;
  static constexpr Index NotStored = std::numeric_limits<Index>::max();

  std::array<Index, 6> conformalMetric{
    NotStored, NotStored, NotStored, NotStored, NotStored, NotStored
  };
  std::array<Index, 6> conformalExtrinsicCurvature{
    NotStored, NotStored, NotStored, NotStored, NotStored, NotStored
  };
  Index traceK = NotStored;
  Index conformalFactor = NotStored;

  std::array<Index, NumberOfCompactFields> componentsInCompactOrder() const {
    return {
      conformalMetric[0], conformalMetric[1], conformalMetric[2],
      conformalMetric[3], conformalMetric[4], conformalMetric[5],
      conformalExtrinsicCurvature[0], conformalExtrinsicCurvature[1],
      conformalExtrinsicCurvature[2], conformalExtrinsicCurvature[3],
      conformalExtrinsicCurvature[4], conformalExtrinsicCurvature[5],
      traceK, conformalFactor
    };
  }

  bool isComplete() const {
    const auto components = componentsInCompactOrder();
    return std::all_of(components.begin(), components.end(), [](Index index) {
      return index != NotStored;
    });
  }

  void validateRequiredFields() const {
    const auto components = componentsInCompactOrder();
    for (std::size_t field = 0; field < components.size(); ++field) {
      if (components[field] == NotStored) {
        throw std::invalid_argument(
          "SourceVariableMapping: required field "
          + std::string(CompactFieldNames[field]) + " is NotStored");
      }
    }

    for (std::size_t first = 0; first < components.size(); ++first) {
      for (std::size_t second = first + 1; second < components.size(); ++second) {
        if (components[first] == components[second]) {
          throw std::invalid_argument(
            "SourceVariableMapping: required fields "
            + std::string(CompactFieldNames[first]) + " and "
            + CompactFieldNames[second] + " both map to source component "
            + std::to_string(components[first]));
        }
      }
    }
  }

  void validateAgainstSourceTuple(std::size_t numberOfSourceComponents) const {
    validateRequiredFields();
    const auto components = componentsInCompactOrder();
    for (std::size_t field = 0; field < components.size(); ++field) {
      if (components[field] >= numberOfSourceComponents) {
        throw std::invalid_argument(
          "SourceVariableMapping: required field "
          + std::string(CompactFieldNames[field]) + " selects source component "
          + std::to_string(components[field]) + ", but the source tuple has only "
          + std::to_string(numberOfSourceComponents) + " components");
      }
    }
  }

  bool isValid(std::size_t numberOfSourceComponents) const {
    const auto components = componentsInCompactOrder();
    for (std::size_t field = 0; field < components.size(); ++field) {
      if (components[field] >= numberOfSourceComponents) {
        return false;
      }
      for (std::size_t previous = 0; previous < field; ++previous) {
        if (components[field] == components[previous]) {
          return false;
        }
      }
    }
    return true;
  }
};

/**
 * Mapping for output of the exagrype.api solvers, in the order of
 * FOCCZ4Solver.Unknowns (src/exagrype/api/FOCCZ4Solver.py): G 0-5, K 6-11,
 * traceK 23, phi 24. FOSOCCZ4Solver stores the same first 25 unknowns.
 */
inline SourceVariableMapping ExaGRyPESourceMapping() {
  return {
    .conformalMetric = {0, 1, 2, 3, 4, 5},
    .conformalExtrinsicCurvature = {6, 7, 8, 9, 10, 11},
    .traceK = 23,
    .conformalFactor = 24
  };
}

/**
 * Mapping for output of the legacy applications/exahype2/ccz4/ccz4.py, which
 * stores traceK at 53 and phi at 54. The GitLab-era snapshots used by the
 * integration tests have this layout.
 */
inline SourceVariableMapping LegacyCCZ4SourceMapping() {
  return {
    .conformalMetric = {0, 1, 2, 3, 4, 5},
    .conformalExtrinsicCurvature = {6, 7, 8, 9, 10, 11},
    .traceK = 53,
    .conformalFactor = 54
  };
}

/**
 * Names the CCZ4 solvers give the AHF fields, in CompactField order. Peano's
 * patch-file converter (src/tarch/plotter/api/render.py) labels each component
 * of the unknowns array with the solver's names, so a converted snapshot says
 * where each field is, whatever the layout.
 */
inline constexpr std::array<const char*, NumberOfCompactFields> CCZ4SourceComponentNames{
  "G[0]", "G[1]", "G[2]", "G[3]", "G[4]", "G[5]",
  "K[0]", "K[1]", "K[2]", "K[3]", "K[4]", "K[5]",
  "traceK", "phi"
};

/**
 * Build the mapping from the component names of a source tuple. Returns
 * nothing unless each name in CCZ4SourceComponentNames occurs exactly once.
 */
inline std::optional<SourceVariableMapping> SourceVariableMappingFromComponentNames(
  std::span<const std::string> componentNames
) {
  std::array<SourceVariableMapping::Index, NumberOfCompactFields> components;
  components.fill(SourceVariableMapping::NotStored);
  for (std::size_t component = 0; component < componentNames.size(); ++component) {
    for (std::size_t field = 0; field < NumberOfCompactFields; ++field) {
      if (componentNames[component] == CCZ4SourceComponentNames[field]) {
        if (components[field] != SourceVariableMapping::NotStored) {
          return std::nullopt;
        }
        components[field] = component;
      }
    }
  }
  if (std::find(components.begin(), components.end(), SourceVariableMapping::NotStored)
      != components.end()) {
    return std::nullopt;
  }

  SourceVariableMapping mapping;
  for (std::size_t i = 0; i < NumberOfSymmetricTensorComponents; ++i) {
    mapping.conformalMetric[i] = components[i];
    mapping.conformalExtrinsicCurvature[i] = components[NumberOfSymmetricTensorComponents + i];
  }
  mapping.traceK = components[static_cast<std::size_t>(CompactField::TraceK)];
  mapping.conformalFactor = components[static_cast<std::size_t>(CompactField::Phi)];
  return mapping;
}

/**
 * Reject a mapping that contradicts the component names of the source tuple.
 * Names that do not identify every field (see
 * SourceVariableMappingFromComponentNames) are not checked, so data without
 * names, or with names from another convention, is accepted as before.
 */
inline void ValidateSourceMappingAgainstComponentNames(
  const SourceVariableMapping& sourceMapping,
  std::span<const std::string> componentNames
) {
  const auto named = SourceVariableMappingFromComponentNames(componentNames);
  if (!named) {
    return;
  }

  const auto given = sourceMapping.componentsInCompactOrder();
  const auto expected = named->componentsInCompactOrder();
  for (std::size_t field = 0; field < given.size(); ++field) {
    if (given[field] != expected[field]) {
      const std::string givenComponent = given[field] < componentNames.size()
        ? "which the data names " + componentNames[given[field]]
        : std::string("which the data does not have");
      throw std::invalid_argument(
        "SourceVariableMapping: required field "
        + std::string(CompactFieldNames[field]) + " selects source component "
        + std::to_string(given[field]) + ", " + givenComponent + ", but "
        + CCZ4SourceComponentNames[field] + " is component "
        + std::to_string(expected[field])
        + "; the mapping does not match this data's layout");
    }
  }
}

/**
 * Apply the sole source-index to compact-index transformation. Both symmetric
 * tensors use xx, xy, xz, yy, yz, zz order; K and direct phi occupy columns 12
 * and 13 respectively.
 */
inline std::array<double, NumberOfCompactFields> CompactSourceTuple(
  std::span<const double> sourceTuple,
  const SourceVariableMapping& sourceMapping
) {
  sourceMapping.validateAgainstSourceTuple(sourceTuple.size());

  const auto sourceComponents = sourceMapping.componentsInCompactOrder();

  std::array<double, NumberOfCompactFields> compact{};
  for (std::size_t column = 0; column < compact.size(); ++column) {
    compact[column] = sourceTuple[sourceComponents[column]];
  }
  return compact;
}

/**
 * Read-only, zero-copy view of one fixed-layout compact cell row.
 */
class CompactCellView {
public:
  explicit CompactCellView(
    std::span<const double, NumberOfCompactFields> values
  ): _values(values) {}

  std::span<const double, NumberOfCompactFields> row() const {
    return _values;
  }

  double conformalMetricComponent(std::size_t component) const {
    return tensorComponent(component, ConformalMetricCompactFields, "conformal metric");
  }

  double conformalMetricComponent(SymmetricTensorComponent component) const {
    return conformalMetricComponent(static_cast<std::size_t>(component));
  }

  double conformalExtrinsicCurvatureComponent(std::size_t component) const {
    return tensorComponent(
      component,
      ConformalExtrinsicCurvatureCompactFields,
      "conformal extrinsic curvature"
    );
  }

  double conformalExtrinsicCurvatureComponent(SymmetricTensorComponent component) const {
    return conformalExtrinsicCurvatureComponent(static_cast<std::size_t>(component));
  }

  double traceK() const {
    return compactValue(CompactField::TraceK);
  }

  double phi() const {
    return compactValue(CompactField::Phi);
  }

private:
  double compactValue(CompactField field) const {
    return _values[static_cast<std::size_t>(field)];
  }

  double tensorComponent(
    std::size_t component,
    const std::array<CompactField, NumberOfSymmetricTensorComponents>& fields,
    const char* tensorName
  ) const {
    if (component >= fields.size()) {
      throw std::out_of_range(
        "CompactCellView: " + std::string(tensorName)
        + " component index " + std::to_string(component)
        + " is outside xx, xy, xz, yy, yz, zz");
    }
    return compactValue(fields[component]);
  }

  std::span<const double, NumberOfCompactFields> _values;
};

/**
 * Pointwise Cartesian grid data.
 *
 * Values are stored as a flat point-major array:
 *   values[pointIndex * numberOfVariables + variableIndex]
 */
struct GridData {
  std::vector<GridCoordinate> coordinates;
  std::vector<double> values;

  static constexpr std::size_t numberOfVariables = NumberOfCompactFields;

  std::size_t numberOfPoints() const {
    return coordinates.size();
  }

  void resize(std::size_t numberOfPoints) {
    coordinates.resize(numberOfPoints);
    values.resize(numberOfPoints * numberOfVariables);
  }

  bool isConsistent() const {
    return values.size() == coordinates.size() * numberOfVariables;
  }

  CompactCellView cell(std::size_t cellIndex) const {
    if (!isConsistent()) {
      throw std::logic_error(
        "GridData::cell: malformed compact data; expected "
        + std::to_string(coordinates.size() * numberOfVariables) + " values for "
        + std::to_string(coordinates.size()) + " cells, but found "
        + std::to_string(values.size()));
    }
    if (cellIndex >= coordinates.size()) {
      throw std::out_of_range(
        "GridData::cell: cell index " + std::to_string(cellIndex)
        + " is outside " + std::to_string(coordinates.size()) + " cells");
    }

    return CompactCellView(
      std::span<const double, NumberOfCompactFields>(
        values.data() + cellIndex * numberOfVariables,
        numberOfVariables
      )
    );
  }

  double& value(const GridCoordinate& coordinate, std::size_t variableIndex) {
    return values[valueIndex(coordinate, variableIndex)];
  }

  const double& value(const GridCoordinate& coordinate, std::size_t variableIndex) const {
    return values[valueIndex(coordinate, variableIndex)];
  }

  double& value(const GridCoordinate& coordinate, CompactField field) {
    return value(coordinate, static_cast<std::size_t>(field));
  }

  const double& value(const GridCoordinate& coordinate, CompactField field) const {
    return value(coordinate, static_cast<std::size_t>(field));
  }

private:
  std::size_t valueIndex(const GridCoordinate& coordinate, std::size_t variableIndex) const {
    if (variableIndex >= numberOfVariables) {
      throw std::out_of_range("GridData variable index is out of range");
    }

    const auto point = std::find_if(
      coordinates.begin(),
      coordinates.end(),
      [&coordinate](const GridCoordinate& candidate) {
        return coordinatesEqual(candidate.x, coordinate.x)
            && coordinatesEqual(candidate.y, coordinate.y)
            && coordinatesEqual(candidate.z, coordinate.z);
      }
    );
    if (point == coordinates.end()) {
      throw std::out_of_range("GridData coordinate was not found");
    }

    const auto pointIndex = static_cast<std::size_t>(point - coordinates.begin());
    return pointIndex * numberOfVariables + variableIndex;
  }

  static bool coordinatesEqual(double first, double second) {
    constexpr double AbsoluteTolerance = 1.0e-12;
    constexpr double RelativeTolerance = 1.0e-10;
    return std::abs(first - second)
        <= AbsoluteTolerance
         + RelativeTolerance * std::max(std::abs(first), std::abs(second));
  }
};

using Vector3 = std::array<double, 3>;
using Tensor3 = std::array<std::array<double, 3>, 3>;
using Tensor3Derivative = std::array<Tensor3, 3>;  // [k][i][j] = partial_k tensor_ij

/**
 * Expand symmetric components ordered xx, xy, xz, yy, yz, zz into a 3x3
 * tensor. Off-diagonal entries are assigned explicitly to preserve symmetry.
 */
inline Tensor3 UnpackSymmetricTensor(
  std::span<const double, NumberOfSymmetricTensorComponents> components
) {
  return Tensor3{{
    {{components[0], components[1], components[2]}},
    {{components[1], components[3], components[4]}},
    {{components[2], components[4], components[5]}}
  }};
}

/** Active kernels use 1e-2 as the safe lower bound for direct phi. */
inline constexpr double MinimumPhysicalPhi = 1.0e-2;

inline void ValidatePhysicalPhi(double phi) {
  if (!std::isfinite(phi)) {
    throw std::invalid_argument(
      "Geometry conversion: direct phi must be finite");
  }
  if (phi <= 0.0) {
    throw std::invalid_argument(
      "Geometry conversion: direct phi must be strictly positive");
  }
  if (phi < MinimumPhysicalPhi) {
    throw std::invalid_argument(
      "Geometry conversion: direct phi is below the safe division threshold 1e-2");
  }
}

/** Compute gamma_ij = gammaTilde_ij / phi^2 for direct (not logarithmic) phi. */
inline Tensor3 ComputePhysicalSpatialMetric(const Tensor3& gammaTilde, double phi) {
  ValidatePhysicalPhi(phi);
  const double inversePhiSquared = 1.0 / (phi * phi);

  Tensor3 gamma{};
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = i; j < 3; ++j) {
      gamma[i][j] = gammaTilde[i][j] * inversePhiSquared;
      gamma[j][i] = gamma[i][j];
    }
  }
  return gamma;
}

/**
 * Compute K_ij = (ATilde_ij + K gammaTilde_ij / 3) / phi^2 using direct phi.
 */
inline Tensor3 ComputePhysicalExtrinsicCurvature(
  const Tensor3& ATilde,
  double traceK,
  const Tensor3& gammaTilde,
  double phi
) {
  ValidatePhysicalPhi(phi);
  const double inversePhiSquared = 1.0 / (phi * phi);

  Tensor3 extrinsicCurvature{};
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = i; j < 3; ++j) {
      extrinsicCurvature[i][j] =
        (ATilde[i][j] + traceK * gammaTilde[i][j] / 3.0) * inversePhiSquared;
      extrinsicCurvature[j][i] = extrinsicCurvature[i][j];
    }
  }
  return extrinsicCurvature;
}

/**
 * Compute Cartesian physical-metric derivatives in [k][i][j] order:
 * partial_k gamma_ij = partial_k gammaTilde_ij / phi^2
 *                    - 2 gammaTilde_ij partial_k phi / phi^3.
 */
inline Tensor3Derivative ComputePhysicalSpatialMetricDerivative(
  const Tensor3& gammaTilde,
  const Tensor3Derivative& gammaTildeDerivative,
  double phi,
  const Vector3& phiDerivative
) {
  ValidatePhysicalPhi(phi);
  const double inversePhiSquared = 1.0 / (phi * phi);
  const double inversePhiCubed = inversePhiSquared / phi;

  Tensor3Derivative gammaDerivative{};
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = i; j < 3; ++j) {
        gammaDerivative[k][i][j] =
          gammaTildeDerivative[k][i][j] * inversePhiSquared
          - 2.0 * gammaTilde[i][j] * phiDerivative[k] * inversePhiCubed;
        gammaDerivative[k][j][i] = gammaDerivative[k][i][j];
      }
    }
  }
  return gammaDerivative;
}

/**
 * Physical and geometrical quantities shared by shell and surface samples.
 *
 * The first index of coordinateGradient and coordinateHessian selects the
 * surface coordinate u: 0 for rho and 1 for sigma.
 */
struct InterpolatedGeometryData {
  Tensor3 T{};
  std::array<Vector3, 2> coordinateGradient{};  // X^u_i
  std::array<Tensor3, 2> coordinateHessian{};   // X^u_ij
  Tensor3 spatialMetric{};                     // gamma_ij
  Tensor3Derivative spatialMetricDerivative{};  // [k][i][j] = partial_k gamma_ij
  bool spatialMetricDerivativeAvailable = false;
  Tensor3 extrinsicCurvature{};                // K_ij
  double traceK = 0.0;
};

/**
 * Interpolated quantities at one spherical shell sample (r, rho, sigma).
 */
struct InterpolatedShellPoint {
  double r = 0.0;
  double rho = 0.0;
  double sigma = 0.0;
  InterpolatedGeometryData geometry{};
};

/**
 * Interpolated data on the shell-like subdomain around an AH guess.
 *
 * Each entry in points owns both its spherical coordinate and all quantities
 * interpolated at that coordinate. Points belonging to one angular sample
 * should be stored consecutively and ordered by increasing r so that the
 * shell-to-surface step can perform one-dimensional radial interpolation.
 */
struct InterpolatedShell {
  double innerRadius = 0.0;
  double outerRadius = 0.0;
  std::vector<InterpolatedShellPoint> points;

  std::size_t numberOfPoints() const {
    return points.size();
  }

  InterpolatedShellPoint& value(const AHGuessPoint& coordinate) {
    return *findPoint(coordinate);
  }

  const InterpolatedShellPoint& value(const AHGuessPoint& coordinate) const {
    return *findPoint(coordinate);
  }

  bool isConsistent() const {
    if (!std::isfinite(innerRadius) || !std::isfinite(outerRadius)
        || innerRadius > outerRadius || points.empty()) {
      return false;
    }

    for (const InterpolatedShellPoint& point : points) {
      if (!std::isfinite(point.r) || !std::isfinite(point.rho)
          || !std::isfinite(point.sigma) || point.r < innerRadius
          || point.r > outerRadius) {
        return false;
      }
    }
    return true;
  }

private:
  std::vector<InterpolatedShellPoint>::iterator findPoint(const AHGuessPoint& coordinate) {
    const auto point = std::find_if(
      points.begin(),
      points.end(),
      [&coordinate](const InterpolatedShellPoint& candidate) {
        return coordinatesEqual(candidate.r, coordinate.r)
            && coordinatesEqual(candidate.rho, coordinate.rho)
            && coordinatesEqual(candidate.sigma, coordinate.sigma);
      }
    );
    if (point == points.end()) {
      throw std::out_of_range("InterpolatedShell coordinate was not found");
    }
    return point;
  }

  std::vector<InterpolatedShellPoint>::const_iterator findPoint(const AHGuessPoint& coordinate) const {
    const auto point = std::find_if(
      points.begin(),
      points.end(),
      [&coordinate](const InterpolatedShellPoint& candidate) {
        return coordinatesEqual(candidate.r, coordinate.r)
            && coordinatesEqual(candidate.rho, coordinate.rho)
            && coordinatesEqual(candidate.sigma, coordinate.sigma);
      }
    );
    if (point == points.end()) {
      throw std::out_of_range("InterpolatedShell coordinate was not found");
    }
    return point;
  }

  static bool coordinatesEqual(double first, double second) {
    constexpr double AbsoluteTolerance = 1.0e-12;
    constexpr double RelativeTolerance = 1.0e-10;
    return std::abs(first - second)
        <= AbsoluteTolerance
         + RelativeTolerance * std::max(std::abs(first), std::abs(second));
  }
};

/**
 * Interpolated quantities at one point on the current AH surface.
 */
struct InterpolatedSurfacePoint {
  double r = 0.0;
  double rho = 0.0;
  double sigma = 0.0;

  Vector3 surfaceNormal{};             // s_i
  Tensor3 surfaceNormalDerivative{};   // partial_i s_j
  InterpolatedGeometryData geometry{};
  InterpolationDiagnostics interpolationDiagnostics{};
};

/**
 * Interpolated data on the current AH surface r = h(rho, sigma).
 *
 * Surface point i corresponds to AHGuess::points[i].
 */
struct InterpolatedSurface {
  std::vector<InterpolatedSurfacePoint> points;

  std::size_t numberOfPoints() const {
    return points.size();
  }

  InterpolatedSurfacePoint& value(const AHGuessPoint& coordinate) {
    return *findPoint(coordinate);
  }

  const InterpolatedSurfacePoint& value(const AHGuessPoint& coordinate) const {
    return *findPoint(coordinate);
  }

  bool isConsistent() const {
    if (points.empty()) {
      return false;
    }

    for (const InterpolatedSurfacePoint& point : points) {
      if (!std::isfinite(point.r) || !std::isfinite(point.rho)
          || !std::isfinite(point.sigma) || point.r < 0.0) {
        return false;
      }
    }
    return true;
  }

  bool matches(const AHGuess& ahGuess) const {
    if (points.size() != ahGuess.points.size()) {
      return false;
    }

    for (std::size_t i = 0; i < points.size(); ++i) {
      if (points[i].r != ahGuess.points[i].r
          || points[i].rho != ahGuess.points[i].rho
          || points[i].sigma != ahGuess.points[i].sigma) {
        return false;
      }
    }
    return true;
  }

private:
  std::vector<InterpolatedSurfacePoint>::iterator findPoint(const AHGuessPoint& coordinate) {
    const auto point = std::find_if(
      points.begin(),
      points.end(),
      [&coordinate](const InterpolatedSurfacePoint& candidate) {
        return coordinatesEqual(candidate.r, coordinate.r)
            && coordinatesEqual(candidate.rho, coordinate.rho)
            && coordinatesEqual(candidate.sigma, coordinate.sigma);
      }
    );
    if (point == points.end()) {
      throw std::out_of_range("InterpolatedSurface coordinate was not found");
    }
    return point;
  }

  std::vector<InterpolatedSurfacePoint>::const_iterator findPoint(const AHGuessPoint& coordinate) const {
    const auto point = std::find_if(
      points.begin(),
      points.end(),
      [&coordinate](const InterpolatedSurfacePoint& candidate) {
        return coordinatesEqual(candidate.r, coordinate.r)
            && coordinatesEqual(candidate.rho, coordinate.rho)
            && coordinatesEqual(candidate.sigma, coordinate.sigma);
      }
    );
    if (point == points.end()) {
      throw std::out_of_range("InterpolatedSurface coordinate was not found");
    }
    return point;
  }

  static bool coordinatesEqual(double first, double second) {
    constexpr double AbsoluteTolerance = 1.0e-12;
    constexpr double RelativeTolerance = 1.0e-10;
    return std::abs(first - second)
        <= AbsoluteTolerance
         + RelativeTolerance * std::max(std::abs(first), std::abs(second));
  }
};

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_AHF_DATA_H_
