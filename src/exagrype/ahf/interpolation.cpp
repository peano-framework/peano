#include "interpolation.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>

namespace exahype2::ccz4::ahf {

double WendlandC2Weight(double q) {
  if (!std::isfinite(q) || q < 0.0) {
    throw std::invalid_argument(
      "WendlandC2Weight: normalized distance must be finite and non-negative"
    );
  }
  if (q >= 1.0) return 0.0;
  const double oneMinus = 1.0 - q;
  return oneMinus * oneMinus * oneMinus * oneMinus * (4.0 * q + 1.0);
}

double LinearReconstructionWeight(
  double distance,
  double lengthScale,
  const LinearReconstructionOptions& options
) {
  if (!std::isfinite(distance) || distance < 0.0
      || !std::isfinite(lengthScale) || lengthScale <= 0.0) {
    throw std::invalid_argument(
      "LinearReconstructionWeight: distance and length scale must be finite and valid");
  }
  if (options.method == LinearReconstructionMethod::Unweighted) {
    return 1.0;
  }
  if (options.method == LinearReconstructionMethod::RadiusWeightedWendlandC2) {
    if (!std::isfinite(options.supportRadius) || options.supportRadius <= 0.0) {
      throw std::invalid_argument(
        "LinearReconstructionWeight: support radius must be finite and strictly positive"
      );
    }
    return WendlandC2Weight(distance / options.supportRadius);
  }
  if (!std::isfinite(options.weightSupport) || options.weightSupport <= 0.0) {
    throw std::invalid_argument(
      "LinearReconstructionWeight: weight support must be finite and strictly positive");
  }
  const double scaledDistance = distance / (options.weightSupport * lengthScale);
  const double weight = std::exp(-scaledDistance * scaledDistance);
  if (!std::isfinite(weight) || weight <= 0.0) {
    throw std::invalid_argument(
      "LinearReconstructionWeight: computed weight must be finite and strictly positive");
  }
  return weight;
}

GridCoordinate SurfacePointToCartesian(
  const AHGuessPoint& surfacePoint,
  const ReferencePoint& referencePoint
) {
  if (!std::isfinite(surfacePoint.r)
      || !std::isfinite(surfacePoint.rho)
      || !std::isfinite(surfacePoint.sigma)
      || !std::isfinite(referencePoint.x)
      || !std::isfinite(referencePoint.y)
      || !std::isfinite(referencePoint.z)) {
    throw std::invalid_argument(
      "SurfacePointToCartesian: radius, angles, and reference point must be finite");
  }
  if (surfacePoint.r <= 0.0) {
    throw std::invalid_argument(
      "SurfacePointToCartesian: apparent-horizon radius must be strictly positive");
  }

  const double cosSigma = std::cos(surfacePoint.sigma);
  return GridCoordinate{
    referencePoint.x + surfacePoint.r * cosSigma * std::cos(surfacePoint.rho),
    referencePoint.y + surfacePoint.r * cosSigma * std::sin(surfacePoint.rho),
    referencePoint.z + surfacePoint.r * std::sin(surfacePoint.sigma)
  };
}

Tensor3 ComputeRadialProjectionTensor(const GridCoordinate& relativeCoordinate) {
  if (!std::isfinite(relativeCoordinate.x)
      || !std::isfinite(relativeCoordinate.y)
      || !std::isfinite(relativeCoordinate.z)) {
    throw std::invalid_argument(
      "ComputeRadialProjectionTensor: relative coordinate must be finite");
  }
  const Vector3 x{relativeCoordinate.x, relativeCoordinate.y, relativeCoordinate.z};
  const double radiusSquared = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  Tensor3 T{};
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = i; j < 3; ++j) {
      T[i][j] = (i == j ? radiusSquared : 0.0) - x[i] * x[j];
      T[j][i] = T[i][j];
    }
  }
  return T;
}

AnalyticCoordinateGeometry ComputeAnalyticCoordinateGeometry(
  const GridCoordinate& globalCoordinate,
  const ReferencePoint& referencePoint
) {
  constexpr double MinimumAbsoluteCosSigma = 1.0e-12;
  if (!std::isfinite(globalCoordinate.x) || !std::isfinite(globalCoordinate.y)
      || !std::isfinite(globalCoordinate.z) || !std::isfinite(referencePoint.x)
      || !std::isfinite(referencePoint.y) || !std::isfinite(referencePoint.z)) {
    throw std::invalid_argument(
      "ComputeAnalyticCoordinateGeometry: coordinates must be finite");
  }
  const GridCoordinate relative{
    globalCoordinate.x - referencePoint.x,
    globalCoordinate.y - referencePoint.y,
    globalCoordinate.z - referencePoint.z
  };
  const double cylindricalRadius = std::hypot(relative.x, relative.y);
  const double radius = std::hypot(cylindricalRadius, relative.z);
  if (!(radius > 0.0) || !std::isfinite(radius)) {
    throw std::invalid_argument(
      "ComputeAnalyticCoordinateGeometry: radius must be finite and positive");
  }
  const double cosSigma = cylindricalRadius / radius;
  if (std::abs(cosSigma) < MinimumAbsoluteCosSigma) {
    throw std::domain_error(
      "ComputeAnalyticCoordinateGeometry: longitude coordinate is singular near cos(sigma)=0");
  }

  const double rho = std::atan2(relative.y, relative.x);
  const double sigma = std::atan2(relative.z, cylindricalRadius);
  const double sinRho = std::sin(rho);
  const double cosRho = std::cos(rho);
  const double sinSigma = std::sin(sigma);
  const double inverseRadius = 1.0 / radius;
  const double inverseRadiusSquared = inverseRadius * inverseRadius;
  const double inverseRadiusCosSigma = inverseRadius / cosSigma;
  const double inverseRadiusSquaredCosSigmaSquared =
    inverseRadiusSquared / (cosSigma * cosSigma);

  AnalyticCoordinateGeometry result;
  result.T = ComputeRadialProjectionTensor(relative);

  // X^u_i is stored as [u][i], with u=(rho,sigma) and i=(x,y,z).
  result.gradient[0] = {
    -sinRho * inverseRadiusCosSigma,
    cosRho * inverseRadiusCosSigma,
    0.0
  };
  result.gradient[1] = {
    -cosRho * sinSigma * inverseRadius,
    -sinRho * sinSigma * inverseRadius,
    cosSigma * inverseRadius
  };

  // X^u_ij is stored symmetrically as [u][i][j].
  Tensor3& rhoHessian = result.hessian[0];
  rhoHessian[0][0] = std::sin(2.0 * rho) * inverseRadiusSquaredCosSigmaSquared;
  rhoHessian[0][1] =
    -std::cos(2.0 * rho) * inverseRadiusSquaredCosSigmaSquared;
  rhoHessian[1][0] = rhoHessian[0][1];
  rhoHessian[1][1] = -rhoHessian[0][0];

  Tensor3& sigmaHessian = result.hessian[1];
  const double tanSigma = std::tan(sigma);
  const double cosTwoSigma = std::cos(2.0 * sigma);
  sigmaHessian[0][0] = tanSigma * inverseRadiusSquared
    * (2.0 * cosSigma * cosSigma * cosRho * cosRho - sinRho * sinRho);
  sigmaHessian[0][1] = 0.5 * std::sin(2.0 * rho) * inverseRadiusSquared
    * (std::sin(2.0 * sigma) + tanSigma);
  sigmaHessian[1][0] = sigmaHessian[0][1];
  sigmaHessian[0][2] = -cosRho * cosTwoSigma * inverseRadiusSquared;
  sigmaHessian[2][0] = sigmaHessian[0][2];
  sigmaHessian[1][1] = tanSigma * inverseRadiusSquared
    * (2.0 * cosSigma * cosSigma * sinRho * sinRho - cosRho * cosRho);
  sigmaHessian[1][2] = -sinRho * cosTwoSigma * inverseRadiusSquared;
  sigmaHessian[2][1] = sigmaHessian[1][2];
  sigmaHessian[2][2] = -std::sin(2.0 * sigma) * inverseRadiusSquared;
  return result;
}

std::vector<std::size_t> FindNearestGridCellsBruteForce(
  const GridData& gridData,
  const GridCoordinate& target,
  std::size_t neighbourCount
) {
  if (gridData.coordinates.empty()) {
    throw std::invalid_argument(
      "FindNearestGridCellsBruteForce: cannot search an empty Cartesian grid");
  }
  if (neighbourCount == 0) {
    throw std::invalid_argument(
      "FindNearestGridCellsBruteForce: neighbour count must be greater than zero");
  }
  if (neighbourCount > gridData.coordinates.size()) {
    throw std::invalid_argument(
      "FindNearestGridCellsBruteForce: requested " + std::to_string(neighbourCount)
      + " neighbours from only " + std::to_string(gridData.coordinates.size())
      + " cells");
  }
  if (!std::isfinite(target.x)
      || !std::isfinite(target.y)
      || !std::isfinite(target.z)) {
    throw std::invalid_argument(
      "FindNearestGridCellsBruteForce: target coordinate must be finite");
  }

  std::vector<std::size_t> indices(gridData.coordinates.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});

  const auto squaredDistance = [&gridData, &target](std::size_t cellIndex) {
    const GridCoordinate& coordinate = gridData.coordinates[cellIndex];
    const double dx = coordinate.x - target.x;
    const double dy = coordinate.y - target.y;
    const double dz = coordinate.z - target.z;
    return dx * dx + dy * dy + dz * dz;
  };
  const auto nearer = [&squaredDistance](std::size_t first, std::size_t second) {
    const double firstDistance = squaredDistance(first);
    const double secondDistance = squaredDistance(second);
    return firstDistance < secondDistance
        || (firstDistance == secondDistance && first < second);
  };

  std::partial_sort(
    indices.begin(),
    indices.begin() + neighbourCount,
    indices.end(),
    nearer
  );
  indices.resize(neighbourCount);
  return indices;
}

struct CartesianSpatialIndex::Implementation {
  // 1. Store one grid cell and the links/splitting axis of its k-d tree node.
  struct Node {
    std::size_t cellIndex = 0;
    std::size_t left = std::numeric_limits<std::size_t>::max();
    std::size_t right = std::numeric_limits<std::size_t>::max();
    unsigned char axis = 0;
  };

  static constexpr std::size_t NoNode = std::numeric_limits<std::size_t>::max();

  // 2. Validate and copy coordinates, then build the immutable tree once.
  explicit Implementation(const GridData& gridData): coordinates(gridData.coordinates) {
    if (coordinates.empty()) {
      throw std::invalid_argument("CartesianSpatialIndex: cannot search an empty grid");
    }
    for (const GridCoordinate& coordinate : coordinates) {
      if (!std::isfinite(coordinate.x) || !std::isfinite(coordinate.y)
          || !std::isfinite(coordinate.z)) {
        throw std::invalid_argument("CartesianSpatialIndex: coordinates must be finite");
      }
    }
    std::vector<std::size_t> indices(coordinates.size());
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    nodes.reserve(coordinates.size());
    root = build(indices, 0, indices.size(), 0);
  }

  // Read one Cartesian component selected by axis=(x,y,z).
  double component(std::size_t cellIndex, std::size_t axis) const {
    const GridCoordinate& coordinate = coordinates[cellIndex];
    return axis == 0 ? coordinate.x : axis == 1 ? coordinate.y : coordinate.z;
  }

  // 3. Recursively median-split the cells, cycling through x, y, and z.
  std::size_t build(
    std::vector<std::size_t>& indices,
    std::size_t begin,
    std::size_t end,
    std::size_t depth
  ) {
    if (begin == end) {
      return NoNode;
    }
    const std::size_t axis = depth % 3;
    const std::size_t middle = begin + (end - begin) / 2;
    std::nth_element(
      indices.begin() + begin,
      indices.begin() + middle,
      indices.begin() + end,
      [this, axis](std::size_t first, std::size_t second) {
        const double firstCoordinate = component(first, axis);
        const double secondCoordinate = component(second, axis);
        return firstCoordinate < secondCoordinate
            || (firstCoordinate == secondCoordinate && first < second);
      }
    );
    const std::size_t nodeIndex = nodes.size();
    nodes.push_back(Node{.cellIndex = indices[middle], .axis = static_cast<unsigned char>(axis)});
    const std::size_t left = build(indices, begin, middle, depth + 1);
    const std::size_t right = build(indices, middle + 1, end, depth + 1);
    nodes[nodeIndex].left = left;
    nodes[nodeIndex].right = right;
    return nodeIndex;
  }

  // 4. Compare candidates without an unnecessary square root.
  double squaredDistance(std::size_t cellIndex, const GridCoordinate& target) const {
    const GridCoordinate& coordinate = coordinates[cellIndex];
    const double dx = coordinate.x - target.x;
    const double dy = coordinate.y - target.y;
    const double dz = coordinate.z - target.z;
    return dx * dx + dy * dy + dz * dz;
  }

  using Candidate = std::pair<double, std::size_t>;
  using CandidateHeap = std::priority_queue<Candidate>;

  // 5. Search the near branch first and prune the far branch when possible.
  void query(
    std::size_t nodeIndex,
    const GridCoordinate& target,
    std::size_t neighbourCount,
    CandidateHeap& candidates
  ) const {
    if (nodeIndex == NoNode) {
      return;
    }
    const Node& node = nodes[nodeIndex];
    const Candidate candidate{squaredDistance(node.cellIndex, target), node.cellIndex};
    if (candidates.size() < neighbourCount) {
      candidates.push(candidate);
    } else if (candidate < candidates.top()) {
      candidates.pop();
      candidates.push(candidate);
    }

    const double targetComponent = node.axis == 0 ? target.x
      : node.axis == 1 ? target.y : target.z;
    const double difference = targetComponent - component(node.cellIndex, node.axis);
    const std::size_t nearChild = difference <= 0.0 ? node.left : node.right;
    const std::size_t farChild = difference <= 0.0 ? node.right : node.left;
    query(nearChild, target, neighbourCount, candidates);
    const double worstDistance = candidates.size() < neighbourCount
      ? std::numeric_limits<double>::infinity() : candidates.top().first;
    if (difference * difference <= worstDistance) {
      query(farChild, target, neighbourCount, candidates);
    }
  }

  void queryRadius(
    std::size_t nodeIndex,
    const GridCoordinate& target,
    double squaredRadius,
    std::vector<Candidate>& candidates
  ) const {
    if (nodeIndex == NoNode) return;
    const Node& node = nodes[nodeIndex];
    const double candidateDistance = squaredDistance(node.cellIndex, target);
    if (candidateDistance < squaredRadius) {
      candidates.emplace_back(candidateDistance, node.cellIndex);
    }
    const double targetComponent = node.axis == 0 ? target.x
      : node.axis == 1 ? target.y : target.z;
    const double difference = targetComponent - component(node.cellIndex, node.axis);
    const std::size_t nearChild = difference <= 0.0 ? node.left : node.right;
    const std::size_t farChild = difference <= 0.0 ? node.right : node.left;
    queryRadius(nearChild, target, squaredRadius, candidates);
    if (difference * difference < squaredRadius) {
      queryRadius(farChild, target, squaredRadius, candidates);
    }
  }

  // Immutable storage shared by all read-only queries.
  std::vector<GridCoordinate> coordinates;
  std::vector<Node> nodes;
  std::size_t root = NoNode;
};

CartesianSpatialIndex::CartesianSpatialIndex(const GridData& gridData):
  _implementation(std::make_shared<const Implementation>(gridData)) {}

std::size_t CartesianSpatialIndex::size() const {
  return _implementation->coordinates.size();
}

std::vector<std::size_t> CartesianSpatialIndex::findNearest(
  const GridCoordinate& target,
  std::size_t neighbourCount
) const {
  if (!std::isfinite(target.x) || !std::isfinite(target.y)
      || !std::isfinite(target.z)) {
    throw std::invalid_argument("CartesianSpatialIndex: target must be finite");
  }
  if (neighbourCount == 0) {
    throw std::invalid_argument(
      "CartesianSpatialIndex: neighbour count must be greater than zero");
  }
  if (neighbourCount > size()) {
    throw std::invalid_argument(
      "CartesianSpatialIndex: requested " + std::to_string(neighbourCount)
      + " neighbours from only " + std::to_string(size()) + " cells");
  }
  Implementation::CandidateHeap candidates;
  _implementation->query(_implementation->root, target, neighbourCount, candidates);
  std::vector<Implementation::Candidate> ordered;
  ordered.reserve(candidates.size());
  while (!candidates.empty()) {
    ordered.push_back(candidates.top());
    candidates.pop();
  }
  std::sort(ordered.begin(), ordered.end());
  std::vector<std::size_t> indices;
  indices.reserve(ordered.size());
  for (const auto& candidate : ordered) {
    indices.push_back(candidate.second);
  }
  return indices;
}

std::vector<std::size_t> CartesianSpatialIndex::findWithinRadius(
  const GridCoordinate& target,
  double radius
) const {
  if (!std::isfinite(target.x) || !std::isfinite(target.y)
      || !std::isfinite(target.z)) {
    throw std::invalid_argument("CartesianSpatialIndex: target must be finite");
  }
  if (!std::isfinite(radius) || radius <= 0.0) {
    throw std::invalid_argument(
      "CartesianSpatialIndex: radius must be finite and strictly positive"
    );
  }
  std::vector<Implementation::Candidate> ordered;
  _implementation->queryRadius(
    _implementation->root, target, radius * radius, ordered
  );
  std::sort(ordered.begin(), ordered.end());
  std::vector<std::size_t> indices;
  indices.reserve(ordered.size());
  for (const auto& candidate : ordered) indices.push_back(candidate.second);
  return indices;
}

std::vector<std::size_t> FindNearestGridCells(
  const CartesianSpatialIndex& spatialIndex,
  const GridCoordinate& target,
  std::size_t neighbourCount
) {
  return spatialIndex.findNearest(target, neighbourCount);
}

std::vector<std::size_t> FindGridCellsWithinRadius(
  const CartesianSpatialIndex& spatialIndex,
  const GridCoordinate& target,
  double radius
) {
  return spatialIndex.findWithinRadius(target, radius);
}

std::vector<std::size_t> FindNearestGridCells(
  const GridData& gridData,
  const GridCoordinate& target,
  std::size_t neighbourCount
) {
  return CartesianSpatialIndex(gridData).findNearest(target, neighbourCount);
}

std::size_t PolynomialBasisSize(std::size_t degree) {
  return (degree + 1) * (degree + 2) * (degree + 3) / 6;
}

PolynomialBasis GeneratePolynomialBasis(std::size_t degree) {
  PolynomialBasis basis;
  basis.terms.reserve(PolynomialBasisSize(degree));
  for (std::size_t totalDegree = 0; totalDegree <= degree; ++totalDegree) {
    for (std::size_t xExponent = totalDegree;; --xExponent) {
      const std::size_t remaining = totalDegree - xExponent;
      for (std::size_t yExponent = remaining;; --yExponent) {
        const PolynomialExponent exponent{
          .x = xExponent,
          .y = yExponent,
          .z = remaining - yExponent
        };
        const std::size_t index = basis.terms.size();
        basis.terms.push_back(exponent);
        if (exponent == PolynomialExponent{1, 0, 0}) basis.linearX = index;
        if (exponent == PolynomialExponent{0, 1, 0}) basis.linearY = index;
        if (exponent == PolynomialExponent{0, 0, 1}) basis.linearZ = index;
        if (yExponent == 0) break;
      }
      if (xExponent == 0) break;
    }
  }
  if (basis.terms.size() != PolynomialBasisSize(degree)
      || basis.terms.empty() || basis.terms.front() != PolynomialExponent{0, 0, 0}) {
    throw std::logic_error("GeneratePolynomialBasis: inconsistent basis generation");
  }
  return basis;
}

namespace {

double integerPower(double value, std::size_t exponent) {
  double result = 1.0;
  for (std::size_t power = 0; power < exponent; ++power) result *= value;
  return result;
}

class PolynomialFactorization {
public:
  PolynomialFactorization(
    std::span<const GridCoordinate> coordinates,
    const GridCoordinate& target,
    const LinearReconstructionOptions& options,
    const char* caller
  ): _sampleCount(coordinates.size()) {
    if (options.polynomialDegree > 3) {
      throw std::invalid_argument(std::string(caller) + ": supported polynomial degree is 0 through 3");
    }
    _basis = GeneratePolynomialBasis(options.polynomialDegree);
    if (coordinates.size() < _basis.terms.size()) {
      const std::string requiredSamples = options.polynomialDegree == 1
        ? "four" : std::to_string(_basis.terms.size());
      throw std::invalid_argument(
        std::string(caller) + ": degree-" + std::to_string(options.polynomialDegree)
        + " reconstruction requires at least " + requiredSamples
        + " samples");
    }
    if (!std::isfinite(target.x) || !std::isfinite(target.y) || !std::isfinite(target.z)) {
      throw std::invalid_argument(std::string(caller) + ": target must be finite");
    }
    if (!std::isfinite(options.rankToleranceMultiplier)
        || options.rankToleranceMultiplier <= 0.0) {
      throw std::invalid_argument(
        std::string(caller) + ": rank tolerance multiplier must be finite and positive");
    }

    for (const GridCoordinate& sample : coordinates) {
      if (!std::isfinite(sample.x) || !std::isfinite(sample.y) || !std::isfinite(sample.z)) {
        throw std::invalid_argument(std::string(caller) + ": sample coordinates must be finite");
      }
      _lengthScale = std::max(
        _lengthScale,
        std::hypot(sample.x - target.x, sample.y - target.y, sample.z - target.z));
    }
    if (options.method == LinearReconstructionMethod::RadiusWeightedWendlandC2) {
      if (!std::isfinite(options.supportRadius) || options.supportRadius <= 0.0) {
        throw std::invalid_argument(
          std::string(caller) + ": support radius must be finite and strictly positive"
        );
      }
      _lengthScale = options.supportRadius;
    }
    if (!std::isfinite(_lengthScale)
        || (_lengthScale <= 0.0 && options.polynomialDegree > 0)) {
      throw std::invalid_argument(std::string(caller) + ": samples are geometrically rank deficient");
    }
    if (_lengthScale == 0.0) _lengthScale = 1.0;

    _sqrtWeights.resize(coordinates.size());
    _q.assign(_basis.terms.size(), std::vector<double>(coordinates.size()));
    for (std::size_t row = 0; row < coordinates.size(); ++row) {
      const GridCoordinate& sample = coordinates[row];
      const double dx = sample.x - target.x;
      const double dy = sample.y - target.y;
      const double dz = sample.z - target.z;
      const double distance = std::hypot(dx, dy, dz);
      _sqrtWeights[row] = std::sqrt(
        LinearReconstructionWeight(distance, _lengthScale, options));
      const double X = dx / _lengthScale;
      const double Y = dy / _lengthScale;
      const double Z = dz / _lengthScale;
      for (std::size_t column = 0; column < _basis.terms.size(); ++column) {
        const PolynomialExponent& exponent = _basis.terms[column];
        _q[column][row] = integerPower(X, exponent.x)
                        * integerPower(Y, exponent.y)
                        * integerPower(Z, exponent.z) * _sqrtWeights[row];
      }
    }

    const auto dot = [](const std::vector<double>& first, const std::vector<double>& second) {
      return std::inner_product(first.begin(), first.end(), second.begin(), 0.0);
    };
    std::vector<double> squaredNorm(_basis.terms.size());
    for (std::size_t column = 0; column < _basis.terms.size(); ++column) {
      squaredNorm[column] = dot(_q[column], _q[column]);
    }
    const double largestInitialNorm =
      std::sqrt(*std::max_element(squaredNorm.begin(), squaredNorm.end()));
    const double rankTolerance = options.rankToleranceMultiplier
      * std::numeric_limits<double>::epsilon() * static_cast<double>(coordinates.size())
      * largestInitialNorm;
    _r.assign(_basis.terms.size(), std::vector<double>(_basis.terms.size()));
    _permutation.resize(_basis.terms.size());
    std::iota(_permutation.begin(), _permutation.end(), std::size_t{0});

    double minimumDiagonal = std::numeric_limits<double>::infinity();
    double maximumDiagonal = 0.0;
    for (std::size_t pivot = 0; pivot < _basis.terms.size(); ++pivot) {
      const auto largest = std::max_element(squaredNorm.begin() + pivot, squaredNorm.end());
      const std::size_t selected = static_cast<std::size_t>(largest - squaredNorm.begin());
      if (selected != pivot) {
        std::swap(_q[pivot], _q[selected]);
        std::swap(squaredNorm[pivot], squaredNorm[selected]);
        std::swap(_permutation[pivot], _permutation[selected]);
        for (std::size_t previous = 0; previous < pivot; ++previous) {
          std::swap(_r[previous][pivot], _r[previous][selected]);
        }
      }
      const double diagonal = std::sqrt(squaredNorm[pivot]);
      if (!(diagonal > rankTolerance) || !std::isfinite(diagonal)) {
        throw std::invalid_argument(std::string(caller) + ": samples are geometrically rank deficient");
      }
      _r[pivot][pivot] = diagonal;
      minimumDiagonal = std::min(minimumDiagonal, diagonal);
      maximumDiagonal = std::max(maximumDiagonal, diagonal);
      for (double& entry : _q[pivot]) entry /= diagonal;
      for (std::size_t column = pivot + 1; column < _basis.terms.size(); ++column) {
        _r[pivot][column] = dot(_q[pivot], _q[column]);
        for (std::size_t row = 0; row < coordinates.size(); ++row) {
          _q[column][row] -= _r[pivot][column] * _q[pivot][row];
        }
        squaredNorm[column] = dot(_q[column], _q[column]);
      }
    }
    _conditionEstimate = maximumDiagonal / minimumDiagonal;
  }

  std::vector<double> solve(std::span<const double> values, const char* caller) const {
    if (values.size() != _sampleCount) {
      throw std::invalid_argument(std::string(caller) + ": right-hand-side size mismatch");
    }
    std::vector<double> projected(_basis.terms.size());
    for (std::size_t column = 0; column < _basis.terms.size(); ++column) {
      for (std::size_t row = 0; row < values.size(); ++row) {
        if (!std::isfinite(values[row])) {
          throw std::invalid_argument(std::string(caller) + ": sample values must be finite");
        }
        projected[column] += _q[column][row] * values[row] * _sqrtWeights[row];
      }
    }
    std::vector<double> pivoted(_basis.terms.size());
    for (std::size_t reverse = _basis.terms.size(); reverse-- > 0;) {
      double residual = projected[reverse];
      for (std::size_t column = reverse + 1; column < _basis.terms.size(); ++column) {
        residual -= _r[reverse][column] * pivoted[column];
      }
      pivoted[reverse] = residual / _r[reverse][reverse];
    }
    std::vector<double> coefficients(_basis.terms.size());
    for (std::size_t column = 0; column < _basis.terms.size(); ++column) {
      coefficients[_permutation[column]] = pivoted[column];
    }
    return coefficients;
  }

  const PolynomialBasis& basis() const { return _basis; }
  std::size_t rank() const { return _basis.terms.size(); }
  double lengthScale() const { return _lengthScale; }
  double conditionEstimate() const { return _conditionEstimate; }

private:
  PolynomialBasis _basis;
  std::size_t _sampleCount = 0;
  double _lengthScale = 0.0;
  double _conditionEstimate = 1.0;
  std::vector<double> _sqrtWeights;
  std::vector<std::vector<double>> _q;
  std::vector<std::vector<double>> _r;
  std::vector<std::size_t> _permutation;
};

}  // namespace

LinearPolynomialFit FitLinearPolynomial(
  std::span<const GridCoordinate> sampleCoordinates,
  std::span<const double> sampleValues,
  const GridCoordinate& target,
  const LinearReconstructionOptions& options
) {
  if (sampleCoordinates.size() != sampleValues.size()) {
    throw std::invalid_argument(
      "FitLinearPolynomial: coordinate and value counts must match");
  }
  const std::size_t requestedCount = options.numberOfNeighbours == 0
    ? sampleCoordinates.size() : options.numberOfNeighbours;
  if (requestedCount > sampleCoordinates.size()) {
    throw std::invalid_argument(
      "FitLinearPolynomial: requested neighbour count exceeds supplied samples");
  }
  sampleCoordinates = sampleCoordinates.first(requestedCount);
  sampleValues = sampleValues.first(requestedCount);
  PolynomialFactorization factorization(
    sampleCoordinates, target, options, "FitLinearPolynomial");
  const std::vector<double> coefficients =
    factorization.solve(sampleValues, "FitLinearPolynomial");
  const PolynomialBasis& basis = factorization.basis();
  LinearPolynomialFit result{
    .value = coefficients[basis.constant],
    .lengthScale = factorization.lengthScale()
  };
  if (basis.linearX != PolynomialBasis::NoTerm) {
    result.derivative = {
      coefficients[basis.linearX] / result.lengthScale,
      coefficients[basis.linearY] / result.lengthScale,
      coefficients[basis.linearZ] / result.lengthScale
    };
  }
  if (!std::isfinite(result.value)
      || !std::isfinite(result.derivative[0])
      || !std::isfinite(result.derivative[1])
      || !std::isfinite(result.derivative[2])) {
    throw std::runtime_error("FitLinearPolynomial: QR solve produced a non-finite result");
  }
  return result;
}

CompactLinearFit FitCompactLinearPolynomial(
  const GridData& gridData,
  std::span<const std::size_t> neighbourIndices,
  const GridCoordinate& target,
  const LinearReconstructionOptions& options
) {
  constexpr std::size_t FieldCount = NumberOfCompactFields;
  const std::size_t requestedCount = options.numberOfNeighbours == 0
    ? neighbourIndices.size() : options.numberOfNeighbours;
  if (requestedCount > neighbourIndices.size()) {
    throw std::invalid_argument(
      "FitCompactLinearPolynomial: requested neighbour count exceeds supplied neighbours");
  }
  neighbourIndices = neighbourIndices.first(requestedCount);
  if (!gridData.isConsistent()) {
    throw std::invalid_argument(
      "FitCompactLinearPolynomial: GridData must contain fourteen values per cell");
  }

  std::vector<GridCoordinate> coordinates;
  coordinates.reserve(neighbourIndices.size());
  std::array<std::vector<double>, FieldCount> rightHandSides;
  for (auto& values : rightHandSides) values.reserve(neighbourIndices.size());
  for (std::size_t cellIndex : neighbourIndices) {
    if (cellIndex >= gridData.coordinates.size()) {
      throw std::out_of_range(
        "FitCompactLinearPolynomial: neighbour cell index is out of range");
    }
    coordinates.push_back(gridData.coordinates[cellIndex]);
    const auto row = gridData.cell(cellIndex).row();
    for (std::size_t field = 0; field < FieldCount; ++field) {
      rightHandSides[field].push_back(row[field]);
    }
  }

  PolynomialFactorization factorization(
    coordinates, target, options, "FitCompactLinearPolynomial");
  CompactLinearFit result{
    .numberOfNeighbours = neighbourIndices.size(),
    .polynomialDegree = options.polynomialDegree,
    .rank = factorization.rank(),
    .lengthScale = factorization.lengthScale(),
    .conditionEstimate = factorization.conditionEstimate()
  };
  const PolynomialBasis& basis = factorization.basis();
  for (std::size_t field = 0; field < FieldCount; ++field) {
    const std::vector<double> coefficients =
      factorization.solve(rightHandSides[field], "FitCompactLinearPolynomial");
    result.values[field] = coefficients[basis.constant];
    if (basis.linearX != PolynomialBasis::NoTerm) {
      result.gradients[field] = {
        coefficients[basis.linearX] / result.lengthScale,
        coefficients[basis.linearY] / result.lengthScale,
        coefficients[basis.linearZ] / result.lengthScale
      };
    }
  }
  return result;
}

void InterpolateToSurfaceNearestNeighbour(
  const GridData& gridData,
  const AHGuess& ahGuess,
  const ReferencePoint& referencePoint,
  InterpolatedSurface& interpolatedSurface
) {
  InterpolateCartesianToSurface(
    gridData,
    ahGuess,
    referencePoint,
    interpolatedSurface,
    {.method = CartesianToSurfaceMethod::NearestNeighbour, .numberOfNeighbours = 1}
  );
}

CartesianPointInterpolator::CartesianPointInterpolator(
  const GridData& gridData,
  const CartesianToSurfaceOptions& options
):
  _gridData(&gridData),
  _options(options),
  _spatialIndex(gridData),
  _domainMinimum(gridData.coordinates.front()),
  _domainMaximum(gridData.coordinates.front()) {
  if (!gridData.isConsistent()) {
    throw std::invalid_argument(
      "CartesianPointInterpolator: GridData value count must equal cell count times fourteen");
  }
  if (options.method != CartesianToSurfaceMethod::RadiusWeightedPolynomial
      && options.numberOfNeighbours == 0) {
    throw std::invalid_argument(
      "CartesianPointInterpolator: neighbour count must be greater than zero");
  }
  if (options.method != CartesianToSurfaceMethod::NearestNeighbour
      && options.polynomialDegree > 3) {
    throw std::invalid_argument(
      "CartesianPointInterpolator: supported polynomial degree is 0 through 3");
  }
  if (options.method == CartesianToSurfaceMethod::RadiusWeightedPolynomial
      && (!std::isfinite(options.supportRadius)
          || options.supportRadius <= 0.0)) {
    throw std::invalid_argument(
      "CartesianPointInterpolator: support radius must be finite and strictly positive"
    );
  }
  if (!std::isfinite(options.domainRelativeTolerance)
      || options.domainRelativeTolerance < 0.0) {
    throw std::invalid_argument(
      "CartesianPointInterpolator: domain relative tolerance must be finite and non-negative");
  }
  for (const GridCoordinate& coordinate : gridData.coordinates) {
    _domainMinimum.x = std::min(_domainMinimum.x, coordinate.x);
    _domainMinimum.y = std::min(_domainMinimum.y, coordinate.y);
    _domainMinimum.z = std::min(_domainMinimum.z, coordinate.z);
    _domainMaximum.x = std::max(_domainMaximum.x, coordinate.x);
    _domainMaximum.y = std::max(_domainMaximum.y, coordinate.y);
    _domainMaximum.z = std::max(_domainMaximum.z, coordinate.z);
  }
  _domainDiagonal = std::hypot(
    _domainMaximum.x - _domainMinimum.x,
    _domainMaximum.y - _domainMinimum.y,
    _domainMaximum.z - _domainMinimum.z
  );
}

bool CartesianPointInterpolator::coversTarget(const GridCoordinate& target
) const {
  if (!std::isfinite(target.x) || !std::isfinite(target.y)
      || !std::isfinite(target.z)) {
    return false;
  }
  const double outsideX = std::max(
    {_domainMinimum.x - target.x, 0.0, target.x - _domainMaximum.x}
  );
  const double outsideY = std::max(
    {_domainMinimum.y - target.y, 0.0, target.y - _domainMaximum.y}
  );
  const double outsideZ = std::max(
    {_domainMinimum.z - target.z, 0.0, target.z - _domainMaximum.z}
  );
  return std::hypot(outsideX, outsideY, outsideZ)
         <= _options.domainRelativeTolerance * _domainDiagonal;
}

void CartesianPointInterpolator::interpolatePoints(
  std::span<const GridCoordinate> targets,
  std::vector<InterpolatedCartesianPoint>& output
) const {
  std::vector<InterpolatedCartesianPoint> reconstructed;
  reconstructed.reserve(targets.size());

  for (std::size_t pointIndex = 0; pointIndex < targets.size(); ++pointIndex) {
    const GridCoordinate target = targets[pointIndex];
    try {
      if (!std::isfinite(target.x) || !std::isfinite(target.y)
          || !std::isfinite(target.z)) {
        throw std::invalid_argument("target coordinate must be finite");
      }
      if (!coversTarget(target)) {
        throw std::out_of_range(
          "target lies outside the sampled Cartesian domain tolerance");
      }

      InterpolationDiagnostics diagnostics{
        .requestedMethod = _options.method,
        .requestedDegree = _options.method == CartesianToSurfaceMethod::NearestNeighbour
          ? 0u : _options.polynomialDegree,
        .degreeUsed = _options.method == CartesianToSurfaceMethod::NearestNeighbour
          ? 0u : _options.polynomialDegree,
        .neighbourCount = _options.method == CartesianToSurfaceMethod::NearestNeighbour
          ? 1u : (_options.method == CartesianToSurfaceMethod::RadiusWeightedPolynomial
                    ? 0u : _options.numberOfNeighbours),
        .target = target,
        .rank = _options.method == CartesianToSurfaceMethod::NearestNeighbour ? 1u : 0u,
        .conditionEstimate = 1.0,
        .orderReduced = false
      };

      bool useNearestNeighbour =
        _options.method == CartesianToSurfaceMethod::NearestNeighbour;
      if (!useNearestNeighbour
          && _options.method != CartesianToSurfaceMethod::RadiusWeightedPolynomial
          && (_options.numberOfNeighbours < PolynomialBasisSize(_options.polynomialDegree)
              || _options.numberOfNeighbours > _gridData->coordinates.size())) {
        if (!_options.allowOrderReduction) {
          throw std::invalid_argument(
            "polynomial reconstruction has insufficient neighbours and order reduction is disabled");
        }
        useNearestNeighbour = true;
        diagnostics.degreeUsed = 0;
        diagnostics.neighbourCount = 1;
        diagnostics.rank = 1;
        diagnostics.orderReduced = true;
      }

      std::vector<std::size_t> neighbours =
        _options.method == CartesianToSurfaceMethod::RadiusWeightedPolynomial
          ? FindGridCellsWithinRadius(_spatialIndex, target, _options.supportRadius)
          : FindNearestGridCells(
              _spatialIndex, target,
              useNearestNeighbour ? 1 : _options.numberOfNeighbours
            );
      if (!useNearestNeighbour
          && neighbours.size() < PolynomialBasisSize(_options.polynomialDegree)) {
        if (!_options.allowOrderReduction) {
          throw std::invalid_argument(
            "polynomial reconstruction has insufficient cells inside support radius and order reduction is disabled"
          );
        }
        useNearestNeighbour = true;
        neighbours = FindNearestGridCells(_spatialIndex, target, 1);
        diagnostics.degreeUsed = 0;
        diagnostics.neighbourCount = 1;
        diagnostics.rank = 1;
        diagnostics.orderReduced = true;
      }
      diagnostics.neighbourCount = neighbours.size();
      const auto validateStencil = [this](std::span<const std::size_t> stencil) {
        for (std::size_t cellIndex : stencil) {
          for (double value : _gridData->cell(cellIndex).row()) {
            if (!std::isfinite(value)) {
              throw std::invalid_argument(
                "selected interpolation stencil contains a non-finite compact value");
            }
          }
        }
      };
      validateStencil(neighbours);

      std::array<double, NumberOfCompactFields> compactValues{};
      std::array<Vector3, NumberOfCompactFields> compactGradients{};
      bool derivativesAvailable = false;
      if (useNearestNeighbour) {
        const auto row = _gridData->cell(neighbours.front()).row();
        std::copy(row.begin(), row.end(), compactValues.begin());
      } else {
        const LinearReconstructionOptions reconstructionOptions{
          .method = _options.method == CartesianToSurfaceMethod::WeightedPolynomial
            ? LinearReconstructionMethod::WeightedMovingLeastSquares
            : (_options.method == CartesianToSurfaceMethod::RadiusWeightedPolynomial
                ? LinearReconstructionMethod::RadiusWeightedWendlandC2
                : LinearReconstructionMethod::Unweighted),
          .numberOfNeighbours = neighbours.size(),
          .polynomialDegree = _options.polynomialDegree,
          .weightSupport = _options.weightSupport,
          .supportRadius = _options.supportRadius,
          .rankToleranceMultiplier = _options.rankToleranceMultiplier
        };
        try {
          const CompactLinearFit fit = FitCompactLinearPolynomial(
            *_gridData, neighbours, target, reconstructionOptions);
          compactValues = fit.values;
          compactGradients = fit.gradients;
          derivativesAvailable =
            _options.method != CartesianToSurfaceMethod::RadiusWeightedPolynomial
            || _options.polynomialDegree >= 1;
          diagnostics.rank = fit.rank;
          diagnostics.conditionEstimate = fit.conditionEstimate;
        } catch (const std::invalid_argument& error) {
          const bool rankFailure =
            std::string(error.what()).find("rank deficient") != std::string::npos;
          if (!rankFailure || !_options.allowOrderReduction) throw;
          neighbours = FindNearestGridCells(_spatialIndex, target, 1);
          validateStencil(neighbours);
          const auto row = _gridData->cell(neighbours.front()).row();
          std::copy(row.begin(), row.end(), compactValues.begin());
          diagnostics.degreeUsed = 0;
          diagnostics.neighbourCount = 1;
          diagnostics.rank = 1;
          diagnostics.conditionEstimate = 1.0;
          diagnostics.orderReduced = true;
        }
      }

      std::array<double, NumberOfSymmetricTensorComponents> gammaTildeComponents{};
      std::array<double, NumberOfSymmetricTensorComponents> ATildeComponents{};
      for (std::size_t component = 0;
           component < NumberOfSymmetricTensorComponents; ++component) {
        gammaTildeComponents[component] = compactValues[component];
        ATildeComponents[component] = compactValues[component + 6];
      }
      const Tensor3 gammaTilde = UnpackSymmetricTensor(gammaTildeComponents);
      const Tensor3 ATilde = UnpackSymmetricTensor(ATildeComponents);
      const double traceK =
        compactValues[static_cast<std::size_t>(CompactField::TraceK)];
      const double phi = compactValues[static_cast<std::size_t>(CompactField::Phi)];

      InterpolatedCartesianPoint point{.target = target, .diagnostics = diagnostics};
      point.geometry.spatialMetric = ComputePhysicalSpatialMetric(gammaTilde, phi);
      point.geometry.extrinsicCurvature =
        ComputePhysicalExtrinsicCurvature(ATilde, traceK, gammaTilde, phi);
      point.geometry.traceK = traceK;
      point.geometry.spatialMetricDerivativeAvailable = derivativesAvailable;
      if (derivativesAvailable) {
        Tensor3Derivative gammaTildeDerivative{};
        for (std::size_t k = 0; k < 3; ++k) {
          std::array<double, NumberOfSymmetricTensorComponents> components{};
          for (std::size_t component = 0;
               component < NumberOfSymmetricTensorComponents; ++component) {
            components[component] = compactGradients[component][k];
          }
          gammaTildeDerivative[k] = UnpackSymmetricTensor(components);
        }
        point.geometry.spatialMetricDerivative =
          ComputePhysicalSpatialMetricDerivative(
            gammaTilde,
            gammaTildeDerivative,
            phi,
            compactGradients[static_cast<std::size_t>(CompactField::Phi)]
          );
      }

      const auto requireFinite = [](double value, const char* name) {
        if (!std::isfinite(value)) {
          throw std::runtime_error(std::string("produced non-finite ") + name);
        }
      };
      requireFinite(point.geometry.traceK, "trace K");
      for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
          requireFinite(point.geometry.spatialMetric[i][j], "physical metric");
          requireFinite(point.geometry.extrinsicCurvature[i][j], "extrinsic curvature");
          for (std::size_t k = 0; k < 3; ++k) {
            requireFinite(
              point.geometry.spatialMetricDerivative[k][i][j], "metric derivative");
          }
        }
      }
      reconstructed.push_back(point);
    } catch (const std::exception& error) {
      std::ostringstream message;
      message << "CartesianPointInterpolator: target point " << pointIndex
              << " target (" << target.x << ", " << target.y << ", " << target.z
              << ") failed: " << error.what();
      throw std::runtime_error(message.str());
    }
  }
  output = std::move(reconstructed);
}

void InterpolateCartesianToSurface(
  const CartesianPointInterpolator& interpolator,
  const AHGuess& ahGuess,
  const ReferencePoint& referencePoint,
  InterpolatedSurface& interpolatedSurface
) {
  std::vector<GridCoordinate> targets;
  targets.reserve(ahGuess.points.size());
  for (std::size_t pointIndex = 0; pointIndex < ahGuess.points.size(); ++pointIndex) {
    try {
      targets.push_back(SurfacePointToCartesian(ahGuess.points[pointIndex], referencePoint));
    } catch (const std::exception& error) {
      std::ostringstream message;
      message << "InterpolateCartesianToSurface: surface point " << pointIndex
              << " target (nan, nan, nan) failed: " << error.what();
      throw std::runtime_error(message.str());
    }
  }

  std::vector<InterpolatedCartesianPoint> cartesianPoints;
  try {
    interpolator.interpolatePoints(targets, cartesianPoints);
  } catch (const std::exception& error) {
    std::string message = error.what();
    const std::string marker = "target point ";
    if (const std::size_t position = message.find(marker); position != std::string::npos) {
      message.replace(position, marker.size(), "surface point ");
    }
    throw std::runtime_error("InterpolateCartesianToSurface: " + message);
  }

  std::vector<InterpolatedSurfacePoint> points;
  points.reserve(ahGuess.points.size());
  for (std::size_t pointIndex = 0; pointIndex < ahGuess.points.size(); ++pointIndex) {
    const AHGuessPoint& guess = ahGuess.points[pointIndex];
    InterpolatedSurfacePoint point{
      .r = guess.r,
      .rho = guess.rho,
      .sigma = guess.sigma,
      .geometry = cartesianPoints[pointIndex].geometry,
      .interpolationDiagnostics = cartesianPoints[pointIndex].diagnostics
    };
    const AnalyticCoordinateGeometry analytic =
      ComputeAnalyticCoordinateGeometry(targets[pointIndex], referencePoint);
    point.geometry.T = analytic.T;
    point.geometry.coordinateGradient = analytic.gradient;
    point.geometry.coordinateHessian = analytic.hessian;
    points.push_back(point);
  }
  interpolatedSurface.points = std::move(points);
}

void InterpolateCartesianToSurface(
  const GridData& gridData,
  const AHGuess& ahGuess,
  const ReferencePoint& referencePoint,
  InterpolatedSurface& interpolatedSurface,
  const CartesianToSurfaceOptions& options
) {
  const CartesianPointInterpolator interpolator(gridData, options);
  InterpolateCartesianToSurface(
    interpolator, ahGuess, referencePoint, interpolatedSurface);
}

void InterpolateFromCartesianToShell(const GridData& gridData,
                                     const AHGuess& ahGuess,
                                     const ReferencePoint& referencePoint,
                                     InterpolatedShell& interpolatedShell,
                                     std::span<const AngularSample> angularSamples) {
  (void)gridData;
  (void)ahGuess;
  (void)referencePoint;
  (void)angularSamples;
  (void)interpolatedShell;
  // Interpolate and calculate all necessary quantities from the 3D Cartesian grid to the 3D spherical
  //          coordinates using the given angular coordinates sample ($\{\rho_I,\sigma_I\}$) within a radius range
  //          (thickness) determined by AH guess input, i.e., we prepare a shell-like subdomain around
  //          the AH guess with spherical coordinates. The angular coordinate sample is given as input.
  // The radial coordinate sample can be determined by this function. Notice we need to calculate derivatives 
  //          here, so the subdomain may require halos if we do the process in parallel.
  // – Input: Full grid dataset ($Q\{x_I,y_I,z_I\}$), AH Guess function ($ \{h^{(k)}(\rho _I, \sigma _I)\}$),
  //          Reference point($s_0$), optional angular coordinate samples, Variable mapping (see below).
  //          If angularSamples is empty, use the angular coordinates stored in ahGuess.
  //– Output: the data structure InterpolatedShell that contains $T_{ij}$, $X^u_i$, $X^u_{ij}$, $ \gamma_{ij}$,
  //          $\partial_i \gamma_{jk}$, $K$ for each sample point ($\{r_I,\rho_I,\sigma_I\}$) within the
  //          shell-like subdomain, the structure also contains the shell outer and inner radii and the 
  //          angular coordinate sample.
  // The variable mapping input is required for generality; the user needs to specify how their quantities
  //          are ordered in the grid solution. The output quantities are calculated from the interpolated
  //          variables (the expressions of $T_{ij}$, $X^u_i$, and $X^u_{ij}$ are given below).
  // The data structure InterpolatedShell would be created (initialization) when this function is called
  //          for the first time during iteration. Then, during the following iteration (see below), the
  //          structure only needs to be updated in two cases: 
  //          1. the guess radius of the AH($h$) is approaching the radial boundary of the shell, 
  //             we need to thicken the shell in the corresponding direction.
  //          2. the resolution of the angular coordinates samples gets changed, we need to re-initialize
  //             the structure correspondingly.
  //          Except for these two cases, the structure remains unchanged.
  // We need derivatives of $Q$ over Cartesian coordinates and the derivative of $h$ over surface coordinates
  //          during the calculation.
}

void InterpolateFromShellToSurface(const InterpolatedShell& interpolatedShell,
                                   const AHGuess& ahGuess,
                                   const ReferencePoint& referencePoint,
                                   InterpolatedSurface& interpolatedSurface) {
  (void)interpolatedShell;
  (void)ahGuess;
  (void)referencePoint;
  (void)interpolatedSurface;
  // nterpolate and calculate all necessary quantities from the subdomain stored in InterpolatedShell
  //          to the surface points $\left\{\left( {h^{(k)}(\rho _I, \sigma _I)},\rho_I,\sigma_I \right)\right\}$.
  // Notice the angular sample needs to be identical, thus we actually only need to do the 1D radial
  //          interpolation and could increase the interpolation order for accuracy.
  // – Input: the data structure InterpolatedShell, AH Guess function($ \{h^{(k)}(\rho _I, \sigma _I)\}$),
  //          Reference point($s_0$).
  // – Output: the data structure InterpolatedSurface that contains $s_i$, $\partial_i s_j$, $T_{ij}$,
  //          $X^u_i$, $X^u_{ij}$, $ \gamma_{ij}$, $\partial_i \gamma_{jk}$, $K$ for each sample point
  //          ($\left\{\left( {h^{(k)}(\rho _I, \sigma _I)},\rho_I,\sigma_I \right)\right\}$) on the
  //          surface, and the angular coordinate sample
}

}  // namespace exahype2::ccz4::ahf
