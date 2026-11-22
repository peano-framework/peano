#ifndef APPARENT_HORIZON_FINDER_DENSE_LINEAR_ALGEBRA_H_
#define APPARENT_HORIZON_FINDER_DENSE_LINEAR_ALGEBRA_H_

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

// Small dense linear solve for the spectral Newton-Raphson step (the dense N x N system
// J delta a = -R). N is small (e.g. 81 at lmax = 8), so a plain Gaussian
// elimination with partial pivoting is fine. 

namespace exahype2::ccz4::ahf {

/**
 * Solve the dense square system A x = b by Gaussian elimination with partial
 * pivoting. A is passed by value (elimination happens on the copy); rows of A are
 * length b.size(). 
 * Throws std::invalid_argument on a shape mismatch and
 * std::runtime_error if the matrix is (numerically) singular.
 */
inline std::vector<double> SolveLinearSystem(std::vector<std::vector<double>> A,
                                             std::vector<double> b) {
  // Checks for A
  const std::size_t n = b.size();
  if (A.size() != n) {
    throw std::invalid_argument("SolveLinearSystem: matrix/vector size mismatch");
  }
  for (const std::vector<double>& row : A) {
    if (row.size() != n) {
      throw std::invalid_argument("SolveLinearSystem: matrix is not square");
    }
  }

// Gaussian elimination with partial pivoting.
  for (std::size_t col = 0; col < n; ++col) {
    // Partial pivot: pick the largest-magnitude entry in this column.
    std::size_t pivot = col;
    double best = std::abs(A[col][col]);
    for (std::size_t row = col + 1; row < n; ++row) {
      const double candidate = std::abs(A[row][col]);
      if (candidate > best) {
        best = candidate;
        pivot = row;
      }
    }
    if (!(best > 0.0)) {
      throw std::runtime_error("SolveLinearSystem: singular matrix");
    }
    if (pivot != col) {
      std::swap(A[pivot], A[col]);
      std::swap(b[pivot], b[col]);
    }

    // Eliminate below the pivot.
    for (std::size_t row = col + 1; row < n; ++row) {
      const double factor = A[row][col] / A[col][col];
      if (factor != 0.0) {
        for (std::size_t c = col; c < n; ++c) {
          A[row][c] -= factor * A[col][c];
        }
        b[row] -= factor * b[col];
      }
    }
  }

  // Back-substitution.
  std::vector<double> x(n, 0.0);
  for (std::size_t i = n; i-- > 0;) {
    double sum = b[i];
    for (std::size_t c = i + 1; c < n; ++c) {
      sum -= A[i][c] * x[c];
    }
    x[i] = sum / A[i][i];
  }
  return x;
}

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_DENSE_LINEAR_ALGEBRA_H_
