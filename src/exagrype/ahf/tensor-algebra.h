#ifndef APPARENT_HORIZON_FINDER_TENSOR_ALGEBRA_H_
#define APPARENT_HORIZON_FINDER_TENSOR_ALGEBRA_H_

#include "ahf-data.h"

// Some handful of 3x3 operations needed to compute the expansion Theta.
// These work on the AHF Vector3 / Tensor3 types. 
// At the moment, we do not use make use of tarch::la::Matrix here. 
// Instead we use the same raw-loop form used by CCZ4Kernels.cpp.
//
// Index convention: spatial indices i, j, k, ... run over {0:x, 1:y, 2:z}; 
// Tensor3 is indexed as [i][j].

namespace exahype2::ccz4::ahf {

/**
 * Determinant of a 3x3 matrix (general; used for the metric determinant).
 */
inline double determinant(const Tensor3& m) {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
       - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
       + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

/**
 * Inverse of a 3x3 matrix via the closed-form adjugate / determinant. For a
 * symmetric input (as gamma_ij always is here) the result is symmetric. This is
 * the same expression the CCZ4 PDE kernels use for the (inverse) spatial metric.
 *
 * The determinant is assumed non-zero (a valid spatial metric is positive definite).
 */
inline Tensor3 inverse(const Tensor3& m) {
  const double invDet = 1.0 / determinant(m);
  Tensor3 result{};
  result[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet;
  result[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * invDet;
  result[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;
  result[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * invDet;
  result[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
  result[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * invDet;
  result[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet;
  result[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * invDet;
  result[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet;
  return result;
}

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_TENSOR_ALGEBRA_H_
