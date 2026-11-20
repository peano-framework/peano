#ifndef APPARENT_HORIZON_FINDER_EXPANSION_H_
#define APPARENT_HORIZON_FINDER_EXPANSION_H_

#include <array>

#include "ahf-data.h"

namespace exahype2::ccz4::ahf {

/**
 * Local assembly of the outgoing null expansion Theta at one surface point.
 * This should vanish for an apparent horizon.
 *
 * This will be invoked by the AssembleGlobalMatrix kernel
 *
 * This kernel is discretisation-agnostic: given the (non-unit)
 * surface normal s_i and its gradient partial_i s_j,
 * together with the local ADM geometry, it returns
 *
 *   Theta = A / D^{3/2} + B / D^{1/2} + C / D - trK,
 *
 * with A, B, C, D given by the appendix expressions
 *
 *   A  = -gamma^{ik} gamma^{jl} s_k s_l (partial_i s_j)
 *        - 1/2 gamma^{ij} s_j s_k s_l (partial_i gamma^{kl}),
 *   B  = s_j (partial_i gamma^{ij}) + gamma^{ij} (partial_i s_j)
 *        + (partial_i ln sqrt(gamma)) gamma^{ij} s_j,
 *   C  = K^{ij} s_i s_j.
 *   D  = gamma^{ij} s_i s_j,
 *
 * s_i and partial_i s_j are supplied by the
 * shell-to-surface step (they are pure functions of the guess geometry), so this
 * kernel starts from s_i, partial_i s_j, gamma_ij, partial_k gamma_ij, K_ij, trK
 * already present at the point.
 *
 * @param surfaceNormal            s_i.
 * @param surfaceNormalDerivative  partial_i s_j, indexed [i][j].
 * @param spatialMetric            gamma_ij.
 * @param spatialMetricDerivative  partial_k gamma_ij, indexed [k][i][j].
 * @param extrinsicCurvature       K_ij.
 * @param traceK                   trK = gamma^{ij} K_ij.
 * @return the expansion Theta at the point.
 */
double computeExpansion(
  const Vector3& surfaceNormal,
  const Tensor3& surfaceNormalDerivative,
  const Tensor3& spatialMetric,
  const std::array<Tensor3, 3>& spatialMetricDerivative,
  const Tensor3& extrinsicCurvature,
  double traceK);

/**
 * Convenience overload taking a fully populated InterpolatedSurfacePoint.
 */
double computeExpansion(const InterpolatedSurfacePoint& point);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_EXPANSION_H_
