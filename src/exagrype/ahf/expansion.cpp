#include "expansion.h"

#include <cmath>

#include "tensor-algebra.h"

// Note: Index conventions match ahf-data.h:
//   surfaceNormalDerivative[i][j]      = partial_i s_j
//   spatialMetricDerivative[k][i][j]   = partial_k gamma_ij

namespace exahype2::ccz4::ahf {

double computeExpansion(
  const Vector3& s,
  const Tensor3& ds,
  const Tensor3& gamma,
  const std::array<Tensor3, 3>& dgamma,
  const Tensor3& K,
  double traceK) {
  const Tensor3 ginv = inverse(gamma);

  // partial_k gamma^{ij} = -gamma^{ia} gamma^{jb} partial_k gamma_ab.
  std::array<Tensor3, 3> dginv{};
  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        double value = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            value -= ginv[i][a] * ginv[j][b] * dgamma[k][a][b];
          }
        }
        dginv[k][i][j] = value;
      }
    }
  }

  // partial_k ln sqrt(gamma) = 1/2 gamma^{ab} partial_k gamma_ab.
  Vector3 dlnsqrtg{};
  for (int k = 0; k < 3; ++k) {
    double value = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        value += ginv[a][b] * dgamma[k][a][b];
      }
    }
    dlnsqrtg[k] = 0.5 * value;
  }

  // D = gamma^{ij} s_i s_j.
  double D = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      D += ginv[i][j] * s[i] * s[j];
    }
  }

  // A1 = -gamma^{ik} gamma^{jl} s_k s_l (partial_i s_j).
  double A1 = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          A1 -= ginv[i][k] * ginv[j][l] * s[k] * s[l] * ds[i][j];
        }
      }
    }
  }

  // A2 = -1/2 gamma^{ij} s_j s_k s_l (partial_i gamma^{kl}).
  double A2 = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          A2 -= 0.5 * ginv[i][j] * s[j] * s[k] * s[l] * dginv[i][k][l];
        }
      }
    }
  }

  const double A = A1 + A2;

  // B1 = s_j (partial_i gamma^{ij}), with partial_i gamma^{ij} = sum_i dginv[i][i][j].
  double B1 = 0.0;
  for (int j = 0; j < 3; ++j) {
    double divGinv = 0.0;
    for (int i = 0; i < 3; ++i) {
      divGinv += dginv[i][i][j];
    }
    B1 += s[j] * divGinv;
  }

  // B2 = gamma^{ij} (partial_i s_j).
  double B2 = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      B2 += ginv[i][j] * ds[i][j];
    }
  }

  // B3 = (partial_i ln sqrt(gamma)) gamma^{ij} s_j.
  double B3 = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      B3 += dlnsqrtg[i] * ginv[i][j] * s[j];
    }
  }

  const double B = B1 + B2 + B3;

  // C = K^{ij} s_i s_j, with K^{ij} = gamma^{ia} gamma^{jb} K_ab.
  double C = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double Kup = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          Kup += ginv[i][a] * ginv[j][b] * K[a][b];
        }
      }
      C += Kup * s[i] * s[j];
    }
  }

  return A / std::pow(D, 1.5) + B / std::sqrt(D) + C / D - traceK;
}

double computeExpansion(const InterpolatedSurfacePoint& point) {
  return computeExpansion(
    point.surfaceNormal,
    point.surfaceNormalDerivative,
    point.geometry.spatialMetric,
    point.geometry.spatialMetricDerivative,
    point.geometry.extrinsicCurvature,
    point.geometry.traceK);
}

}  // namespace exahype2::ccz4::ahf
