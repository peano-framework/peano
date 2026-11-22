#include "spacetimes.h"

#include "tensor-algebra.h"

#include <complex>

namespace exahype2::ccz4::ahf {

  namespace {

    // Complex-step size. Complex-step differentiation is truncation-free, so
    // any sufficiently small step is machine-exact; the only constraint is
    // avoiding underflow of the imaginary part in std::sqrt. 
    // 1e-100 safely above the underflow seems to avoid this underflow threshold.
    constexpr double ComplexStep = 1.0e-100;

    using Complex = std::complex<double>;

    /** Promote a real Cartesian point to a complex-typed FieldVector. */
    FieldVector<Complex> toComplex(const Vector3& X) {
      return FieldVector<Complex>{Complex(X[0]), Complex(X[1]), Complex(X[2])};
    }

  } // namespace

  template <class Spacetime>
  InterpolatedGeometryData MetricData(
    const Spacetime& spacetime,
    const Vector3&   X
  ) {
    const FieldVector<double> point{X[0], X[1], X[2]};

    // Real-valued fields.
    const FieldMatrix<double> gamma = spacetime.template gamma<double>(point);
    const FieldVector<double> beta  = spacetime.template beta<double>(point);
    const double              alpha = spacetime.template alpha<double>(point);

    Tensor3 spatialMetric{};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        spatialMetric[i][j] = gamma[i][j];
      }
    }
    const Tensor3 ginv = inverse(spatialMetric);

    // Complex-step gradient of the metric: dgamma[k][i][j] = partial_k
    // gamma_ij.
    Tensor3Derivative dgamma{};
    for (int k = 0; k < 3; ++k) {
      FieldVector<Complex> perturbed = toComplex(X);
      perturbed[k] += Complex(0.0, ComplexStep);
      const FieldMatrix<Complex> gc = spacetime.template gamma<Complex>(
        perturbed
      );
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          dgamma[k][i][j] = gc[i][j].imag() / ComplexStep;
        }
      }
    }

    // Complex-step gradient of the shift: dbeta[i][j] = partial_i beta_j.
    Tensor3 dbeta{};
    for (int i = 0; i < 3; ++i) {
      FieldVector<Complex> perturbed = toComplex(X);
      perturbed[i] += Complex(0.0, ComplexStep);
      const FieldVector<Complex> bc = spacetime.template beta<Complex>(perturbed
      );
      for (int j = 0; j < 3; ++j) {
        dbeta[i][j] = bc[j].imag() / ComplexStep;
      }
    }

    // Christoffel symbols of the 3-metric:
    //   Gamma^a_ij = 1/2 gamma^{ab} (partial_i gamma_bj + partial_j gamma_bi
    //                                - partial_b gamma_ij).
    Tensor3Derivative christoffel{}; // [a][i][j]
    for (int a = 0; a < 3; ++a) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          double value = 0.0;
          for (int b = 0; b < 3; ++b) {
            const double term = dgamma[i][b][j] + dgamma[j][b][i]
                                - dgamma[b][i][j];
            value += ginv[a][b] * term;
          }
          christoffel[a][i][j] = 0.5 * value;
        }
      }
    }

    // Covariant derivative of the shift and the extrinsic curvature (stationary
    // ADM):
    //   D_i beta_j = partial_i beta_j - Gamma^k_ij beta_k,
    //   K_ij       = (D_i beta_j + D_j beta_i) / (2 alpha).
    Tensor3 covariantBeta{};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        double value = dbeta[i][j];
        for (int k = 0; k < 3; ++k) {
          value -= christoffel[k][i][j] * beta[k];
        }
        covariantBeta[i][j] = value;
      }
    }

    Tensor3 extrinsicCurvature{};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        extrinsicCurvature[i][j] = (covariantBeta[i][j] + covariantBeta[j][i])
                                   / (2.0 * alpha);
      }
    }

    double traceK = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        traceK += ginv[i][j] * extrinsicCurvature[i][j];
      }
    }

    InterpolatedGeometryData data;
    data.spatialMetric                    = spatialMetric;
    data.spatialMetricDerivative          = dgamma;
    data.spatialMetricDerivativeAvailable = true;
    data.extrinsicCurvature               = extrinsicCurvature;
    data.traceK                           = traceK;
    return data;
  }

  // Explicit instantiations for the concrete spacetimes used in the tests
  template InterpolatedGeometryData MetricData<
    SchwarzschildIsotropic>(const SchwarzschildIsotropic&, const Vector3&);
  template InterpolatedGeometryData MetricData<
    SchwarzschildKerrSchild>(const SchwarzschildKerrSchild&, const Vector3&);
  template InterpolatedGeometryData MetricData<
    KerrKerrSchild>(const KerrKerrSchild&, const Vector3&);
  template InterpolatedGeometryData MetricData<
    BrillLindquist>(const BrillLindquist&, const Vector3&);

} // namespace exahype2::ccz4::ahf
