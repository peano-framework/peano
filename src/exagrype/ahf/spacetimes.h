#ifndef APPARENT_HORIZON_FINDER_SPACETIMES_H_
#define APPARENT_HORIZON_FINDER_SPACETIMES_H_

#include "ahf-data.h"

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace exahype2::ccz4::ahf {

  /**
   * Analytic spacetimes and the geometry they induce.
   * Useful for testing.
   *
   * Each spacetime exposes three exact field functions on a Cartesian point X
   * (a three-vector), templated on the scalar type so they can be evaluated at
   * real (double) or complex (std::complex<double>) argument:
   *
   *   gamma(X) -> gamma_ij   spatial 3-metric
   *   beta(X)  -> beta_i     shift (lower index)
   *   alpha(X) -> alpha      lapse
   *
   * From these, MetricData() derives everything the expansion needs
   * (gamma^{ij}, partial_k gamma_ij, K_ij, trK) using complex-step
   * differentiation, which is exact to machine precision for these holomorphic
   * functions. The extrinsic curvature uses the stationary ADM relation K_ij =
   * (D_i beta_j + D_j beta_i) / (2 alpha), valid because partial_t gamma_ij = 0
   * on these slices.
   */

  /** Small vector/matrix aliases templated on the scalar type (real or
   * complex). */
  template <class T> using FieldVector = std::array<T, 3>;
  template <class T> using FieldMatrix = std::array<std::array<T, 3>, 3>;

  namespace detail {

    /** Euclidean radius |X| = sqrt(sum_i X_i^2); valid for real or complex T.
     */
    template <class T> T radius(const FieldVector<T>& X) {
      return std::sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);
    }

    /** 3x3 identity in the scalar type T. */
    template <class T> FieldMatrix<T> identity() {
      FieldMatrix<T> result{};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          result[i][j] = (i == j) ? T(1.0) : T(0.0);
        }
      }
      return result;
    }

  } // namespace detail

  /**
   * Schwarzschild in isotropic coordinates (time symmetric, K_ij = 0).
   * Horizon at the isotropic coordinate radius r = M/2.
   */
  struct SchwarzschildIsotropic {
    double M = 1.0;

    template <class T> T psi(const FieldVector<T>& X) const {
      return T(1.0) + T(M) / (T(2.0) * detail::radius(X));
    }

    template <class T> FieldMatrix<T> gamma(const FieldVector<T>& X) const {
      const T        p4     = psi(X) * psi(X) * psi(X) * psi(X);
      FieldMatrix<T> result = detail::identity<T>();
      for (int i = 0; i < 3; ++i) {
        result[i][i] = p4;
      }
      return result;
    }

    template <class T> FieldVector<T> beta(const FieldVector<T>&) const {
      return FieldVector<T>{T(0.0), T(0.0), T(0.0)};
    }

    // Time-symmetric slice: beta = 0 and partial_t gamma = 0, so K_ij = 0 for
    // any lapse.
    // The static isotropic lapse vanishes at the horizon, which
    // would make the K_ij = (D beta)/(2 alpha) evaluation a spurious 0/0.
    // use alpha = 1.
    template <class T> T alpha(const FieldVector<T>&) const { return T(1.0); }

    double horizonRadius() const { return 0.5 * M; }
    double horizonArea() const { return 16.0 * std::numbers::pi * M * M; }
  };

  /**
   * Schwarzschild in Kerr-Schild coordinates (K_ij != 0, horizon at r = 2M).
   */
  struct SchwarzschildKerrSchild {
    double M = 1.0;

    template <class T> FieldMatrix<T> gamma(const FieldVector<T>& X) const {
      const T              r = detail::radius(X);
      const T              H = T(M) / r;
      const FieldVector<T> l{X[0] / r, X[1] / r, X[2] / r};
      FieldMatrix<T>       result = detail::identity<T>();
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          result[i][j] += T(2.0) * H * l[i] * l[j];
        }
      }
      return result;
    }

    template <class T> FieldVector<T> beta(const FieldVector<T>& X) const {
      const T r = detail::radius(X);
      const T H = T(M) / r;
      return FieldVector<
        T>{T(2.0) * H * X[0] / r, T(2.0) * H * X[1] / r, T(2.0) * H * X[2] / r};
    }

    template <class T> T alpha(const FieldVector<T>& X) const {
      const T H = T(M) / detail::radius(X);
      return T(1.0) / std::sqrt(T(1.0) + T(2.0) * H);
    }

    double horizonRadius() const { return 2.0 * M; }
    double horizonArea() const { return 16.0 * std::numbers::pi * M * M; }
  };

  /**
   * Kerr in Kerr-Schild Cartesian coordinates, spin a along z. The horizon is a
   * coordinate ellipsoid; the Kerr-Schild radius on it equals the
   * Boyer-Lindquist outer horizon r+ = M + sqrt(M^2 - a^2), with area 8 pi M
   * r+.
   */
  struct KerrKerrSchild {
    double M = 1.0;
    double a = 0.6;

    template <class T> T ksRadius(const FieldVector<T>& X) const {
      const T rho2 = X[0] * X[0] + X[1] * X[1] + X[2] * X[2];
      const T a2   = T(a) * T(a);
      const T r2
        = T(0.5)
          * ((rho2 - a2) + std::sqrt((rho2 - a2) * (rho2 - a2) + T(4.0) * a2 * X[2] * X[2]));
      return std::sqrt(r2);
    }

    template <class T>
    void HandL(const FieldVector<T>& X, T& H, FieldVector<T>& l) const {
      const T r  = ksRadius(X);
      const T a2 = T(a) * T(a);
      H          = T(M) * r * r * r / (r * r * r * r + a2 * X[2] * X[2]);
      l          = FieldVector<T>{
        (r * X[0] + T(a) * X[1]) / (r * r + a2),
        (r * X[1] - T(a) * X[0]) / (r * r + a2),
        X[2] / r};
    }

    template <class T> FieldMatrix<T> gamma(const FieldVector<T>& X) const {
      T              H;
      FieldVector<T> l;
      HandL(X, H, l);
      FieldMatrix<T> result = detail::identity<T>();
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          result[i][j] += T(2.0) * H * l[i] * l[j];
        }
      }
      return result;
    }

    template <class T> FieldVector<T> beta(const FieldVector<T>& X) const {
      T              H;
      FieldVector<T> l;
      HandL(X, H, l);
      return FieldVector<
        T>{T(2.0) * H * l[0], T(2.0) * H * l[1], T(2.0) * H * l[2]};
    }

    template <class T> T alpha(const FieldVector<T>& X) const {
      T              H;
      FieldVector<T> l;
      HandL(X, H, l);
      return T(1.0) / std::sqrt(T(1.0) + T(2.0) * H);
    }

    double horizonRadius() const { return M + std::sqrt(M * M - a * a); }
    double horizonArea() const {
      return 8.0 * std::numbers::pi * M * horizonRadius();
    }
  };

  /**
   * Brill-Lindquist multi-puncture initial data (time symmetric, K_ij = 0). 
   * A conformally flat slice with psi = 1 + sum_n m_n / (2 |X - p_n|).
   * Note: A single puncture at the origin reduces exactly to
   * SchwarzschildIsotropic.
   */
  struct BrillLindquist {
    std::vector<std::pair<double, FieldVector<double>>> punctures;

    template <class T> T psi(const FieldVector<T>& X) const {
      T out = T(1.0);
      for (const auto& puncture : punctures) {
        const double               m = puncture.first;
        const FieldVector<double>& p = puncture.second;
        const FieldVector<T>
          delta{X[0] - T(p[0]), X[1] - T(p[1]), X[2] - T(p[2])};
        out += T(m) / (T(2.0) * detail::radius(delta));
      }
      return out;
    }

    template <class T> FieldMatrix<T> gamma(const FieldVector<T>& X) const {
      const T        p4     = psi(X) * psi(X) * psi(X) * psi(X);
      FieldMatrix<T> result = detail::identity<T>();
      for (int i = 0; i < 3; ++i) {
        result[i][i] = p4;
      }
      return result;
    }

    template <class T> FieldVector<T> beta(const FieldVector<T>&) const {
      return FieldVector<T>{T(0.0), T(0.0), T(0.0)};
    }

    template <class T> T alpha(const FieldVector<T>&) const { return T(1.0); }

    /** Exact ADM mass sum_n m_n. */
    double admMass() const {
      double total = 0.0;
      for (const auto& puncture : punctures) {
        total += puncture.first;
      }
      return total;
    }

    /** Bare mass of the sole puncture; defined only for single-puncture slices.
     */
    double M() const {
      if (punctures.size() != 1) {
        throw std::logic_error(
          "BrillLindquist::M is only defined for a single-puncture slice"
        );
      }
      return punctures.front().first;
    }
    double horizonRadius() const { return 0.5 * M(); }
    double horizonArea() const { return 16.0 * std::numbers::pi * M() * M(); }

    /** Mass-weighted centroid of the punctures (natural common-horizon centre).
     */
    FieldVector<double> centroid() const {
      FieldVector<double> sum{0.0, 0.0, 0.0};
      for (const auto& puncture : punctures) {
        for (int i = 0; i < 3; ++i) {
          sum[i] += puncture.first * puncture.second[i];
        }
      }
      const double mass = admMass();
      for (int i = 0; i < 3; ++i) {
        sum[i] /= mass;
      }
      return sum;
    }

    /** Puncture positions, in construction order. */
    std::vector<FieldVector<double>> puncturePositions() const {
      std::vector<FieldVector<double>> positions;
      positions.reserve(punctures.size());
      for (const auto& puncture : punctures) {
        positions.push_back(puncture.second);
      }
      return positions;
    }

    enum class Axis { X = 0, Y = 1, Z = 2 };
    enum class Plane { XY, XZ, YZ };

    /** One puncture of mass m at center (equals Schwarzschild isotropic). */
    static BrillLindquist single(
      double              m      = 1.0,
      FieldVector<double> center = {0.0, 0.0, 0.0}
    ) {
      return BrillLindquist{{{m, center}}};
    }

    /** Two equal punctures of mass m at +/- separation/2 along axis. */
    static BrillLindquist binary(
      double m          = 1.0,
      double separation = 2.0,
      Axis   axis       = Axis::Z
    ) {
      FieldVector<double> half{0.0, 0.0, 0.0};
      half[static_cast<int>(axis)] = 0.5 * separation;
      const FieldVector<double> minusHalf{-half[0], -half[1], -half[2]};
      return BrillLindquist{{{m, half}, {m, minusHalf}}};
    }

    /**
     * Three equal punctures of mass m on an equilateral triangle of side length
     * separation, centred on the origin, in the chosen coordinate plane.
     * Placing the triangle in-plane makes the common horizon genuinely
     * non-axisymmetric (three-fold symmetry, exciting the m = +/- 3 azimuthal
     * harmonics).
     */
    static BrillLindquist triangle(
      double m          = 1.0,
      double separation = 1.5,
      Plane  plane      = Plane::XY
    ) {
      const double circumradius = separation / std::sqrt(3.0); // from side
                                                               // length
      const double degToRad  = std::numbers::pi / 180.0;
      const double angles[3] = {
        90.0 * degToRad,
        210.0 * degToRad,
        330.0 * degToRad};
      int a0 = 0;
      int a1 = 1;
      switch (plane) {
      case Plane::XY:
        a0 = 0;
        a1 = 1;
        break;
      case Plane::XZ:
        a0 = 0;
        a1 = 2;
        break;
      case Plane::YZ:
        a0 = 1;
        a1 = 2;
        break;
      }
      std::vector<std::pair<double, FieldVector<double>>> punctures;
      for (double angle : angles) {
        FieldVector<double> p{0.0, 0.0, 0.0};
        p[a0] = circumradius * std::cos(angle);
        p[a1] = circumradius * std::sin(angle);
        punctures.push_back({m, p});
      }
      return BrillLindquist{punctures};
    }
  };

  /**
   * Geometry derived from an analytic spacetime at the Cartesian point X,
   * packed into the same InterpolatedGeometryData that the grid-interpolation
   * path uses.
   * Fills spatialMetric gamma_ij, spatialMetricDerivative partial_k
   * gamma_ij, extrinsicCurvature K_ij and traceK; the coordinate-transform
   * fields (T, coordinateGradient, coordinateHessian) are left untouched
   * (populated separately). partial_k gamma_ij and K_ij are computed by
   * complex-step differentiation of the spacetime field functions.
   */
  template <class Spacetime>
  InterpolatedGeometryData MetricData(
    const Spacetime& spacetime,
    const Vector3&   X
  );

} // namespace exahype2::ccz4::ahf

#endif // APPARENT_HORIZON_FINDER_SPACETIMES_H_
