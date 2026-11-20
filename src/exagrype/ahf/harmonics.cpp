#include "harmonics.h"

#include <cmath>
#include <numbers>

// Conventions:
//  - Azimuth phi = rho.
//  - Elevation sigma:
//    - Colatitude theta = pi/2 - sigma, 
//    - mu = sin(sigma) = cos(theta), 

namespace exahype2::ccz4::ahf {

namespace {

/**
 * Normalisation making the real Y_lm orthonormal on the unit sphere.
 *
 * base = sqrt((2l+1)/(4 pi) * (l-|m|)!/(l+|m|)!)
 *
 * with an extra sqrt(2) for m != 0.
 * The factorial ratio is formed as a running product to avoid overflow.
 */
double Normalization(int l, int m) {
  const int am = std::abs(m);
  double ratio = 1.0;  // (l-am)! / (l+am)!
  for (int k = l - am + 1; k <= l + am; ++k) {
    ratio /= static_cast<double>(k);
  }
  const double base = std::sqrt((2.0 * l + 1.0) / (4.0 * std::numbers::pi) * ratio);
  return (m == 0) ? base : base * std::sqrt(2.0);
}

/**
 * Associated Legendre function P_l^m(mu) for 0 <= m <= l, with the Condon-Shortley
 * phase (matching scipy.special.lpmv). 
 * Also returns P_{l-1}^m (zero when l-1 < m), which the derivative recurrence needs.
 */
void AssociatedLegendrePair(int m, int l, double mu, double& plm, double& plm1) {
  // P_m^m = (-1)^m (2m-1)!! (1-mu^2)^{m/2}.
  double pmm = 1.0;
  if (m > 0) {
    const double somx2 = std::sqrt((1.0 - mu) * (1.0 + mu));  // = cos(sigma) > 0
    double fact = 1.0;
    for (int i = 1; i <= m; ++i) {
      pmm *= -fact * somx2;
      fact += 2.0;
    }
  }
  if (l == m) {
    plm = pmm;
    plm1 = 0.0;
    return;
  }
  const double pmmp1 = mu * (2.0 * m + 1.0) * pmm;  // P_{m+1}^m
  if (l == m + 1) {
    plm = pmmp1;
    plm1 = pmm;
    return;
  }
  double previous2 = pmm;    // P_{ll-2}^m
  double previous1 = pmmp1;  // P_{ll-1}^m
  double current = 0.0;
  for (int ll = m + 2; ll <= l; ++ll) {
    current = ((2.0 * ll - 1.0) * mu * previous1 - (ll + m - 1.0) * previous2)
            / (ll - m);
    previous2 = previous1;
    previous1 = current;
  }
  plm = previous1;   // P_l^m
  plm1 = previous2;  // P_{l-1}^m
}

/**
 * P_l^{|m|}(mu) and its first two mu-derivatives.
 * one_minus = 1 - mu^2 = cos^2(sigma) is strictly positive on the Gauss-Legendre
 * nodes, so the divisions are safe.
 */
void LegendreStack(int l, int m, double mu,
                   double& P, double& dPdmu, double& d2Pdmu2) {
  const int am = std::abs(m);
  double plm1 = 0.0;
  AssociatedLegendrePair(am, l, mu, P, plm1);
  const double oneMinus = 1.0 - mu * mu;
  if (l == 0) {
    dPdmu = 0.0;
  } else {
    dPdmu = (l * mu * P - (l + am) * plm1) / (-oneMinus);
  }
  d2Pdmu2 = (2.0 * mu * dPdmu - (l * (l + 1) - am * am / oneMinus) * P) / oneMinus;
}

/**
 * Azimuthal factor and its first two rho-derivatives.
 */
void AzimuthFactor(int m, double rho, double& f, double& df, double& d2f) {
  const int am = std::abs(m);
  if (m > 0) {
    f = std::cos(am * rho);
    df = -am * std::sin(am * rho);
    d2f = -(am * am) * std::cos(am * rho);
  } else if (m < 0) {
    f = std::sin(am * rho);
    df = am * std::cos(am * rho);
    d2f = -(am * am) * std::sin(am * rho);
  } else {
    f = 1.0;
    df = 0.0;
    d2f = 0.0;
  }
}

}  // namespace

std::vector<std::pair<int, int>> EnumerateModes(int lmax) {
  std::vector<std::pair<int, int>> modes;
  modes.reserve(static_cast<std::size_t>((lmax + 1) * (lmax + 1)));
  for (int l = 0; l <= lmax; ++l) {
    for (int m = -l; m <= l; ++m) {
      modes.emplace_back(l, m);
    }
  }
  return modes;
}

std::pair<std::vector<double>, std::vector<double>> GaussLegendre(int n) {
  std::vector<double> nodes(static_cast<std::size_t>(n));
  std::vector<double> weights(static_cast<std::size_t>(n));
  constexpr double kTolerance = 1.0e-15;

  // Roots are symmetric about 0; solve for the upper half by Newton iteration on
  // the Legendre polynomial P_n, evaluated by its three-term recurrence.
  const int halfCount = (n + 1) / 2;
  for (int i = 0; i < halfCount; ++i) {
    double z = std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5));
    double z1 = 0.0;
    double pp = 0.0;
    do {
      double p1 = 1.0;  // P_0
      double p2 = 0.0;  // P_{-1}
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);  // P_n'(z)
      z1 = z;
      z = z1 - p1 / pp;
    } while (std::abs(z - z1) > kTolerance);

    // Ascending order: the largest positive root goes to the last slot.
    const std::size_t low = static_cast<std::size_t>(i);
    const std::size_t high = static_cast<std::size_t>(n - 1 - i);
    nodes[low] = -z;
    nodes[high] = z;
    const double w = 2.0 / ((1.0 - z * z) * pp * pp);
    weights[low] = w;
    weights[high] = w;
  }
  return {nodes, weights};
}

// Y_lm alone. Deliberately not routed through EvalMode: LegendreStack divides
// by 1 - mu^2 to build the derivatives, which is zero at the poles, so the
// cheap-looking EvalMode(...).Y would be computed via non-finite intermediates
// there. Only AssociatedLegendrePair and AzimuthFactor are needed for the
// value, and neither divides by anything. EvalMode calls back into this for its
// own Y, so the formula still lives in one place.
double EvalModeValue(int l, int m, double rho, double sigma) {
  const double mu = std::sin(sigma);

  double P      = 0.0;
  double unused = 0.0;
  AssociatedLegendrePair(std::abs(m), l, mu, P, unused);

  double f   = 0.0;
  double df  = 0.0;
  double d2f = 0.0;
  AzimuthFactor(m, rho, f, df, d2f);

  return Normalization(l, m) * P * f;
}

ModeValues EvalMode(int l, int m, double rho, double sigma) {
  const double mu = std::sin(sigma);
  const double cs = std::cos(sigma);
  const double N = Normalization(l, m);

  double P = 0.0;
  double dPdmu = 0.0;
  double d2Pdmu2 = 0.0;
  LegendreStack(l, m, mu, P, dPdmu, d2Pdmu2);

  double f = 0.0;
  double df = 0.0;
  double d2f = 0.0;
  AzimuthFactor(m, rho, f, df, d2f);

  // sigma-derivatives of the Legendre part L(sigma) = P(sin sigma).
  const double L = P;
  const double L_s = cs * dPdmu;
  const double L_ss = -mu * dPdmu + cs * cs * d2Pdmu2;

  ModeValues values;
  values.Y = EvalModeValue(l, m, rho, sigma);
  values.Yrho = N * L * df;
  values.Ysigma = N * L_s * f;
  values.Yrhorho = N * L * d2f;
  values.Yrhosigma = N * L_s * df;
  values.Ysigmasigma = N * L_ss * f;
  return values;
}

SpectralBasis BuildSpectralBasis(int lmax, int nSigma, int nRho) {
  if (nSigma <= 0) {
    nSigma = lmax + 1;
  }
  if (nRho <= 0) {
    nRho = 2 * lmax + 1;
  }

  const auto [muNodes, muWeights] = GaussLegendre(nSigma);
  const double dRho = 2.0 * std::numbers::pi / nRho;

  SpectralBasis basis;
  basis.lmax = lmax;
  basis.nSigma = nSigma;
  basis.nRho = nRho;
  basis.modes = EnumerateModes(lmax);
  basis.nModes = basis.modes.size();
  basis.nPoints = static_cast<std::size_t>(nSigma) * static_cast<std::size_t>(nRho);

  basis.rho.resize(basis.nPoints);
  basis.sigma.resize(basis.nPoints);
  basis.weight.resize(basis.nPoints);

  // Flattened tensor-product grid, point index p = i * nRho + j (sigma outer,
  // rho inner), matching numpy meshgrid(indexing="xy") + ravel in build_basis.
  for (int i = 0; i < nSigma; ++i) {
    const double sigmaValue = std::asin(muNodes[static_cast<std::size_t>(i)]);
    const double weightValue = muWeights[static_cast<std::size_t>(i)] * dRho;
    for (int j = 0; j < nRho; ++j) {
      const std::size_t p = static_cast<std::size_t>(i) * nRho + j;
      basis.rho[p] = dRho * j;
      basis.sigma[p] = sigmaValue;
      basis.weight[p] = weightValue;
    }
  }

  const std::size_t size = basis.nPoints * basis.nModes;
  basis.Y.resize(size);
  basis.Yrho.resize(size);
  basis.Ysigma.resize(size);
  basis.Yrhorho.resize(size);
  basis.Yrhosigma.resize(size);
  basis.Ysigmasigma.resize(size);

  for (std::size_t p = 0; p < basis.nPoints; ++p) {
    for (std::size_t j = 0; j < basis.nModes; ++j) {
      const auto [l, m] = basis.modes[j];
      const ModeValues v = EvalMode(l, m, basis.rho[p], basis.sigma[p]);
      const std::size_t index = p * basis.nModes + j;
      basis.Y[index] = v.Y;
      basis.Yrho[index] = v.Yrho;
      basis.Ysigma[index] = v.Ysigma;
      basis.Yrhorho[index] = v.Yrhorho;
      basis.Yrhosigma[index] = v.Yrhosigma;
      basis.Ysigmasigma[index] = v.Ysigmasigma;
    }
  }

  return basis;
}

namespace {

// h_field[p] = sum_j matrix[p, j] * coefficients[j].
std::vector<double> ApplyDesignMatrix(const SpectralBasis& basis,
                                      const std::vector<double>& matrix,
                                      const std::vector<double>& coefficients) {
  std::vector<double> result(basis.nPoints, 0.0);
  for (std::size_t p = 0; p < basis.nPoints; ++p) {
    double value = 0.0;
    const std::size_t rowStart = p * basis.nModes;
    for (std::size_t j = 0; j < basis.nModes; ++j) {
      value += matrix[rowStart + j] * coefficients[j];
    }
    result[p] = value;
  }
  return result;
}

}  // namespace

SynthesizedField Synthesize(const SpectralBasis& basis,
                            const std::vector<double>& coefficients) {
  SynthesizedField field;
  field.h = ApplyDesignMatrix(basis, basis.Y, coefficients);
  field.hRho = ApplyDesignMatrix(basis, basis.Yrho, coefficients);
  field.hSigma = ApplyDesignMatrix(basis, basis.Ysigma, coefficients);
  field.hRhoRho = ApplyDesignMatrix(basis, basis.Yrhorho, coefficients);
  field.hRhoSigma = ApplyDesignMatrix(basis, basis.Yrhosigma, coefficients);
  field.hSigmaSigma = ApplyDesignMatrix(basis, basis.Ysigmasigma, coefficients);
  return field;
}

std::vector<double> Project(const SpectralBasis& basis,
                            const std::vector<double>& field) {
  std::vector<double> coefficients(basis.nModes, 0.0);
  for (std::size_t p = 0; p < basis.nPoints; ++p) {
    const double weightedField = basis.weight[p] * field[p];
    const std::size_t rowStart = p * basis.nModes;
    for (std::size_t j = 0; j < basis.nModes; ++j) {
      coefficients[j] += basis.Y[rowStart + j] * weightedField;
    }
  }
  return coefficients;
}

std::vector<double> SphereCoefficients(const SpectralBasis& basis, double radius) {
  std::vector<double> coefficients(basis.nModes, 0.0);
  // Y_00 = 1/sqrt(4 pi)  =>  a_00 = radius * sqrt(4 pi).
  coefficients[0] = radius * std::sqrt(4.0 * std::numbers::pi);
  return coefficients;
}

}  // namespace exahype2::ccz4::ahf
