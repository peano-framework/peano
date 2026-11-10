#include "exagrype/euler/EulerKernels.h"

#include <cmath>

double exagrype::euler::pressure(const double* const Q, const double gamma) {
  const double invRho = 1.0 / Q[0];
  return (gamma - 1.0) * (Q[4] - 0.5 * invRho * (Q[1] * Q[1] + Q[2] * Q[2] + Q[3] * Q[3]));
}

void exagrype::euler::flux(
  const double* const        Q,
  [[maybe_unused]] const double* const ccz4Fields,
  const int                  normal,
  const double               gamma,
  double* const              F
) {
  const double invRho = 1.0 / Q[0];
  const double u      = Q[1 + normal] * invRho;
  const double p      = pressure(Q, gamma);

  F[0] = Q[1 + normal];
  F[1] = Q[1] * u;
  F[2] = Q[2] * u;
  F[3] = Q[3] * u;
  F[4] = (Q[4] + p) * u;
  F[1 + normal] += p;
}

double exagrype::euler::maxEigenvalue(
  const double* const        Q,
  [[maybe_unused]] const double* const ccz4Fields,
  const int                  normal,
  const double               gamma
) {
  const double invRho = 1.0 / Q[0];
  const double p      = pressure(Q, gamma);
  const double c      = std::sqrt(gamma * p * invRho);
  return std::abs(Q[1 + normal] * invRho) + c;
}

void exagrype::euler::source(
  [[maybe_unused]] const double* const        Q,
  [[maybe_unused]] const double* const        ccz4Fields,
  [[maybe_unused]] const double               gamma,
  double* const                               S
) {
  for (int i = 0; i < 5; i++) {
    S[i] = 0.0;
  }
}
