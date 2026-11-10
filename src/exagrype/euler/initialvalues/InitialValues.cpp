#include "exagrype/euler/initialvalues/InitialValues.h"

#include <cmath>

void exagrype::euler::initialvalues::smoothWave(
  double* __restrict__                         Q,
  const tarch::la::Vector<Dimensions, double>& x,
  [[maybe_unused]] double                      t,
  double                                       amplitude,
  double                                       velocity,
  double                                       pressure,
  double                                       gamma
) {
  const double rho = 1.0 + amplitude * std::sin(2.0 * M_PI * x(0));
  Q[0]             = rho;
  Q[1]             = rho * velocity;
  Q[2]             = 0.0;
  Q[3]             = 0.0;
  Q[4]             = pressure / (gamma - 1.0) + 0.5 * rho * velocity * velocity;
}
