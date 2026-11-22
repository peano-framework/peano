#include "CentralAdvection.h"


void exahype2::fd::addCentralAdvection(
  const double* __restrict__ Q,
  const double* __restrict__ deltaQ,
  double* __restrict__       BTimesDeltaQ,
  int                        normal,
  const int*                 velocity,
  const int*                 quantities,
  int                        numberOfQuantities
) {
  const double v = Q[velocity[normal]];
  for (int i = 0; i < numberOfQuantities; i++) {
    BTimesDeltaQ[quantities[i]] -= v * deltaQ[quantities[i]];
  }
}
