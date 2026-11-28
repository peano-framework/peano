#include "exagrype/scalarfield/ScalarFieldKernels.h"

#include <cmath>


double exagrype::scalarfield::advectionVelocity(const double* const ccz4Fields, const int normal) {
  const double lapse = ccz4Fields[LapseIndex];
  return normal == 0 ? lapse * lapse : 0.0;
}


void exagrype::scalarfield::nonconservativeProduct(
  const double* const ccz4Fields,
  const double* const deltaQ,
  const int           normal,
  double* const       BTimesDeltaQ
) {
  const double velocity = advectionVelocity(ccz4Fields, normal);
  for (int i = 0; i < NumberOfUnknowns; i++) {
    BTimesDeltaQ[i] = velocity * deltaQ[i];
  }
}


double exagrype::scalarfield::maxEigenvalue(const double* const ccz4Fields, const int normal) {
  return std::abs(advectionVelocity(ccz4Fields, normal));
}


void exagrype::scalarfield::initialValues(
  double* const                               Q,
  const tarch::la::Vector<Dimensions, double>& x
) {
  for (int i = 0; i < NumberOfUnknowns; i++) {
    Q[i] = std::sin(2.0 * M_PI * x(0) + 0.25 * M_PI * i);
  }
}


void exagrype::scalarfield::initialAuxiliaryValues(
  double* const ccz4Fields
) {
  ccz4Fields[LapseIndex] = 1.0;
}
