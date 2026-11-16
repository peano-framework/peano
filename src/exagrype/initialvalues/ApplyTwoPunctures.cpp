// This file is part of the ExaGRyPE project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "exagrype/initialvalues/ApplyTwoPunctures.h"

#include "tarch/Assertions.h"

#ifdef UseGSL
void exagrype::initialvalues::ApplyTwoPunctures(
  double* __restrict__ Q,
  const tarch::la::Vector<Dimensions, double>& X,
  double                                       t,
  TP::TwoPunctures*                            tp,
  bool                                         low_res
) {
  constexpr int nVars   = 58;
  const double  coor[3] = {X[0], X[1], X[2]};
  double        LgradQ[3 * nVars];
  tarch::memset(Q, .0, nVars * sizeof(double));
  tp->Interpolate(coor, Q, low_res);
  // calculate corresponding soccz4 quantities
  TP_bindding::SOCCZ4Cal(Q);
  // calculate gradient for auxiliary variables
  TP_bindding::GradientCal(coor, Q, LgradQ, nVars, tp, low_res);
  // calculate the auxiliary variables
  TP_bindding::AuxiliaryCal(Q, LgradQ, nVars);
}

namespace {
  TP::TwoPunctures* createSingleBlackHole(double mass) {
    TP::TwoPunctures* result = new TP::TwoPunctures();

    result->par_b            = 1.0;
    result->center_offset[0] = -1.0;
    result->center_offset[1] = 0.0;
    result->center_offset[2] = 0.0;
    result->target_M_plus    = mass;
    result->target_M_minus   = 0.0;
    for (int d = 0; d < 3; d++) {
      result->par_P_plus[d]  = 0.0;
      result->par_S_plus[d]  = 0.0;
      result->par_P_minus[d] = 0.0;
      result->par_S_minus[d] = 0.0;
    }
    result->grid_setup_method = "evaluation";
    result->TP_epsilon        = 1e-6;

    result->PrintParameters();
    result->Run();
    return result;
  }
} // namespace

void exagrype::initialvalues::singleBlackHole(
  double* __restrict__ Q,
  const tarch::la::Vector<Dimensions, double>& x,
  double                                       t,
  double                                       mass
) {
  static TP::TwoPunctures* twoPunctures = createSingleBlackHole(mass);
  assertionNumericalEquals2(twoPunctures->target_M_plus, mass, x, t);
  ApplyTwoPunctures(Q, x, t, twoPunctures, false);
}
#endif
