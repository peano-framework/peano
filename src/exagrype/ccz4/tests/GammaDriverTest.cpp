// This file is part of the ExaGRyPE project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "GammaDriverTest.h"

#include <cmath>

#include "exagrype/ccz4/Kernels.h"
#include "exagrype/initialvalues/InitialValues.h"
#include "tarch/tests/TestMacros.h"


namespace {
  constexpr int Unknowns = 58;

  constexpr int Ghat            = 13;
  constexpr int Shift           = 17;
  constexpr int VectorB         = 20;
  constexpr int ShiftDerivative = 28;

  constexpr int    CCZ4LapseType = 1;
  constexpr double CCZ4c         = 1.0;
  constexpr double CCZ4e         = 1.0;
  constexpr double CCZ4f         = 0.75;
  constexpr double CCZ4mu        = 0.2;
  constexpr double CCZ4itau      = 1.0;
  constexpr double CCZ4eta       = 1.0;
  constexpr double CCZ4k1        = 0.1;
  constexpr double CCZ4k2        = 0.0;
  constexpr double CCZ4k3        = 0.5;

  constexpr double Tolerance = 1e-12;

  /**
   * Flat spacetime plus a reproducible perturbation of all entries, so the
   * shift, b, Ghat and all derivatives differ from zero.
   */
  void initPerturbedSpacetime(double* Q, double* gradQ) {
    exagrype::initialvalues::flat(Q, tarch::la::Vector<Dimensions, double>(0.0), 0.0);
    unsigned int state = 20261116u;
    auto         noise = [&state]() {
      state = 1664525u * state + 1013904223u;
      return 0.4 * (state / 4294967296.0) - 0.2;
    };
    for (int i = 0; i < Unknowns; i++) {
      Q[i]     += noise();
      gradQ[i]  = noise();
    }
  }

  /** @f$ B_j^i = \partial_j \beta^i @f$ */
  double shiftDerivative(const double* Q, int j, int i) { return Q[ShiftDerivative + 3 * j + i]; }
} // namespace


exagrype::ccz4::tests::GammaDriverTest::GammaDriverTest():
  TestCase("exagrype::ccz4::tests::GammaDriverTest") {}


void exagrype::ccz4::tests::GammaDriverTest::run() {
  testMethod(testSourceOfBothVariants);
  testMethod(testNonconservativeProductOfBothVariants);
  testMethod(testFrozenShift);
}


void exagrype::ccz4::tests::GammaDriverTest::testSourceOfBothVariants() {
  constexpr int CCZ4s = 1;
  double        Q[Unknowns], gradQ[Unknowns];
  initPerturbedSpacetime(Q, gradQ);

  double shifting[Unknowns], nonShifting[Unknowns];
  exagrype::ccz4::sourceFirstOrder(shifting, shifting + 25, Q, Q + 25, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, 1, CCZ4s, CCZ4itau, CCZ4eta, CCZ4k1, CCZ4k2, CCZ4k3);
  exagrype::ccz4::sourceFirstOrder(nonShifting, nonShifting + 25, Q, Q + 25, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, 0, CCZ4s, CCZ4itau, CCZ4eta, CCZ4k1, CCZ4k2, CCZ4k3);

  double expectedDifference[Unknowns] = {0.0};
  for (int i = 0; i < 3; i++) {
    for (int k = 0; k < 3; k++) {
      expectedDifference[Shift + i] += Q[Shift + k] * shiftDerivative(Q, k, i);
    }
    for (int j = 0; j < 3; j++)
    for (int k = 0; k < 3; k++) {
      expectedDifference[ShiftDerivative + 3 * j + i] += shiftDerivative(Q, j, k) * shiftDerivative(Q, k, i);
    }
  }
  for (int i = 0; i < Unknowns; i++) {
    validateNumericalEqualsWithEpsWithParams1(shifting[i] - nonShifting[i], expectedDifference[i], Tolerance, i);
  }

  double shiftingSecondOrder[25], nonShiftingSecondOrder[25];
  exagrype::ccz4::sourceSecondOrder(shiftingSecondOrder, Q, Q + 25, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, 1, CCZ4s, CCZ4itau, CCZ4eta, CCZ4k1, CCZ4k2, CCZ4k3);
  exagrype::ccz4::sourceSecondOrder(nonShiftingSecondOrder, Q, Q + 25, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, 0, CCZ4s, CCZ4itau, CCZ4eta, CCZ4k1, CCZ4k2, CCZ4k3);
  for (int i = 0; i < 25; i++) {
    validateNumericalEqualsWithEpsWithParams1(shiftingSecondOrder[i], nonShiftingSecondOrder[i], Tolerance, i);
  }
}


void exagrype::ccz4::tests::GammaDriverTest::testNonconservativeProductOfBothVariants() {
  constexpr int CCZ4s = 1;
  double        Q[Unknowns], gradQ[Unknowns];
  initPerturbedSpacetime(Q, gradQ);

  for (int normal = 0; normal < 3; normal++) {
    // The kernels return B(Q) grad(Q), i.e. the negative right-hand side
    double expectedDifference[Unknowns] = {0.0};
    for (int i = 0; i < 3; i++) {
      expectedDifference[VectorB + i] = -Q[Shift + normal] * gradQ[Ghat + i];
    }

    double shifting[Unknowns], nonShifting[Unknowns];
    exagrype::ccz4::ncpFirstOrder(shifting, shifting + 25, Q, Q + 25, gradQ, gradQ + 25, normal, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, 1, CCZ4s, CCZ4mu);
    exagrype::ccz4::ncpFirstOrder(nonShifting, nonShifting + 25, Q, Q + 25, gradQ, gradQ + 25, normal, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, 0, CCZ4s, CCZ4mu);
    for (int i = 0; i < Unknowns; i++) {
      validateNumericalEqualsWithEpsWithParams2(nonShifting[i] - shifting[i], expectedDifference[i], Tolerance, normal, i);
    }

    exagrype::ccz4::ncpSecondOrder(shifting, Q, Q + 25, gradQ, gradQ + 25, normal, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, 1, CCZ4s, CCZ4mu);
    exagrype::ccz4::ncpSecondOrder(nonShifting, Q, Q + 25, gradQ, gradQ + 25, normal, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, 0, CCZ4s, CCZ4mu);
    for (int i = 0; i < 25; i++) {
      validateNumericalEqualsWithEpsWithParams2(nonShifting[i] - shifting[i], expectedDifference[i], Tolerance, normal, i);
    }
  }
}


void exagrype::ccz4::tests::GammaDriverTest::testFrozenShift() {
  constexpr int CCZ4s = 0;
  double        Q[Unknowns], gradQ[Unknowns];
  initPerturbedSpacetime(Q, gradQ);

  for (int CCZ4b = 0; CCZ4b < 2; CCZ4b++) {
    double S[Unknowns];
    exagrype::ccz4::sourceFirstOrder(S, S + 25, Q, Q + 25, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, CCZ4b, CCZ4s, CCZ4itau, CCZ4eta, CCZ4k1, CCZ4k2, CCZ4k3);
    for (int i = 0; i < 3; i++) {
      validateNumericalEqualsWithParams2(S[Shift + i], 0.0, CCZ4b, i);
      validateNumericalEqualsWithParams2(S[VectorB + i], 0.0, CCZ4b, i);
    }
    for (int i = 0; i < 9; i++) {
      validateNumericalEqualsWithParams2(S[ShiftDerivative + i], 0.0, CCZ4b, i);
    }

    for (int normal = 0; normal < 3; normal++) {
      double BgradQ[Unknowns];
      exagrype::ccz4::ncpFirstOrder(BgradQ, BgradQ + 25, Q, Q + 25, gradQ, gradQ + 25, normal, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, CCZ4b, CCZ4s, CCZ4mu);
      for (int i = 0; i < 3; i++) {
        validateNumericalEqualsWithParams3(BgradQ[Shift + i], 0.0, CCZ4b, normal, i);
        validateNumericalEqualsWithParams3(BgradQ[VectorB + i], 0.0, CCZ4b, normal, i);
      }
      for (int i = 0; i < 9; i++) {
        validateNumericalEqualsWithParams3(BgradQ[ShiftDerivative + i], 0.0, CCZ4b, normal, i);
      }
    }
  }
}
