// This file is part of the ExaGRyPE project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "tarch/tests/TestCase.h"


namespace exagrype {
  namespace ccz4 {
    namespace tests {
      class FlatSpacetimeTest;
    } // namespace tests
  } // namespace ccz4
} // namespace exagrype


/**
 * Sanity checks of the CCZ4 kernels against flat (Minkowski) spacetime
 *
 * Flat spacetime with a unit lapse and zero shift is a stationary solution
 * of the CCZ4 equations. If we take exagrype::initialvalues::flat() and set
 * all gradients to zero, both the non-conservative product and the source
 * therefore have to vanish. The test furthermore checks the maximum
 * eigenvalue estimate for flat spacetime, and that the gauge wave initial
 * data yields the analytically known lapse.
 *
 * We also test two helpers of the second-order formulation here, as they do
 * not warrant a test case of their own: that
 * reconstructFirstDerivativesFD4() differentiates cubic polynomials exactly
 * (for a halo of two, three and four), that
 * reconstructFirstDerivativesCentralDifferences() does the same for quadratic
 * polynomials (halo of one and two), both for each row of a table of
 * variables, axes and factors, and that
 * enforceCCZ4constraintsOfEvolvedVariables() does the same to the first 25
 * entries as enforceCCZ4constraints().
 */
class exagrype::ccz4::tests::FlatSpacetimeTest: public tarch::tests::TestCase {
private:
  void testNonconservativeProductVanishes();
  void testSourceVanishes();
  void testSecondaryEntriesInSeparateMemory();
  void testMaxEigenvalueOfFlatSpacetime();
  void testGaugeWaveLapse();
  void testFirstDerivativeReconstructionIsExactForCubics();
  void testCentralDifferenceReconstructionIsExactForQuadratics();
  void testConstraintEnforcementOfEvolvedVariablesMatchesFullEnforcement();

public:
  FlatSpacetimeTest();

  virtual void run() override;
};
