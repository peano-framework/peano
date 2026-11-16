// This file is part of the ExaGRyPE project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "tarch/tests/TestCase.h"


namespace exagrype {
  namespace ccz4 {
    namespace tests {
      class GammaDriverTest;
    } // namespace tests
  } // namespace ccz4
} // namespace exagrype


/**
 * Checks of the two variants of the Gamma driver in the CCZ4 kernels
 *
 * CCZ4b picks the shifting shift (1) or the non-shifting shift (0), see
 * exagrype::ccz4::ncpFirstOrder(). The kernels of the two variants must
 * differ in exactly those terms in which the equations differ. The test
 * evaluates the kernels for both variants on a perturbed flat spacetime
 * with a shift and compares the difference of the results with these
 * terms:
 *
 * - The source of the shifting shift holds the advection term
 *   @f$ \beta^k B_k^i @f$ of the shift and its derivative
 *   @f$ B_j^k B_k^i @f$ in the equation of @f$ B_j^i @f$. The second-order
 *   source holds neither, as the numerical scheme advects the shift.
 * - The non-conservative product of the non-shifting shift holds the
 *   advection of Ghat in the equation of b, as the equation of b needs
 *   the complete time derivative of Ghat, whereas the kernels leave the
 *   advection of Ghat itself to the numerical scheme.
 * - All other entries are the same for both variants.
 *
 * It also checks that a shift which does not evolve (CCZ4s=0) switches
 * off the equations of the shift, of b and of the derivatives of the shift
 * for both variants.
 *
 * The rationale behind the terms is written down in
 * src/exagrype/ccz4/adrs/0002-ccz4b-selects-the-shifting-or-the-non-shifting-shift.md.
 */
class exagrype::ccz4::tests::GammaDriverTest: public tarch::tests::TestCase {
private:
  void testSourceOfBothVariants();
  void testNonconservativeProductOfBothVariants();
  void testFrozenShift();

public:
  GammaDriverTest();

  virtual void run() override;
};
