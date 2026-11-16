/**
 *
 * This code is taken from the original ExaHyPE project written by
 * colleagues from the University of Trento.
 *
 */

#pragma once

#include "tarch/la/Vector.h"

#include "peano4/utils/Globals.h"


namespace exagrype {
  /**
   * Analytical initial conditions for the CCZ4 formulation
   *
   * All routines fill the 58 unknowns of the first-order CCZ4 formulation
   * at point x and time t.
   */
  namespace initialvalues {
      void gaugeWave(
        double * __restrict__ Q, // Q[58+0],
        const tarch::la::Vector<Dimensions,double>&  x,
        double t
      );
      void diagonal_gaugeWave(
        double * __restrict__ Q, // Q[58+0],
        const tarch::la::Vector<Dimensions,double>&  x,
        double t
      );
      void linearWave(
        double * __restrict__ Q, // Q[58+0],
        const tarch::la::Vector<Dimensions,double>&  X,
        double t
      );
      void flat(
        double * __restrict__ Q, // Q[58+0],
        const tarch::la::Vector<Dimensions,double>&  X,
        double t
      );
  }
}
