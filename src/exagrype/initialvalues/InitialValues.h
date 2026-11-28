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
   * Each solution comes in two variants, which evaluate it at point x and
   * time t. The one without a suffix fills the 58 unknowns of the
   * first-order CCZ4 formulation. The one with the suffix EvolvedVariables
   * fills the first 25 of them only, i.e. the unknowns which the
   * second-order formulation (exagrype.api.FOSOCCZ4Solver) evolves. The
   * other 33 entries are first derivatives, which that solver reconstructs.
   * The variant without suffix calls the other one and adds the analytical
   * derivatives, i.e. the formulae of the first 25 entries exist only once.
   */
  namespace initialvalues {
      void gaugeWave(
        double * __restrict__ Q, // Q[58+0]
        const tarch::la::Vector<Dimensions,double>&  x,
        double t
      );
      void gaugeWaveEvolvedVariables(
        double * __restrict__ Q, // Q[25]
        const tarch::la::Vector<Dimensions,double>&  x,
        double t
      );
      void diagonal_gaugeWave(
        double * __restrict__ Q, // Q[58+0]
        const tarch::la::Vector<Dimensions,double>&  x,
        double t
      );
      void diagonal_gaugeWaveEvolvedVariables(
        double * __restrict__ Q, // Q[25]
        const tarch::la::Vector<Dimensions,double>&  x,
        double t
      );
      void linearWave(
        double * __restrict__ Q, // Q[58+0]
        const tarch::la::Vector<Dimensions,double>&  X,
        double t
      );
      void linearWaveEvolvedVariables(
        double * __restrict__ Q, // Q[25]
        const tarch::la::Vector<Dimensions,double>&  X,
        double t
      );
      void flat(
        double * __restrict__ Q, // Q[58+0]
        const tarch::la::Vector<Dimensions,double>&  x,
        double t
      );
      void flatEvolvedVariables(
        double * __restrict__ Q, // Q[25]
        const tarch::la::Vector<Dimensions,double>&  x,
        double t
      );
  }
}
