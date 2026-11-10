// This file is part of the ExaGRyPE project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "tarch/la/Vector.h"

#include "peano4/utils/Globals.h"

namespace exagrype {
  namespace euler {
    /**
     * Analytical initial conditions for the Euler equations
     *
     * All routines fill the first five entries of Q (density, momentum,
     * total energy) at point x and time t. Entries behind them (auxiliary
     * variables) are left untouched.
     */
    namespace initialvalues {
      /**
       * Density wave on constant pressure
       *
       * The density is 1 + amplitude * sin(2 pi x(0)) and the fluid moves
       * along the x-axis with constant velocity. Pressure and velocity are
       * constant, i.e. the wave is an exact solution of the Euler equations
       * which is advected. The data are independent of t.
       *
       * @param Q Five conserved quantities (output).
       * @param x Position.
       * @param t Time stamp (ignored).
       * @param amplitude Amplitude of the density wave, smaller than one in modulus.
       * @param velocity Speed along the x-axis.
       * @param pressure Constant pressure.
       * @param gamma Adiabatic index.
       */
      void smoothWave(
        double* __restrict__                         Q,
        const tarch::la::Vector<Dimensions, double>& x,
        double                                       t,
        double                                       amplitude,
        double                                       velocity,
        double                                       pressure,
        double                                       gamma
      );
    } // namespace initialvalues
  }   // namespace euler
} // namespace exagrype
