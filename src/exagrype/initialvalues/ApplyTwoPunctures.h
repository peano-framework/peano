// This file is part of the ExaGRyPE project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "config.h"
#include "peano4/utils/Globals.h"
#include "tarch/la/Vector.h"

#if defined(IncludeTwoPunctures) and not defined(UseGSL)
#error \
  "TwoPunctures initial data requires GSL: configure Peano with --with-gsl (or -DWITH_GSL=ON)"
#endif

#ifdef UseGSL
#include "exagrype/twopunctures/TP_bindding.h"

namespace exagrype {
  namespace initialvalues {
    /**
     * Initial conditions for a (binary) black hole from TwoPunctures
     *
     * TwoPunctures (namespaces TP and TP_bindding in src/exagrype/twopunctures)
     * is taken from the EinsteinToolkit and needs GSL. Therefore, it is only
     * part of the ExaGRyPE library if Peano is configured with --with-gsl
     * (-DWITH_GSL=ON for CMake), which defines UseGSL.
     *
     * Applications that use this routine typically guard their TwoPunctures
     * code with IncludeTwoPunctures, so they can switch between scenarios
     * per build. If an application sets IncludeTwoPunctures but Peano lacks
     * GSL, this header aborts the translation with a meaningful error.
     */
    void ApplyTwoPunctures(
      double* __restrict__ Q,
      const tarch::la::Vector<Dimensions, double>& X,
      double                                       t,
      TP::TwoPunctures*                            tp,
      bool                                         low_res
    );

    /**
     * Single Schwarzschild black hole at the origin
     *
     * Fills the 58 unknowns of the first-order CCZ4 formulation at point x
     * with the TwoPunctures initial data of one non-spinning black hole
     * at rest. The puncture sits at the coordinate origin.
     *
     * TwoPunctures solves an elliptic problem before it can interpolate
     * anything. We do this once, upon the very first call, and keep the
     * solution in a function-local static object. C++ guarantees that
     * this initialisation is thread-safe, so you can call the routine from
     * concurrent tasks right from the start. All subsequent calls only
     * interpolate, i.e. read the TwoPunctures solution.
     *
     * The setup mirrors the legacy benchmark
     * @ref benchmarks_exahype2_ccz4_single_black_hole: TwoPunctures always
     * models two punctures at +/- par_b along the x-axis. We place them at
     * 0 and -2 by shifting the centre by -1, and give the second one zero
     * mass.
     *
     * We always use TwoPunctures' high-resolution interpolation, as we
     * evaluate the routine once per degree of freedom only, i.e. at
     * startup.
     *
     * @param Q    Output: the 58 CCZ4 unknowns at x.
     * @param x    Position. It should not coincide with the puncture, i.e.
     *             with the origin, where the data is singular.
     * @param t    Time stamp. Ignored, as the data is static initial data.
     * @param mass ADM mass of the black hole. All calls have to use the
     *             same mass, as the solution is computed only once.
     */
    void singleBlackHole(
      double* __restrict__ Q,
      const tarch::la::Vector<Dimensions, double>& x,
      double                                       t,
      double                                       mass
    );
  } // namespace initialvalues
} // namespace exagrype
#endif
