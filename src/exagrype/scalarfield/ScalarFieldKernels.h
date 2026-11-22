// This file is part of the ExaGRyPE project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "tarch/tarch.h"
#include "tarch/la/Vector.h"

namespace exagrype {
  namespace scalarfield {
    /**
     * Number of unknowns of the scalar field solver (exagrype.api.scalarfield.ScalarField.Unknowns).
     */
    constexpr int NumberOfUnknowns = 4;

    /**
     * Position of the lapse in the 58 entries of first-order CCZ4
     * (exagrype.api.FOCCZ4Solver.Unknowns). The solver of the scalar field
     * holds a copy of these entries as auxiliary variables.
     */
    constexpr int LapseIndex = 16;

    /**
     * Advection velocity of the scalar field along one coordinate axis
     *
     * The scalar field is advected along the x-axis with the square of the
     * lapse, which is the metric component g_xx of the gauge wave, i.e.
     * the scalar field sees the spacetime, but does not feed back into it
     * (fake coupling). Along the other axes, the velocity is zero.
     *
     * @param ccz4Fields The 58 entries of first-order CCZ4 in this volume.
     * @param normal Axis (0, 1 or 2).
     */
    #if defined(SharedOMP)
    #pragma omp declare simd
    #endif
    double advectionVelocity(const double* const ccz4Fields, const int normal);

    /**
     * Non-conservative product of the advection equation
     *
     * For @f$ \partial_t q + v \partial_x q = 0 @f$, the product of the
     * matrix with deltaQ is the velocity times deltaQ. Overwrites the four
     * entries of BTimesDeltaQ.
     *
     * @param ccz4Fields The 58 entries of first-order CCZ4 in this volume.
     * @param deltaQ Jump (Finite Volumes) or derivative times mesh width (Finite Differences) of the four unknowns.
     * @param normal Axis along which the product is evaluated.
     * @param BTimesDeltaQ Result, four entries.
     */
    #if defined(SharedOMP)
    #pragma omp declare simd
    #endif
    void nonconservativeProduct(
      const double* const ccz4Fields,
      const double* const deltaQ,
      const int           normal,
      double* const       BTimesDeltaQ
    );

    /**
     * Maximum eigenvalue, i.e. the modulus of the advection velocity along the axis
     */
    #if defined(SharedOMP)
    #pragma omp declare simd
    #endif
    double maxEigenvalue(const double* const ccz4Fields, const int normal);

    /**
     * Initial values: four sine waves along the x-axis with phase shifts
     *
     * Unknown m (m=0,...,3) is sin(2 pi x + m pi/4). The solution of the
     * advection equation on the gauge wave is known analytically (see
     * benchmarks/exagrype/gauge-wave-with-scalar-field). The routine also
     * sets the lapse of the auxiliary variables to one, so the first time
     * step size is well-defined before any data has been exchanged with CCZ4.
     *
     * @param Q The four unknowns.
     * @param ccz4Fields The 58 entries of first-order CCZ4 in this volume.
     * @param x Position of the volume.
     */
    void initialValues(
      double* const                               Q,
      double* const                               ccz4Fields,
      const tarch::la::Vector<Dimensions, double>& x
    );
  }
}
