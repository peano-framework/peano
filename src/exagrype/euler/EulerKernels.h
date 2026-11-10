// This file is part of the ExaGRyPE project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "tarch/tarch.h"
#include "tarch/multicore/multicore.h"

namespace exagrype {
  namespace euler {

    /**
     * Pressure of an ideal gas
     *
     * The five conserved quantities are ordered density, the three momentum
     * components and the total energy (exagrype.api.euler.Euler.Unknowns),
     * and the pressure is
     * @f$ p = (\gamma-1) ( E - |j|^2 / (2 \rho) ) @f$.
     *
     * @param Q The five conserved quantities.
     * @param gamma Adiabatic index.
     */
    #if defined(SharedOMP)
    #pragma omp declare simd
    #endif
    double pressure(const double* const Q, const double gamma);

    /**
     * Flux of the Euler equations along one coordinate axis
     *
     * Overwrites all five entries of F. The caller may modify the result
     * afterwards, e.g. scale it by the lapse of a spacetime.
     *
     * @param Q The five conserved quantities.
     * @param ccz4Fields The 59 entries of first-order CCZ4 in this volume
     *   (exagrype.api.FOCCZ4Solver.Unknowns), i.e. the auxiliary variables
     *   through which the spacetime enters. The Euler equations do not use
     *   them yet; the argument is the hook through which the coupling does.
     * @param normal Axis along which the flux is evaluated (0, 1 or 2).
     * @param gamma Adiabatic index.
     * @param F Flux, five entries.
     */
    #if defined(SharedOMP)
    #pragma omp declare simd
    #endif
    void flux(
      const double* const        Q,
      const double* const        ccz4Fields,
      const int                  normal,
      const double               gamma,
      double* const              F
    );

    /**
     * Maximum eigenvalue of the flux Jacobian along one coordinate axis
     *
     * This is the modulus of the fluid velocity along the axis plus the speed
     * of sound.
     *
     * @param Q The five conserved quantities.
     * @param ccz4Fields The 59 entries of first-order CCZ4 in this volume
     *   (exagrype.api.FOCCZ4Solver.Unknowns), i.e. the auxiliary variables
     *   through which the spacetime enters. The Euler equations do not use
     *   them yet; the argument is the hook through which the coupling does.
     * @param normal Axis along which the eigenvalue is evaluated (0, 1 or 2).
     * @param gamma Adiabatic index.
     * @return Maximum eigenvalue.
     */
    #if defined(SharedOMP)
    #pragma omp declare simd
    #endif
    double maxEigenvalue(
      const double* const        Q,
      const double* const        ccz4Fields,
      const int                  normal,
      const double               gamma
    );

    /**
     * Volumetric source of the Euler equations
     *
     * The source is zero at the moment. It accepts the solution and the
     * auxiliary variables already, so that a coupling can fill it without
     * changing the interface.
     *
     * @param Q The five conserved quantities.
     * @param ccz4Fields The 59 entries of first-order CCZ4 in this volume
     *   (exagrype.api.FOCCZ4Solver.Unknowns), i.e. the auxiliary variables
     *   through which the spacetime enters. The Euler equations do not use
     *   them yet; the argument is the hook through which the coupling does.
     * @param gamma Adiabatic index.
     * @param S Source, five entries (output).
     */
    #if defined(SharedOMP)
    #pragma omp declare simd
    #endif
    void source(
      const double* const        Q,
      const double* const        ccz4Fields,
      const double               gamma,
      double* const              S
    );

  } // namespace euler
} // namespace exagrype
