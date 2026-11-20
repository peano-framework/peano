// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once


namespace exahype2 {
  namespace fd {
    /**
     * Add advection terms discretised through central differences to a non-conservative product
     *
     * Many PDEs carry advection terms
     *
     * @f$ \partial _t Q_q = \ldots + \sum _d v_d \partial _d Q_q @f$
     *
     * with a velocity v which is part of the solution itself, such as the
     * shift in the Einstein equations. These terms are linear in the
     * derivative of the advected entry, i.e. they are a non-conservative
     * product @f$ B(Q) \cdot \nabla Q @f$ with @f$ B = -v_d @f$ on the
     * diagonal of the advected entries. The routine adds them to the
     * non-conservative product along one normal, which the right-hand side
     * kernels of the Finite Differences (exahype2.api.rkfd4.FD4Kernels) hand
     * the derivative along this normal, computed with their central
     * differences:
     *
     * @f$ (B \cdot \Delta Q)_q \mathrel{-}= v_{normal} \Delta Q_q @f$
     *
     * for each selected quantity q. The velocity is taken from the point
     * itself.
     *
     * This is the counterpart of addLopsidedAdvection(), which adds the same
     * terms with lopsided stencils to the right-hand side of a whole patch.
     * A solver uses either of the two for a quantity, never both, and the
     * PDE terms of the solver must not contain these advection terms
     * themselves. Both routines take the velocity and the quantities in the
     * same form, so a solver can hand the same lists to either of them
     * (exagrype.api.CCZ4SolverBase is an example, see
     * src/exagrype/ccz4/adrs/0003-the-solver-picks-who-adds-the-advection.md).
     *
     * The routine handles one normal per call, as the right-hand side
     * kernels call the non-conservative product once per normal. If the
     * non-conservative product receives all normals at once one day, it has
     * to sum over them.
     *
     * @param Q The unknowns of the point.
     * @param deltaQ The derivative of the unknowns along normal, times the
     *   mesh width, as the non-conservative product receives it.
     * @param BTimesDeltaQ The non-conservative product along normal, which
     *   the routine adds the advection terms to.
     * @param normal Axis along which deltaQ is given.
     * @param velocity Indices of the Dimensions entries of Q which hold the
     *   velocity components.
     * @param quantities Indices of the advected entries of Q.
     * @param numberOfQuantities Number of entries of quantities.
     */
    void addCentralAdvection(
      const double* __restrict__ Q,
      const double* __restrict__ deltaQ,
      double* __restrict__       BTimesDeltaQ,
      int                        normal,
      const int*                 velocity,
      const int*                 quantities,
      int                        numberOfQuantities
    );
  }
}
