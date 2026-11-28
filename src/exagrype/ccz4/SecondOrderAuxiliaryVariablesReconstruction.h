// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once


#include "tarch/la/Vector.h"


namespace exagrype {
  namespace ccz4 {
      /**
       * Reconstruct first derivatives with fourth-order central differences
       *
       * This is the entry point of exagrype.api.FOSOReconstructionSolver for
       * the Finite Differences. The second-order solver evolves the first 25
       * of the 58 CCZ4 entries; the remaining 33 entries are first derivatives
       * of the lapse, shift, metric and phi. The routine computes these
       * derivatives of one cell from the evolved unknowns with their halo.
       *
       * Which unknown a derivative differentiates, along which axis and with
       * which factor, is not known to this routine: The caller hands over a
       * table with one row per derivative d, i.e.
       *
       * @f$ \mathrm{firstDerivatives}_d = \mathrm{factor}_d \,
       *     \partial _{\mathrm{axis}_d} Q_{\mathrm{differentiatedVariable}_d} @f$.
       *
       * exagrype.api.FOSOCCZ4Solver.FOSOCCZ4Solver.auxiliary_derivatives()
       * is the only definition of this table, which the generated call passes
       * on
       * (src/exagrype/ccz4/adrs/0005-one-table-defines-the-auxiliary-derivatives.md).
       *
       * The routine applies the stencil
       *
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       * [1 -8 0 8 -1]
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       *
       * scaled with @f$ \frac{1}{12h} @f$, which reaches two layers around
       * every grid cell. It computes the derivatives of the cell's own grid
       * cells only and needs a halo of two. The CCZ4 solver obtains the
       * derivatives of its halo from the neighbouring cells via the faces, as
       * for the Finite Volumes
       * (reconstructFirstDerivativesCentralDifferences()).
       *
       * ## Why not one layer outside of the cell
       *
       * @image html recomputeAuxiliaryVariablesFD4_4thOrder.png
       *
       * An earlier version also evaluated the derivatives one layer outside of
       * the cell (brownish layer to the right), with the argument that the
       * fluxes and non-conservative products on the cell's faces need the
       * derivatives on both sides. This required a halo of three. The Finite
       * Differences of exahype2.api.rkfd4 evaluate no terms on faces, though,
       * and the derivatives of the halo come from the neighbouring cell
       * (src/exagrype/ccz4/adrs/0004-the-foso-reconstruction-takes-the-halo-of-the-scheme.md).
       * Furthermore, Peano holds no diagonal neighbours: What we really get is
       * not the illustration above to the left but the one in the centre. The
       * diagonal blocks hold garbage, so in the outer layer only the
       * derivatives along its normal would be correct.
       *
       * @param primaryVariablesWithHalo The evolved unknowns of the cell with
       *   its halo, i.e. (numberOfGridCellsPerPatchPerAxis+2*haloSize)^3 grid
       *   cells with numberOfPrimaryVariables entries each, the first axis
       *   running fastest.
       * @param patchSize Extent of the cell (without halo).
       * @param numberOfGridCellsPerPatchPerAxis Grid cells per axis within
       *   the cell.
       * @param haloSize Has to be at least two.
       * @param numberOfPrimaryVariables Entries per grid cell of
       *   primaryVariablesWithHalo, 25 for CCZ4.
       * @param numberOfDerivatives Entries per grid cell of firstDerivatives,
       *   i.e. the number of rows of the table, 33 for CCZ4.
       * @param differentiatedVariable Per derivative, the index of the
       *   differentiated entry of primaryVariablesWithHalo.
       * @param axis Per derivative, the axis of the derivative.
       * @param factor Per derivative, the factor of the derivative.
       * @param firstDerivatives Output, numberOfDerivatives entries per grid
       *   cell of the cell (without halo), the first axis running fastest.
       */
      void reconstructFirstDerivativesFD4(
        const double* __restrict__                  primaryVariablesWithHalo,
        const tarch::la::Vector<Dimensions,double>& patchSize,
        int                                         numberOfGridCellsPerPatchPerAxis,
        int                                         haloSize,
        int                                         numberOfPrimaryVariables,
        int                                         numberOfDerivatives,
        const int* __restrict__                     differentiatedVariable,
        const int* __restrict__                     axis,
        const double* __restrict__                  factor,
        double* __restrict__                        firstDerivatives
      );

      /**
       * Reconstruct first derivatives with second-order central differences
       *
       * Same contract and table as reconstructFirstDerivativesFD4(), but the
       * stencil is @f$ [-1\ 0\ 1]/(2h) @f$. It reaches one layer out only, so a
       * halo of one grid cell suffices. This is what the Finite Volumes
       * (overlap one) of exagrype.api.FOSOCCZ4Solver use: the derivatives of
       * the halo layer are not computed here but exchanged via the faces, as
       * they are computed by the neighbouring cell.
       *
       * @param primaryVariablesWithHalo See reconstructFirstDerivativesFD4().
       * @param patchSize See reconstructFirstDerivativesFD4().
       * @param numberOfGridCellsPerPatchPerAxis See reconstructFirstDerivativesFD4().
       * @param haloSize Has to be at least one.
       * @param numberOfPrimaryVariables See reconstructFirstDerivativesFD4().
       * @param numberOfDerivatives See reconstructFirstDerivativesFD4().
       * @param differentiatedVariable See reconstructFirstDerivativesFD4().
       * @param axis See reconstructFirstDerivativesFD4().
       * @param factor See reconstructFirstDerivativesFD4().
       * @param firstDerivatives See reconstructFirstDerivativesFD4().
       */
      void reconstructFirstDerivativesCentralDifferences(
        const double* __restrict__                  primaryVariablesWithHalo,
        const tarch::la::Vector<Dimensions,double>& patchSize,
        int                                         numberOfGridCellsPerPatchPerAxis,
        int                                         haloSize,
        int                                         numberOfPrimaryVariables,
        int                                         numberOfDerivatives,
        const int* __restrict__                     differentiatedVariable,
        const int* __restrict__                     axis,
        const double* __restrict__                  factor,
        double* __restrict__                        firstDerivatives
      );
  }
}
