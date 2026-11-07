// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include <vector>

#include "DGUtils.h"
#include "Functors.h"

#include "tarch/la/Vector.h"
#include "tarch/multicore/multicore.h"

#include "peano4/grid/GridControlEvent.h"
#include "peano4/utils/Loop.h"
#include "peano4/utils/Globals.h"

namespace exahype2 {
  namespace dg {
    /**
     * ## Interpolation matrix
     *
     * Core ingredient to this routine is the interpolation matrix.
     * I describes how to interpolate a 1d polynomial over the unit
     * interval into the three subcell polynomials. Consequently,
     * the matrix points to three matrices. One for the left, one
     * for the middle, one for the right subcell.
     *
     * Each matrix accepts the order+1 values at the quadrature points
     * as input, i.e. each matrix has order+1 columns. Each matrix
     * returns the the output for one quadrature point at the chosen
     * subcell. So each matrix has order+1 rows.
     *
     * In the example above, we work with polynomials of order three.
     * We hence have four quadrature points.
     *
     * @image html src/exahype2/dg/interpolation.png
     *
     * In the example above, the weights in c1, c2, c3 and c4 span a
     * polynomial (blue). InterpolationMatrix1d points to
     * @f$ 3 \cdot 4^2 @f$ entries. If we want to know the interpolated
     * value for f31, we have to pick the third matrix (matrix index 2
     * as we start counting at zero), then take the first row of this
     * matrix, and multiply its four entries with c1, c2, c3, c4.
     *
     * All entries in all faces are enumerated lexicographically, i.e.
     * from left to right, bottom-up, and always first along the x-axis,
     * then y, then z.
     *
     *
     * ## Implementation
     *
     * I first construct a
     * face enumerator. So I don't really have to care about the serialisation
     * of the dofs anymore. This is all done by the enumerator.
     *
     * Next, I compute the total number of unknowns (doubles) in the fine grid
     * face and set them all to zero. From now on, I can accumulate the result.
     * There is a factor of 2 in the calculation of the total number of unknowns.
     * It results from the fact that I have a left and a right projection.
     *
     * The main part of the routine is a nested loop, where I run over all
     * combinations of input and output dofs. Again, these are three loops
     * nested, as I always have left and right.
     *
     * Per combination, I have to compute the d-1-dimensional tensor product
     * over the interpolation matrix. For 2d, this degenerates to a simple
     * lookup.
     *
     *
     * ## Evaluation of interpolation/restriction matrix
     *
     * We run over the d-1-dimensional dofs of both coarse and fine grid face.
     * For a 2d simulation, we hence run over a one-dimensional thing. The
     * interpolation and restriction matrices in ExaHyPE are written down as
     * 1d operators, as any dimension then results from a tensor-product
     * approach.
     *
     * So in 2d, we can only count which index in the coarse and fine grid
     * face we are currently handling and thus effectively run through the
     * interpolation matrix row by row and column by column. In 3d, where
     * the face is actually a 2d object, this does not work. Here, the row
     * and column indices are defined by the entry along one direction and
     * d-1 entries have to be multiplied (tensor product). So effectively,
     * we have to map the dof counter for coarse and fine first to the
     * indices of a 1d projection operator.
     *
     *
     * @param numberOfProjectedQuantities This is one if we only project the
     *   solution. It is two if we project the solution and the F-value or
     *   the solution and itsgradient along the normal.
     */
    void interpolateRiemannSolution(
      const peano4::datamanagement::FaceMarker&  marker,
      int                                        order,
      int                                        unknowns,
      const double* __restrict__                 InterpolationMatrix1d,
      const double* __restrict__                 coarseGridFaceQ,
      double* __restrict__                       fineGridFaceQ
    );

    /**
     * Counterpart of interpolateRiemannSolution().
     *
     * The restriction matrix has exactly the same form as the interpolation
     * matrix. However, this time we multiply the corresponding row of the
     * submatrix with a weight from the fine grid.
     *
     * The image below illustrates the same example as used in
     * interpolateRiemannSolution():
     *
     * @image html src/exahype2/dg/interpolation.png
     *
     * Different to the interpolation, we pick one of the three submatrices
     * of RestrictionMatrix1d, compute the c1, c2, c3 and c4 values and then
     * add them to the coarse representation. The interpolation can overwrite.
     *
     * Different to the interpolation, we only restrict one half of the face
     * and not both. The interpolation can set both sides, as it overwrites
     * a hanging face (or a new one). When we restrict, we know that a
     * persistent coarse grid face might already hold some projected solution.
     * We are interested in the other half. If we accumulate both, we'd add
     * the interpolated value (or any other rubbish) again to the one face
     * part which had been valid before.
     *
     *
     * ## Differences to interpolateRiemannSolution()
     *
     * - In the final accummulation, we write to the coarse grid while the
     *   interpolation writes to the fine grid. Consequently, coarse grid
     *   and fine grid data are permuted in the signature.
     * - The restriction works on the solution that is extrapolated to the
     *   faces. Consequently, these data hold auxiliary variables. The
     *   Riemann solution instead does not hold any auxiliary variables.
     * - The restriction does not clear the target field, i.e. write zeros
     *   to it. That has to be done earlier by the mesh traversal, as
     *   multiple restriction calls accummulate in one variable, i.e. if
     *   we clear we loose all the data from previous restrictions.
     * - Row and column indices are permuted in the matrix access, i.e. the
     *   coarse grid dof determines the target (image) and hence defines the
     *   row.
     */
    void restrictAndAccumulateProjectedFacePolynomial(
      const peano4::datamanagement::FaceMarker&  marker,
      int                                        order,
      int                                        numberOfProjectedQuantities,
      int                                        unknowns,
      int                                        auxiliaryVariables,
      const double* __restrict__                 RestrictionMatrix1d,
      const double* __restrict__                 fineGridFaceQ,
      double* __restrict__                       coarseGridFaceQ
    );

    /**
     * Take polynomial within cell and project it onto the faces
     *
     * ## Ordering
     *
     *                           2 3
     *                           0 1
     *                     2 3 [ o o ] 2 3
     *                     0 1 [ o o ] 0 1
     *                           2 3
     *                           0 1
     *
     * Each individual face is ordered lexicographically. So we go from left to
     * right, and we always start closest to the origin or the coordinate system
     * if we interpret the cell as the unit square.
     *
     * The sketch above illustrates the ordering of the four faces for a p=1
     * discretisation where we have two dofs per direction per face per side.
     *
     * @param cellQ Pointer to cell data. The array has the size
     *   @f$ unknowns \times (order+1)^d@f$.
     *
     * @param order Integer greater or equal to 0. The number of nodes in each
     *  dimension will be equal to order+1
     *
     * @param BasisFunctionValuesLeft Array of size order+1 which clarifies (in 1d)
     *   how to project the order+1 quadrature weights onto a value on the left
     *   face. For Gauss Lobatto, this is a trivial array (you can read out the
     *   value directly from the leftmost quadrature point, but for any other
     *   basis, you need a linear combination of all the quadrature points in the
     *   cell.
     *
     * @param faceQLeft Left unknowns. This array hosts @f$ 2 \times (order+1)^(d-1) * unknowns@f$
     *   values, as it holds the (projected) values from the left side of the face
     *   and those from the right. The ordering of these values is lexicographic,
     *   i.e. the first unknowns entries are those at the bottom of the vertical
     *   face from the left adjacent cell. The next unknowns entries are those
     *   at the bottom of the face from the right adjacent cell. The next are then
     *   again left, then right, then left and so on.
     *   This function will fill the right values since the current cell is to the
     *   right of its own left face.
     *
     * @param faceQBottom Bottom unknowns. See faceQLeft. Again, we order
     *   lexicographically. This time, the first half of the entries are those
     *   from the adjacent cell below. The remaining entries are the projections
     *   from above.
     */
    void projectVolumetricDataOntoFaces(
      const double* __restrict__          cellQ,
      int                                 order,
      int                                 unknowns,
      int                                 auxiliaryVariables,
      const double* const __restrict__    BasisFunctionValuesLeft,
      double* const __restrict__          faceQLeft,
      double* const __restrict__          faceQRight,
      double* const __restrict__          faceQBottom,
      double* const __restrict__          faceQUp
    );

    /**
     * Set the whole solution projection on the face from left and right to zero
     *
     * @param numberOfProjectedQUantities Equals one if we project only the
     *   solution to the face, it equals two if we project both the solution
     *   and the gradient or the F-extrapolation, and so forth.
     */
    void clearSolutionProjection(
      int                           order,
      int                           unknowns,
      int                           auxiliaryVariables,
      int                           numberOfProjectedQuantities,
      double* __restrict__          faceQ
    );

    /**
     * @see clearSolutionProjection()
     */
    void clearRiemannResult(
      int                           order,
      int                           unknowns,
      double* __restrict__          faceQ
    );

    /**
     * 3d counterpart to other projectVolumetricDataOntoFaces() variant.
     */
    void projectVolumetricDataOntoFaces(
      const double* __restrict__          cellQ,
      int                                 order,
      int                                 unknowns,
      int                                 auxiliaryVariables,
      const double* const __restrict__    BasisFunctionValuesLeft,
      double* const __restrict__          faceQLeft,
      double* const __restrict__          faceQRight,
      double* const __restrict__          faceQBottom,
      double* const __restrict__          faceQUp,
      double* const __restrict__          faceQFront,
      double* const __restrict__          faceQBack
    );

    void projectVolumetricDataAndGradientOntoFaces(
      const double* __restrict__          cellQ,
      int                                 order,
      int                                 unknowns,
      int                                 auxiliaryVariables,
      const double* const __restrict__    BasisFunctionValuesLeft,
      double* const __restrict__          faceQLeft,
      double* const __restrict__          faceQRight,
      double* const __restrict__          faceQBottom,
      double* const __restrict__          faceQUp
    );

    /**
     * 3d counterpart to other projectVolumetricDataOntoFaces() variant.
     */
    void projectVolumetricDataAndGradientOntoFaces(
      const double* __restrict__          cellQ,
      int                                 order,
      int                                 unknowns,
      int                                 auxiliaryVariables,
      const double* const __restrict__    BasisFunctionValuesLeft,
      double* const __restrict__          faceQLeft,
      double* const __restrict__          faceQRight,
      double* const __restrict__          faceQBottom,
      double* const __restrict__          faceQUp,
      double* const __restrict__          faceQFront,
      double* const __restrict__          faceQBack
    );

    /**
     * Given a numerical flux at the various faces, this computes and adds the Riemann
     * integral of this flux to the nodes of the cell.
     *
     *
     * ## Realisation
     *
     * The code is a nested for loop, i.e. a 2d Cartesian loop. Per loop body, we add
     * the results from left, right, bottom and top.
     *
     *
     * Computes this flux for a 2-dimensional cell
     *
     * @param cellQ the current solution, expected to be the solution already updated
     *   with local contributions. (See Euler.h)
     *
     * @param faceQX This should contain a value for the numerical flux at each node along
     * the corresponding face. This is typically computed by a Riemann solver such as
     * a Rusanov solver.
     *
     * @param: BasisFunctionValuesLeft (respectively Right): as the name implies, this is
     *  the numerical value of the basis function value evaluated at the left node in
     *  1 dimension. This has a part in determining how much the value at the boundary
     *  influences the value at a given node.
     *  Note that the Basis function values on the right side are symmetric to those on the
     *  left side, so the BasisFunctionValuesLeft are also used there.
     *
     * @param quadratureWeights: quadrature weight of the given quadrature node 1 dimension
     *
     * @see projectVolumetricDataOntoFaces()
     */
    void integrateOverRiemannSolutionsAndAddToVolume_GaussLegendre(
      const double* const __restrict__    faceQLeft,
      const double* const __restrict__    faceQRight,
      const double* const __restrict__    faceQBottom,
      const double* const __restrict__    faceQUp,
      int                                 order,
      int                                 unknowns,
      const int                           auxiliaryVariables,
      const tarch::la::Vector<2,double>&  cellSize,
      const double* const __restrict__    BasisFunctionValuesLeft,
      const double* __restrict__          MassMatrixDiagonal1d,
      double* __restrict__                cellQ
    );

    /**
     * 3D counterpart of integrateOverRiemannSolutionsAndAddToVolume_GaussLegendre()
     *
     * The routine relies on a left-handed coordinate system. So we have a cube
     * with a left and a right face. Those are the faxes 0 and 3
     * (=0+Dimensions). Their normal points along the x axis. Next, we have
     * bottom and top face, as they have a normal along the y axis and y is the
     * second coordinate axis. Finally, we handle the front and the back face.
     * They are the faces 2 and 2+Dimensions=5. The front face comes first, as
     * we work with a left-handed coordinate system, i.e. the normal points
     * into the screen.
     */
    void integrateOverRiemannSolutionsAndAddToVolume_GaussLegendre(
      const double* const __restrict__    faceQLeft,
      const double* const __restrict__    faceQRight,
      const double* const __restrict__    faceQBottom,
      const double* const __restrict__    faceQUp,
      const double* const __restrict__    faceQFront,
      const double* const __restrict__    faceQBack,
      int                                 order,
      int                                 unknowns,
      const int                           auxiliaryVariables,
      const tarch::la::Vector<3,double>&  cellSize,
      const double* const __restrict__    BasisFunctionValuesLeft,
      const double* __restrict__          MassMatrixDiagonal1d,
      double* __restrict__                cellQ
    );
  }
}
