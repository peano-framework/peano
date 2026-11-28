#include "exagrype/ccz4/SecondOrderAuxiliaryVariablesReconstruction.h"

#include "exahype2/enumerator/AoSLexicographicEnumerator.h"
#include "exahype2/fd/FD_Helper.cpph"
#include "tarch/Assertions.h"


namespace {
  /**
   * Apply a one-dimensional stencil to the table of derivatives
   *
   * For each grid cell of the cell and each derivative d, the routine hands
   * the stencil a functor which yields the differentiated variable shifted
   * along the derivative's axis, the mesh width along this axis and the
   * factor of the derivative.
   */
  template <typename Stencil>
  void reconstructFirstDerivatives(
    const double* __restrict__                  primaryVariablesWithHalo,
    const tarch::la::Vector<Dimensions,double>& patchSize,
    int                                         numberOfGridCellsPerPatchPerAxis,
    int                                         haloSize,
    int                                         numberOfPrimaryVariables,
    int                                         numberOfDerivatives,
    const int* __restrict__                     differentiatedVariable,
    const int* __restrict__                     axis,
    const double* __restrict__                  factor,
    double* __restrict__                        firstDerivatives,
    Stencil                                     stencil
  ) {
    static_assert( Dimensions==3, "ExaGRyPE is three-dimensional only" );

    const int n = numberOfGridCellsPerPatchPerAxis;
    const ::exahype2::enumerator::AoSLexicographicEnumerator enumerator(
      1, n, haloSize, numberOfPrimaryVariables, 0
    );

    for (int z=0; z<n; z++)
    for (int y=0; y<n; y++)
    for (int x=0; x<n; x++) {
      const tarch::la::Vector<Dimensions,int> volume = gridCellIndex3d(x,y,z);
      const int serialisedVolume = x + n*y + n*n*z;
      for (int d=0; d<numberOfDerivatives; d++) {
        const int    variable = differentiatedVariable[d];
        const int    normal   = axis[d];
        const double cellSize = patchSize(normal) / n;
        auto shifted = [&](int shift) -> double {
          tarch::la::Vector<Dimensions,int> neighbour = volume;
          neighbour(normal) += shift;
          return primaryVariablesWithHalo[ enumerator(0,neighbour,variable) ];
        };
        firstDerivatives[ serialisedVolume*numberOfDerivatives + d ] =
          stencil( shifted, cellSize, factor[d] );
      }
    }
  }
}


void exagrype::ccz4::reconstructFirstDerivativesFD4(
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
) {
  assertion( haloSize>=2 );
  reconstructFirstDerivatives(
    primaryVariablesWithHalo, patchSize, numberOfGridCellsPerPatchPerAxis,
    haloSize, numberOfPrimaryVariables, numberOfDerivatives,
    differentiatedVariable, axis, factor, firstDerivatives,
    [](auto shifted, double cellSize, double factor) -> double {
      // The weights are rounded as in the former, hard-coded version: factor
      // is 1 or 1/2, i.e. the products below are exact.
      const double gradcoef = 1.0/(12.0*cellSize);
      const double w1       = factor * gradcoef;
      const double w8       = (factor * 8.0) * gradcoef;
      return w1 * shifted(-2) - w8 * shifted(-1)
           + w8 * shifted(1)  - w1 * shifted(2);
    }
  );
}


void exagrype::ccz4::reconstructFirstDerivativesCentralDifferences(
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
) {
  assertion( haloSize>=1 );
  reconstructFirstDerivatives(
    primaryVariablesWithHalo, patchSize, numberOfGridCellsPerPatchPerAxis,
    haloSize, numberOfPrimaryVariables, numberOfDerivatives,
    differentiatedVariable, axis, factor, firstDerivatives,
    [](auto shifted, double cellSize, double factor) -> double {
      return factor * ((shifted(1) - shifted(-1)) / 2.0 / cellSize);
    }
  );
}
