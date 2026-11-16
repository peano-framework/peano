#include "exagrype/ccz4/SecondOrderAuxiliaryVariablesReconstruction.h"
#include "exahype2/fd/FD_Helper.cpph"

#include <vector>


namespace {
  /**
   * Templated recompute function which works with a function.
   */
  template <void (*recompute_LoopBody)(
      double* __restrict__                                       QIn,
      const ::exahype2::enumerator::AoSLexicographicEnumerator& QInEnumerator,
      const tarch::la::Vector<Dimensions,double>&  patchCentre,
      const tarch::la::Vector<Dimensions,double>&  patchSize,
      int                                          patchIndex,
      const tarch::la::Vector<Dimensions,int>&     volumeIndex,
      int                                          normal
  )>
  void recomputeAuxiliaryVariablesFD4(
    ::exahype2::CellData<double, double>&   patchData,
    int                                     numberOfGridCellsPerPatchPerAxis,
    int                                     haloSize,
    int                                     unknowns,
    int                                     auxiliaryVariables
  ) {
    assertion( haloSize >= 1 );
    ::exahype2::enumerator::AoSLexicographicEnumerator QInEnumerator (1,numberOfGridCellsPerPatchPerAxis,haloSize,unknowns,auxiliaryVariables);

    for (int patchIndex=0; patchIndex<patchData.numberOfCells; patchIndex++) {
      if (Dimensions==2) {
          for (int x = 0; x < numberOfGridCellsPerPatchPerAxis; x++)
          for (int y = 0; y < numberOfGridCellsPerPatchPerAxis; y++) {
            recompute_LoopBody(
              patchData.QIn[patchIndex],
              QInEnumerator,
              patchData.cellCentre[patchIndex],
              patchData.cellSize[patchIndex],
              patchIndex,
              gridCellIndex2d(x,y),
              0  // normal
            );
          }

          for (int x = 0; x < numberOfGridCellsPerPatchPerAxis; x++)
          for (int y = 0; y < numberOfGridCellsPerPatchPerAxis; y++) {
            recompute_LoopBody(
              patchData.QIn[patchIndex],
              QInEnumerator,
              patchData.cellCentre[patchIndex],
              patchData.cellSize[patchIndex],
              patchIndex,
              gridCellIndex2d(x,y),
              1 // normal
            );
          }
      }
      else {
          for (int x = 0; x < numberOfGridCellsPerPatchPerAxis; x++)
          for (int y = 0; y < numberOfGridCellsPerPatchPerAxis; y++)
          for (int z = 0; z < numberOfGridCellsPerPatchPerAxis; z++) {
            recompute_LoopBody(
              patchData.QIn[patchIndex],
              QInEnumerator,
              patchData.cellCentre[patchIndex],
              patchData.cellSize[patchIndex],
              patchIndex,
              gridCellIndex3d(x,y,z),
              0 // normal
            );
          }

          for (int x = 0; x < numberOfGridCellsPerPatchPerAxis; x++)
          for (int y = 0; y < numberOfGridCellsPerPatchPerAxis; y++)
          for (int z = 0; z < numberOfGridCellsPerPatchPerAxis; z++) {
            recompute_LoopBody(
              patchData.QIn[patchIndex],
              QInEnumerator,
              patchData.cellCentre[patchIndex],
              patchData.cellSize[patchIndex],
              patchIndex,
              gridCellIndex3d(x,y,z),
              1 // normal
            );
          }

          for (int x = 0; x < numberOfGridCellsPerPatchPerAxis; x++)
          for (int y = 0; y < numberOfGridCellsPerPatchPerAxis; y++)
          for (int z = 0; z < numberOfGridCellsPerPatchPerAxis; z++) {
            recompute_LoopBody(
              patchData.QIn[patchIndex],
              QInEnumerator,
              patchData.cellCentre[patchIndex],
              patchData.cellSize[patchIndex],
              patchIndex,
              gridCellIndex3d(x,y,z),
              2 // normal
            );
          }
      }
    }
  }
}


void exagrype::ccz4::recomputeAuxiliaryVariablesFD4_centralDifferences(
  ::exahype2::CellData<double, double>&   patchData,
  int                                     numberOfGridCellsPerPatchPerAxis,
  int                                     haloSize,
  int                                     unknowns,
  int                                     auxiliaryVariables
) {
  recomputeAuxiliaryVariablesFD4<internal::recomputeAuxiliaryVariablesFD4_centralDifferences_LoopBody>(
    patchData,
    numberOfGridCellsPerPatchPerAxis,
    haloSize,
    unknowns,
    auxiliaryVariables
  );
}


void exagrype::ccz4::recomputeAuxiliaryVariablesFD4_leftDifferences(
  ::exahype2::CellData<double, double>&   patchData,
  int                                     numberOfGridCellsPerPatchPerAxis,
  int                                     haloSize,
  int                                     unknowns,
  int                                     auxiliaryVariables
) {
  recomputeAuxiliaryVariablesFD4<internal::recomputeAuxiliaryVariablesFD4_leftDifferences_LoopBody>(
    patchData,
    numberOfGridCellsPerPatchPerAxis,
    haloSize,
    unknowns,
    auxiliaryVariables
  );
}


void exagrype::ccz4::recomputeAuxiliaryVariablesFD4_rightDifferences(
  ::exahype2::CellData<double, double>&   patchData,
  int                                     numberOfGridCellsPerPatchPerAxis,
  int                                     haloSize,
  int                                     unknowns,
  int                                     auxiliaryVariables
) {
  recomputeAuxiliaryVariablesFD4<internal::recomputeAuxiliaryVariablesFD4_rightDifferences_LoopBody>(
    patchData,
    numberOfGridCellsPerPatchPerAxis,
    haloSize,
    unknowns,
    auxiliaryVariables
  );
}


void exagrype::ccz4::recomputeAuxiliaryVariablesFD4_4thOrder(
  ::exahype2::CellData<double, double>&   patchData,
  int                                     numberOfGridCellsPerPatchPerAxis,
  int                                     haloSize,
  int                                     unknowns,
  int                                     auxiliaryVariables
) {
  assertion( haloSize >= 3 );
  ::exahype2::enumerator::AoSLexicographicEnumerator QInEnumerator (1,numberOfGridCellsPerPatchPerAxis,haloSize,unknowns,auxiliaryVariables);

  for (int patchIndex=0; patchIndex<patchData.numberOfCells; patchIndex++) {
    if (Dimensions==2) {
        for (int x = -1; x < numberOfGridCellsPerPatchPerAxis+1; x++)
        for (int y =  0; y < numberOfGridCellsPerPatchPerAxis;   y++) {
          internal::recomputeAuxiliaryVariablesFD4_4thOrder_LoopBody(
            patchData.QIn[patchIndex],
            QInEnumerator,
            patchData.cellCentre[patchIndex],
            patchData.cellSize[patchIndex],
            patchIndex,
            gridCellIndex2d(x,y),
            0  // normal
          );
        }

        for (int x =  0; x < numberOfGridCellsPerPatchPerAxis;   x++)
        for (int y = -1; y < numberOfGridCellsPerPatchPerAxis+1; y++) {
          internal::recomputeAuxiliaryVariablesFD4_4thOrder_LoopBody(
            patchData.QIn[patchIndex],
            QInEnumerator,
            patchData.cellCentre[patchIndex],
            patchData.cellSize[patchIndex],
            patchIndex,
            gridCellIndex2d(x,y),
            1 // normal
          );
        }
    }
    else {
        for (int x = -1; x < numberOfGridCellsPerPatchPerAxis+1; x++)
        for (int y =  0; y < numberOfGridCellsPerPatchPerAxis;   y++)
        for (int z =  0; z < numberOfGridCellsPerPatchPerAxis;   z++) {
          internal::recomputeAuxiliaryVariablesFD4_4thOrder_LoopBody(
            patchData.QIn[patchIndex],
            QInEnumerator,
            patchData.cellCentre[patchIndex],
            patchData.cellSize[patchIndex],
            patchIndex,
            gridCellIndex3d(x,y,z),
            0 // normal
          );
        }

        for (int x =  0; x < numberOfGridCellsPerPatchPerAxis;   x++)
        for (int y = -1; y < numberOfGridCellsPerPatchPerAxis+1; y++)
        for (int z =  0; z < numberOfGridCellsPerPatchPerAxis;   z++) {
          internal::recomputeAuxiliaryVariablesFD4_4thOrder_LoopBody(
            patchData.QIn[patchIndex],
            QInEnumerator,
            patchData.cellCentre[patchIndex],
            patchData.cellSize[patchIndex],
            patchIndex,
            gridCellIndex3d(x,y,z),
            1 // normal
          );
        }

        for (int x =  0; x < numberOfGridCellsPerPatchPerAxis;   x++)
        for (int y =  0; y < numberOfGridCellsPerPatchPerAxis;   y++)
        for (int z = -1; z < numberOfGridCellsPerPatchPerAxis+1; z++) {
          internal::recomputeAuxiliaryVariablesFD4_4thOrder_LoopBody(
            patchData.QIn[patchIndex],
            QInEnumerator,
            patchData.cellCentre[patchIndex],
            patchData.cellSize[patchIndex],
            patchIndex,
            gridCellIndex3d(x,y,z),
            2 // normal
          );
        }
    }
  }
}


namespace {
  /**
   * Shared by the reconstruction routines: copy the 25 primary variables into
   * a 58-entry buffer, let recompute fill the derivatives and copy them back.
   */
  template <typename Recompute>
  void reconstructFirstDerivatives(
    const double* __restrict__                  primaryVariablesWithHalo,
    const tarch::la::Vector<Dimensions,double>& patchSize,
    int                                         numberOfGridCellsPerPatchPerAxis,
    int                                         haloSize,
    double* __restrict__                        firstDerivatives,
    Recompute                                   recompute
  ) {
    static_assert( Dimensions==3, "ExaGRyPE is three-dimensional only" );

    constexpr int NumberOfPrimaryVariables = 25;
    constexpr int NumberOfDerivatives      = 33;
    constexpr int NumberOfEntries          = NumberOfPrimaryVariables + NumberOfDerivatives;

    const int n = numberOfGridCellsPerPatchPerAxis;

    const ::exahype2::enumerator::AoSLexicographicEnumerator sourceEnumerator(1,n,haloSize,NumberOfPrimaryVariables,0);
    const ::exahype2::enumerator::AoSLexicographicEnumerator targetEnumerator(1,n,haloSize,NumberOfPrimaryVariables,NumberOfDerivatives);

    const int volumesPerAxis = n + 2*haloSize;
    std::vector<double> buffer( volumesPerAxis*volumesPerAxis*volumesPerAxis*NumberOfEntries, 0.0 );

    for (int x=-haloSize; x<n+haloSize; x++)
    for (int y=-haloSize; y<n+haloSize; y++)
    for (int z=-haloSize; z<n+haloSize; z++) {
      const tarch::la::Vector<Dimensions,int> volume = gridCellIndex3d(x,y,z);
      for (int k=0; k<NumberOfPrimaryVariables; k++) {
        buffer[ targetEnumerator(0,volume,k) ] = primaryVariablesWithHalo[ sourceEnumerator(0,volume,k) ];
      }
    }

    ::exahype2::CellData<double,double> cellData(
      buffer.data(),
      tarch::la::Vector<Dimensions,double>(0.0),
      patchSize,
      0.0,
      0.0,
      nullptr
    );
    recompute( cellData, n, haloSize, NumberOfPrimaryVariables, NumberOfDerivatives );

    for (int x=0; x<n; x++)
    for (int y=0; y<n; y++)
    for (int z=0; z<n; z++) {
      const tarch::la::Vector<Dimensions,int> volume = gridCellIndex3d(x,y,z);
      const int serialisedVolume = x + n*y + n*n*z;
      for (int k=0; k<NumberOfDerivatives; k++) {
        firstDerivatives[ serialisedVolume*NumberOfDerivatives + k ] = buffer[ targetEnumerator(0,volume,NumberOfPrimaryVariables+k) ];
      }
    }
  }
}


void exagrype::ccz4::reconstructFirstDerivativesFD4(
  const double* __restrict__                  primaryVariablesWithHalo,
  const tarch::la::Vector<Dimensions,double>& patchSize,
  int                                         numberOfGridCellsPerPatchPerAxis,
  int                                         haloSize,
  double* __restrict__                        firstDerivatives
) {
  assertion( haloSize>=3 );
  reconstructFirstDerivatives(
    primaryVariablesWithHalo, patchSize, numberOfGridCellsPerPatchPerAxis, haloSize, firstDerivatives,
    recomputeAuxiliaryVariablesFD4_4thOrder
  );
}


void exagrype::ccz4::reconstructFirstDerivativesCentralDifferences(
  const double* __restrict__                  primaryVariablesWithHalo,
  const tarch::la::Vector<Dimensions,double>& patchSize,
  int                                         numberOfGridCellsPerPatchPerAxis,
  int                                         haloSize,
  double* __restrict__                        firstDerivatives
) {
  assertion( haloSize>=1 );
  reconstructFirstDerivatives(
    primaryVariablesWithHalo, patchSize, numberOfGridCellsPerPatchPerAxis, haloSize, firstDerivatives,
    recomputeAuxiliaryVariablesFD4_centralDifferences
  );
}


void exagrype::ccz4::internal::recomputeAuxiliaryVariablesFD4_centralDifferences_LoopBody(
  double* __restrict__                                       QIn,
  const ::exahype2::enumerator::AoSLexicographicEnumerator& QInEnumerator,
  const tarch::la::Vector<Dimensions,double>&  patchCentre,
  const tarch::la::Vector<Dimensions,double>&  patchSize,
  int                                          patchIndex,
  const tarch::la::Vector<Dimensions,int>&     volumeIndex,
  int                                          normal
) {
  tarch::la::Vector<Dimensions,int>       centralVolume  = volumeIndex;
  tarch::la::Vector<Dimensions,int>  leftAdjacentVolume  = volumeIndex;
  tarch::la::Vector<Dimensions,int> rightAdjacentVolume  = volumeIndex;

  rightAdjacentVolume(normal)++;
   leftAdjacentVolume(normal)--;

  double cellSize = patchSize(normal) / QInEnumerator._numberOfDoFsPerAxisInCell;

  auto finiteDifferences = [&](int source, int target) {
     double dx = QIn[ QInEnumerator(patchIndex, rightAdjacentVolume,source) ]
               - QIn[ QInEnumerator(patchIndex, leftAdjacentVolume, source) ];
     QIn[ QInEnumerator(patchIndex,centralVolume,target) ] = dx / 2.0 / cellSize;
  };

  finiteDifferences(16,25+normal);
  finiteDifferences(24,55+normal);

  for (int i=0; i<3; i++) {
    finiteDifferences(17+i,28+3*normal+i);
  }

  for (int i=0; i<3; i++)
  for (int j=i; j<3; j++) {
    int k = (i==0?j:i+j+1);
    finiteDifferences(k,37+6*normal+k);
  }
}




void exagrype::ccz4::internal::recomputeAuxiliaryVariablesFD4_leftDifferences_LoopBody(
  double* __restrict__                                       QIn,
  const ::exahype2::enumerator::AoSLexicographicEnumerator& QInEnumerator,
  const tarch::la::Vector<Dimensions,double>&  patchCentre,
  const tarch::la::Vector<Dimensions,double>&  patchSize,
  int                                          patchIndex,
  const tarch::la::Vector<Dimensions,int>&     volumeIndex,
  int                                          normal
) {
  tarch::la::Vector<Dimensions,int>       centralVolume  = volumeIndex;
  tarch::la::Vector<Dimensions,int>  leftAdjacentVolume  = volumeIndex;

   leftAdjacentVolume(normal)--;

  double cellSize = patchSize(normal) / QInEnumerator._numberOfDoFsPerAxisInCell;

  auto finiteDifferences = [&](int source, int target) {
     double dx = QIn[ QInEnumerator(patchIndex, centralVolume,source) ]
               - QIn[ QInEnumerator(patchIndex, leftAdjacentVolume, source) ];
     QIn[ QInEnumerator(patchIndex,centralVolume,target) ] = dx / cellSize;
  };

  finiteDifferences(16,25+normal);
  finiteDifferences(24,55+normal);

  for (int i=0; i<3; i++) {
    finiteDifferences(17+i,28+3*normal+i);
  }

  for (int i=0; i<3; i++)
  for (int j=i; j<3; j++) {
    int k = (i==0?j:i+j+1);
    finiteDifferences(k,37+6*normal+k);
  }
}




void exagrype::ccz4::internal::recomputeAuxiliaryVariablesFD4_rightDifferences_LoopBody(
  double* __restrict__                                       QIn,
  const ::exahype2::enumerator::AoSLexicographicEnumerator& QInEnumerator,
  const tarch::la::Vector<Dimensions,double>&  patchCentre,
  const tarch::la::Vector<Dimensions,double>&  patchSize,
  int                                          patchIndex,
  const tarch::la::Vector<Dimensions,int>&     volumeIndex,
  int                                          normal
) {
  tarch::la::Vector<Dimensions,int>       centralVolume  = volumeIndex;
  tarch::la::Vector<Dimensions,int> rightAdjacentVolume  = volumeIndex;

  rightAdjacentVolume(normal)++;

  double cellSize = patchSize(normal) / QInEnumerator._numberOfDoFsPerAxisInCell;

  auto finiteDifferences = [&](int source, int target) {
     double dx = QIn[ QInEnumerator(patchIndex, rightAdjacentVolume,source) ]
               - QIn[ QInEnumerator(patchIndex, centralVolume, source) ];
     QIn[ QInEnumerator(patchIndex,centralVolume,target) ] = dx / cellSize;
  };

  finiteDifferences(16,25+normal);
  finiteDifferences(24,55+normal);

  for (int i=0; i<3; i++) {
    finiteDifferences(17+i,28+3*normal+i);
  }

  for (int i=0; i<3; i++)
  for (int j=i; j<3; j++) {
    int k = (i==0?j:i+j+1);
    finiteDifferences(k,37+6*normal+k);
  }
}


void exagrype::ccz4::internal::recomputeAuxiliaryVariablesFD4_4thOrder_LoopBody(
  double* __restrict__                                       QIn,
  const ::exahype2::enumerator::AoSLexicographicEnumerator& QInEnumerator,
  const tarch::la::Vector<Dimensions,double>&  patchCentre,
  const tarch::la::Vector<Dimensions,double>&  patchSize,
  int                                          patchIndex,
  const tarch::la::Vector<Dimensions,int>&     volumeIndex,
  int                                          normal
) {
  tarch::la::Vector<Dimensions,int>       centralVolume  = volumeIndex;
  tarch::la::Vector<Dimensions,int>  leftAdjacentVolume1 = volumeIndex;
  tarch::la::Vector<Dimensions,int>  leftAdjacentVolume2 = volumeIndex;
  tarch::la::Vector<Dimensions,int> rightAdjacentVolume1 = volumeIndex;
  tarch::la::Vector<Dimensions,int> rightAdjacentVolume2 = volumeIndex;

  rightAdjacentVolume1(normal)++;
  rightAdjacentVolume2(normal)++;
  rightAdjacentVolume2(normal)++;
   leftAdjacentVolume1(normal)--;
   leftAdjacentVolume2(normal)--;
   leftAdjacentVolume2(normal)--;

  double cellsize = patchSize(normal) / QInEnumerator._numberOfDoFsPerAxisInCell;
  double gradcoef = 1.0/(12.0*cellsize);

  QIn[ QInEnumerator(patchIndex,centralVolume,25+normal) ] =     gradcoef * QIn[ QInEnumerator(patchIndex, leftAdjacentVolume2,16) ] -
                                                             8.0*gradcoef * QIn[ QInEnumerator(patchIndex, leftAdjacentVolume1,16) ] +
                                                             8.0*gradcoef * QIn[ QInEnumerator(patchIndex,rightAdjacentVolume1,16) ] -
                                                                 gradcoef * QIn[ QInEnumerator(patchIndex,rightAdjacentVolume2,16) ];

  // I'm afraid I misunderstood the enumeration
  QIn[ QInEnumerator(patchIndex,centralVolume,55+normal) ] =     gradcoef * QIn[ QInEnumerator(patchIndex, leftAdjacentVolume2,24) ] -
                                                             8.0*gradcoef * QIn[ QInEnumerator(patchIndex, leftAdjacentVolume1,24) ] +
                                                             8.0*gradcoef * QIn[ QInEnumerator(patchIndex,rightAdjacentVolume1,24) ] -
                                                                 gradcoef * QIn[ QInEnumerator(patchIndex,rightAdjacentVolume2,24) ];

  for (int i=0; i<3; i++) {
    QIn[ QInEnumerator(patchIndex,centralVolume,28+3*normal+i) ] =     gradcoef * QIn[ QInEnumerator(patchIndex, leftAdjacentVolume2,17+i) ] -
                                                                   8.0*gradcoef * QIn[ QInEnumerator(patchIndex, leftAdjacentVolume1,17+i) ] +
                                                                   8.0*gradcoef * QIn[ QInEnumerator(patchIndex,rightAdjacentVolume1,17+i) ] -
                                                                       gradcoef * QIn[ QInEnumerator(patchIndex,rightAdjacentVolume2,17+i) ];
  }

  for (int i=0; i<3; i++)
  for (int j=i; j<3; j++) {
    int k = (i==0?j:i+j+1);
    QIn[ QInEnumerator(patchIndex,centralVolume,37+6*normal+k) ] = 0.5*gradcoef * QIn[ QInEnumerator(patchIndex, leftAdjacentVolume2,k) ] -
                                                                   4.0*gradcoef * QIn[ QInEnumerator(patchIndex, leftAdjacentVolume1,k) ] +
                                                                   4.0*gradcoef * QIn[ QInEnumerator(patchIndex,rightAdjacentVolume1,k) ] -
                                                                   0.5*gradcoef * QIn[ QInEnumerator(patchIndex,rightAdjacentVolume2,k) ];
  }
}
