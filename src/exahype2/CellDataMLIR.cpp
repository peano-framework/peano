#include "CellData.h"

exahype2::MLIRCellDataFlattened::MLIRCellDataFlattened(
  const exahype2::CellData<double, double>& cellData,
  int inputPatchSize,
  int outputPatchSize
):
  numberOfCells(cellData.numberOfCells),
  QInStride(inputPatchSize),
  QOutStride(outputPatchSize)
  {
  QIn = createMemRefDescriptor2D(cellData.QIn, cellData.numberOfCells, inputPatchSize);
  QOut = createMemRefDescriptor2D(cellData.QOut, cellData.numberOfCells, outputPatchSize);
  t = createMemRefDescriptor1D(cellData.t, cellData.numberOfCells);
  dt = createMemRefDescriptor1D(cellData.dt, cellData.numberOfCells);

  cellCentre = exahype2::MemRefDescriptor2D();
  double* cellCentrePtr = new double[cellData.numberOfCells * Dimensions];
  for (int i = 0; i < cellData.numberOfCells; i++) {
    for (int d = 0; d < Dimensions; d++) {
      cellCentrePtr[i * Dimensions + d] = cellData.cellCentre[i](d);
    }
  }
  cellCentre.allocated = cellCentrePtr;
  cellCentre.aligned = cellCentrePtr;
  cellCentre.offset = 0;
  cellCentre.sizes[0] = cellData.numberOfCells;
  cellCentre.sizes[1] = Dimensions;
  cellCentre.strides[0] = Dimensions;
  cellCentre.strides[1] = 1;

  cellSize = exahype2::MemRefDescriptor2D();
  double* cellSizePtr = new double[cellData.numberOfCells * Dimensions];
  for (int i = 0; i < cellData.numberOfCells; i++) {
    for (int d = 0; d < Dimensions; d++) {
      cellSizePtr[i * Dimensions + d] = cellData.cellSize[i](d);
    }
  }
  cellSize.allocated = cellSizePtr;
  cellSize.aligned = cellSizePtr;
  cellSize.offset = 0;
  cellSize.sizes[0] = cellData.numberOfCells;
  cellSize.sizes[1] = Dimensions;
  cellSize.strides[0] = Dimensions;
  cellSize.strides[1] = 1;

  maxEigenvalue = createMemRefDescriptor1D(cellData.maxEigenvalue, cellData.numberOfCells);
  QInNested = cellData.QIn;
  QOutNested = cellData.QOut;
}


void exahype2::MLIRCellDataFlattened::copyBackAndDestroy() {
  for (int i = 0; i < numberOfCells; ++i) {
    std::memcpy(
      QOutNested[i],
      &QOut.allocated[i * QOutStride],
      static_cast<std::size_t>(QOutStride) * sizeof(double)
    );
  }

  delete[] QIn.allocated;
  delete[] QOut.allocated;
  QIn.allocated = nullptr;
  QOut.allocated = nullptr;
}

std::string exahype2::MLIRCellDataFlattened::toString() const {
  std::ostringstream msg;
  msg << "[";
  for (int i = 0; i < numberOfCells; i++) {
    
  }
  msg << "]";
  return msg.str();
}


exahype2::MLIRCellDataNested::MLIRCellDataNested(
  const exahype2::CellData<double, double>& cellData,
  int inputPatchSize,
  int outputPatchSize
):
  numberOfCells(cellData.numberOfCells),
  QInStride(inputPatchSize),
  QOutStride(outputPatchSize)
  {
  QIn = createMemRefDescriptorNested(cellData.QIn, cellData.numberOfCells, inputPatchSize);
  QOut = createMemRefDescriptorNested(cellData.QOut, cellData.numberOfCells, outputPatchSize);
  t = createMemRefDescriptor1D(cellData.t, cellData.numberOfCells);
  dt = createMemRefDescriptor1D(cellData.dt, cellData.numberOfCells);

  cellCentre = exahype2::MemRefDescriptor2D();
  double* cellCentrePtr = new double[cellData.numberOfCells * Dimensions];
  for (int i = 0; i < cellData.numberOfCells; i++) {
    for (int d = 0; d < Dimensions; d++) {
      cellCentrePtr[i * Dimensions + d] = cellData.cellCentre[i](d);
    }
  }
  cellCentre.allocated = cellCentrePtr;
  cellCentre.aligned = cellCentrePtr;
  cellCentre.offset = 0;
  cellCentre.sizes[0] = cellData.numberOfCells;
  cellCentre.sizes[1] = Dimensions;
  cellCentre.strides[0] = Dimensions;
  cellCentre.strides[1] = 1;

  cellSize = exahype2::MemRefDescriptor2D();
  double* cellSizePtr = new double[cellData.numberOfCells * Dimensions];
  for (int i = 0; i < cellData.numberOfCells; i++) {
    for (int d = 0; d < Dimensions; d++) {
      cellSizePtr[i * Dimensions + d] = cellData.cellSize[i](d);
    }
  }
  cellSize.allocated = cellSizePtr;
  cellSize.aligned = cellSizePtr;
  cellSize.offset = 0;
  cellSize.sizes[0] = cellData.numberOfCells;
  cellSize.sizes[1] = Dimensions;
  cellSize.strides[0] = Dimensions;
  cellSize.strides[1] = 1;

  maxEigenvalue = createMemRefDescriptor1D(cellData.maxEigenvalue, cellData.numberOfCells);
  QInNested = cellData.QIn;
  QOutNested = cellData.QOut;
}


void exahype2::MLIRCellDataNested::copyBackAndDestroy() {
  delete[] QIn.allocated;
  delete[] QOut.allocated;
}

std::string exahype2::MLIRCellDataNested::toString() const {
  std::ostringstream msg;
  msg << "[";
  for (int i = 0; i < numberOfCells; i++) {
    
  }
  msg << "]";
  return msg.str();
}