#include "file-input.h"

#include <vtkCellCenters.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf {

namespace {

/**
 * check if @p path has the @extension suffix.
 */
bool hasExtension(const std::string& path, const std::string& extension) {
  return path.size() >= extension.size()
      && path.compare(path.size() - extension.size(), extension.size(), extension) == 0;
}

/**
 * The component names of @p unknowns, with an empty string for each unnamed
 * component.
 */
std::vector<std::string> componentNames(vtkDataArray* unknowns) {
  std::vector<std::string> names(static_cast<std::size_t>(unknowns->GetNumberOfComponents()));
  for (std::size_t component = 0; component < names.size(); ++component) {
    const char* name = unknowns->GetComponentName(static_cast<vtkIdType>(component));
    if (name != nullptr) {
      names[component] = name;
    }
  }
  return names;
}

/**
 * Choose the source components to keep, in CompactField order. Without a
 * mapping they are selected by name; a given mapping is checked against the
 * names, when the data has them.
 */
std::vector<std::size_t> selectComponents(vtkDataArray* unknowns,
                                          const std::optional<SourceVariableMapping>& sourceMapping,
                                          const std::string& context) {
  const auto names = componentNames(unknowns);
  SourceVariableMapping mapping;
  if (sourceMapping) {
    ValidateSourceMappingAgainstComponentNames(*sourceMapping, names);
    mapping = *sourceMapping;
  } else {
    const auto named = SourceVariableMappingFromComponentNames(names);
    if (!named) {
      throw std::runtime_error(
        "ReadGridFromFile: the cell-data array in " + context
        + " does not name the components G[0..5], K[0..5], traceK and phi, so they cannot"
          " be selected by name; pass a SourceVariableMapping, such as"
          " ExaGRyPESourceMapping() or LegacyCCZ4SourceMapping()");
    }
    mapping = *named;
  }
  const auto components = mapping.componentsInCompactOrder();
  return {components.begin(), components.end()};
}

/**
 * Append every cell of @p grid to @p gridData: the cell centre as a coordinate
 * and the selected source components as one fixed-size compact value row. The
 * first grid with cells fills @p selected, and later pieces reuse it.
 */
void appendGridCells(vtkUnstructuredGrid* grid,
                     const std::optional<SourceVariableMapping>& sourceMapping,
                     std::vector<std::size_t>& selected,
                     const std::string& context,
                     GridData& gridData) {
  const vtkIdType numberOfCells = (grid != nullptr) ? grid->GetNumberOfCells() : 0;
  if (numberOfCells == 0) {
    return;
  }

  // The CCZ4 solution is stored as cell data: one multi-component tuple per
  // finite volume. Prefer the array flagged as the active scalars, falling back
  // to the first cell-data array.
  vtkCellData* cellData = grid->GetCellData();
  vtkDataArray* unknowns = (cellData != nullptr) ? cellData->GetScalars() : nullptr;
  if (unknowns == nullptr && cellData != nullptr && cellData->GetNumberOfArrays() > 0) {
    unknowns = cellData->GetArray(0);
  }
  if (unknowns == nullptr) {
    throw std::runtime_error("ReadGridFromFile: no cell-data array found in " + context);
  }

  const int numberOfComponents = unknowns->GetNumberOfComponents();
  if (unknowns->GetNumberOfTuples() != numberOfCells) {
    throw std::runtime_error(
      "ReadGridFromFile: cell-data tuple count does not match the number of cells in "
      + context);
  }
  if (selected.empty()) {
    selected = selectComponents(unknowns, sourceMapping, context);
  }
  for (std::size_t compactColumn = 0; compactColumn < selected.size(); ++compactColumn) {
    const std::size_t sourceIndex = selected[compactColumn];
    if (sourceIndex >= static_cast<std::size_t>(numberOfComponents)) {
      throw std::runtime_error(
        "ReadGridFromFile: required field "
        + std::string(CompactFieldNames[compactColumn]) + " selects source component "
        + std::to_string(sourceIndex) + ", but the cell-data tuple in " + context
        + " has only " + std::to_string(numberOfComponents) + " components");
    }
  }

  // The sample coordinate of each finite volume is its cell centre. vtkCellCenters
  // evaluates the cell's parametric centre, which is correct for every cell type
  // (unlike a plain corner average, which only matches for linear box/simplex
  // cells). Output point i corresponds to cell i.
  vtkNew<vtkCellCenters> cellCenters;
  cellCenters->SetInputData(grid);
  cellCenters->Update();
  vtkPoints* centers =
    (cellCenters->GetOutput() != nullptr) ? cellCenters->GetOutput()->GetPoints() : nullptr;
  if (centers == nullptr || centers->GetNumberOfPoints() != numberOfCells) {
    throw std::runtime_error(
      "ReadGridFromFile: failed to compute a cell centre for every cell in " + context);
  }

  const std::size_t base = gridData.coordinates.size();
  constexpr std::size_t numberOfVariables = GridData::numberOfVariables;
  gridData.coordinates.resize(base + static_cast<std::size_t>(numberOfCells));
  gridData.values.resize((base + static_cast<std::size_t>(numberOfCells)) * numberOfVariables);

  // Read one full tuple per cell and keep only the selected components, so the
  // large all-component array is never duplicated into GridData.
  std::vector<double> sourceTuple(static_cast<std::size_t>(numberOfComponents));
  for (vtkIdType cellIndex = 0; cellIndex < numberOfCells; ++cellIndex) {
    const std::size_t index = base + static_cast<std::size_t>(cellIndex);

    double centre[3];
    centers->GetPoint(cellIndex, centre);
    gridData.coordinates[index] = GridCoordinate{centre[0], centre[1], centre[2]};

    unknowns->GetTuple(cellIndex, sourceTuple.data());
    double* row = &gridData.values[index * numberOfVariables];
    for (std::size_t column = 0; column < selected.size(); ++column) {
      row[column] = sourceTuple[selected[column]];
    }
  }
}

/**
 * Read a single-piece .vtu in one shot.
 */
void readWholeVtu(const std::string& inputFile,
                  const std::optional<SourceVariableMapping>& sourceMapping,
                  GridData& gridData) {
  vtkNew<vtkXMLUnstructuredGridReader> reader;
  reader->SetFileName(inputFile.c_str());
  if (reader->CanReadFile(inputFile.c_str()) == 0) {
    throw std::runtime_error("ReadGridFromFile: cannot read file " + inputFile);
  }
  reader->Update();

  vtkUnstructuredGrid* grid = reader->GetOutput();
  if (grid == nullptr || grid->GetNumberOfCells() == 0) {
    throw std::runtime_error(
      "ReadGridFromFile: no unstructured-grid cells found in " + inputFile);
  }

  gridData.coordinates.clear();
  gridData.values.clear();
  static_assert(GridData::numberOfVariables == NumberOfCompactFields);
  gridData.coordinates.reserve(static_cast<std::size_t>(grid->GetNumberOfCells()));
  gridData.values.reserve(
    static_cast<std::size_t>(grid->GetNumberOfCells()) * GridData::numberOfVariables);

  std::vector<std::size_t> selected;
  appendGridCells(grid, sourceMapping, selected, inputFile, gridData);
}

/**
 * Stream a multi-piece .pvtu one piece at a time. A fresh reader is used per
 * piece so that only one piece is resident at once: peak memory is bounded by
 * one piece plus the accumulating (compact) GridData, not by the whole grid.
 */
void readPvtuStreaming(const std::string& inputFile,
                       const std::optional<SourceVariableMapping>& sourceMapping,
                       GridData& gridData) {
  int numberOfPieces = 0;
  {
    vtkNew<vtkXMLPUnstructuredGridReader> summaryReader;
    summaryReader->SetFileName(inputFile.c_str());
    if (summaryReader->CanReadFile(inputFile.c_str()) == 0) {
      throw std::runtime_error("ReadGridFromFile: cannot read file " + inputFile);
    }
    summaryReader->UpdateInformation();
    numberOfPieces = summaryReader->GetNumberOfPieces();
  }
  if (numberOfPieces < 1) {
    throw std::runtime_error("ReadGridFromFile: " + inputFile + " declares no pieces");
  }

  gridData.coordinates.clear();
  gridData.values.clear();

  std::vector<std::size_t> selected;
  for (int piece = 0; piece < numberOfPieces; ++piece) {
    // Fresh reader per piece: reusing one reader across UpdatePiece() calls
    // caches internal buffers and climbs back to the whole-grid footprint.
    vtkNew<vtkXMLPUnstructuredGridReader> reader;
    reader->SetFileName(inputFile.c_str());
    reader->UpdatePiece(piece, numberOfPieces, 0);
    vtkUnstructuredGrid* grid = reader->GetOutput();

    if (piece == 0 && grid != nullptr) {
      // Estimate the total from the first piece and reserve, so the growing
      // GridData is not reallocated (which would transiently double its memory).
      // The "+ numberOfPieces" covers the uneven remainder of an equal split.
      const std::size_t estimate =
        static_cast<std::size_t>(grid->GetNumberOfCells()) * numberOfPieces
        + static_cast<std::size_t>(numberOfPieces);
      gridData.coordinates.reserve(estimate);
      gridData.values.reserve(estimate * gridData.numberOfVariables);
    }

    appendGridCells(grid, sourceMapping, selected, inputFile, gridData);
  }

  if (gridData.coordinates.empty()) {
    throw std::runtime_error("ReadGridFromFile: no cells found across the pieces of " + inputFile);
  }
}

/**
 * Read a .vtu or a .pvtu series. Without @p sourceMapping, the components are
 * selected by name.
 */
void readGrid(const std::string& inputFile,
              GridData& gridData,
              const std::optional<SourceVariableMapping>& sourceMapping) {
  if (hasExtension(inputFile, ".pvtu")) {
    readPvtuStreaming(inputFile, sourceMapping, gridData);
  } else {
    readWholeVtu(inputFile, sourceMapping, gridData);
  }
}

}  // namespace

void ReadGridFromFile(const std::string& inputFile,
                      GridData& gridData,
                      const SourceVariableMapping& sourceMapping) {
  sourceMapping.validateRequiredFields();
  readGrid(inputFile, gridData, sourceMapping);
}

void ReadGridFromFile(const std::string& inputFile, GridData& gridData) {
  readGrid(inputFile, gridData, std::nullopt);
}

}  // namespace exahype2::ccz4::ahf
