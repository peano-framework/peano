#include "../require-asserts.h"

#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../ahf-data.h"
#include "../../file-input.h"
#include "../../output.h"
#include "../../peano-input.h"
#include "../ccz4-component-names.h"
#include "../temporary-directory.h"

namespace exahype2::ccz4::ahf::tests::io {

/**
 * Resolve a hard-coded sample file name relative to this source file so the test
 * works irrespective of the working directory ctest is launched from.
 */
std::string samplePath(const std::string& relative) {
  const std::string thisFile = __FILE__;                       // .../test/io/IoTests.cpp
  const std::string thisDir = thisFile.substr(0, thisFile.find_last_of("/\\"));
  return thisDir + "/../../" + relative;
}

bool fileExists(const std::string& path) {
  std::ifstream probe(path, std::ios::binary);
  return probe.good();
}

// The legacy ccz4.py layout of sample_data_S.vtu.
SourceVariableMapping completeSourceMapping() {
  return LegacyCCZ4SourceMapping();
}

/**
 * Read the sample through ReadGridFromFile() and assert it loaded
 * correctly. The same data and ground truth hold whether it is read from the
 * single-piece .vtu or the repartitioned .pvtu series, so this checker is shared
 * between both modes.
 */
void checkSampleGrid(const std::string& inputFile) {
  // The sample stores 64 cell components; this is the sample's actual
  // mapping. It selects 14 of them, which become GridData's compact columns:
  // conformal metric 0-5 -> columns 0-5, conformal extrinsic curvature 6-11 ->
  // columns 6-11, traceK (source 53) -> column 12, conformal factor (source 54)
  // -> column 13.
  const SourceVariableMapping sourceMapping = completeSourceMapping();

  GridData gridData;
  ReadGridFromFile(inputFile, gridData, sourceMapping);

  // sample_data_S: 3088448 cells; the mapping keeps 14 variables.
  constexpr std::size_t ExpectedNumberOfCells = 3088448;
  constexpr std::size_t ExpectedNumberOfVariables = 14;

  assert(gridData.numberOfPoints() == ExpectedNumberOfCells);
  assert(gridData.numberOfVariables == ExpectedNumberOfVariables);
  assert(gridData.isConsistent());
  assert(gridData.values.size() == ExpectedNumberOfCells * ExpectedNumberOfVariables);

  // Coordinates (cell centroids) and stored unknowns must be finite. Index the
  // value rows directly here; GridData::value() does a linear coordinate search,
  // which is too costly to call per variable over millions of cells.
  const GridCoordinate& firstCoordinate = gridData.coordinates.front();
  assert(std::isfinite(firstCoordinate.x));
  assert(std::isfinite(firstCoordinate.y));
  assert(std::isfinite(firstCoordinate.z));

  const GridCoordinate& lastCoordinate = gridData.coordinates.back();
  assert(std::isfinite(lastCoordinate.x));
  assert(std::isfinite(lastCoordinate.y));
  assert(std::isfinite(lastCoordinate.z));

  const std::size_t lastRow = (gridData.numberOfPoints() - 1) * gridData.numberOfVariables;
  for (std::size_t variable = 0; variable < gridData.numberOfVariables; ++variable) {
    assert(std::isfinite(gridData.values[variable]));            // first cell's row
    assert(std::isfinite(gridData.values[lastRow + variable]));  // last cell's row
  }

  // Exercise the coordinate-based enquiry API once (it is a linear search).
  assert(std::isfinite(gridData.value(firstCoordinate, 0)));

  // Ground truth for three known cells (centroids nearest to {1,0,0}, {0,3,3},
  // {5,5,6}), read directly from the sample. Setting each cell's coordinate must
  // return its stored unknowns, remapped into the compact layout: the checked
  // columns 0, 6, 12, 13 hold source components 0 (metric xx), 6 (extrinsic
  // curvature xx), 53 (traceK) and 54 (conformal factor). Only the conformal
  // factor differs across the three cells; the coordinate lookup pins down cell
  // identity. Repartitioning preserves cells, so these hold in both modes.
  struct KnownComponent {
    GridCoordinate coordinate;
    std::size_t column;
    double expected;
  };
  constexpr GridCoordinate cell1{
    0.98148149251937866, 0.018518491892685951, 0.018518491892685951};
  constexpr GridCoordinate cell2{
    -0.055555548518896103, 2.9444445371627808, 2.9444445371627808};
  constexpr GridCoordinate cell3{
    4.8333337306976318, 4.8333337306976318, 5.8333334922790527};

  const std::array<KnownComponent, 12> knownComponents{{
    {cell1, 0, 1.0},                   // source 0  (metric xx)
    {cell1, 6, 0.0},                   // source 6  (extrinsic curvature xx)
    {cell1, 12, 0.0},                  // source 53 (traceK)
    {cell1, 13, 0.43901168099999999},  // source 54 (conformal factor)
    {cell2, 0, 1.0},
    {cell2, 6, 0.0},
    {cell2, 12, 0.0},
    {cell2, 13, 0.79710270500000002},
    {cell3, 0, 1.0},
    {cell3, 6, 0.0},
    {cell3, 12, 0.0},
    {cell3, 13, 0.89736081099999998},
  }};

  for (const KnownComponent& known : knownComponents) {
    const double actual = gridData.value(known.coordinate, known.column);
    const double tolerance = 1.0e-10 * (1.0 + std::abs(known.expected));
    assert(std::abs(actual - known.expected) <= tolerance);
  }

  std::cout << "ReadGridFromFile() loaded " << gridData.numberOfPoints()
            << " cells with " << gridData.numberOfVariables
            << " variables each from " << inputFile << std::endl;
}

void testCompactSourceTuple() {
  static_assert(static_cast<std::size_t>(CompactField::GammaTildeXX) == 0);
  static_assert(static_cast<std::size_t>(CompactField::GammaTildeXY) == 1);
  static_assert(static_cast<std::size_t>(CompactField::GammaTildeXZ) == 2);
  static_assert(static_cast<std::size_t>(CompactField::GammaTildeYY) == 3);
  static_assert(static_cast<std::size_t>(CompactField::GammaTildeYZ) == 4);
  static_assert(static_cast<std::size_t>(CompactField::GammaTildeZZ) == 5);
  static_assert(static_cast<std::size_t>(CompactField::ATildeXX) == 6);
  static_assert(static_cast<std::size_t>(CompactField::ATildeXY) == 7);
  static_assert(static_cast<std::size_t>(CompactField::ATildeXZ) == 8);
  static_assert(static_cast<std::size_t>(CompactField::ATildeYY) == 9);
  static_assert(static_cast<std::size_t>(CompactField::ATildeYZ) == 10);
  static_assert(static_cast<std::size_t>(CompactField::ATildeZZ) == 11);
  static_assert(static_cast<std::size_t>(CompactField::TraceK) == 12);
  static_assert(static_cast<std::size_t>(CompactField::Phi) == 13);

  SourceVariableMapping sourceMapping;
  sourceMapping.conformalMetric = {17, 3, 42, 8, 31, 1};
  sourceMapping.conformalExtrinsicCurvature = {58, 27, 14, 49, 6, 35};
  sourceMapping.traceK = 53;
  sourceMapping.conformalFactor = 54;

  std::array<double, 64> sourceTuple{};
  for (std::size_t source = 0; source < sourceTuple.size(); ++source) {
    sourceTuple[source] = 1000.0 + 17.0 * static_cast<double>(source);
  }

  const auto compact = CompactSourceTuple(sourceTuple, sourceMapping);
  static_assert(compact.size() == NumberOfCompactFields);

  const std::array<std::size_t, NumberOfCompactFields> expectedSources{
    17, 3, 42, 8, 31, 1,
    58, 27, 14, 49, 6, 35,
    53, 54
  };
  for (std::size_t compactColumn = 0; compactColumn < compact.size(); ++compactColumn) {
    assert(compact[compactColumn] == sourceTuple[expectedSources[compactColumn]]);
  }

  assert(compact[static_cast<std::size_t>(CompactField::TraceK)] == sourceTuple[53]);
  assert(compact[static_cast<std::size_t>(CompactField::Phi)] == sourceTuple[54]);
  assert(GridData::numberOfVariables == 14);

  std::cout << "Source tuple compaction preserves the fixed fourteen-column contract"
            << std::endl;
}

void expectReaderMappingRejected(const SourceVariableMapping& sourceMapping,
                                 const std::string& expectedMessagePart) {
  GridData unchanged;
  unchanged.coordinates = {{1.0, 2.0, 3.0}};
  unchanged.values.assign(GridData::numberOfVariables, 7.0);

  bool rejected = false;
  try {
    ReadGridFromFile("mapping-validation-must-happen-before-opening-this-file.vtu",
                     unchanged, sourceMapping);
  } catch (const std::invalid_argument& error) {
    rejected = true;
    assert(std::string(error.what()).find(expectedMessagePart) != std::string::npos);
  }

  assert(rejected);
  assert(unchanged.coordinates.size() == 1);
  assert(unchanged.values.size() == GridData::numberOfVariables);
  assert(unchanged.values.front() == 7.0);
  assert(unchanged.isConsistent());
}

void testRequiredMappingFieldsAreRejectedBeforeFileOpen() {
  SourceVariableMapping missingMetric = completeSourceMapping();
  missingMetric.conformalMetric[2] = SourceVariableMapping::NotStored;
  expectReaderMappingRejected(missingMetric, "gammaTilde_xz is NotStored");

  SourceVariableMapping missingATilde = completeSourceMapping();
  missingATilde.conformalExtrinsicCurvature[4] = SourceVariableMapping::NotStored;
  expectReaderMappingRejected(missingATilde, "ATilde_yz is NotStored");

  SourceVariableMapping missingK = completeSourceMapping();
  missingK.traceK = SourceVariableMapping::NotStored;
  expectReaderMappingRejected(missingK, "K is NotStored");

  SourceVariableMapping missingPhi = completeSourceMapping();
  missingPhi.conformalFactor = SourceVariableMapping::NotStored;
  expectReaderMappingRejected(missingPhi, "phi is NotStored");

  SourceVariableMapping duplicated = completeSourceMapping();
  duplicated.conformalFactor = duplicated.traceK;
  expectReaderMappingRejected(duplicated, "K and phi both map to source component 53");

  std::array<double, 54> shortSourceTuple{};
  bool unavailableFieldRejected = false;
  try {
    (void)CompactSourceTuple(shortSourceTuple, completeSourceMapping());
  } catch (const std::invalid_argument& error) {
    unavailableFieldRejected = true;
    assert(std::string(error.what()).find("phi selects source component 54")
           != std::string::npos);
  }
  assert(unavailableFieldRejected);

  // Failed mappings cannot alter the fixed positions of later compact fields.
  static_assert(static_cast<std::size_t>(CompactField::TraceK) == 12);
  static_assert(static_cast<std::size_t>(CompactField::Phi) == 13);

  std::cout << "Incomplete, duplicate, and unavailable required mappings are rejected"
            << std::endl;
}

void testReadGridFromFile() {
  const std::string vtuFile = samplePath("sample_data_S.vtu");
  const std::string pvtuFile = samplePath("pieces/sample_data_S.pvtu");

  // Mode 1: single-piece .vtu, read in one shot.
  if (fileExists(vtuFile)) {
    checkSampleGrid(vtuFile);
  } else {
    std::cout << "Skipping .vtu mode: sample not found at " << vtuFile << std::endl;
  }

  // Mode 2: repartitioned .pvtu series, streamed piece by piece. Generate with:
  //   python3 repartition_vtu.py sample_data_S.vtu pieces/sample_data_S.pvtu
  if (fileExists(pvtuFile)) {
    checkSampleGrid(pvtuFile);
  } else {
    std::cout << "Skipping .pvtu mode: series not found at " << pvtuFile << std::endl;
  }
}

/**
 * Write two unit voxels side by side along x as an ASCII .vtu, with one
 * cell-data component per entry of @p names, labelled with those names when
 * @p withNames is set (as render.py does). Component c of cell i holds
 * 100 i + c, so each value shows where it was read from.
 */
void writeTwoCellVtu(const std::filesystem::path& file,
                     const std::vector<std::string>& names,
                     bool withNames) {
  std::ofstream out(file);
  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
      << "<UnstructuredGrid>\n<Piece NumberOfPoints=\"12\" NumberOfCells=\"2\">\n"
      << "<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 3; ++x) {
        out << x << " " << y << " " << z << "\n";
      }
    }
  }
  out << "</DataArray>\n</Points>\n<Cells>\n"
      << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  for (int cell = 0; cell < 2; ++cell) {
    // Voxel corner order: x fastest, then y, then z. Point (x, y, z) is x + 3 (y + 2 z).
    for (int corner = 0; corner < 8; ++corner) {
      const int x = cell + (corner & 1);
      const int y = (corner >> 1) & 1;
      const int z = (corner >> 2) & 1;
      out << x + 3 * (y + 2 * z) << " ";
    }
    out << "\n";
  }
  out << "</DataArray>\n"
      << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">8 16</DataArray>\n"
      << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">11 11</DataArray>\n"
      << "</Cells>\n<CellData Scalars=\"Unknowns\">\n"
      << "<DataArray type=\"Float64\" Name=\"Unknowns\" NumberOfComponents=\"" << names.size() << "\"";
  if (withNames) {
    for (std::size_t component = 0; component < names.size(); ++component) {
      out << " ComponentName" << component << "=\"" << names[component] << "\"";
    }
  }
  out << " format=\"ascii\">\n";
  for (int cell = 0; cell < 2; ++cell) {
    for (std::size_t component = 0; component < names.size(); ++component) {
      out << 100 * cell + static_cast<int>(component) << " ";
    }
    out << "\n";
  }
  out << "</DataArray>\n</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
  assert(out.good());
}

/** Both cells were read, and each compact column holds the component @p expectedMapping selects. */
void checkTwoCellGrid(const GridData& gridData,
                      const SourceVariableMapping& expectedMapping) {
  assert(gridData.numberOfPoints() == 2);
  assert(gridData.isConsistent());
  assert(std::abs(gridData.coordinates[0].x - 0.5) < 1.0e-12);
  assert(std::abs(gridData.coordinates[1].x - 1.5) < 1.0e-12);
  assert(std::abs(gridData.coordinates[1].y - 0.5) < 1.0e-12);
  assert(std::abs(gridData.coordinates[1].z - 0.5) < 1.0e-12);

  const auto components = expectedMapping.componentsInCompactOrder();
  for (std::size_t cell = 0; cell < 2; ++cell) {
    for (std::size_t column = 0; column < GridData::numberOfVariables; ++column) {
      const double expected = 100.0 * static_cast<double>(cell) + static_cast<double>(components[column]);
      assert(gridData.values[cell * GridData::numberOfVariables + column] == expected);
    }
  }
}

void testReadGridSelectsComponentsByName() {
  const TemporaryDirectory directory("ahf-io-tests");
  const auto names = ExaGRyPEComponentNames();
  const auto named = directory.path() / "named.vtu";
  const auto unnamed = directory.path() / "unnamed.vtu";
  writeTwoCellVtu(named, names, true);
  writeTwoCellVtu(unnamed, names, false);

  // Named components need no mapping, and a matching mapping is accepted.
  GridData byName;
  ReadGridFromFile(named.string(), byName);
  checkTwoCellGrid(byName, ExaGRyPESourceMapping());

  GridData byMapping;
  ReadGridFromFile(named.string(), byMapping, ExaGRyPESourceMapping());
  checkTwoCellGrid(byMapping, ExaGRyPESourceMapping());

  // A mapping for another layout is rejected instead of reading the wrong fields.
  bool contradictionRejected = false;
  try {
    GridData wrongLayout;
    ReadGridFromFile(named.string(), wrongLayout, LegacyCCZ4SourceMapping());
  } catch (const std::invalid_argument& error) {
    contradictionRejected = true;
    assert(std::string(error.what()).find("which the data names dzG[4], but traceK is component 23")
           != std::string::npos);
  }
  assert(contradictionRejected);

  // Without names, the reader needs a mapping and says so.
  bool missingNamesRejected = false;
  try {
    GridData withoutNames;
    ReadGridFromFile(unnamed.string(), withoutNames);
  } catch (const std::runtime_error& error) {
    missingNamesRejected = true;
    assert(std::string(error.what()).find("does not name the components") != std::string::npos);
  }
  assert(missingNamesRejected);

  GridData unnamedByMapping;
  ReadGridFromFile(unnamed.string(), unnamedByMapping, ExaGRyPESourceMapping());
  checkTwoCellGrid(unnamedByMapping, ExaGRyPESourceMapping());

  std::cout << "ReadGridFromFile selects the fields by component name and rejects a contradicting mapping"
            << std::endl;
}

void testReadGridFromPeano() {
  std::cout << "Placeholder test for peano-input ReadGridFromPeano()" << std::endl;
}

void testWriteAHToFile() {
  std::cout << "Placeholder test for output WriteAHToFile()" << std::endl;
}

void testWriteAHToVTK() {
  std::cout << "Placeholder test for output WriteAHToVTK()" << std::endl;
}

}  // namespace exahype2::ccz4::ahf::tests::io

int main() {
  using namespace exahype2::ccz4::ahf::tests::io;

  std::cout << "Apparent Horizon Finder I/O tests begin..." << std::endl;

  testCompactSourceTuple();
  testRequiredMappingFieldsAreRejectedBeforeFileOpen();
  testReadGridFromFile();
  testReadGridSelectsComponentsByName();
  testReadGridFromPeano();
  testWriteAHToFile();
  testWriteAHToVTK();

  std::cout << "All I/O tests completed." << std::endl;
  return 0;
}
