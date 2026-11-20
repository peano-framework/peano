#ifndef APPARENT_HORIZON_FINDER_FILE_INPUT_H_
#define APPARENT_HORIZON_FINDER_FILE_INPUT_H_

#include "ahf-data.h"

#include <string>

namespace exahype2::ccz4::ahf {

/**
 * Load a grid dataset from VTK XML unstructured-grid data, keeping only the
 * variables selected by @p sourceMapping.
 *
 * The input may be either a single-piece `.vtu` (read in one shot) or a
 * multi-piece `.pvtu` series (dispatched by the `.pvtu` extension). The `.pvtu`
 * path is streamed one piece at a time with a fresh reader per piece, so peak
 * memory is bounded by one piece plus the accumulating GridData rather than by
 * the whole grid. Use repartition_vtu.py to turn a single-piece snapshot into a
 * `.pvtu` series for low-memory reads.
 *
 * The CCZ4 finite-volume output stores its unknowns as a single multi-component
 * cell-data array (one finite volume per cell). Rather than copying all
 * components, this reader stores only the source components referenced by the
 * source mapping. The reader requires all fourteen AHF fields and translates
 * them into the fixed CompactField layout. Source component numbers do not
 * survive this boundary and must never be used by interpolation.
 *
 * Output layout. GridData::coordinates[i] is the centre of cell i. GridData is
 * compact: every row has exactly fourteen columns in this fixed order:
 *   conformalMetric              xx, xy, xz, yy, yz, zz   (columns 0..5)
 *   conformalExtrinsicCurvature  xx, xy, xz, yy, yz, zz   (columns 6..11)
 *   traceK                                                (column 12)
 *   conformalFactor                                       (column 13)
 * CompactField names these columns. SourceVariableMapping names components of
 * the input tuple only.
 *
 * If the cell-data array names its components, as snapshots converted with
 * src/tarch/plotter/api/render.py do, the mapping must agree with the names
 * (see ValidateSourceMappingAgainstComponentNames).
 *
 * @param inputFile       path to a .vtu file.
 * @param gridData        output; any previous content is overwritten.
 * @param sourceMapping identifies all fourteen source tuple components, for
 *        example ExaGRyPESourceMapping() or LegacyCCZ4SourceMapping().
 * @throws std::runtime_error if the file cannot be read, holds no cell data,
 *         or a selected index is out of range.
 * @throws std::invalid_argument if the mapping is incomplete or contradicts
 *         the component names.
 */
void ReadGridFromFile(const std::string& inputFile,
                      GridData& gridData,
                      const SourceVariableMapping& sourceMapping);

/**
 * Load a grid dataset as above, selecting the fourteen AHF fields by the names
 * of the cell-data components (CCZ4SourceComponentNames), so no mapping is
 * needed.
 *
 * @throws std::runtime_error if the file cannot be read, holds no cell data,
 *         or does not name each of the fourteen components exactly once.
 */
void ReadGridFromFile(const std::string& inputFile, GridData& gridData);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_FILE_INPUT_H_
