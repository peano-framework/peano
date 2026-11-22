#ifndef APPARENT_HORIZON_FINDER_OUTPUT_H_
#define APPARENT_HORIZON_FINDER_OUTPUT_H_

#include "ah-finder-result.h"
#include "ah-finder-options.h"
#include "ah-search.h"
#include "merger-orchestration.h"
#include "common-horizon-search.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf {

struct AHGuess;

/** Resolution of the output mesh only; it does not affect the AH solve. */
struct AHVisualizationOptions {
  int numberOfRhoPoints = 64;
  int numberOfSigmaPoints = 33;
};

/** Dense closed surface reconstructed directly from the spectral coefficients. */
struct AHVisualizationSurface {
  std::vector<std::array<double, 3>> points;
  std::vector<double> radius;
  std::vector<std::array<std::size_t, 3>> triangles;
};

/** Filenames and visualisation resolution for the single-horizon façade. */
struct AHOutputOptions {
  std::string summaryFilename = "apparent-horizon.txt";
  std::string surfaceFilename = "apparent-horizon.vtp";
  AHVisualizationOptions visualization{};
};

/**
 * Output configuration for a search containing several physical horizons.
 *
 * The aggregate diagnostic is written to summaryFilename. For every
 * successful target i, the per-horizon files are derived deterministically as
 *
 *   horizonFilenamePrefix + "-" + i + "-" + target-id + ".txt/.vtp".
 *
 * Characters in target-id which are unsafe in a filename are replaced by '_'.
 */
struct AHMultiHorizonOutputOptions {
  std::string summaryFilename = "apparent-horizons.txt";
  std::string horizonFilenamePrefix = "apparent-horizon";
  AHVisualizationOptions visualization{};
};

/** Validate output filenames and visualisation resolution before solving. */
void ValidateAHOutputOptions(const AHOutputOptions& options);

/** Write all solver and reconstructed-surface diagnostics as deterministic text. */
void WriteAHSummary(const AHFinderResult& result, const std::string& filename);

/**
 * Evaluate the retained spectral representation on an independent dense grid.
 * The grid has one vertex at each pole and periodic azimuthal rings between.
 */
AHVisualizationSurface BuildAHVisualizationSurface(
  const AHFinderResult& result,
  const AHVisualizationOptions& options = {}
);

/** Write a converged dense horizon surface as VTK XML PolyData (.vtp). */
void WriteAHSurfaceToVTP(
  const AHFinderResult& result,
  const std::string& filename,
  const AHVisualizationOptions& options = {}
);

/**
 * Complete single-horizon user workflow.
 *
 * The grid is referenced by one persistent CartesianPointInterpolator for the
 * solve and final reconstruction. A summary is always written after a solver
 * result is obtained. The normal VTP horizon is written only for a converged,
 * reconstructable surface.
 */
AHFinderResult WriteAHToFile(
  const GridData& gridData,
  const AHFinderOptions& finderOptions,
  const AHOutputOptions& outputOptions = {}
);

/**
 * Write ordered diagnostics for every executed attempt and, on success, the
 * selected horizon diagnostics.
 */
void WriteAHSearchSummary(
  const AHSearchResult& result,
  const std::string& filename
);

/** Write every executed target/attempt and any selected horizon diagnostics. */
void WriteAHMultiHorizonSummary(
  const AHMultiHorizonResult& result,
  const std::string& filename
);

/**
 * Multi-attempt user workflow. Attempts are paired entries tried in vector
 * order; the first converged, reconstructable horizon stops the search.
 */
AHSearchResult WriteAHToFile(
  const GridData& gridData,
  const AHSearchOptions& searchOptions,
  const AHOutputOptions& outputOptions = {}
);

/**
 * Complete multi-horizon workflow. All targets are searched. The aggregate
 * summary is always written after the searches complete; each converged,
 * reconstructable target receives its own numerical summary and VTP surface.
 */
AHMultiHorizonResult WriteAHToFile(
  const GridData& gridData,
  const AHMultiHorizonOptions& searchOptions,
  const AHMultiHorizonOutputOptions& outputOptions = {}
);

/** Merger-aware binary workflow with the same per-target output convention. */
AHBinaryMergerResult WriteAHToFile(
  const GridData& gridData,
  const AHBinaryMergerOptions& mergerOptions,
  const AHMultiHorizonOutputOptions& outputOptions = {}
);

/**
 * The per-horizon file the catalogue workflow writes for entry @p index of
 * @p kind ("individual" or "common"):
 *
 *   horizonFilenamePrefix + "-" + kind + "-" + index + "-" + id + extension,
 *
 * with the characters of @p id that are unsafe in a filename replaced by '_'.
 */
std::string AHCatalogueFilename(const AHMultiHorizonOutputOptions& options,
                                const char* kind, std::size_t index,
                                const std::string& id, const char* extension);

/** Catalogue summary retaining individual failures and common validation. */
void WriteAHHorizonCatalogueSummary(
  const AHHorizonCatalogue& result,
  const std::string& filename
);

/** N-black-hole individual/common workflow and validated output façade. */
AHHorizonCatalogue WriteAHToFile(
  const GridData& gridData,
  const AHCommonHorizonSearchOptions& searchOptions,
  const AHMultiHorizonOutputOptions& outputOptions = {}
);

/**
 * Convert the ASCII apparent-horizon hypersurface output into a VTK file.
 */
void WriteAHToVTK(const std::string& inputFile, const std::string& outputFile);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_OUTPUT_H_
