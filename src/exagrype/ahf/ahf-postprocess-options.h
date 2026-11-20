#ifndef APPARENT_HORIZON_FINDER_AHF_POSTPROCESS_OPTIONS_H_
#define APPARENT_HORIZON_FINDER_AHF_POSTPROCESS_OPTIONS_H_

#include "ah-search.h"
#include "ahf-data.h"
#include "common-horizon-search.h"

#include <optional>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf {

/** How ahf-postprocess finds the AHF fields in a snapshot. */
enum class PostprocessMapping {
  /** By the component names of the cell data (ReadGridFromFile without a mapping). */
  Names,
  /** ExaGRyPESourceMapping(). */
  ExaGRyPE,
  /** LegacyCCZ4SourceMapping(). */
  LegacyCCZ4
};

/** A black hole given with --puncture x,y,z:m. */
struct PostprocessPuncture {
  ReferencePoint position{0.0, 0.0, 0.0};
  double mass = 1.0;
};

/**
 * Command line of ahf-postprocess. Without punctures it searches for one
 * horizon, trying the initial radii in order around one centre, and stops at
 * the first that converges. With punctures it searches for each puncture's
 * horizon and then for the common horizons that FindApparentHorizons plans
 * from them:
 *
 * @dot
 * digraph PunctureSearch {
 *   rankdir=LR; node [shape=box, fontsize=10];
 *   own [label="each puncture's horizon,\nfrom radiiPerMass times m"];
 *   plan [label="pairs with gap\n<= 0.5 (m1 + m2)"];
 *   common [label="common search\nand validation"];
 *   own -> plan -> common;
 * }
 * @enddot
 */
struct PostprocessOptions {
  std::string inputFile;
  /**
   * One search writes <outputPrefix>.txt (summary) and, on success,
   * <outputPrefix>.vtp (surface). With punctures, <outputPrefix>.txt is the
   * catalogue summary, and each horizon gets its own files, named by
   * AHCatalogueFilename with outputPrefix as horizonFilenamePrefix.
   */
  std::string outputPrefix = "apparent-horizon";
  PostprocessMapping mapping = PostprocessMapping::Names;
  ReferencePoint centre{0.0, 0.0, 0.0};
  std::vector<double> initialRadii;
  std::vector<PostprocessPuncture> punctures;
  /**
   * Initial radii of each puncture's search, in units of its mass. An isolated
   * puncture of bare mass m has its horizon at coordinate radius m/2; a
   * companion makes it smaller, about 0.33 m for the equal-mass binary at
   * separation m (the Brill-Lindquist case the defaults were tried on).
   */
  std::vector<double> radiiPerMass{0.35, 0.5, 0.25};
  /**
   * Highest spherical-harmonic degree; with punctures and no --lmax it is
   * PunctureLMax instead. A common horizon near the separation where it
   * disappears is strongly distorted: for equal masses at 0.7 M, lMax 4 left a
   * relative expansion error of 0.126 and the irreducible mass 1e-4 low, lMax 8
   * 0.014 and 1e-6, for four times the run time.
   */
  int lMax = 4;
  static constexpr int PunctureLMax = 8;
  /** Interpolation support radius; without one, three times the finest cell spacing. */
  std::optional<double> supportRadius;
  double tolerance = 1.0e-8;
  bool help = false;
};

/**
 * Parse the arguments that follow the program name.
 *
 * @throws std::invalid_argument for an unknown option, a missing or malformed
 *         value, a missing --input, neither --radius nor --puncture, or
 *         --puncture together with --radius or --centre (unless --help is
 *         given).
 */
PostprocessOptions ParsePostprocessArguments(const std::vector<std::string>& arguments);

/** The usage text printed by --help and after an argument error. */
std::string PostprocessUsage(const std::string& executable);

/** The explicit source mapping selected by @p mapping; nothing for Names. */
std::optional<SourceVariableMapping> PostprocessSourceMapping(PostprocessMapping mapping);

/**
 * The smallest distance between neighbouring cell centres along a coordinate
 * axis. On a Peano grid this is the cell size of the finest level present.
 *
 * @throws std::invalid_argument if no axis has two distinct coordinates.
 */
double FinestGridSpacing(const GridData& gridData);

/**
 * The search configuration: one attempt per initial radius around the centre,
 * radius-weighted cubic interpolation within @p supportRadius, and the given
 * lMax and tolerance.
 */
AHSearchOptions MakePostprocessSearchOptions(const PostprocessOptions& options,
                                             double supportRadius);

/**
 * The catalogue configuration for punctures: black holes bh0, bh1, ... at the
 * punctures, each searched from radiiPerMass times its mass, with the
 * interpolation, lMax and tolerance of MakePostprocessSearchOptions and the
 * default candidate planning and validation.
 *
 * The solver stops at tolerance / massScale, and a common search has the
 * members' total mass as its massScale, while the validation compares the
 * residual with an absolute bound. That bound is therefore tolerance divided
 * by the smallest mass, which no common horizon's total mass can be below, so
 * that a converged common solve is never rejected for its residual.
 */
AHCommonHorizonSearchOptions MakePostprocessCatalogueOptions(
  const PostprocessOptions& options, double supportRadius);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_AHF_POSTPROCESS_OPTIONS_H_
