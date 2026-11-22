#ifndef APPARENT_HORIZON_FINDER_ANALYTIC_RESIDUAL_H_
#define APPARENT_HORIZON_FINDER_ANALYTIC_RESIDUAL_H_

#include "ahf-data.h"
#include "harmonics.h"
#include "spacetimes.h"
#include "spectral-solver.h"
#include "surface-residual.h"

#include <vector>

namespace exahype2::ccz4::ahf {

  /**
   * Wrappers for the general residual assembly (surface-residual.h)
   * to work with an analytic spacetime as the geometry source. 
   * The only spacetime-specific step is the physical geometry, supplied here by
   * MetricData; everything else (synthesis, coordinate transform, surface-shape
   * geometry, spectral projection, Newton) is shared with the grid-backed path.
   * These wrappers exist so the analytic tests exercise the exact same
   * BuildSurface / MakeResidual pipeline the production solver uses.
   */

  /** BuildSurface using MetricData(spacetime, .) as the geometry source. */
  template <class Spacetime>
  InterpolatedSurface BuildAnalyticSurface(
    const SpectralBasis&       basis,
    const std::vector<double>& coefficients,
    const Spacetime&           spacetime,
    const ReferencePoint&      referencePoint
  ) {
    return BuildSurface(
      basis,
      coefficients,
      [spacetime](const Vector3& X) { return MetricData(spacetime, X); },
      referencePoint
    );
  }

  /** MakeResidual using MetricData(spacetime, .) as the geometry source. */
  template <class Spacetime>
  SpectralResidual MakeAnalyticResidual(
    const SpectralBasis&  basis,
    const Spacetime&      spacetime,
    const ReferencePoint& referencePoint
  ) {
    return MakeResidual(
      basis,
      [spacetime](const Vector3& X) { return MetricData(spacetime, X); },
      referencePoint
    );
  }

} // namespace exahype2::ccz4::ahf

#endif // APPARENT_HORIZON_FINDER_ANALYTIC_RESIDUAL_H_
