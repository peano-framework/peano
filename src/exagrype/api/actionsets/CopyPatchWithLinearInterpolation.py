""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

from peano4.api.solversteps.ActionSet import ActionSet


class CopyPatchWithLinearInterpolation(ActionSet):
    """!

    Copy the data of one patch into another patch, with a different number of volumes per axis

    The two patches hold the same entries per volume, but they may resolve the
    mesh cell with a different number of volumes per axis, e.g. 8 Finite
    Difference points and 17 Finite Volumes. The action set hooks into
    touchCellFirstTime() and fills each volume of the target with the
    piecewise (tri)linear interpolant of the source, which is evaluated at the
    centre of the target volume. Both patches discretise the same mesh cell,
    and the values sit in the centres of their volumes.

    The interpolation uses the source patch only, i.e. it does not look into
    neighbouring cells. Targets in the outermost half source volume are
    extrapolated linearly from the two outermost source volumes. The result is
    second-order accurate for smooth data, but it is not conservative, and it
    is not continuous across cell boundaries. If both patches have the same
    size, the action set copies the data (the weights are 0 and 1).

    """

    def __init__(
        self,
        source_patch,
        target_patch,
        guard: str = "true",
        additional_includes: str = "",
    ):
        """!

        @param source_patch: peano4.api.datamodel.Patch
          The cell-associated patch the action set reads. Has to be
          cubic, without halo, with at least two volumes per axis.

        @param target_patch: peano4.api.datamodel.Patch
          The cell-associated patch the action set writes. Has to be cubic,
          without halo, and has to hold as many entries per volume as the source.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime.

        @param additional_includes: str
          Includes that the generated code needs.

        """
        super().__init__(descend_invocation_order=1, parallel=False)
        if source_patch.no_of_unknowns != target_patch.no_of_unknowns:
            raise ValueError(
                f"Source patch {source_patch.name} holds {source_patch.no_of_unknowns} entries per volume, "
                f"but target patch {target_patch.name} holds {target_patch.no_of_unknowns}"
            )
        if source_patch.dim[0] < 2:
            raise ValueError("The source patch needs at least two volumes per axis")
        self._source = source_patch
        self._target = target_patch
        self._guard = guard
        self.additional_includes = additional_includes

    def get_constructor_body(self):
        return ""

    def get_destructor_body(self):
        return ""

    def get_body_of_createGridControlEventsForNextTraversal(self):
        return "  return std::vector< peano4::grid::GridControlEvent >();\n"

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")

    def user_should_modify_template(self):
        return False

    def get_body_of_operation(self, operation_name):
        if operation_name != ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return "\n"
        source_size = self._source.dim[0]
        target_size = self._target.dim[0]
        entries = self._source.no_of_unknowns
        return f"""
  constexpr int SourceSize = {source_size};
  constexpr int TargetSize = {target_size};
  constexpr int Entries    = {entries};
  const ::exahype2::enumerator::AoSLexicographicEnumerator sourceEnumerator(1,SourceSize,0,Entries,0);
  const ::exahype2::enumerator::AoSLexicographicEnumerator targetEnumerator(1,TargetSize,0,Entries,0);
  const double* __restrict__ source = fineGridCell{self._source.name}.value;
  double* __restrict__      target = fineGridCell{self._target.name}.value;

  // Index of the lower source volume and the weight of the upper one. The weight
  // is outside [0,1] for the outermost target volumes (linear extrapolation).
  auto locate = [](int targetIndex, int& lower, double& upperWeight) -> void {{
    const double position = (targetIndex + 0.5) * SourceSize / TargetSize - 0.5;
    lower       = std::min( std::max( static_cast<int>( std::floor(position) ), 0 ), SourceSize - 2 );
    upperWeight = position - lower;
  }};

  for (int z=0; z<TargetSize; z++) {{
    int    z0;
    double wz;
    locate(z, z0, wz);
    for (int y=0; y<TargetSize; y++) {{
      int    y0;
      double wy;
      locate(y, y0, wy);
      for (int x=0; x<TargetSize; x++) {{
        int    x0;
        double wx;
        locate(x, x0, wx);
        tarch::la::Vector<Dimensions,int> targetVolume = {{x, y, z}};
        for (int entry=0; entry<Entries; entry++) {{
          double value = 0.0;
          for (int dz=0; dz<2; dz++)
          for (int dy=0; dy<2; dy++)
          for (int dx=0; dx<2; dx++) {{
            tarch::la::Vector<Dimensions,int> sourceVolume = {{x0+dx, y0+dy, z0+dz}};
            const double weight = (dx==0 ? 1.0-wx : wx) * (dy==0 ? 1.0-wy : wy) * (dz==0 ? 1.0-wz : wz);
            value += weight * source[ sourceEnumerator(0, sourceVolume, entry) ];
          }}
          target[ targetEnumerator(0, targetVolume, entry) ] = value;
        }}
      }}
    }}
  }}
"""

    def get_guard_of_operation(self, operation_name):
        if operation_name == ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self._guard
        return "true"

    def get_attributes(self):
        return ""

    def get_includes(self):
        return (
            self.additional_includes
            + """
#include <algorithm>
#include <cmath>
#include "exahype2/enumerator/AoSLexicographicEnumerator.h"
"""
        )
