# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractFVActionSet import AbstractFVActionSet
from .DiscardPatchInterior import wait_for_patch_interior_update, WAIT_INCLUDES

import peano4.api


class InsertPatchInterior(AbstractFVActionSet):
    """!

    Insert the patch interior into the patch

    Copies solver._patch_interior, i.e. the result of
    exahype2.api.fv.actionsets.UpdateCellInterior, into the patch interior
    of solver._patch. The patch
    then is the updated patch of the whole cell, so
    exahype2.api.fv.actionsets.ComputeMaxEigenvalue can determine its
    maximum eigenvalue (@see
    exahype2.api.fv.RusanovWithOptimisticExplicitEuler).

    If the patch interior is updated by a task (@see
    exahype2.api.fv.actionsets.SpawnTaskToUpdateCellInterior), the action
    set first waits for this task.

    """

    def __init__(self, solver, patch_size, overlap, guard, task_number=None):
        """!

        @param solver: exahype2.api.Solver
          Has to have solver._patch, solver._patch_interior and
          solver._max_eigenvalue_and_time_step_sizes set - @see
          exahype2.api.fv.RusanovWithOptimisticExplicitEuler.

        @param patch_size: int
          Number of Finite Volumes per axis within one patch.

        @param overlap: int
          Thickness of the shell, i.e. the patch interior has
          patch_size-2*overlap Finite Volumes per axis.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime,
          guarding whether that cell's patch interior is inserted.

        @param task_number: peano4.api.datamodel.DaStGen2
          Cell data holding the number of the task updating the patch
          interior. If given, the action set waits for this task before it
          copies the patch interior. None if the patch interior is not
          updated by a task.

        """
        AbstractFVActionSet.__init__(self, solver)
        self.task_number = task_number
        self.patch_size = patch_size
        self.overlap = overlap
        self.guard = guard

    def get_body_of_operation(self, operation_name):
        result = ""
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            unknowns = self._solver.unknowns
            interior_size = self.patch_size - 2 * self.overlap
            if self.task_number is not None:
                result = wait_for_patch_interior_update(self._solver, self.task_number)
            result += f"""
  dfor(interiorVolume,{interior_size}) {{
    tarch::la::Vector<Dimensions,int> patchVolume = interiorVolume + tarch::la::Vector<Dimensions,int>({self.overlap});
    int interiorVolumeSerialised = peano4::utils::dLinearised(interiorVolume,{interior_size});
    int patchVolumeSerialised    = peano4::utils::dLinearised(patchVolume,{self.patch_size});
    for (int j=0; j<{unknowns}; j++) {{
      fineGridCell{self._solver._patch.name}.value[patchVolumeSerialised*{unknowns}+j] = fineGridCell{self._solver._patch_interior.name}.value[interiorVolumeSerialised*{unknowns}+j];
    }}
  }}
"""
        return result

    def get_guard_of_operation(self, operation_name):
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self.guard
        return "true"

    def get_includes(self):
        result = super().get_includes() + """
#include "peano4/utils/Loop.h"
"""
        if self.task_number is not None:
            result += WAIT_INCLUDES
        return result

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
