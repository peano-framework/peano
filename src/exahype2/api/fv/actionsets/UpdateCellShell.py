# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractFVActionSet import AbstractFVActionSet

import peano4.api


class UpdateCellShell(AbstractFVActionSet):
    """!

    Update the shell of a Finite Volume patch

    Counterpart of exahype2.api.fv.actionsets.UpdateCell which updates only
    the shell of the patch (@see
    exahype2.api.fv.RusanovWithOptimisticExplicitEuler). It reads the
    halo-augmented reconstruction solver._patch_with_halo, invokes the
    shell compute kernel, and writes the shell of solver._patch. The patch
    interior of solver._patch remains unchanged. The kernel does not return any
    eigenvalue, as exahype2.api.fv.actionsets.ComputeMaxEigenvalue determines
    it for the whole patch afterwards.

    """

    def __init__(self, solver, guard):
        """!

        @param solver: exahype2.api.Solver
          Has to have solver._patch, solver._patch_with_halo,
          and solver.compute_kernel_call_on_shell
          set - @see exahype2.api.fv.RusanovWithOptimisticExplicitEuler.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime,
          guarding whether that cell's shell is updated.

        """
        AbstractFVActionSet.__init__(self, solver)
        self.guard = guard

    def get_body_of_operation(self, operation_name):
        result = ""
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            result = f"""
  double timeStamp    = repositories::{self._solver.get_name_of_global_instance()}.getMinTimeStamp();
  double timeStepSize = repositories::{self._solver.get_name_of_global_instance()}.getMinTimeStepSize();
  ::exahype2::CellData patchData(fineGridCell{self._solver._patch_with_halo.name}.value, marker.x(), marker.h(), timeStamp, timeStepSize, fineGridCell{self._solver._patch.name}.value);
  {self._solver.compute_kernel_call_on_shell}
"""
        return result

    def get_guard_of_operation(self, operation_name):
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self.guard
        return "true"

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
