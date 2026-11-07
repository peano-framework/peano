# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from ...AuxiliaryVariables import code_to_assemble_kernel_input
import peano4.api

from .AbstractRKFD4ActionSet import AbstractRKFD4ActionSet


class ComputeMaxEigenvalue(AbstractRKFD4ActionSet):
    """!

    Store the maximum eigenvalue over all grid cells of a patch

    The action set runs after the final linear combination of a Runge-Kutta
    step, i.e. it sees the solution of the next time step in solver._patch.
    It stores the maximum eigenvalue in the cell data solver._max_eigenvalue.
    It does nothing else: the reduction over the cells is a task of its own
    (@ref exahype2.api.rkfd4.actionsets.ReduceMaxEigenvalue), and
    exahype2.api.rkfd4.actionsets.ComputeTimeStepSize derives the next time
    step size from its outcome. Why: The eigenvalue per cell stays available
    for others, and the cell update, the eigenvalue and the reduction are
    independent steps of the task graph that the compiler can overlap.

    """

    def __init__(self, solver, guard):
        """!

        @param solver: exahype2.api.Solver
          Has to have an exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
          with adaptive time stepping as numerical scheme, which has already
          constructed its kernel calls.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime.

        """
        AbstractRKFD4ActionSet.__init__(self, solver)
        self.guard = guard

    def get_body_of_operation(self, operation_name):
        if (
            operation_name
            != peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
        ):
            return ""
        instance = self._solver.get_name_of_global_instance()
        unknowns = f"fineGridCell{self._solver._patch.name}.value"
        return f"""
  {code_to_assemble_kernel_input(self._solver, "kernelIn", unknowns, 0, lambda auxiliary: f"fineGridCell{auxiliary.patch.name}.value")}
  ::exahype2::CellData patchData(
    {"kernelIn.data()" if self._solver.auxiliary_variables else unknowns},
    marker.x(),
    marker.h(),
    repositories::{instance}.getMinTimeStamp(),
    repositories::{instance}.getMinTimeStepSize(),
    fineGridCell{self._solver._patch.name}.value
  );
  {self._solver.numerical_scheme.max_eigenvalue_kernel_call}
  fineGridCell{self._solver._max_eigenvalue.name}.setMaxEigenvalue( 0, patchData.maxEigenvalue[0] );
"""

    def get_guard_of_operation(self, operation_name):
        if (
            operation_name
            == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
        ):
            return self.guard
        return "true"

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
