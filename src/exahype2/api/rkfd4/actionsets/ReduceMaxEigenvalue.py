# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api

from .AbstractRKFD4ActionSet import AbstractRKFD4ActionSet


class ReduceMaxEigenvalue(AbstractRKFD4ActionSet):
    """!

    Reduce the maximum eigenvalues of the cells into the solver

    exahype2.api.rkfd4.actionsets.ComputeMaxEigenvalue stores the maximum
    eigenvalue per cell in solver._max_eigenvalue. The action set visits each
    cell and hands this value to the solver's reduceMaxEigenvalue(), which
    takes the maximum over all cells and threads of the rank. It writes the
    global datum from which
    exahype2.api.rkfd4.actionsets.ComputeTimeStepSize derives the next time
    step size at the end of the time step. The action set does not touch
    anything else, i.e. neither the solution nor the faces.

    """

    def __init__(self, solver, guard):
        """!

        @param solver: exahype2.api.Solver
          Has to have an exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
          with adaptive time stepping as numerical scheme.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime.
          Has to match the guard of exahype2.api.rkfd4.actionsets.ComputeMaxEigenvalue.

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
        return f"""
  repositories::{instance}.reduceMaxEigenvalue( fineGridCell{self._solver._max_eigenvalue.name}.getMaxEigenvalue(0) );
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
