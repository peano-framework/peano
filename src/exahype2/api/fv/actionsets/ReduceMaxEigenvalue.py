# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractFVActionSet import AbstractFVActionSet

import peano4.api


class ReduceMaxEigenvalue(AbstractFVActionSet):
    """!

    Reduce the maximum eigenvalues of the cells into the solver

    exahype2.api.fv.actionsets.ComputeMaxEigenvalue stores the maximum
    eigenvalue per cell. The action set visits each cell and hands the
    first entry of this value to the solver's reduceMaxEigenvalue(), which
    takes the maximum over all cells and threads of the rank (@ref
    exahype2.api.fv.MaxEigenvalueReduction). It writes the global datum from
    which exahype2.api.fv.actionsets.ComputeGlobalCFLCondition derives the
    next time step size at the end of the time step. The action set does not
    touch anything else.

    """

    def __init__(self, solver, guard, max_eigenvalue=None):
        """!

        @param solver: exahype2.api.Solver
          Has to host the reduction (@ref exahype2.api.fv.MaxEigenvalueReduction).

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime.
          Has to match the guard of exahype2.api.fv.actionsets.ComputeMaxEigenvalue.

        @param max_eigenvalue: peano4.api.datamodel.DaStGen2
          The cell data holding the maximum eigenvalue (first entry).
          Defaults to solver._max_eigenvalue.

        """
        AbstractFVActionSet.__init__(self, solver)
        self.guard = guard
        self._max_eigenvalue = (
            max_eigenvalue if max_eigenvalue is not None else solver._max_eigenvalue
        )

    def get_body_of_operation(self, operation_name):
        if (
            operation_name
            != peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
        ):
            return ""
        return f"""
  repositories::{self._solver.get_name_of_global_instance()}.reduceMaxEigenvalue( fineGridCell{self._max_eigenvalue.name}.getMaxEigenvalue(0) );
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
