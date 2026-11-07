# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractFVActionSet import AbstractFVActionSet

from ...AuxiliaryVariables import code_to_assemble_kernel_input
import peano4.api


class ComputeMaxEigenvalue(AbstractFVActionSet):
    """!

    Store the maximum eigenvalue over the volumes of a patch

    The action set runs after the patch update, i.e. it sees the updated
    solution in solver._patch. It invokes the eigenvalue kernel, which is a
    kernel of its own and not a by-product of the time step kernel, and
    stores the outcome in the first entry of the cell data
    max_eigenvalue (by default solver._max_eigenvalue). It does nothing
    else: the reduction over the cells is a task of its own
    (@ref exahype2.api.fv.actionsets.ReduceMaxEigenvalue), and
    exahype2.api.fv.actionsets.ComputeGlobalCFLCondition derives the next
    time step size from its outcome. The task graph compiler can fuse the
    update, the eigenvalue computation and the reduction into one mesh
    traversal.

    """

    def __init__(self, solver, guard, max_eigenvalue=None):
        """!

        @param solver: exahype2.api.Solver
          Has to have solver._patch and solver._max_eigenvalue set, and
          solver.numerical_scheme.max_eigenvalue_kernel_call constructed -
          @see exahype2.api.fv.RusanovWithExplicitEuler.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime.

        @param max_eigenvalue: peano4.api.datamodel.DaStGen2
          The cell data the maximum eigenvalue goes into (first entry).
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
        solver = self._solver
        unknowns = f"fineGridCell{solver._patch.name}.value"
        has_parameters = len(solver.auxiliary_variables) > 0
        kernel_in = "kernelIn.data()" if has_parameters else unknowns
        return f"""
  {code_to_assemble_kernel_input(solver, "kernelIn", unknowns, 0, lambda auxiliary: f"fineGridCell{auxiliary.patch.name}.value")}
  ::exahype2::CellData patchData(
    {kernel_in},
    marker.x(),
    marker.h(),
    repositories::{solver.get_name_of_global_instance()}.getMinTimeStamp(),
    repositories::{solver.get_name_of_global_instance()}.getMinTimeStepSize(),
    {unknowns}
  );
  {solver.numerical_scheme.max_eigenvalue_kernel_call}
  fineGridCell{self._max_eigenvalue.name}.setMaxEigenvalue( 0, patchData.maxEigenvalue[0] );
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
