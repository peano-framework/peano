# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from ...AuxiliaryVariables import code_to_assemble_kernel_input
import peano4.api

from .AbstractRKFD4ActionSet import AbstractRKFD4ActionSet


class EvaluateRungeKuttaRightHandSide(AbstractRKFD4ActionSet):
    """!

    Evaluate the right-hand side of one Runge-Kutta stage over a patch

    The action set hands the input of the stage with its halo
    (solver._patch_with_halo) to the fourth-order Finite Differences kernel
    (@ref exahype2.api.rkfd4.FD4Kernels) and lets it write the right-hand
    side into the stage's slot of solver._rhs_estimates. The kernel sees
    the time stamp of the stage, i.e. @f$ t^n + c_k \\Delta t @f$, and the
    time step size of the whole Runge-Kutta step.

    Global time stepping: the time stamp and the time step size are those
    of the solver (getMinTimeStamp() and getMinTimeStepSize()).

    """

    def __init__(self, solver, stage, guard):
        """!

        @param solver: exahype2.api.Solver
          Has to have an exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
          as numerical scheme which has already constructed its kernel call,
          i.e. solver.compute_kernel_call.

        @param stage: int
          Number of the Runge-Kutta stage, starting with zero.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime.

        """
        AbstractRKFD4ActionSet.__init__(self, solver)
        self.stage = stage
        self.guard = guard

    def get_body_of_operation(self, operation_name):
        if (
            operation_name
            != peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
        ):
            return ""
        scheme = self._solver.numerical_scheme
        n = scheme.patch_size
        unknowns = self._solver.unknowns
        relative_time_step_size = scheme.butcher_tableau.relative_time_step_sizes()[
            self.stage
        ]
        instance = self._solver.get_name_of_global_instance()
        return f"""
  const double timeStepSize = repositories::{instance}.getMinTimeStepSize();
  const double timeStamp    = repositories::{instance}.getMinTimeStamp() + {relative_time_step_size!r} * timeStepSize;
  #if Dimensions==2
  double* estimate = fineGridCell{self._solver._rhs_estimates.name}.value + {self.stage * unknowns * n * n};
  #else
  double* estimate = fineGridCell{self._solver._rhs_estimates.name}.value + {self.stage * unknowns * n * n * n};
  #endif
  {code_to_assemble_kernel_input(self._solver, "kernelIn", f"fineGridCell{self._solver._patch_with_halo.name}.value", scheme.HALO_SIZE)}
  ::exahype2::CellData patchData({"kernelIn.data()" if self._solver.auxiliary_variables else f"fineGridCell{self._solver._patch_with_halo.name}.value"}, marker.x(), marker.h(), timeStamp, timeStepSize, estimate);
  {self._solver.compute_kernel_call}
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
