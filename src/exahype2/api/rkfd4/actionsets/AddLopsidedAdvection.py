# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api

from .AbstractRKFD4ActionSet import AbstractRKFD4ActionSet


class AddLopsidedAdvection(AbstractRKFD4ActionSet):
    """!

    Add advection terms with lopsided stencils to the right-hand side of one Runge-Kutta stage

    The action set reads the input of the stage with its halo
    (solver._patch_with_halo) and adds the advection terms to the stage's
    slot of solver._rhs_estimates via
    exahype2::fd::addLopsidedAdvection(). It has to run after
    exahype2.api.rkfd4.actionsets.EvaluateRungeKuttaRightHandSide, which overwrites
    the slot, within the same task. @ref
    exahype2.api.rkfd4.CentralDifferencesWithRungeKutta arranges this if
    the user selects the kernel CentralDifferencesWithLopsidedAdvection.

    """

    TASK_NAME = "Add lopsided advection"
    """! Name of the task which hosts the action set (@ref exahype2.api.rkfd4.CentralDifferencesWithRungeKutta) """

    def __init__(self, solver, stage, velocity, quantities, guard):
        """!

        @param solver: exahype2.api.Solver
          Has to have an exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
          as numerical scheme.

        @param stage: int
          Number of the Runge-Kutta stage, starting with zero.

        @param velocity: List[int]
          Indices of the entries of Q which hold the velocity, one per
          dimension.

        @param quantities: List[int]
          Indices of the unknowns which are advected.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime.

        """
        super().__init__(solver)
        self.stage = stage
        self.velocity = velocity
        self.quantities = quantities
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
        velocity = ", ".join(str(index) for index in self.velocity)
        quantities = ", ".join(str(quantity) for quantity in self.quantities)
        return f"""
  constexpr int advectionVelocity[] = {{ {velocity} }};
  constexpr int advectedQuantities[] = {{ {quantities} }};
  static_assert(sizeof(advectionVelocity) / sizeof(int) == Dimensions, "The advection velocity needs one entry per dimension");
  #if Dimensions==2
  double* estimate = fineGridCell{self._solver._rhs_estimates.name}.value + {self.stage * unknowns * n * n};
  #else
  double* estimate = fineGridCell{self._solver._rhs_estimates.name}.value + {self.stage * unknowns * n * n * n};
  #endif
  ::exahype2::fd::addLopsidedAdvection(
    fineGridCell{self._solver._patch_with_halo.name}.value,
    estimate,
    marker.h(),
    {n},
    {scheme.overlap},
    {unknowns},
    {0},
    advectionVelocity,
    advectedQuantities,
    {len(self.quantities)}
  );
"""

    def get_guard_of_operation(self, operation_name):
        if (
            operation_name
            == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
        ):
            return self.guard
        return "true"

    def get_includes(self):
        return super().get_includes() + '\n#include "exahype2/fd/LopsidedAdvection.h"\n'

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
