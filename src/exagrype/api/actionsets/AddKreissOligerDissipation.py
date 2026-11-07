""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

from typing import List

import exahype2.api.rkfd4.actionsets
import peano4.api


class AddKreissOligerDissipation(exahype2.api.rkfd4.actionsets.AbstractRKFD4ActionSet):
    """!

    Add the Kreiss-Oliger dissipation to the right-hand side of one Runge-Kutta stage

    The action set reads the input of the stage with its halo
    (solver._patch_with_halo) and adds the dissipation to the stage's slot
    of solver._rhs_estimates via exagrype::ccz4::addKreissOligerDissipation().
    It has to run after exahype2.api.rkfd4.actionsets.EvaluateRungeKuttaRightHandSide,
    which overwrites the slot. As every action set of the scheme has a task of
    its own, the dissipation task depends on the evaluation task. @ref
    exagrype.api.FD4WithKODissipation arranges this.

    """

    TASK_NAME = "Add Kreiss-Oliger dissipation"
    """! Name of the task which hosts the action set (@ref exahype2.api.rkfd4.CentralDifferencesWithRungeKutta) """

    def __init__(
        self, solver, stage: int, sigma: float, quantities: List[int], guard: str
    ):
        """!

        @param solver: exahype2.api.Solver
          Has to have an exagrype.api.FD4WithKODissipation as numerical
          scheme.

        @param stage: int
          Number of the Runge-Kutta stage, starting with zero.

        @param sigma: float
          Scaling of the dissipation.

        @param quantities: List[int]
          Indices of the unknowns which are subject to the dissipation.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime.

        """
        super().__init__(solver)
        self.stage = stage
        self.sigma = sigma
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
        quantities = ", ".join(str(quantity) for quantity in self.quantities)
        return f"""
  static_assert(Dimensions == 3, "The Kreiss-Oliger dissipation of ExaGRyPE is three-dimensional only");
  constexpr int dissipativeQuantities[] = {{ {quantities} }};
  ::exagrype::ccz4::addKreissOligerDissipation(
    fineGridCell{self._solver._patch_with_halo.name}.value,
    fineGridCell{self._solver._rhs_estimates.name}.value + {self.stage * unknowns * n * n * n},
    marker.h(),
    {n},
    {scheme.overlap},
    {unknowns},
    {0},
    {self.sigma!r},
    dissipativeQuantities,
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
        return super().get_includes() + '\n#include "exagrype/ccz4/Kernels.h"\n'

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
