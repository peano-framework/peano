""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api

from ..FOCCZ4Solver import FOCCZ4Solver
from .InnerCouplingGenerator import InnerCouplingGenerator


class _RungeKuttaFD4(InnerCouplingGenerator):
    """!

    Fourth-order Finite Differences with Runge-Kutta for both solvers (@ref exahype2.api.rkfd4.CentralDifferencesWithRungeKutta)

    The solvers exchange data in each Runge-Kutta stage. CCZ4 uses the lopsided advection of its shift.

    """

    def __init__(self, patch_size: int, rk_order: int):
        """!

        @param patch_size: int
          Number of grid cells per patch and axis. The same for both solvers.

        @param rk_order: int
          Order of the Runge-Kutta scheme, i.e. its number of stages (one to four).

        """
        super().__init__(patch_size)
        self.rk_order = rk_order

    def _create_scheme(self, solver):
        Scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
        if solver is self._host.ccz4:
            return Scheme(
                solver,
                self.patch_size,
                rk_order=self.rk_order,
                right_hand_side_kernel=Scheme.RightHandSideKernel.CentralDifferencesWithLopsidedAdvection,
                advection_velocity=FOCCZ4Solver.ADVECTION_VELOCITY,
                advected_quantities=solver.lopsided_advection_quantities(),
            )
        return Scheme(solver, self.patch_size, rk_order=self.rk_order)

    def _number_of_stages(self) -> int:
        return self.rk_order

    def _stage_input(self, solver, stage):
        return solver._patch if stage == 0 else solver._stage_input

    def _end_of_stage(self, solver, graph, stage):
        if stage == self.rk_order - 1:
            return [task for task in graph.sink_tasks() if not task.entity.is_global]
        name = f"{solver.name}_Project stage {stage + 1} onto faces"
        return [task for task in graph.tasks if task.name == name]


class RungeKuttaFD4InnerMultiplicative(_RungeKuttaFD4):
    """! Runge-Kutta; in each stage, the scalar field's stage follows the one of CCZ4 """

    MULTIPLICATIVE = True


class RungeKuttaFD4InnerAdditive(_RungeKuttaFD4):
    """! Runge-Kutta; in each stage, the stages of CCZ4 and the scalar field run in parallel """

    MULTIPLICATIVE = False
