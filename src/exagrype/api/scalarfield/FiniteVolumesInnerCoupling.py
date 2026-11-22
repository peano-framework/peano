""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api

from .InnerCouplingGenerator import InnerCouplingGenerator


class _FiniteVolumes(InnerCouplingGenerator):
    """!

    Explicit Euler with a Rusanov flux for both solvers (@ref exahype2.api.fv.RusanovWithExplicitEuler)

    The time integrator has one stage, i.e. the solvers exchange data once per time step.

    """

    def _create_scheme(self, solver):
        return exahype2.api.fv.RusanovWithExplicitEuler(solver, self.patch_size)

    def _number_of_stages(self) -> int:
        return 1

    def _stage_input(self, solver, stage):
        return solver._patch

    def _end_of_stage(self, solver, graph, stage):
        return [task for task in graph.sink_tasks() if not task.entity.is_global]


class FiniteVolumesInnerMultiplicative(_FiniteVolumes):
    """! Finite Volumes; the scalar field's step follows the one of CCZ4 """

    MULTIPLICATIVE = True


class FiniteVolumesInnerAdditive(_FiniteVolumes):
    """! Finite Volumes; the steps of CCZ4 and the scalar field run in parallel """

    MULTIPLICATIVE = False
