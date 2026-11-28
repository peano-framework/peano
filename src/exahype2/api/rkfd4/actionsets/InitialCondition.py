# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractRKFD4ActionSet import AbstractRKFD4ActionSet

import peano4.api
import jinja2


class InitialCondition(AbstractRKFD4ActionSet):
    """!

    Fill a patch's initial condition, one grid cell at a time, by calling
    the solver's exahype2.api.InitPointWisely PDE term once per grid cell
    with that grid cell's centre.

    The Finite Differences of @ref exahype2.api.rkfd4 are cell-centred, i.e.
    each patch hosts patch_size^d grid cells and the unknowns are located at
    their centres. The action set is a copy of
    exahype2.api.fv.actionsets.InitialCondition, which places the unknowns
    in the very same way, as rkfd4 does not depend on the Finite Volume
    package.

    Checkpoint-restart support has deliberately not been ported: it relies
    on Solver attributes that do not exist in the new exahype2.api yet.

    """

    def __init__(self, solver, guard):
        """!

        @param solver: exahype2.api.Solver
          Has to have an InitPointWisely PDE term registered - @ref
          exahype2.api.rkfd4.SetupSimulationOverRegularGrid asserts this
          before it attaches an instance of this action set.

        @param guard: str
          C++ boolean expression, evaluated per cell, guarding whether the
          initial condition is (re)written for that cell.

        """
        AbstractRKFD4ActionSet.__init__(self, solver)
        self.guard = guard

    def get_body_of_operation(self, operation_name):
        result = ""

        if (
            operation_name
            == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
        ):
            d = {
                "NUMBER_OF_GRID_CELLS_PER_AXIS": self._solver.numerical_scheme.patch_size,
                "NUMBER_OF_UNKNOWNS": self._solver.unknowns,
                "UNKNOWN_IDENTIFIER": self._solver._unknown_identifier(),
                "SOLVER_INSTANCE": self._solver.get_name_of_global_instance(),
            }
            result = jinja2.Template(self.TemplateInitialCondition).render(**d)

        return result

    def get_guard_of_operation(self, operation_name):
        if (
            operation_name
            == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
        ):
            return self.guard
        return "true"

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")

    TemplateInitialCondition = """
  logTraceIn( "touchCellFirstTime(...)---InitialCondition" );
  parallelDfor( gridCell, {{NUMBER_OF_GRID_CELLS_PER_AXIS}} ) {
    int index = peano4::utils::dLinearised(gridCell,{{NUMBER_OF_GRID_CELLS_PER_AXIS}}) * {{NUMBER_OF_UNKNOWNS}};
    repositories::{{SOLVER_INSTANCE}}.initPointWisely(
      ::exahype2::fd::getGridCellCentre( marker.x(), marker.h(), {{NUMBER_OF_GRID_CELLS_PER_AXIS}}, gridCell),
      fineGridCell{{UNKNOWN_IDENTIFIER}}.value + index
    );
  } endParallelDfor

  logTraceOut( "touchCellFirstTime(...)---InitialCondition" );
"""
