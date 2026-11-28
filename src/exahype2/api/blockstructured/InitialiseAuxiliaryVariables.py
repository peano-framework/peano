# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from peano4.api.solversteps.ActionSet import ActionSet

from ..AuxiliaryVariables import code_to_initialise_auxiliary_variables


class InitialiseAuxiliaryVariables(ActionSet):
    """!

    Call the solver's exahype2.api.InitAuxiliaryVariablesPointWisely term for
    each volume or grid cell of a patch

    The action set hooks into touchCellFirstTime(). It has to run after the
    initial condition, as the term may read the unknowns. It is used by the
    setup of the Finite Volumes and of the Finite Differences.

    """

    def __init__(self, solver, cell_centre_function: str, includes: str, guard: str):
        """!

        @param solver: exahype2.api.Solver

        @param cell_centre_function: str
          C++ function which yields the centre of a volume or grid cell,
          @ref exahype2.api.AuxiliaryVariables.code_to_initialise_auxiliary_variables().

        @param includes: str
          Includes which declare cell_centre_function and the solver.

        @param guard: str
          C++ boolean expression, evaluated per cell.

        """
        super().__init__(descend_invocation_order=1, parallel=False)
        self._solver = solver
        self._cell_centre_function = cell_centre_function
        self._includes = includes
        self._guard = guard

    def get_body_of_operation(self, operation_name):
        if operation_name != ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return "\n"
        return code_to_initialise_auxiliary_variables(
            self._solver, self._cell_centre_function
        )

    def get_guard_of_operation(self, operation_name):
        if operation_name == ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self._guard
        return "true"

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")

    def user_should_modify_template(self):
        return False

    def get_includes(self):
        return self._includes
