# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .InitialCondition import InitialCondition
from ...AuxiliaryVariables import code_to_initialise_auxiliary_variables
from ...InitAuxiliaryVariablesPointWisely import InitAuxiliaryVariablesPointWisely

import peano4.api


class InitialiseNewCells(InitialCondition):
    """!

    Set the initial condition within cells which the mesh refinement creates

    Same as InitialCondition, i.e. the action set invokes the solver's
    exahype2.api.InitPointWisely term per volume, followed by its
    exahype2.api.InitAuxiliaryVariablesPointWisely term if it has one, but it plugs into
    createCell() instead of touchCellFirstTime(). It hence initialises
    exactly those cells which a mesh sweep adds to the mesh, while existing
    cells keep their data.

    """

    def __init__(self, solver, guard):
        """!

        @param solver: exahype2.api.Solver
          Has to have an InitPointWisely PDE term registered.

        @param guard: str
          C++ boolean expression, evaluated per new cell.

        """
        super().__init__(solver, guard)

    def get_body_of_operation(self, operation_name):
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_CREATE_CELL:
            auxiliary_variables = ""
            if any(
                isinstance(term, InitAuxiliaryVariablesPointWisely)
                for term in self._solver.pde_terms
            ):
                auxiliary_variables = code_to_initialise_auxiliary_variables(
                    self._solver, "::exahype2::fv::getVolumeCentre"
                )
            return (
                super()
                .get_body_of_operation(
                    peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
                )
                .replace(
                    "touchCellFirstTime(...)---InitialCondition",
                    "createCell(...)---InitialiseNewCells",
                )
                + auxiliary_variables
            )
        return ""

    def get_guard_of_operation(self, operation_name):
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_CREATE_CELL:
            return self.guard
        return "true"

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
