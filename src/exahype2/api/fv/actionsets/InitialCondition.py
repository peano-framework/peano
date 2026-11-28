# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractFVActionSet import AbstractFVActionSet

import peano4.api
import jinja2


class InitialCondition(AbstractFVActionSet):
    """!

    Fill a Finite Volume patch's initial condition, one volume at a time, by
    calling the solver's exahype2.api.InitPointWisely PDE term once per
    volume with that volume's centre.

    Ported from the legacy, cell-wise
    python/exahype2/solvers/fv/actionsets/InitialCondition.py, which is
    handed a whole mesh cell's centre and extent, gridIsConstructed, and
    fills the whole patch itself via the legacy initialCondition() PDE term.
    This variant instead loops over the patch's volumes itself and calls
    InitPointWisely's initPointWisely() once per volume, exactly like Flux
    and MaxEigenvalue are evaluated once per degree of freedom - @see
    exahype2.api.InitPointWisely. The loop over the volumes is a generated
    kernel (@ref exahype2.api.fv.InitialConditionsKernel). The action set
    sets the unknowns only, i.e. no auxiliary variables.

    Checkpoint-restart support (restart_from_checkpoint/loading_scheme in
    the legacy version) has deliberately not been ported: it relies on
    Solver attributes (CheckpointTimeStamp, CheckpointFilesOf...) that do
    not exist in the new exahype2.api yet.

    """

    def __init__(self, solver, guard):
        """!

        solver: exahype2.api.Solver
          Has to have an InitPointWisely PDE term registered - @see
          exahype2.api.fv.SetupSimulationOverRegularGrid, which asserts
          this before it attaches an instance of this action set.

        guard: str
          C++ boolean expression, evaluated per cell, guarding whether the
          initial condition is (re)written for that cell.

          Unlike the legacy, cell-wise version, this action set takes no
          grid_is_constructed argument: InitPointWisely fills in a degree of
          freedom from its coordinate alone, so - unlike the legacy
          cell-wise initialCondition(), which had to know whether it was
          still filling in default values during AMR mesh construction or
          the real initial condition once the grid was done - it produces a
          well-defined value regardless of how far mesh construction has
          progressed.

        """
        AbstractFVActionSet.__init__(self, solver)
        self.guard = guard

    def get_body_of_operation(self, operation_name):
        result = ""

        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            d = {
                "UNKNOWN_IDENTIFIER": self._solver._unknown_identifier(),
                "SOLVER_INSTANCE": self._solver.get_name_of_global_instance(),
                "SOLVER_NAME": self._solver.name,
            }
            result = jinja2.Template(self.TemplateInitialCondition).render(**d)

        return result

    def get_guard_of_operation(self, operation_name):
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self.guard
        return "true"

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")

    TemplateInitialCondition = """
  #if defined (OMPUnifiedSharedMemory)
    if (device != tarch::accelerator::Device::HostDevice){
      cpp::flattened::{{SOLVER_NAME}}_initialiseCentreOfPatchStateless<Abstract{{SOLVER_NAME}}>(device, fineGridCell{{UNKNOWN_IDENTIFIER}}.value, marker.x());
    else{
      std::function<void(const tarch::la::Vector<Dimensions, double>&, double* __restrict__)> initFunction =
      [&](const tarch::la::Vector<Dimensions, double>& x, double* __restrict__ I) -> void {
          repositories::{{SOLVER_INSTANCE}}.initPointWisely(x, I);
      };
      cpp::flattened::{{SOLVER_NAME}}_initialiseCentreOfPatch(fineGridCell{{UNKNOWN_IDENTIFIER}}.value, marker.x(), marker.h(), initFunction);
    }
  #else
    std::function<void(const tarch::la::Vector<Dimensions, double>&, double* __restrict__)> initFunction =
    [&](const tarch::la::Vector<Dimensions, double>& x, double* __restrict__ I) -> void {
        repositories::{{SOLVER_INSTANCE}}.initPointWisely(x, I);
    };
    cpp::flattened::{{SOLVER_NAME}}_initialiseCentreOfPatch(fineGridCell{{UNKNOWN_IDENTIFIER}}.value, marker.x(), marker.h(), initFunction);
  #endif

  logTraceOut( "touchCellFirstTime(...)---InitialCondition" );
"""
