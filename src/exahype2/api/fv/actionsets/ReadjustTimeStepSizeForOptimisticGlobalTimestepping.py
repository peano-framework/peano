# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractFVActionSet import AbstractFVActionSet

import peano4.api


class ReadjustTimeStepSizeForOptimisticGlobalTimestepping(AbstractFVActionSet):
    """!

    Validate the estimated time step size of an optimistic patch interior update

    The second entry of solver._max_eigenvalue_and_time_step_sizes holds the
    time step size the patch interior has been updated with optimistically
    (@see exahype2.api.fv.actionsets.SpawnTaskToUpdateCellInterior). The
    action set compares it to the admissible time step size, i.e. the time
    step size the solver would use if it had not been enforced to use a
    smaller one (@see exahype2::Solver::getAdmissibleTimeStepSize()):

    - If the admissible time step size is zero, i.e. in the very first time
      step, there is no estimate yet. The action set sets the third entry
      to zero and does not read the second entry.
    - If the estimate is bigger than the admissible time step size, we have
      overestimated the admissible time step size. The action set
      invalidates the estimate, i.e. sets it to the maximum double value.
    - If the estimate is smaller than or equal to the admissible time step
      size, subject to floating point accuracy (tarch::la::equals()), the
      action set enforces the estimate as the solver's minimum and maximum
      time step size (@see exahype2::Solver::enforceTimeStepSize()).

    In the latter two cases, the action set stores the admissible time step
    size in the third entry. It is the time step size the solver could have
    chosen, whether or not the estimate is enforced.

    """

    def __init__(self, solver, guard):
        """!

        @param solver: exahype2.api.Solver
          Has to have solver._max_eigenvalue_and_time_step_sizes set - @see
          exahype2.api.fv.RusanovWithOptimisticExplicitEuler.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime,
          guarding whether that cell's estimate is checked.

        """
        AbstractFVActionSet.__init__(self, solver)
        self.guard = guard

    def get_body_of_operation(self, operation_name):
        result = ""
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            solver = f"repositories::{self._solver.get_name_of_global_instance()}"
            data = f"fineGridCell{self._solver._max_eigenvalue_and_time_step_sizes.name}"
            result = f"""
  const double admissibleTimeStepSize = {solver}.getAdmissibleTimeStepSize();
  if ( tarch::la::equals(admissibleTimeStepSize, 0.0) ) {{
    {data}.setMaxEigenvalue(2, 0.0);
  }}
  else {{
    const double estimatedTimeStepSize = {data}.getMaxEigenvalue(1);
    if (estimatedTimeStepSize < admissibleTimeStepSize or tarch::la::equals(estimatedTimeStepSize, admissibleTimeStepSize)) {{
      {solver}.enforceTimeStepSize(estimatedTimeStepSize);
    }}
    else {{
      {data}.setMaxEigenvalue(1, std::numeric_limits<double>::max());
    }}
    {data}.setMaxEigenvalue(2, admissibleTimeStepSize);
  }}
"""
        return result

    def get_guard_of_operation(self, operation_name):
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self.guard
        return "true"

    def may_run_in_background_task(self):
        """!

        Never spawn this action set as a task

        The check is a handful of floating point comparisons per cell, i.e.
        far too cheap to justify the overhead of a task. Task graph compilers
        such as @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer
        therefore keep it within the mesh traversal. It does not touch the
        patch data, so it does not depend on the reconstruction task which
        the producer spawns before.

        """
        return False

    def get_includes(self):
        return super().get_includes() + """
#include <limits>
#include "tarch/la/ScalarOperations.h"
"""

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
