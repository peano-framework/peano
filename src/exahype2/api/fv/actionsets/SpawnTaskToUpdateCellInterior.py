# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractFVActionSet import AbstractFVActionSet
from .UpdateCellInterior import UpdateCellInterior

import peano4.api


class SpawnTaskToUpdateCellInterior(AbstractFVActionSet):
    """!

    Update the patch interior with a damped time step size

    The action set takes the admissible time step size of the current time
    step, i.e. the third entry of solver._max_eigenvalue_and_time_step_sizes
    (@see exahype2.api.fv.actionsets.ReadjustTimeStepSizeForOptimisticGlobalTimestepping),
    multiplies it with damp_time_step_size, and stores the result as
    estimated time step size in the second entry. Afterwards, it updates the
    patch interior with this damped time step size (@see
    exahype2.api.fv.actionsets.UpdateCellInterior and
    exahype2.api.fv.RusanovWithOptimisticExplicitEuler). As the admissible
    time step size is never damped itself, the estimates do not shrink from
    time step to time step.

    The action set has to run before
    exahype2.api.fv.actionsets.ComputeGlobalCFLCondition, which is why
    exahype2.api.fv.RusanovWithOptimisticExplicitEuler makes the CFL
    condition's task depend on the spawn's task. This way, the order in
    which the tasks are added to the task graph does not matter. The CFL
    condition advances the solver's time stamp and time step size
    straightaway. Before it runs, the solver still holds the time stamp
    and time step size of the current time step. The patch, however, has
    already been advanced by the current time step. The patch interior
    update hence starts from the solver's time stamp plus its time step
    size.

    If the admissible time step size is zero, i.e. in the very first time
    step, there is nothing to estimate from. The action set then does not
    update the patch interior. It sets the estimated time step size
    (second entry) to the maximum double value, so the next time step
    takes the traditional route.

    ## Spawning the update as a task

    If the action set is given the cell data holding a task number, it does
    not update the patch interior straightaway. Instead, it takes a new task
    number, stores it in the cell data, and spawns the update as a task
    with this number. exahype2.api.fv.actionsets.InsertPatchInterior and
    exahype2.api.fv.actionsets.DiscardPatchInterior wait for this task.
    If there is nothing to update, the action set stores
    tarch::multicore::TaskEnumerator::NoNumber instead.

    The task captures the cell data by value. As the patch interior and the
    eigenvalue data are managed via smart pointers, the task writes into the
    cell's data. The patch, however, is handed to the task as a private copy,
    as the next time step updates the patch's shell, which the task reads,
    while the task might still be running. The time stamp and the time step
    size are evaluated before the task is spawned, as the CFL condition
    changes the solver's values afterwards.

    """

    def __init__(self, solver, damp_time_step_size, guard, task_number=None):
        """!

        @param solver: exahype2.api.Solver
          Has to have solver._patch, solver._patch_interior,
          solver._max_eigenvalue_and_time_step_sizes and
          solver.compute_kernel_call_on_patch_interior set - @see
          exahype2.api.fv.RusanovWithOptimisticExplicitEuler.

        @param damp_time_step_size: float
          Factor the solver's current time step size is scaled with to
          obtain the estimated time step size of the patch interior update.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime,
          guarding whether the patch interior of that cell is updated.

        @param task_number: peano4.api.datamodel.DaStGen2
          Cell data holding the number of the task updating the patch
          interior. If None, the update runs straightaway.

        """
        AbstractFVActionSet.__init__(self, solver)
        self.damp_time_step_size = damp_time_step_size
        self.guard = guard
        self.task_number = task_number
        if task_number is None:
            self._update_cell_interior = UpdateCellInterior(
                solver,
                guard,
                time_step_size="dampedTimeStepSize",
                time_stamp=f"repositories::{solver.get_name_of_global_instance()}.getMinTimeStamp() + "
                f"repositories::{solver.get_name_of_global_instance()}.getMinTimeStepSize()",
            )
        else:
            self._update_cell_interior = UpdateCellInterior(
                solver,
                guard,
                time_step_size="dampedTimeStepSize",
                time_stamp="timeStampOfPatchInteriorUpdate",
                input="patchCopy->data()",
            )

    def get_body_of_operation(self, operation_name):
        result = ""
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            solver = f"repositories::{self._solver.get_name_of_global_instance()}"
            data = f"fineGridCell{self._solver._max_eigenvalue_and_time_step_sizes.name}"
            update = self._update_cell_interior.get_body_of_operation(operation_name)
            if self.task_number is None:
                result = f"""
  if ( tarch::la::equals({data}.getMaxEigenvalue(2), 0.0) ) {{
    {data}.setMaxEigenvalue( 1, std::numeric_limits<double>::max() );
  }}
  else {{
    double dampedTimeStepSize = {self.damp_time_step_size} * {data}.getMaxEigenvalue(2);
    {data}.setMaxEigenvalue( 1, dampedTimeStepSize );
""" + update + """
  }
"""
            else:
                task_number = f"fineGridCell{self.task_number.name}"
                patch = f"fineGridCell{self._solver._patch.name}"
                result = f"""
  if ( tarch::la::equals({data}.getMaxEigenvalue(2), 0.0) ) {{
    {data}.setMaxEigenvalue( 1, std::numeric_limits<double>::max() );
    {task_number}.setTaskNumber( tarch::multicore::TaskEnumerator::NoNumber );
  }}
  else {{
    const double dampedTimeStepSize             = {self.damp_time_step_size} * {data}.getMaxEigenvalue(2);
    const double timeStampOfPatchInteriorUpdate = {solver}.getMinTimeStamp() + {solver}.getMinTimeStepSize();
    {data}.setMaxEigenvalue( 1, dampedTimeStepSize );

    // The next time step updates the patch's shell while the task might still run
    auto patchCopy = std::make_shared< std::vector<double> >( {patch}.Cardinality );
    for (int i=0; i<{patch}.Cardinality; i++) {{
      (*patchCopy)[i] = {patch}.value[i];
    }}

    const int taskNumber = {task_number}.getTaskEnumerator().getNumber();
    {task_number}.setTaskNumber( taskNumber );
    tarch::multicore::spawnTask(
      new tarch::multicore::TaskWithCopyOfFunctor(
        tarch::multicore::Task::DontFuse,
        tarch::multicore::Task::DefaultPriority,
        [=]() mutable {{
""" + update + f"""
        }}
      ),
      tarch::multicore::NoInDependencies,
      taskNumber
    );
  }}
"""
        return result

    def get_guard_of_operation(self, operation_name):
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self.guard
        return "true"

    def get_includes(self):
        result = super().get_includes() + """
#include <limits>
#include "tarch/la/ScalarOperations.h"
"""
        if self.task_number is not None:
            result += """
#include <memory>
#include <vector>
#include "tarch/multicore/multicore.h"
#include "tarch/multicore/Task.h"
#include "tarch/multicore/TaskEnumerator.h"
"""
        return result

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
