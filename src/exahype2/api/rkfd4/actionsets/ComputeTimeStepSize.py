# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractRKFD4ActionSet import AbstractRKFD4ActionSet

import peano4.api


class ComputeTimeStepSize(AbstractRKFD4ActionSet):
    """!

    Determine the time step size of the next Runge-Kutta step and advance
    the solver's time stamp

    The action set supports two modes:

    - Fixed time step sizes: the next time step size is the one passed to
      the constructor.
    - Adaptive time step sizes: exahype2.api.rkfd4.actionsets.ComputeMaxEigenvalue
      reduces the maximum eigenvalue of every patch into the solver. The
      action set reads and resets it (getAndResetMaxEigenvalue()), takes the
      maximum over all ranks, and derives the next time step size from it
      through the CFL condition
      @f$ \\Delta t = C h / (d \\lambda _{max}) @f$, with h the minimal grid
      cell size, d the dimension and C the CFL constant.

    In both modes, unprepareTraversal() feeds the next time step size into
    ::exahype2::Solver's own time step bookkeeping
    (updateCell()/finishTimeStep()/startTimeStep()). The time stamp
    advances by the time step size the Runge-Kutta step has just used. The
    last time step is not truncated: it may overshoot MaxTerminalTime, and
    the simulation stops afterwards, i.e. MaxTerminalTime is a lower bound
    on the final time. The solver starts with a time step size of zero, so
    the very first Runge-Kutta step does not alter the solution: it only
    yields the first time step size.

    Global time stepping: all cells share one time stamp and time step size.
    As getMinTimeStamp()==getMaxTimeStamp() and
    getMinTimeStepSize()==getMaxTimeStepSize() always hold, a single
    updateCell() call per time step is enough.

    The action set only acts in unprepareTraversal() and touches no mesh data.
    The task that hosts it hence is a GLOBAL_END_OF_STEP task
    (@ref peano4.api.taskgraph.Task.Task.GridEntity), which the task graph
    compiler fuses into the last traversal of the time step instead of
    spending a traversal on it. Why the reduction over the cells happens
    elsewhere: a task that touches cells could not be of this kind.

    The class is derived from exahype2.api.fv.actionsets.ComputeGlobalCFLCondition,
    as rkfd4 does not depend on the Finite Volume package. Consult the
    latter for details on the reduction over ranks.

    """

    def __init__(self, solver, patch_size, cfl_constant=None, time_step_size=None):
        """!

        @param solver: exahype2.api.Solver
          With adaptive time stepping, the solver has to host the reduction of
          the maximum eigenvalue (@ref exahype2.api.rkfd4.CentralDifferencesWithRungeKutta).

        @param patch_size: int
          Number of grid cells per patch and axis. The CFL condition is
          evaluated over the grid cell size, i.e. the mesh cell size divided
          by patch_size.

        @param cfl_constant: float or None
          CFL constant for adaptive time stepping.

        @param time_step_size: float or None
          Time step size for fixed time stepping. Exactly one of
          cfl_constant and time_step_size has to be set.

        """
        AbstractRKFD4ActionSet.__init__(self, solver)
        if (cfl_constant is None) == (time_step_size is None):
            raise ValueError("Set exactly one of cfl_constant and time_step_size")
        self.patch_size = patch_size
        self.cfl_constant = cfl_constant
        self.time_step_size = time_step_size

    @property
    def is_adaptive(self) -> bool:
        return self.cfl_constant is not None

    def get_body_of_unprepareTraversal(self):
        instance = "repositories::" + self._solver.get_name_of_global_instance()
        if self.is_adaptive:
            admissible_time_step_size = f"""
  const peano4::grid::GridStatistics globalStatistics = peano4::parallel::SpacetreeSet::getInstance().getGridStatistics();
  const double                       globalMinGridCellH = tarch::la::max( globalStatistics.getMinH() ) / {self.patch_size};

  double localMaxEigenvalue  = {instance}.getAndResetMaxEigenvalue();
  double globalMaxEigenvalue = localMaxEigenvalue;
  #ifdef Parallel
  tarch::mpi::Rank::getInstance().allReduce(
    &localMaxEigenvalue, &globalMaxEigenvalue, 1, MPI_DOUBLE, MPI_MAX,
    [&]() -> void {{ tarch::services::ServiceRepository::getInstance().receiveDanglingMessages(); }}
  );
  #endif
  const double admissibleTimeStepSize = {self.cfl_constant!r} * globalMinGridCellH / globalMaxEigenvalue / Dimensions;
"""
        else:
            admissible_time_step_size = f"""
  const double admissibleTimeStepSize = {self.time_step_size!r};
"""
        return (
            admissible_time_step_size
            + f"""
  // The time stamp advances by the time step size the update used,
  // newTimeStepSize is the one the next update will use
  const double usedTimeStepSize = {instance}.getMaxTimeStepSize();
  const double newTimeStamp     = {instance}.getMaxTimeStamp() + usedTimeStepSize;
  // The last time step is not truncated, i.e. it may overshoot MaxTerminalTime
  const double newTimeStepSize  = admissibleTimeStepSize;

  {instance}.updateCell( newTimeStepSize, newTimeStamp );
  {instance}.finishTimeStep();
  {instance}.startTimeStep(
    {instance}.getMinTimeStamp(),
    {instance}.getMaxTimeStamp(),
    {instance}.getMinTimeStepSize(),
    {instance}.getMaxTimeStepSize()
  );
  logInfo( "unprepareTraversal()", "t=" << {instance}.getMaxTimeStamp() << ", new dt=" << newTimeStepSize );
"""
        )

    def get_includes(self):
        return (
            AbstractRKFD4ActionSet.get_includes(self)
            + """
#include <algorithm>
#include "tarch/mpi/Rank.h"
#include "tarch/services/ServiceRepository.h"
#include "tarch/la/VectorOperations.h"
#include "peano4/parallel/SpacetreeSet.h"
#include "peano4/grid/grid.h"
"""
        )

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
