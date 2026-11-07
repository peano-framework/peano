# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractFVActionSet import AbstractFVActionSet

import peano4.api


class ComputeGlobalCFLCondition(AbstractFVActionSet):
    """!

    Derive the next global time step size from the CFL condition.

    exahype2.api.fv.actionsets.ComputeMaxEigenvalue stores the maximum
    eigenvalue of every patch per cell, and
    exahype2.api.fv.actionsets.ReduceMaxEigenvalue reduces these values into
    the solver. This action set acts in unprepareTraversal() only: it reads
    and resets the solver's reduced maximum eigenvalue, turns it into the
    next time step size, and feeds it into ::exahype2::Solver's own time
    step bookkeeping (updateCell()/finishTimeStep()/startTimeStep()). There
    is deliberately no separate, solver-level time stamp/time step size
    state on top of ::exahype2::Solver.

    The action set touches no mesh data. The task that hosts it hence is a
    GLOBAL_END_OF_STEP task (@ref peano4.api.taskgraph.Task.Task.GridEntity),
    which the task graph compiler fuses into the last traversal of the time
    step instead of spending a traversal on it.

    Global time stepping: all cells share one time stamp and time step
    size, which are taken from the solver (getMinTimeStamp() and
    getMinTimeStepSize()) rather than from per-cell bookkeeping.

    The kernel updates all dimensions in one explicit Euler step, so the CFL
    condition reads dt * sum_d lambda_d/h_d <= 1, i.e. the admissible time
    step size is cfl_constant * h / (Dimensions * lambda_max), with h the
    minimal volume size and lambda_max the maximum eigenvalue.

    As this is global time stepping, getMinTimeStamp()==getMaxTimeStamp()
    and getMinTimeStepSize()==getMaxTimeStepSize() always hold. Hence a
    single updateCell() call per time step, not one per cell, is enough to
    keep both the min and the max in sync.

    """

    def __init__(self, solver, cfl_constant, patch_size):
        """!

        @param solver: exahype2.api.Solver
          Has to host the reduction of the maximum eigenvalue (@ref
          exahype2.api.fv.MaxEigenvalueReduction).

        @param cfl_constant: float
          Safety factor the admissible time step size is scaled down by,
          relative to the theoretical CFL=1 stability limit (which already
          accounts for the number of dimensions, see class docstring) - @see
          exahype2.api.fv.RusanovWithExplicitEuler's own cfl_constant
          parameter, which this is meant to be passed straight through
          from.

        @param patch_size: int
          Number of Finite Volumes per patch and dimension. The CFL
          condition is evaluated over the volume size, i.e. the cell size
          divided by patch_size.

        """
        AbstractFVActionSet.__init__(self, solver)
        self.cfl_constant = cfl_constant
        self.patch_size = patch_size

    def get_body_of_unprepareTraversal(self):
        """!

        Called once per rank, after the sweep - the one safe place both to
        read the solver's reduced maximum eigenvalue (nothing can still be
        writing it) and to issue an MPI reduction:
        tarch::mpi::Rank::allReduce()'s own documentation requires "only one
        thread per rank" to ever call it, which a once-per-rank static method
        already guarantees.

        This combines two different kinds of "global", reduced two different
        ways:

        - The mesh size comes out of peano4::grid::GridStatistics, which
          peano4::parallel::SpacetreeSet reduces over all ranks at the end
          of every traversal (@ref page_grid_statistics). Every rank hence
          reads the same, global value. GridStatistics holds the cell size;
          the CFL condition needs the volume size, i.e. the cell size
          divided by the patch size.
        - The solver's maximum eigenvalue is only a local, per-rank
          accumulation - that one *is* reduced directly via
          tarch::mpi::Rank::allReduce(..., MPI_MAX, ...).

        Once newTimeStepSize is known, it folds it into the solver's own
        time step bookkeeping (see this class' docstring for why a single
        updateCell() call, not one per cell, is correct for this solver).
        newTimeStepSize is not truncated by MaxTerminalTime: the last time
        step may overshoot the end time, and the simulation stops afterwards,
        i.e. MaxTerminalTime is a lower bound on the final time. The time
        stamp is advanced by the time step size the update used
        (getMaxTimeStepSize() before updateCell()), not by newTimeStepSize,
        which only the next update uses. The very first time step therefore
        runs with time step size 0 and leaves the time stamp at 0: it only
        yields the first admissible time step size.

        - updateCell(newTimeStepSize, newTimeStamp) folds the new values
          into exahype2::Solver's ...UpdatedThisTimeStep accumulators.
        - finishTimeStep() then MPI-all-reduces that accumulator across
          ranks - a no-op given newTimeStepSize/newTimeStamp are already
          identical on every rank, but calling it anyway keeps the full
          ::exahype2::Solver lifecycle honest for any future override that
          relies on it actually running.
        - startTimeStep(...) immediately promotes that same accumulator into
          the stable _minTimeStamp/_maxTimeStamp/_minTimeStepSize/
          _maxTimeStepSize fields getMinTimeStamp()/.../getMaxTimeStepSize()
          expose, so there is no one-phase lag: the very first TimeStep
          phase must already leave a genuinely positive getMaxTimeStepSize()
          behind for the generated main loop's
          tarch::la::greater(maxTimeStepSize,0.0) guard to see, or the
          simulation could never advance past its first iteration.

        """
        return f"""
  const peano4::grid::GridStatistics globalStatistics = peano4::parallel::SpacetreeSet::getInstance().getGridStatistics();
  const double                       globalMinVolumeH = tarch::la::max( globalStatistics.getMinH() ) / {self.patch_size};

  double localMaxEigenvalue  = repositories::{self._solver.get_name_of_global_instance()}.getAndResetMaxEigenvalue();
  double globalMaxEigenvalue = localMaxEigenvalue;
  #ifdef Parallel
  tarch::mpi::Rank::getInstance().allReduce(
    &localMaxEigenvalue, &globalMaxEigenvalue, 1, MPI_DOUBLE, MPI_MAX,
    [&]() -> void {{ tarch::services::ServiceRepository::getInstance().receiveDanglingMessages(); }}
  );
  #endif

  // The time stamp advances by the time step size the update used,
  // newTimeStepSize is the one the next update will use
  const double usedTimeStepSize = repositories::{self._solver.get_name_of_global_instance()}.getMaxTimeStepSize();
  const double newTimeStamp     = repositories::{self._solver.get_name_of_global_instance()}.getMaxTimeStamp() + usedTimeStepSize;
  // The last time step is not truncated, i.e. it may overshoot MaxTerminalTime
  const double newTimeStepSize  = {self.cfl_constant} * globalMinVolumeH / globalMaxEigenvalue / Dimensions;

  repositories::{self._solver.get_name_of_global_instance()}.updateCell( newTimeStepSize, newTimeStamp );
  repositories::{self._solver.get_name_of_global_instance()}.finishTimeStep();
  repositories::{self._solver.get_name_of_global_instance()}.startTimeStep(
    repositories::{self._solver.get_name_of_global_instance()}.getMinTimeStamp(),
    repositories::{self._solver.get_name_of_global_instance()}.getMaxTimeStamp(),
    repositories::{self._solver.get_name_of_global_instance()}.getMinTimeStepSize(),
    repositories::{self._solver.get_name_of_global_instance()}.getMaxTimeStepSize()
  );
  logInfo( "unprepareTraversal()", "t=" << repositories::{self._solver.get_name_of_global_instance()}.getMaxTimeStamp() << ", new dt=" << newTimeStepSize );
"""

    def get_includes(self):
        return (
            AbstractFVActionSet.get_includes(self)
            + """
#include "tarch/mpi/Rank.h"
#include "tarch/services/ServiceRepository.h"
#include "tarch/la/VectorOperations.h"
#include "peano4/parallel/SpacetreeSet.h"
#include "peano4/grid/grid.h"
"""
        )

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
