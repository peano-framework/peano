# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import os

import peano4.api
import peano4.api.taskgraph
import toolbox.blockstructured.api
import toolbox.loadbalancing.api.actionsets
from peano4.api.solversteps.ActionSet import ActionSet

from ..Solver import Solver


def _create_reader(index_file_name, time_stamp, halo_width):
    """!

    C++ expression which yields the exahype2::CheckpointReader of the
    checkpoint

    """
    return (
        "::exahype2::CheckpointReader::getInstance("
        f'"{index_file_name}", {repr(float(time_stamp))}, {halo_width}, DomainOffset, DomainSize)'
    )


class _RefineFromCheckpoint(ActionSet):
    """!

    Refine the mesh such that it matches the mesh of a checkpoint

    The action set hands the refinement events of the checkpoint
    (@ref exahype2::CheckpointReader::getRefinementEvents()) to Peano. As the
    mesh setup over a regular grid, it pauses the refinement while the load
    balancing has just split.

    """

    def __init__(self, reader, includes):
        super().__init__(descend_invocation_order=1, parallel=False)
        self._reader = reader
        self._includes = includes

    def get_body_of_createGridControlEventsForNextTraversal(self):
        return f"""
  if (
    ::toolbox::loadbalancing::LoadBalancingService::getInstance().hasLoadBalancing()
    and
    ::toolbox::loadbalancing::LoadBalancingService::getInstance().getLoadBalancing()->hasSplitRecently()
  ) {{
    return std::vector< peano4::grid::GridControlEvent >();
  }}
  return {self._reader}.getRefinementEvents();
"""

    def get_body_of_operation(self, operation_name):
        return "\n"

    def get_body_of_prepareTraversal(self):
        return "\n"

    def get_body_of_unprepareTraversal(self):
        return "\n"

    def get_constructor_body(self):
        return ""

    def get_action_set_name(self):
        return "RefineFromCheckpoint"

    def user_should_modify_template(self):
        return False

    def get_attributes(self):
        return ""

    def get_includes(self):
        return (
            """
#include "exahype2/CheckpointReader.h"
#include "toolbox/loadbalancing/AbstractLoadBalancing.h"
#include "toolbox/loadbalancing/LoadBalancingService.h"
"""
            + self._includes
        )


class _LoadPatchesFromCheckpoint(ActionSet):
    """!

    Read each cell's patch with halo from a checkpoint and restore the
    solver's state

    In touchCellFirstTime(), the action set copies the cell's patch with halo
    from the checkpoint into solver._patch_with_halo, and the patch without
    the halo into solver._patch. Every cell selected by the guard has to have
    a patch in the checkpoint. Otherwise, the mesh does not match the
    checkpoint, and the action set terminates the code.

    In unprepareTraversal(), it hands the checkpoint's meta data to the
    solver (@ref exahype2::Solver::fromCheckpointMetaData()) and frees the
    reader.

    """

    def __init__(self, solver, reader, index_file_name, halo_width, guard, includes):
        super().__init__(descend_invocation_order=1, parallel=False)
        self._solver = solver
        self._reader = reader
        self._index_file_name = index_file_name
        self._halo_width = halo_width
        self._guard = guard
        self._includes = includes

    def get_body_of_operation(self, operation_name):
        if operation_name != ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return "\n"
        patch = self._solver._patch
        patch_with_halo = self._solver._patch_with_halo
        return f"""
  constexpr int Unknowns           = {patch.no_of_unknowns};
  constexpr int DoFsPerAxis        = {patch.dim[0]};
  constexpr int Halo               = {self._halo_width};
  constexpr int EntriesPerAxisHalo = DoFsPerAxis + 2 * Halo;

  int numberOfEntries = Unknowns;
  for (int d=0; d<Dimensions; d++) {{
    numberOfEntries *= EntriesPerAxisHalo;
  }}

  const bool found = {self._reader}.readPatch(
    marker.x(),
    marker.h(),
    numberOfEntries,
    fineGridCell{patch_with_halo.name}.value
  );
  if (not found) {{
    logError(
      "touchCellFirstTime(...)",
      "checkpoint {self._index_file_name} holds no patch for cell " << marker.toString()
      << ". The mesh does not match the checkpoint"
    );
    std::abort();
  }}

  dfor(cell, DoFsPerAxis) {{
    const int cellSerialised     = peano4::utils::dLinearised(cell, DoFsPerAxis);
    const int cellWithHaloSerial = peano4::utils::dLinearised(cell + tarch::la::Vector<Dimensions,int>(Halo), EntriesPerAxisHalo);
    for (int unknown=0; unknown<Unknowns; unknown++) {{
      fineGridCell{patch.name}.value[cellSerialised*Unknowns+unknown] = fineGridCell{patch_with_halo.name}.value[cellWithHaloSerial*Unknowns+unknown];
    }}
  }}
"""

    def get_guard_of_operation(self, operation_name):
        if operation_name == ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self._guard
        return "true"

    def get_body_of_prepareTraversal(self):
        return "\n"

    def get_body_of_unprepareTraversal(self):
        return f"""
  repositories::{self._solver.get_name_of_global_instance()}.fromCheckpointMetaData( {self._reader}.getMetaData() );
  ::exahype2::CheckpointReader::release( "{self._index_file_name}" );
"""

    def get_constructor_body(self):
        return ""

    def get_action_set_name(self):
        return "LoadPatchesFromCheckpoint"

    def user_should_modify_template(self):
        return False

    def get_attributes(self):
        return ""

    def get_includes(self):
        return (
            """
#include <cstdlib>
#include "exahype2/CheckpointReader.h"
"""
            + self._includes
        )


class LoadCheckpoint(Solver.GridConstruction):
    """!

    Set up a simulation from a checkpoint

    Alternative to the setups over a regular grid
    (@ref exahype2.api.fv.SetupSimulationOverRegularGrid,
    @ref exahype2.api.rkfd4.SetupSimulationOverRegularGrid): Instead of
    constructing a mesh and setting the initial condition, the setup reads a
    checkpoint written by exahype2.api.fv.WriteCheckpoint or
    exahype2.api.rkfd4.WriteCheckpoint and continues the simulation from
    there. It is available for both schemes as exahype2.api.fv.LoadCheckpoint
    and exahype2.api.rkfd4.LoadCheckpoint, as both use the same data layout.

    The setup consists of three phases:

    - It reconstructs the checkpoint's mesh. The refinement sweep repeats
      until the mesh is stationary and the load balancing has stagnated,
      exactly as for the regular grid. Instead of refining down to a given
      mesh size, it refines the parents of all cells in the checkpoint
      (@ref exahype2::CheckpointReader).
    - Once the mesh is complete, it switches off the load balancing and
      waits until the last splits have been carried out.
    - A last sweep loads the data. For every cell, it reads the patch with
      halo into the temporary solver._patch_with_halo, copies the patch into
      solver._patch and writes the patch with halo onto the overlaps of the
      faces (toolbox.blockstructured.api.ProjectPatchWithHaloOntoFaces).
      The faces hence also hold the ghost data along the domain boundary and
      the data at coarse-fine transitions as they have been at the time of
      the checkpoint. In unprepareTraversal(), the sweep restores the
      solver's time stamps, time step sizes and mesh sizes, plus the
      attributes the Python API has added, e.g. the time stamps of the next
      outputs of the plotters. Plotters which have written output at the
      time of the checkpoint move on to their next output, so the restarted
      simulation does not write them again
      (@ref exahype2.api.Solver.Solver._add_checkpoint_meta_data_to_solver()).

    The setup neither uses the initial condition nor the boundary conditions.
    The checkpoint has to be written by a solver with the same unknowns,
    auxiliary variables, patch size and halo, over the same domain. The code
    reads the checkpoint at runtime, i.e. the files have to exist when the
    simulation starts, but not when it is generated or built.

    The time stamps of all solvers have to match, i.e. a project with
    several solvers has to load checkpoints of all of them, written at the
    same time.

    ## Why the refinement task informs the load balancing itself

    The refinement sweep repeats until the load balancing has stagnated and
    has not split recently. Only finishStep() updates both, i.e. the action
    set toolbox.loadbalancing.api.actionsets.LoadBalancing which calls it has
    to be part of this very task. In a task of its own, the setup terminates
    only if the task graph compiler happens to fuse both tasks. We have
    observed the endless repetition with the setup over a regular grid
    (@ref exahype2.api.fv.SetupSimulationOverRegularGrid), which followed the
    same pattern. We have not observed it with the setup from a checkpoint.

    @image html repeating-task-load-balancing-before.png
    @image html repeating-task-load-balancing-after.png

    The sources of the graphs are the .dot files next to the images in
    src/exahype2/api/blockstructured.

    """

    def __init__(
        self,
        solver: Solver,
        filename: str = "checkpoint",
        output_path: str = ".",
        time_stamp: float = None,
    ):
        """!

        @param solver: exahype2.api.Solver
          The solver whose data is loaded.

        @param filename: str
          File name of the checkpoint as passed to WriteCheckpoint. The setup
          reads the index file output_path/filename-SolverName.peano-patch-file.

        @param output_path: str
          Directory of the checkpoint, relative to the directory the
          simulation is started from, or absolute.

        @param time_stamp: float or None
          Time stamp of the checkpoint to load. If None, the setup loads the
          latest checkpoint in the index file. Otherwise, it loads the
          checkpoint whose time stamp is closest to time_stamp.

        """
        super().__init__(solver)
        self._filename = filename
        self._output_path = output_path
        self._time_stamp = time_stamp

    def index_file_name(self) -> str:
        """!
        Path of the checkpoint's index file
        """
        return os.path.join(
            self._output_path,
            self._filename + "-" + self._solver.name + ".peano-patch-file",
        )

    def construct_output(self, exahype2_project, peano_project, setup_simulation):
        """!

        Add the mesh construction and the loading of the data to the setup
        task graph

        """
        if self._solver.auxiliary_variables:
            raise NotImplementedError(
                f"Solver {self._solver.name}: checkpoints do not cover auxiliary variables yet."
            )
        peano4.api.Logger().log(
            "Create setup task graph which loads a checkpoint into a block-structured mesh"
        )

        for attribute in [
            "_patch",
            "_patch_with_halo",
            "_patch_overlap_current_time_step",
        ]:
            assert hasattr(self._solver, attribute), (
                f"Solver {self._solver.name} has no attribute {attribute}. "
                "LoadCheckpoint assumes the numerical scheme has added this field "
                "to the solver immediately upon construction."
            )

        # The solver has to be able to restore all of its attributes
        self._solver._add_checkpoint_meta_data_to_solver()

        Access = peano4.api.taskgraph.Task.DataAccess
        Entity = peano4.api.taskgraph.Task.GridEntity
        halo_width = (
            self._solver._patch_with_halo.dim[0] - self._solver._patch.dim[0]
        ) // 2
        includes = (
            self._solver._get_default_includes() + self._solver.user_solver_includes
        )
        reader = _create_reader(
            self.index_file_name(),
            -1.0 if self._time_stamp is None else self._time_stamp,
            halo_width,
        )

        # Repeat while the mesh still changes, while the load balancing still
        # wants to split, or while a split is still being carried out, as in
        # the setup over a regular grid
        refine_task = peano4.api.taskgraph.Task(
            name=f"{self._solver.name}_Refine mesh from checkpoint",
            entity=Entity.CELL,
            task_has_to_repeat=(
                "not peano4::grid::isGridStationary( peano4::parallel::SpacetreeSet::getInstance().getGridStatistics() )"
                " or ( ::toolbox::loadbalancing::LoadBalancingService::getInstance().hasLoadBalancing()"
                " and ( not ::toolbox::loadbalancing::LoadBalancingService::getInstance().getLoadBalancing()->hasStagnated()"
                " or ::toolbox::loadbalancing::LoadBalancingService::getInstance().getLoadBalancing()->hasSplitRecently() ) )"
            ),
        )
        refine_task.add_action_set(_RefineFromCheckpoint(reader, includes))
        # The task repeats until the load balancing has stagnated, which only
        # finishStep() tells. The action set hence has to be part of this task,
        # as in the setup over a regular grid.
        refine_task.add_action_set(toolbox.loadbalancing.api.actionsets.LoadBalancing())

        consolidate_load_balancing_task = peano4.api.taskgraph.Task(
            name=f"{self._solver.name}_Consolidate mesh",
            entity=Entity.CELL,
            task_has_to_repeat=(
                "not peano4::grid::isGridStationary( peano4::parallel::SpacetreeSet::getInstance().getGridStatistics() )"
                " or ( ::toolbox::loadbalancing::LoadBalancingService::getInstance().hasLoadBalancing()"
                " and ::toolbox::loadbalancing::LoadBalancingService::getInstance().getLoadBalancing()->hasSplitRecently() )"
            ),
        )
        consolidate_balance_task = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Consolidate load balancing", entity=peano4.api.taskgraph.Task.GridEntity.GLOBAL)
        consolidate_balance_task.add_action_set(
            toolbox.loadbalancing.api.actionsets.ToggleLoadBalancing(
                enable_load_balancing=False, plug_into_prepare_traversal=True
            )
        )
        consolidate_balance_task.add_action_set(
            toolbox.loadbalancing.api.actionsets.LoadBalancing()
        )
        consolidate_balance_task.add_dependency(consolidate_load_balancing_task)
        consolidate_load_balancing_task.add_dependency(refine_task)

        load_task = peano4.api.taskgraph.Task(
            name=f"{self._solver.name}_Load checkpoint",
            entity=Entity.CELL,
            may_spawn_background_tasks=False,
        )
        load_task.add_dependency(consolidate_load_balancing_task)
        load_task.use_cell(self._solver._patch, Access.WRITE)
        load_task.use_cell(self._solver._patch_with_halo, Access.TEMPORARY)
        load_task.use_face(self._solver._patch_overlap_current_time_step, Access.WRITE)
        load_task.add_action_set(
            _LoadPatchesFromCheckpoint(
                self._solver,
                reader,
                self.index_file_name(),
                halo_width,
                guard=self._solver.grid_selector.createCellGuard(),
                includes=includes,
            )
        )
        load_task.add_action_set(
            toolbox.blockstructured.api.ProjectPatchWithHaloOntoFaces(
                self._solver._patch_with_halo,
                self._solver._patch_overlap_current_time_step,
                guard=self._solver.grid_selector.createCellGuard(),
                additional_includes=includes,
            )
        )

        setup_simulation.add_task(refine_task)
        setup_simulation.add_task(consolidate_load_balancing_task)
        setup_simulation.add_task(consolidate_balance_task)
        setup_simulation.add_task(load_task)

        if peano4.api.Logger().is_visualisation:
            setup_simulation.visualise(
                title="Setup task graph for {}".format(self._solver.name)
            )
        if peano4.api.Logger().is_text:
            setup_simulation.log_as_text(
                title="Setup task graph for {}".format(self._solver.name)
            )
