# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api

from ..AdaptivityCriterion import get_adaptivity_criterion
from ..blockstructured.create_dynamic_mesh_refinement_action_set import (
    create_dynamic_mesh_refinement_action_set,
)
from ..Solver import Solver
from .SetupSimulationOverRegularGrid import SetupSimulationOverRegularGrid
from .actionsets import InitialiseNewCells


class SetupSimulationWithUserCodedRefinementCriterion(SetupSimulationOverRegularGrid):
    """!

    Setup over a regular mesh which is then refined adaptively by the user's
    adaptivity criterion

    The setup does everything SetupSimulationOverRegularGrid does: It
    constructs a regular mesh and sets the initial condition. If the solver
    has an exahype2.api.AdaptivityCriterion PDE term, the setup then
    adapts this mesh. It inserts one further task after the task tagged with
    SetupSimulationOverRegularGrid.FinishedInitialRegularMeshSetup, i.e.
    after the initial condition. The projection onto the faces, which has
    followed the initial condition so far, follows this new task instead and
    hence remains the very last task of the setup.

    The new task repeats until the mesh is stationary. It hosts two action
    sets:

    - The action set which also evaluates the criterion within the time
      step (@ref exahype2.api.blockstructured.create_dynamic_mesh_refinement_action_set.create_dynamic_mesh_refinement_action_set())
      turns the criterion's outcome into refine or erase events for
      ExaHyPE's refinement service. Different to the time step, it also
      hands the service's events over to Peano.
      The mesh is never refined below min_volume_size.
      The setup vetoes all erase commands, i.e. it only ever refines the
      regular mesh: It has no restriction which would hand the patches of
      erased cells over to their parent. Erase commands of the criterion
      hence only take effect once the time stepping runs.
    - exahype2.api.fv.actionsets.InitialiseNewCells sets the initial
      condition within each cell which the refinement creates.

    Without an adaptivity criterion, the setup is the same as
    SetupSimulationOverRegularGrid.

    Peano realises a refinement event one to two mesh sweeps after the
    criterion has asked for it (@ref exahype2::RefinementControl). The mesh
    hence might still be stationary in the sweeps right after the criterion
    has issued an event. The task therefore repeats as long as the mesh
    changes or the refinement service still holds events.

    """

    def __init__(self, solver: Solver, volume_size: float, min_volume_size: float):
        """!

        @param volume_size: float
          Maximum edge length of one Finite Volume. The setup first
          constructs a regular mesh with this volume size
          (@ref exahype2.api.fv.SetupSimulationOverRegularGrid). The mesh
          refinement never erases cells of this regular mesh.

        @param min_volume_size: float
          Minimum edge length of one Finite Volume. The setup refines a cell
          only if its volumes are at least of this size, i.e. the resulting
          volumes are strictly smaller than min_volume_size. The
          semantics are the same as for volume_size.

        """
        super().__init__(solver, volume_size)
        assert min_volume_size <= volume_size, (
            f"min_volume_size={min_volume_size} has to be smaller than or equal to "
            f"volume_size={volume_size}"
        )
        self.min_volume_size = min_volume_size

    ## Repeat condition of the mesh adaptation task
    RepeatCondition = (
        "not peano4::grid::isGridStationary( peano4::parallel::SpacetreeSet::getInstance().getGridStatistics() )"
        " or not ::exahype2::RefinementControlService::getInstance().createGridControlEventsForNextTraversal().empty()"
    )

    def construct_output(self, exahype2_project, peano_project, setup_simulation):
        """!

        Construct the setup over the regular mesh and insert the mesh
        adaptation after it, if the solver has an adaptivity criterion

        """
        peano4.api.Logger().log(
            "Create setup task graph for Finite Volume scheme with user-coded refinement criterion"
        )
        self._construct_task_graph(exahype2_project, peano_project, setup_simulation)

        if get_adaptivity_criterion(self._solver) is not None:
            anchors = setup_simulation.find_tasks(
                SetupSimulationOverRegularGrid.FinishedInitialRegularMeshSetup
            )
            assert len(anchors) == 1, (
                f"Expected exactly one task tagged with "
                f"{SetupSimulationOverRegularGrid.FinishedInitialRegularMeshSetup}, "
                f"got {len(anchors)}"
            )
            setup_simulation.insert_after(anchors, self._create_adapt_mesh_task())

        self._report_task_graph(setup_simulation)

    def _create_adapt_mesh_task(self):
        """!

        Create the repeating task which adapts the mesh

        """
        patch_size = self._solver.numerical_scheme.patch_size
        adapt_mesh_task = peano4.api.taskgraph.Task(
            name=f"{self._solver.name} Adapt mesh",
            entity=peano4.api.taskgraph.Task.GridEntity.CELL,
            task_has_to_repeat=self.RepeatCondition,
        )
        adapt_mesh_task.add_action_set(
            create_dynamic_mesh_refinement_action_set(
                self._solver,
                patch_size,
                guard=self._solver.grid_selector.createCellGuard(),
                min_volume_size=self.min_volume_size,
                max_volume_size=self.volume_size,
                hand_events_to_peano=True,
                veto_erase=True,
            )
        )
        adapt_mesh_task.add_action_set(
            InitialiseNewCells(
                self._solver, guard=self._solver.grid_selector.createCellGuard()
            )
        )
        adapt_mesh_task.use_cell(
            self._solver._patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE
        )
        for auxiliary in self._solver.auxiliary_variables:
            adapt_mesh_task.use_cell(
                auxiliary.patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE
            )
        return adapt_mesh_task
