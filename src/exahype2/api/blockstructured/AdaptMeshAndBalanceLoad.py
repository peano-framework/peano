# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api
import toolbox.blockstructured.api
import toolbox.loadbalancing.api.actionsets

from ..Solver import Solver
from .BoundaryConditions import get_access_to_persistent_face_data
from .ConsumeRefinementRequests import ConsumeRefinementRequests


class AdaptMeshAndBalanceLoad(Solver.MeshAdaptation):
    """!

    Default mesh adaptation of the block-structured schemes

    Abstract base class of exahype2.api.fv.AdaptMeshAndBalanceLoad and
    exahype2.api.rkfd4.AdaptMeshAndBalanceLoad, which the Finite Volume and
    the finite difference schemes use unless the user assigns another mesh
    adaptation to solver.mesh_adaptation. The subclasses only add the
    scheme-specific action sets (_add_projection_action_sets() and
    _add_further_adaptation_action_sets()).
    The generated main runs the phase right after a time step if this time
    step has asked for a refinement or erase, or if the load balancing has
    split (@ref exahype2::isMeshAdaptationRequired()). The phase consists of
    two tasks:

    1. Adapt mesh and balance load repeats until the mesh is stationary, the
       refinement service holds no requests anymore, and the load balancing
       has not split recently. It hosts the following action sets:
       - Scheme-specific action sets, such as the one which drops the
         optimistic patch interior updates of the Finite Volumes
         (_add_further_adaptation_action_sets()).
       - exahype2.api.blockstructured.ConsumeRefinementRequests hands the
         requests of the last time step over to Peano. It does not issue any
         new request. If the mesh changes, it resets the solver's time step
         size to zero, so the next time step derives a new admissible time
         step size.
       - toolbox.blockstructured.api.InterpolateAndRestrictCells
         interpolates the patches of new cells from their parent and
         restricts the patches of erased cells into their parent.
       - toolbox.loadbalancing.api.actionsets.LoadBalancing informs the load
         balancing after each mesh sweep. The last task of the time step has
         switched it on.
    2. Project on faces projects the patches onto the faces, as the new
       mesh has new faces. It is the same projection as at the end of a time
       step, including the restriction of the hanging faces if the scheme
       supports adaptive meshes (_add_projection_action_sets()). Sommerfeld
       boundary conditions advance the ghost volumes by a time step size of
       zero, i.e. they keep them. Ghost volumes of faces along the boundary
       which the refinement has created are hence not set properly.

    """

    ## Repeat condition of the task which adapts the mesh and balances the load
    RepeatCondition = (
        "not peano4::grid::isGridStationary( peano4::parallel::SpacetreeSet::getInstance().getGridStatistics() )"
        " or not ::exahype2::RefinementControlService::getInstance().createGridControlEventsForNextTraversal().empty()"
        " or ( ::toolbox::loadbalancing::LoadBalancingService::getInstance().hasLoadBalancing()"
        " and ::toolbox::loadbalancing::LoadBalancingService::getInstance().getLoadBalancing()->hasSplitRecently() )"
    )

    def __init__(
        self,
        solver: Solver,
        interpolation: "toolbox.blockstructured.api.InterpolateAndRestrictCells.Interpolation" = toolbox.blockstructured.api.InterpolateAndRestrictCells.Interpolation.Linear,
        restriction: "toolbox.blockstructured.api.InterpolateAndRestrictCells.Restriction" = toolbox.blockstructured.api.InterpolateAndRestrictCells.Restriction.Averaging,
    ):
        """!

        @param solver: exahype2.api.Solver
          Solver with a block-structured scheme, i.e. with solver._patch,
          solver._patch_overlap_update and
          solver._patch_overlap_current_time_step.

        @param interpolation: toolbox.blockstructured.api.InterpolateAndRestrictCells.Interpolation
          How the patch of a new cell is initialised from its parent. The
          default is d-linear interpolation.

        @param restriction: toolbox.blockstructured.api.InterpolateAndRestrictCells.Restriction
          How the patches of erased cells are handed over to their parent.
          The default is averaging.

        """
        super().__init__(solver)
        self.interpolation = interpolation
        self.restriction = restriction

    def construct_output(self, exahype2_project, peano_project, mesh_adaptation):
        """!

        Add the two tasks of the mesh adaptation to mesh_adaptation

        """
        peano4.api.Logger().log(
            f"Create mesh adaptation and load balancing task graph for {self._solver.name}"
        )
        Access = peano4.api.taskgraph.Task.DataAccess
        includes = (
            self._solver._get_default_includes() + self._solver.user_solver_includes
        )

        adapt_task = peano4.api.taskgraph.Task(
            name=f"{self._solver.name}_Adapt mesh and balance load",
            entity=peano4.api.taskgraph.Task.GridEntity.CELL,
            task_has_to_repeat=self.RepeatCondition,
            may_spawn_background_tasks=False,
        )
        adapt_task.use_cell(self._solver._patch, Access.READ_AND_WRITE)
        self._add_further_adaptation_action_sets(adapt_task)
        adapt_task.add_action_set(ConsumeRefinementRequests(self._solver))
        adapt_task.add_action_set(
            toolbox.blockstructured.api.InterpolateAndRestrictCells(
                self._solver._patch,
                interpolation=self.interpolation,
                restriction=self.restriction,
                additional_includes=includes,
            )
        )
        balance_load_task = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Balance load", entity=peano4.api.taskgraph.Task.GridEntity.GLOBAL)
        balance_load_task.add_action_set(toolbox.loadbalancing.api.actionsets.LoadBalancing())
        balance_load_task.add_dependency(adapt_task)

        project_on_faces_task = peano4.api.taskgraph.Task(
            name=f"{self._solver.name}_Project on faces",
            entity=peano4.api.taskgraph.Task.GridEntity.CELL,
        )
        project_on_faces_task.add_dependency(adapt_task)
        project_on_faces_task.use_cell(self._solver._patch, Access.READ_AND_WRITE)
        project_on_faces_task.use_face(
            self._solver._patch_overlap_update, Access.TEMPORARY
        )
        project_on_faces_task.use_face(
            self._solver._patch_overlap_current_time_step,
            get_access_to_persistent_face_data(
                self._solver, exahype2_project, Access.WRITE
            ),
        )
        self._add_projection_action_sets(project_on_faces_task, exahype2_project)

        mesh_adaptation.add_task(adapt_task)
        mesh_adaptation.add_task(balance_load_task)
        mesh_adaptation.add_task(project_on_faces_task)

        if peano4.api.Logger().is_visualisation:
            mesh_adaptation.visualise(
                title=f"Mesh adaptation task graph for {self._solver.name}"
            )
        if peano4.api.Logger().is_text:
            mesh_adaptation.log_as_text(
                title=f"Mesh adaptation task graph for {self._solver.name}"
            )

    def _add_further_adaptation_action_sets(self, adapt_task):
        """!

        Add scheme-specific action sets to the task which adapts the mesh

        They are added before the generic ones. The default adds nothing.

        """
        pass

    def _add_projection_action_sets(self, project_on_faces_task, exahype2_project):
        """!

        Add the scheme-specific action sets which project the patches onto
        the faces and roll the face data over

        Has to be implemented by the subclasses. The task already declares
        the use of solver._patch, solver._patch_overlap_update and
        solver._patch_overlap_current_time_step.

        """
        raise NotImplementedError(
            f"{type(self).__name__} has to implement _add_projection_action_sets()"
        )
