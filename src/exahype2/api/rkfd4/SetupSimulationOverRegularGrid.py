# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api
import peano4.api.actionsets
import toolbox.loadbalancing.api.actionsets

from ..InitPointWisely import InitPointWisely
from ..Solver import Solver
from ..AuxiliaryVariables import add_auxiliary_variables_to_setup
from ..blockstructured.InitialiseAuxiliaryVariables import (
    InitialiseAuxiliaryVariables,
)
from .actionsets import InitialCondition
from .actionsets import ProjectPatchOntoFaces
from .actionsets import RollOverUpdatedFace
from .actionsets import RollOverUpdatedFaceWithSommerfeldBoundaryConditions


class SetupSimulationOverRegularGrid(Solver.GridConstruction):
    """!

    Setup phase of the fourth-order Finite Differences with Runge-Kutta time
    stepping (@ref exahype2.api.rkfd4): sets the initial condition over a
    regular block-structured mesh and prepares the data required for the
    first time step.

    The mesh is not refined adaptively: setup constructs the regular grid
    the simulation is going to stay on, fills in the initial condition and
    projects it onto the faces, where the first Runge-Kutta stage picks it
    up to reconstruct the halo of each patch.

    The solver this is attached to has to have an exahype2.api.InitPointWisely
    PDE term registered (via solver.add_pde_term(...)), as that is what fills
    in the initial condition this class' setup task graph relies on;
    construct_output() asserts this before it builds anything.

    The class is a copy of exahype2.api.fv.SetupSimulationOverRegularGrid,
    as rkfd4 does not depend on the Finite Volume package. It works against
    the data the numerical scheme assigns to the solver: solver._patch,
    solver._patch_overlap_update and solver._patch_overlap_current_time_step,
    plus the numerical scheme's patch_size.

    """

    def __init__(self, solver: Solver, volume_size: float):
        """!

        @param volume_size: float
          Maximum edge length of one Finite Differences grid cell, i.e. the
          maximum mesh width h of the scheme. Every mesh cell hosts one
          patch of patch_size grid cells per axis. The setup refines every
          mesh cell whose edge length is greater than or equal to
          volume_size times the patch size (@ref
          peano4.api.actionsets.ConstructRegularGrid), i.e. the resulting
          grid cells are strictly smaller than volume_size. If you want grid
          cells of exactly 1/27 with patches of size 9, pass in a value
          slightly bigger than 1/27: 1/27 itself yields mesh cells of size
          1/9 and hence grid cells of size 1/81.

        """
        super().__init__(solver)
        self.volume_size = volume_size

    def construct_output(self, exahype2_project, peano_project, setup_simulation):
        """!

        Create the actual sequence of tasks following the description
        from @ref page_exahype_numerical_scheme "Adding a numerical scheme", as grid construction,
        numerical scheme, but also plotting all follow the same pattern
        of tasks and dependencies.

        """
        peano4.api.Logger().log(
            "Create setup task graph for fourth-order Finite Differences with Runge-Kutta over a regular, block-structured mesh"
        )

        assert any(
            isinstance(term, InitPointWisely) for term in self._solver.pde_terms
        ), (
            f"Solver {self._solver.name} has no InitPointWisely PDE term registered. "
            "SetupSimulationOverRegularGrid needs one to fill in the initial condition - "
            "add one via solver.add_pde_term(exahype2.api.InitPointWisely(...))."
        )

        # Repeat while the mesh still changes, while the load balancing still
        # wants to split, or while a split is still being carried out. Splits
        # within these sweeps are safe, as the sweeps do not hold any data
        # that lives from one sweep to the next.
        refine_task = peano4.api.taskgraph.Task(
            name=f"{self._solver.name}_Refine mesh",
            entity=peano4.api.taskgraph.Task.GridEntity.CELL,
            task_has_to_repeat=(
                "not peano4::grid::isGridStationary( peano4::parallel::SpacetreeSet::getInstance().getGridStatistics() )"
                " or ( ::toolbox::loadbalancing::LoadBalancingService::getInstance().hasLoadBalancing()"
                " and ( not ::toolbox::loadbalancing::LoadBalancingService::getInstance().getLoadBalancing()->hasStagnated()"
                " or ::toolbox::loadbalancing::LoadBalancingService::getInstance().getLoadBalancing()->hasSplitRecently() ) )"
            ),
        )
        consolidate_load_balancing_task = peano4.api.taskgraph.Task(
            name=f"{self._solver.name}_Consolidate mesh",
            entity=peano4.api.taskgraph.Task.GridEntity.CELL,
            task_has_to_repeat=(
                "not peano4::grid::isGridStationary( peano4::parallel::SpacetreeSet::getInstance().getGridStatistics() )"
                " or ( ::toolbox::loadbalancing::LoadBalancingService::getInstance().hasLoadBalancing()"
                " and ::toolbox::loadbalancing::LoadBalancingService::getInstance().getLoadBalancing()->hasSplitRecently() )"
            ),
        )

        assert hasattr(self._solver.numerical_scheme, "patch_size"), (
            f"Solver {self._solver.name}'s numerical scheme has no attribute "
            "patch_size. SetupSimulationOverRegularGrid assumes the numerical "
            "scheme has added this field to itself immediately upon construction."
        )
        # Load balancing is expensive. Once the load balancing has split,
        # we therefore do not add any further mesh elements for a while, so
        # the mesh and the new trees can catch up: if we kept on refining,
        # the freshly split trees would immediately become unbalanced again,
        # and the load balancing would have to split once more. The
        # refinement hence only continues if no load balancing is installed
        # or if the installed load balancing has not split recently.
        refine_task.add_action_set(
            peano4.api.actionsets.ConstructRegularGrid(
                self.volume_size * self._solver.numerical_scheme.patch_size,
                guard=(
                    "not ::toolbox::loadbalancing::LoadBalancingService::getInstance().hasLoadBalancing()"
                    " or not ::toolbox::loadbalancing::LoadBalancingService::getInstance().getLoadBalancing()->hasSplitRecently()"
                ),
                includes="""
#include "toolbox/loadbalancing/AbstractLoadBalancing.h"
#include "toolbox/loadbalancing/LoadBalancingService.h"
""",
            )
        )
        # Inform the load balancing after every mesh refinement sweep. The
        # action set has to be part of the refinement task: The task repeats
        # until the load balancing has stagnated, which only finishStep()
        # tells. In a task of its own, it runs within the repetition only if
        # the task graph compiler happens to fuse both tasks. If it does not,
        # the setup never terminates.
        refine_task.add_action_set(toolbox.loadbalancing.api.actionsets.LoadBalancing())
        # Once the mesh is constructed, switch the load balancing off and let
        # the mesh consolidate: we keep on running mesh sweeps until the
        # mesh is stationary and the most recent splits have been carried out.
        consolidate_balance_task = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Consolidate load balancing", entity=peano4.api.taskgraph.Task.GridEntity.GLOBAL)
        consolidate_balance_task.add_action_set(
            toolbox.loadbalancing.api.actionsets.ToggleLoadBalancing(
                enable_load_balancing=False, plug_into_prepare_traversal=True
            )
        )
        consolidate_balance_task.add_action_set(toolbox.loadbalancing.api.actionsets.LoadBalancing())
        consolidate_balance_task.add_dependency(consolidate_load_balancing_task)

        initial_condition_task = peano4.api.taskgraph.Task(
            name=f"{self._solver.name} Initial condition",
            entity=peano4.api.taskgraph.Task.GridEntity.CELL,
        )
        project_on_faces_task = peano4.api.taskgraph.Task(
            name=f"{self._solver.name} {Solver.GridConstruction.PROJECT_ON_FACES_TASK_NAME}",
            entity=peano4.api.taskgraph.Task.GridEntity.CELL,
        )

        consolidate_load_balancing_task.add_dependency(refine_task)
        initial_condition_task.add_dependency(consolidate_load_balancing_task)
        project_on_faces_task.add_dependency(initial_condition_task)

        initial_condition_task.add_action_set(
            InitialCondition(
                self._solver, guard=self._solver.grid_selector.createCellGuard()
            )
        )
        project_on_faces_task.add_action_set(
            ProjectPatchOntoFaces(
                self._solver._patch,
                self._solver._patch_overlap_update,
                guard=self._solver.grid_selector.createCellGuard(),
                includes=self._solver._get_default_includes()
                + self._solver.user_solver_includes,
            )
        )
        # Sommerfeld boundary conditions initialise the ghost grid cells with
        # the initial condition (@ref exahype2.api.blockstructured.BoundaryConditions)
        project_on_faces_task.add_action_set(
            RollOverUpdatedFaceWithSommerfeldBoundaryConditions.create_roll_over(
                self._solver,
                exahype2_project,
                guard=self._solver.grid_selector.createFaceGuard(),
                includes=self._solver._get_default_includes()
                + self._solver.user_solver_includes,
            )
        )

        initialise_auxiliary_variables_task = add_auxiliary_variables_to_setup(
            self._solver,
            initial_condition_task,
            project_on_faces_task,
            ProjectPatchOntoFaces,
            RollOverUpdatedFace,
            lambda: InitialiseAuxiliaryVariables(
                self._solver,
                "::exahype2::fd::getGridCellCentre",
                includes='#include "exahype2/fd/PatchUtils.h"\n'
                + self._solver._get_default_includes()
                + self._solver.user_solver_includes,
                guard=self._solver.grid_selector.createCellGuard(),
            ),
        )

        setup_simulation.add_task(refine_task)
        setup_simulation.add_task(consolidate_load_balancing_task)
        setup_simulation.add_task(consolidate_balance_task)
        setup_simulation.add_task(initial_condition_task)
        if initialise_auxiliary_variables_task is not None:
            setup_simulation.add_task(initialise_auxiliary_variables_task)
        setup_simulation.add_task(project_on_faces_task)

        initial_condition_task.use_cell(
            self._solver._patch, peano4.api.taskgraph.Task.DataAccess.WRITE
        )
        project_on_faces_task.use_cell(
            self._solver._patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE
        )
        project_on_faces_task.use_face(
            self._solver._patch_overlap_update,
            peano4.api.taskgraph.Task.DataAccess.TEMPORARY,
        )
        project_on_faces_task.use_face(
            self._solver._patch_overlap_current_time_step,
            peano4.api.taskgraph.Task.DataAccess.WRITE,
        )

        if peano4.api.Logger().is_visualisation:
            setup_simulation.visualise(
                title="Setup task graph for {}".format(self._solver.name)
            )
        if peano4.api.Logger().is_text:
            setup_simulation.log_as_text(
                title="Setup task graph for {}".format(self._solver.name)
            )
