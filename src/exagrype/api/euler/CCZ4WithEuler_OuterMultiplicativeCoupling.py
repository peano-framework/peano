""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api
import peano4.api
import peano4.api.taskgraph
import toolbox.blockstructured.api

from .Euler import Euler
from ..FOCCZ4Solver import FOCCZ4Solver
from ..actionsets import CopyPatchWithLinearInterpolation, SynchroniseTimeStepSizes


class CCZ4WithEuler_OuterMultiplicativeCoupling:
    """!

    First-order CCZ4 and an Euler fluid, run one after the other within each time step

    The host is the sibling of exagrype.api.CCZ4WithEuler_OuterAdditiveCoupling, and
    deliberately repeats what the two have in common (solvers, auxiliary
    variables, prep tasks, realisation of the solvers), so that each class can be
    read on its own. It only wires the task graphs differently
    (@ref page_exahype_programmers_guide_coupling). Per time step, we get

        prep (from Euler) -> graph of CCZ4 -> prep (from CCZ4) -> graph of Euler -> synchronise

    i.e. CCZ4 sees the Euler fields of the begin of the time step, and Euler sees the CCZ4
    fields after CCZ4 has completed its step. Each solver brings its own numerical scheme,
    assigned to the solvers. They do not exchange data within
    their Runge-Kutta stages.

    The graphs are chained with peano4.api.taskgraph.Graph.append(): Each graph waits for all sinks of the
    graph built so far. Each solver graph ends in a GLOBAL_END_OF_STEP task that
    computes the next time step size. The next graph depends on it, too, and
    the task compiler therefore does not run it in the mesh sweep of that
    task (@ref peano4.api.taskgraph.compiler.FuseSteps.translate()).
    A final GLOBAL_END_OF_STEP task enforces the minimum of both time step sizes.

    """

    EULER_AUXILIARY_COPIES_NAME = "EulerFields"
    """! Name of the auxiliary variables of CCZ4 which hold the Euler fields """
    CCZ4_AUXILIARY_COPIES_NAME = "CCZ4Fields"
    """! Name of the auxiliary variables of Euler which hold the CCZ4 fields """
    PREP_TASK_NAME = "Prepare auxiliary variables from {}"
    """! Name of the tasks that fill the auxiliary variables; the argument is the name of the solver that provides the data """
    SYNCHRONISE_TASK_NAME = "Synchronise time step sizes"
    """! Name of the final task """

    def __init__(self, ccz4_name: str = "CCZ4", euler_name: str = "Euler"):
        """!

        Create the two solvers and the auxiliary variables through which they see each other

        The solvers exchange data once per time step only, so the numerical
        schemes of the solvers remain flexible.

        @param ccz4_name: str
          Name of the CCZ4 solver.

        @param euler_name: str
          Name of the Euler solver.

        """
        self.ccz4 = FOCCZ4Solver(ccz4_name)
        self.euler = Euler(euler_name)
        Mode = exahype2.api.AuxiliaryVariables.Mode
        self.euler_auxiliary_copies = exahype2.api.AuxiliaryVariables(
            self.EULER_AUXILIARY_COPIES_NAME,
            dict(Euler.Unknowns),
            mode=Mode.STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES,
            data_name="EulerFieldsForCCZ4",
        )
        self.ccz4_auxiliary_copies = exahype2.api.AuxiliaryVariables(
            self.CCZ4_AUXILIARY_COPIES_NAME,
            dict(FOCCZ4Solver.Unknowns),
            mode=Mode.STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES,
            data_name="CCZ4FieldsForEuler",
        )
        self.ccz4.add_auxiliary_variables(self.euler_auxiliary_copies)
        self.euler.add_auxiliary_variables(self.ccz4_auxiliary_copies)
        self._name = f"{ccz4_name}With{euler_name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def solvers(self):
        """ The solvers the project has to register; exahype2.api.Project.add_solver() asks for them """
        return [self.ccz4, self.euler]

    def generate_solver_realisation(
        self,
        exahype2_project,
        peano_project,
        setup_simulation,
        time_step,
        plot,
        mesh_adaptation,
    ):
        """!

        Realise both solvers, chain their time stepping graphs, and add the outcome to the project's graphs

        The arguments are those of exahype2.api.Solver.generate_solver_realisation().
        Both solvers are realised without global tasks. The load balancing and the dynamic
        mesh refinement wrap the chained time step graph afterwards, through the scheme of CCZ4.

        """
        peano4.api.Logger().new_section("Host " + self.name)
        Entity = peano4.api.taskgraph.Task.GridEntity

        ccz4_time_step = peano4.api.taskgraph.Graph()
        self.ccz4.generate_solver_realisation(
            exahype2_project,
            peano_project,
            setup_simulation,
            ccz4_time_step,
            plot,
            mesh_adaptation,
            add_global_tasks=False,
        )
        euler_time_step = peano4.api.taskgraph.Graph()
        self.euler.generate_solver_realisation(
            exahype2_project,
            peano_project,
            setup_simulation,
            euler_time_step,
            plot,
            mesh_adaptation,
            add_global_tasks=False,
        )

        euler_into_ccz4 = peano4.api.taskgraph.Graph()
        self._create_prep_task(
            euler_into_ccz4, self.euler, self.ccz4, self.euler_auxiliary_copies
        )
        ccz4_into_euler = peano4.api.taskgraph.Graph()
        self._create_prep_task(
            ccz4_into_euler, self.ccz4, self.euler, self.ccz4_auxiliary_copies
        )

        ## prep (from Euler) -> graph of CCZ4 -> prep (from CCZ4) -> graph of Euler
        coupled_time_step = peano4.api.taskgraph.Graph()
        coupled_time_step.append(euler_into_ccz4)
        coupled_time_step.append(ccz4_time_step)
        coupled_time_step.append(ccz4_into_euler)
        coupled_time_step.append(euler_time_step)

        ## Both solvers have computed their next time step size by now
        synchronise = peano4.api.taskgraph.Task(
            name=self.SYNCHRONISE_TASK_NAME, entity=Entity.GLOBAL_END_OF_STEP
        )
        for task in coupled_time_step.sink_tasks():
            synchronise.add_dependency(task)
        synchronise.add_action_set(SynchroniseTimeStepSizes([self.ccz4, self.euler]))
        coupled_time_step.add_task(synchronise)

        self.ccz4.numerical_scheme.wrap_task_graph_in_load_balancing_and_amr_tasks(
            coupled_time_step
        )
        if self.ccz4.mesh_adaptation is not None:
            self.ccz4.mesh_adaptation.construct_output(
                exahype2_project, peano_project, mesh_adaptation
            )

        title = f"Time stepping task graph for host {self.name}"
        if peano4.api.Logger().is_visualisation:
            coupled_time_step.visualise(title=title)
        if peano4.api.Logger().is_text:
            coupled_time_step.log_as_text(title=title)
        time_step.merge(coupled_time_step)

    def _create_prep_task(
        self, time_step, source_solver, target_solver, auxiliary_copies
    ):
        """!

        Create the task that fills the auxiliary variables of one solver with the solution of the other one

        auxiliary copies (auxiliary_copies) of the target solver (copy, with
        auxiliary variables auxiliary_copies of the target solver (copy, with
        interpolation if the patches differ in size), projects them onto the
        faces, restricts them on hanging faces, and rolls the faces over. It is
        added to time_step, but no dependencies are set: the caller wires it.
        It is the one place that writes new face data.

        @param time_step: peano4.api.taskgraph.Graph
          Receives the task.

        @param source_solver: exahype2.api.Solver
          Provides the data.

        @param target_solver: exahype2.api.Solver
          Holds the auxiliary variables.

        @param auxiliary_copies: exahype2.api.AuxiliaryVariables
          The auxiliary variables of the target solver.

        @return peano4.api.taskgraph.Task
          The new task.

        """
        Access = peano4.api.taskgraph.Task.DataAccess
        Entity = peano4.api.taskgraph.Task.GridEntity
        actionsets = exahype2.api.rkfd4.actionsets
        includes = (
            target_solver._get_default_includes() + target_solver.user_solver_includes
        )
        prep = peano4.api.taskgraph.Task(
            name=self.PREP_TASK_NAME.format(source_solver.name),
            entity=Entity.CELL,
        )
        prep.use_cell(source_solver._patch, Access.READ_AND_WRITE)
        prep.use_cell(auxiliary_copies.patch, Access.READ_AND_WRITE)
        prep.use_face(auxiliary_copies.patch_overlap_update, Access.TEMPORARY)
        prep.use_face(auxiliary_copies.patch_overlap_current_time_step, Access.READ_AND_WRITE)
        prep.add_action_set(
            CopyPatchWithLinearInterpolation(
                source_patch=source_solver._patch,
                target_patch=auxiliary_copies.patch,
                guard=target_solver.grid_selector.createCellGuard(),
                additional_includes=includes,
            )
        )
        prep.add_action_set(
            actionsets.ProjectPatchOntoFaces(
                auxiliary_copies.patch,
                auxiliary_copies.patch_overlap_update,
                guard=target_solver.grid_selector.createCellGuard(),
                includes=includes,
            )
        )
        prep.add_action_set(
            toolbox.blockstructured.api.RestrictHangingFaces(
                auxiliary_copies.patch_overlap_update,
                scheme=target_solver.numerical_scheme.restriction,
                clear_guard=target_solver.grid_selector.createFaceGuard(),
                additional_includes=includes,
            )
        )
        prep.add_action_set(
            actionsets.RollOverUpdatedFace(
                auxiliary_copies.patch_overlap_current_time_step,
                auxiliary_copies.patch_overlap_update,
                guard=target_solver.grid_selector.createFaceGuard(),
                includes=includes,
            )
        )
        time_step.add_task(prep)
        return prep
