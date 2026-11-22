""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import abc

import peano4.api
import peano4.api.taskgraph

from ..actionsets import SynchroniseTimeStepSizes
from .create_copy_task import create_copy_task


class InnerCouplingGenerator(abc.ABC):
    """!

    Numerical scheme of the coupled solver exagrype.api.scalarfield.CCZ4WithScalarField_InnerCoupling

    The generator builds the task graph of the whole combination of CCZ4 and the
    scalar field. The scheme-specific parts are the numerical schemes of the two
    solvers (_create_scheme()) and what a stage of the time integrator is
    (_number_of_stages(), _stage_input(), _end_of_stage()). The four concrete generators
    only provide these. Whether the coupling is multiplicative or additive
    is the class attribute MULTIPLICATIVE.

    ## Task graph

    attach_to() installs a numerical scheme on each solver. The schemes are
    building blocks: they create the data and the kernels of their solver, and the tasks
    of its stages. The generator's construct_output() then

    1. realises both solvers into graphs of their own (without global tasks) and merges them;
    2. adds per stage k a task "Copy CCZ4 fields into ... stage k". It copies the input of
       CCZ4's stage into the auxiliary variables of the scalar field, projects them onto the faces,
       restricts them on hanging faces, and rolls them over. It reads the stage input before
       CCZ4's reconstruction discards it, and it waits for the end of stage k-1 of both solvers;
    3. lets the first tasks of the stage of the solvers wait for the copy task. This is additive
       coupling: the stages of the solvers run in parallel. For multiplicative coupling, the first
       tasks of the scalar field's stage also wait for the end of CCZ4's stage, i.e. the stage
       of the scalar field follows the one of CCZ4;
    4. adds a final global end-of-step task which enforces the minimum of the time step sizes, and
       wraps the graph into the load balancing and the dynamic mesh refinement (through CCZ4's scheme).

    The scalar field sees the CCZ4 data of the very time level of its own stage in both flavours.
    This is what keeps the order of the time integrator: if the scalar field saw the outcome of CCZ4's stage,
    its Runge-Kutta scheme would see the spacetime at the wrong time level.
    Hence, the data of the two flavours are the same for a one-directional coupling, and
    they differ in the schedule only.

    """

    MULTIPLICATIVE = False
    """! The stage of the scalar field follows the stage of CCZ4 if True, they run in parallel if False """
    COPY_TASK_NAME = "Copy CCZ4 fields into {} stage {}"
    """! Name of the task that fills the auxiliary variables; arguments are the name of the scalar field and the stage """
    SYNCHRONISE_TASK_NAME = "Synchronise time step sizes"
    """! Name of the final task """

    def __init__(self, patch_size: int):
        """!

        @param patch_size: int
          Number of grid cells (volumes) per patch and axis. The same for both solvers.

        """
        self.patch_size = patch_size
        self._host = None
        self._schemes = {}

    def attach_to(self, host):
        """!

        Install the numerical schemes of the building blocks on the solvers of the host

        The host calls it when the user assigns the generator to its numerical_scheme.

        """
        self._host = host
        self._schemes = {solver: self._create_scheme(solver) for solver in host.solvers}
        for solver, scheme in self._schemes.items():
            solver.numerical_scheme = scheme

    def construct_output(
        self,
        exahype2_project,
        peano_project,
        setup_simulation,
        time_step,
        plot,
        mesh_adaptation,
    ):
        """!

        Build the task graph of the coupled system, and add it to time_step

        The arguments are those of exahype2.api.Solver.generate_solver_realisation().

        """
        ccz4, scalar_field = self._host.solvers
        for solver in self._host.solvers:
            if solver.numerical_scheme is not self._schemes[solver]:
                raise ValueError(
                    f"Solver {solver.name} holds a numerical scheme of its own, "
                    f"but the numerical scheme of the coupled solver {self._host.name} rules"
                )

        graphs = {}
        for solver in self._host.solvers:
            graphs[solver] = peano4.api.taskgraph.Graph()
            solver.generate_solver_realisation(
                exahype2_project,
                peano_project,
                setup_simulation,
                graphs[solver],
                plot,
                mesh_adaptation,
                add_global_tasks=False,
            )
        coupled_time_step = peano4.api.taskgraph.Graph()
        coupled_time_step.merge(graphs[ccz4])
        coupled_time_step.merge(graphs[scalar_field])

        ## Wire the stages. End of the previous stage: nothing for the first one.
        previous_end = {ccz4: [], scalar_field: []}
        for stage in range(self._number_of_stages()):
            first = {
                solver: self._first_tasks(graphs[solver], previous_end[solver])
                for solver in self._host.solvers
            }
            end = {
                solver: self._end_of_stage(solver, graphs[solver], stage)
                for solver in self._host.solvers
            }
            copy = create_copy_task(
                self.COPY_TASK_NAME.format(scalar_field.name, stage),
                self._stage_input(ccz4, stage),
                scalar_field,
                self._host.ccz4_auxiliary_copies,
            )
            for task in previous_end[ccz4] + previous_end[scalar_field]:
                copy.add_dependency(task)
            for task in first[ccz4] + first[scalar_field]:
                task.add_dependency(copy)
            if self.MULTIPLICATIVE:
                for task in first[scalar_field]:
                    for dependency in end[ccz4]:
                        task.add_dependency(dependency)
            coupled_time_step.add_task(copy)
            previous_end = end

        synchronise = peano4.api.taskgraph.Task(
            name=self.SYNCHRONISE_TASK_NAME,
            entity=peano4.api.taskgraph.Task.GridEntity.GLOBAL_END_OF_STEP,
        )
        for task in coupled_time_step.sink_tasks():
            synchronise.add_dependency(task)
        synchronise.add_action_set(SynchroniseTimeStepSizes([ccz4, scalar_field]))
        coupled_time_step.add_task(synchronise)

        ccz4.numerical_scheme.wrap_task_graph_in_load_balancing_and_amr_tasks(
            coupled_time_step
        )
        if ccz4.mesh_adaptation is not None:
            ccz4.mesh_adaptation.construct_output(
                exahype2_project, peano_project, mesh_adaptation
            )

        title = f"Time stepping task graph for host {self._host.name}"
        if peano4.api.Logger().is_visualisation:
            coupled_time_step.visualise(title=title)
        if peano4.api.Logger().is_text:
            coupled_time_step.log_as_text(title=title)
        time_step.merge(coupled_time_step)

    @staticmethod
    def _first_tasks(graph, previous_end):
        """!

        The tasks of a solver's graph that start a stage

        @param graph: peano4.api.taskgraph.Graph
          The graph of one solver.

        @param previous_end: List[peano4.api.taskgraph.Task]
          The tasks that end the previous stage, empty for the first stage.
          The first tasks are the sources of the graph in that case, and
          otherwise all tasks that wait for one of previous_end.

        """
        if not previous_end:
            return graph.source_tasks()
        return [
            task
            for task in graph.tasks
            if any(end in task.dependencies for end in previous_end)
        ]

    @abc.abstractmethod
    def _create_scheme(self, solver):
        """! The numerical scheme of one of the solvers, i.e. a building block of the whole scheme """

    @abc.abstractmethod
    def _number_of_stages(self) -> int:
        """! Number of stages of the time integrator """

    @abc.abstractmethod
    def _stage_input(self, solver, stage):
        """! The patch which a solver's stage reads as input (it holds the solver's unknowns of the stage's time level) """

    @abc.abstractmethod
    def _end_of_stage(self, solver, graph, stage):
        """!

        The tasks of a solver's graph that end a stage

        Global tasks do not count.

        """
