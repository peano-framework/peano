# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import abc
import dataclasses
from typing import List

import peano4.api
import peano4.api.taskgraph

from .Solver import Solver


@dataclasses.dataclass
class PhaseGraphs:
    """!

    The task graphs of the four algorithmic phases of an ExaHyPE 2 project

    """

    setup_simulation: peano4.api.taskgraph.Graph = dataclasses.field(
        default_factory=peano4.api.taskgraph.Graph
    )
    time_step: peano4.api.taskgraph.Graph = dataclasses.field(
        default_factory=peano4.api.taskgraph.Graph
    )
    plot: peano4.api.taskgraph.Graph = dataclasses.field(
        default_factory=peano4.api.taskgraph.Graph
    )
    mesh_adaptation: peano4.api.taskgraph.Graph = dataclasses.field(
        default_factory=peano4.api.taskgraph.Graph
    )

    def graphs(self) -> List[peano4.api.taskgraph.Graph]:
        """! The graphs in the order of the arguments of Solver.generate_solver_realisation()"""
        return [self.setup_simulation, self.time_step, self.plot, self.mesh_adaptation]

    def merge(self, other: "PhaseGraphs"):
        """! Merge each graph of other into the corresponding graph of this object, @see peano4.api.taskgraph.Graph.merge()"""
        for mine, theirs in zip(self.graphs(), other.graphs()):
            mine.merge(theirs)


class SolverHost(abc.ABC):
    """!

    Hosts several solvers which are coupled to each other

    A host is what a user adds to an exahype2.api.Project instead of its
    solvers (Project.add_solver() accepts it). The project registers the
    host's solvers (solvers) with its solver repository, README and so forth,
    but asks the host, rather than each solver, to realise them. The host
    follows the recipe of @ref page_exahype_programmers_guide_coupling :

    0. The host, not its solvers, holds the generator of the numerical scheme
       (numerical_scheme). The solvers have no scheme of their own: the
       host raises an exception if a solver has one. The mapping of the
       solvers onto C++ remains solver-specific.
    1. Each solver builds its own task graphs (one per phase, PhaseGraphs).
       The scheme of the host builds the graph of global_tasks_solver. It asks
       every other solver for its task graph, once per stage of the
       time stepping scheme (@ref number_of_stages,
       @ref exahype2.api.Solver.Solver.create_task_graph()).
       Each solver reports its own graph, so the user sees what each
       solver contributes.
    2. The host concatenates the graphs of all solvers into graphs of its own
       (Graph.merge()), and calls couple(), which adds the dependencies and
       tasks that tie the solvers together.
    3. The host reports the concatenated, coupled graphs once more, and then
       appends them to the graphs of the project.

    Whatever belongs to the mesh as a whole, i.e. the load balancing switch and the
    dynamic mesh refinement, shall exist once: The host realises all solvers
    with add_global_tasks=False (@ref
    exahype2.api.Solver.Solver.generate_solver_realisation()), i.e. as
    intermediate steps. After the coupling, it wraps the concatenated graph
    through the numerical scheme of global_tasks_solver
    (@ref exahype2.api.Solver.Solver.NumericalScheme.wrap_task_graph_in_load_balancing_and_amr_tasks()),
    and lets this solver's mesh adaptation fill the mesh adaptation graph.

    """

    class HostedScheme(Solver.NumericalScheme):
        """!

        The numerical scheme which the host hands to a solver that is not
        global_tasks_solver

        The solver has no scheme of its own, but the generic parts of the solver
        still query one for patch size, halo and so forth. The hosted scheme
        forwards all such queries to the scheme of the host. Its
        construct_output() asks the solver for its task graph per stage
        (@ref exahype2.api.Solver.Solver.create_task_graph()) and puts the
        graphs into the solver's own time stepping graph. The graphs are not
        connected: wiring them is up to the host's couple(). The hosted
        scheme remembers them in stage_graphs.

        """

        def __init__(self, solver: Solver, host: "SolverHost"):
            super().__init__(solver)
            self._host = host
            self.stage_graphs: List[peano4.api.taskgraph.Graph] = []
            """! The graph that the solver returned for each stage, filled by construct_output() """

        def __getattr__(self, name):
            # Only called for attributes which are not found otherwise
            if name.startswith("_"):
                raise AttributeError(name)
            return getattr(self._host.numerical_scheme, name)

        def wrap_task_graph_in_load_balancing_and_amr_tasks(self, time_step):
            """! The host wraps the concatenated graph through the scheme of global_tasks_solver"""

        def construct_output(
            self, exahype2_project, peano_project, time_step, add_global_tasks=True
        ):
            """!

            Ask the solver for its task graph once per stage and add the graphs to time_step

            """
            solver = self._solver
            stages = self._host.number_of_stages
            self.stage_graphs = []
            for stage in range(stages):
                graph = solver.create_task_graph(
                    exahype2_project, peano_project, stage, stages
                )
                time_step.merge(graph)
                self.stage_graphs.append(graph)
            title = f"Time stepping task graph for {solver.name}{solver.task_graph_title_suffix}"
            if peano4.api.Logger().is_visualisation:
                time_step.visualise(title=title)
            if peano4.api.Logger().is_text:
                time_step.log_as_text(title=title)

    def __init__(self, name: str, solvers: List[Solver]):
        """!

        @param name: str
          Name of the host. It is used in the titles of the task graphs.

        @param solvers: List[exahype2.api.Solver]
          The solvers in the order in which they are realised. A solver can
          rely on the data of its predecessors.

        """
        if not solvers:
            raise ValueError("A host needs at least one solver")
        self._name = name
        self._solvers = list(solvers)
        self._numerical_scheme = None
        self._attached_schemes = {}
        self._explicit_schemes = {}
        for solver in self._solvers:
            self._assert_has_no_numerical_scheme(solver)

    @property
    def name(self) -> str:
        return self._name

    @property
    def solvers(self) -> List[Solver]:
        """! The solvers in the order of their realisation"""
        return list(self._solvers)

    def _assert_has_no_numerical_scheme(self, solver: Solver):
        attached = self._attached_schemes.get(solver)
        if (
            solver.numerical_scheme is not None
            and solver.numerical_scheme is not attached
        ):
            raise ValueError(
                f"Host {self._name}: solver {solver.name} has a numerical scheme of its own. "
                "Hosted solvers have none, as the host holds the generator of the "
                "numerical scheme. Assign the scheme to the host instead."
            )

    @property
    def numerical_scheme(self) -> Solver.NumericalScheme:
        """!

        The generator of the numerical scheme of global_tasks_solver

        Build it for global_tasks_solver. Assigning it hands it to this solver,
        and a HostedScheme to all other solvers which have no scheme
        (set_numerical_scheme()). Assigning raises an exception if a solver
        has a scheme which the host did not hand out.

        """
        return self._numerical_scheme

    @numerical_scheme.setter
    def numerical_scheme(self, value: Solver.NumericalScheme):
        self.set_numerical_scheme(self.global_tasks_solver, value)

    def set_numerical_scheme(self, solver: Solver, scheme: Solver.NumericalScheme):
        """!

        Let the host hand a numerical scheme to a solver

        Use it for a solver that runs a complete scheme of its own next to the
        others (additive coupling of full solvers). Solvers whose graph is
        built by the scheme of the host
        (@ref exahype2.api.Solver.Solver.create_task_graph()) need none.
        Direct assignment to a hosted solver is an error.

        @param solver: exahype2.api.Solver
          One of the solvers of the host.

        @param scheme: exahype2.api.Solver.NumericalScheme
          The scheme, built for the solver.

        """
        if not isinstance(scheme, Solver.NumericalScheme):
            raise TypeError(
                f"numerical scheme has to be a Solver.NumericalScheme, but got {type(scheme)}"
            )
        if not any(solver is hosted for hosted in self._solvers):
            raise ValueError(f"Host {self._name} does not host solver {solver.name}")
        for hosted in self._solvers:
            self._assert_has_no_numerical_scheme(hosted)
        self._explicit_schemes[solver] = scheme
        if solver is self.global_tasks_solver:
            self._numerical_scheme = scheme
        for hosted in self._solvers:
            attached = self._explicit_schemes.get(hosted)
            if attached is None and self._numerical_scheme is not None:
                attached = SolverHost.HostedScheme(hosted, self)
            if attached is not None:
                self._attached_schemes[hosted] = attached
                hosted.numerical_scheme = attached

    @property
    def number_of_stages(self) -> int:
        """!

        Number of stages of the time stepping scheme, i.e. how often the
        host asks the solvers for a task graph

        It is the Runge-Kutta order (rk_order) of the host's scheme if it has
        one, and one otherwise.

        """
        return getattr(self._numerical_scheme, "rk_order", 1)

    @staticmethod
    def run_in_parallel(
        graph: peano4.api.taskgraph.Graph,
        branches: List[peano4.api.taskgraph.Graph],
        kick_off_name: str = "KickOff",
    ) -> peano4.api.taskgraph.Task:
        """!

        Additive coupling: let the graphs of several solvers run next to each other

        The routine adds all tasks of the branches to graph. It also adds a
        KickOff task which the first tasks (Graph.source_tasks()) of each
        branch depend on, so all branches start together. Call it once per
        Runge-Kutta stage with the graphs that the solvers returned for this stage
        (@ref exahype2.api.Solver.Solver.create_task_graph()), and make the
        KickOff depend on the end of the previous stage. If branches read and
        write the same mesh data, give each branch its own temporary copy
        (@ref page_exahype_programmers_guide_coupling).

        @param graph: peano4.api.taskgraph.Graph
          Receives the KickOff task and the tasks of the branches.

        @param branches: List[peano4.api.taskgraph.Graph]
          One graph per solver and stage.

        @param kick_off_name: str
          Name of the KickOff task. Has to be unique within graph.

        @return peano4.api.taskgraph.Task
          The KickOff task.

        """
        kick_off = peano4.api.taskgraph.Task(
            name=kick_off_name,
            entity=peano4.api.taskgraph.Task.GridEntity.GLOBAL_START_OF_STEP,
        )
        for branch in branches:
            for source in branch.source_tasks():
                source.add_dependency(kick_off)
            graph.merge(branch)
        graph.add_task(kick_off)
        return kick_off

    @property
    def global_tasks_solver(self) -> Solver:
        """!

        The one solver which adds the global tasks (load balancing switch,
        dynamic mesh refinement and mesh adaptation) to the task graphs. By
        default the first one.

        """
        return self._solvers[0]

    @abc.abstractmethod
    def couple(
        self,
        exahype2_project,
        peano_project,
        own_graphs: dict,
        host_graphs: PhaseGraphs,
    ):
        """!

        Tie the solvers together

        Called after the graphs of the solvers have been concatenated into
        host_graphs. The routine adds the tasks and dependencies that
        couple the solvers. It may look up the tasks of the solvers in
        host_graphs or in the graphs they built (own_graphs).

        @param exahype2_project: exahype2.api.Project
          The project that hosts the solvers.

        @param peano_project: peano4.api.Project
          The Peano project that the exahype2 project assembles.

        @param own_graphs: Dict[exahype2.api.Solver, PhaseGraphs]
          The graphs which each solver has built.

        @param host_graphs: PhaseGraphs
          The concatenation of the graphs of all solvers. The routine
          modifies them.

        """

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

        Realise all solvers into graphs of their own, couple them, and add the outcome to the project's graphs

        The arguments are those of exahype2.api.Solver.generate_solver_realisation().

        """
        peano4.api.Logger().new_section("Host " + self._name)
        if self._numerical_scheme is None:
            raise ValueError(
                f"Host {self._name} has no numerical_scheme. Assign one to the host."
            )
        for solver in self._solvers:
            self._assert_has_no_numerical_scheme(solver)
        own_graphs = {}
        host_graphs = PhaseGraphs()
        for solver in self._solvers:
            graphs = PhaseGraphs()
            solver.generate_solver_realisation(
                exahype2_project,
                peano_project,
                *graphs.graphs(),
                add_global_tasks=False,
            )
            own_graphs[solver] = graphs
            host_graphs.merge(graphs)
        self.couple(exahype2_project, peano_project, own_graphs, host_graphs)

        ## The global tasks wrap the coupled graph, i.e. they are added last
        global_solver = self.global_tasks_solver
        global_solver.numerical_scheme.wrap_task_graph_in_load_balancing_and_amr_tasks(
            host_graphs.time_step
        )
        if global_solver.mesh_adaptation is not None:
            global_solver.mesh_adaptation.construct_output(
                exahype2_project, peano_project, host_graphs.mesh_adaptation
            )

        title = (
            "Time stepping task graph for host "
            + self._name
            + " (concatenation of the graphs of "
            + ", ".join(solver.name for solver in self._solvers)
            + ")"
        )
        if peano4.api.Logger().is_visualisation:
            host_graphs.time_step.visualise(title=title)
        if peano4.api.Logger().is_text:
            host_graphs.time_step.log_as_text(title=title)
        for graph, project_graph in zip(
            host_graphs.graphs(),
            [setup_simulation, time_step, plot, mesh_adaptation],
        ):
            project_graph.merge(graph)
