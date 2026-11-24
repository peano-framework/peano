""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

from typing import Dict, List, Union

import exahype2.api
import peano4.api
import peano4.api.taskgraph

from .FOSOCCZ4Solver import FOSOCCZ4Solver
from .FOSOReconstructionSolver import FOSOReconstructionSolver


class _HostedScheme(exahype2.api.Solver.NumericalScheme):
    """!

    The numerical scheme of the reconstruction solver

    The solver has no scheme of its own, but the generic parts of the solver
    still query one for patch size, halo and so forth. The hosted scheme
    forwards all such queries to the scheme of the CCZ4 solver. Its
    construct_output() asks the reconstruction solver for its task graph once
    per Runge-Kutta stage of CCZ4
    (@ref exahype2.api.Solver.Solver.create_task_graph()) and puts the graphs
    into the time stepping graph. The graphs are not connected: wiring them is
    the job of FOSOSolver._couple(). The scheme remembers the graphs in
    stage_graphs.

    """

    def __init__(self, solver, ccz4):
        super().__init__(solver)
        self._ccz4 = ccz4
        self.stage_graphs: List[peano4.api.taskgraph.Graph] = []
        """! The graph that the solver returned for each stage, filled by construct_output() """

    def __getattr__(self, name):
        # Only called for attributes which are not found otherwise
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._ccz4.numerical_scheme, name)

    def wrap_task_graph_in_load_balancing_and_amr_tasks(self, time_step):
        """! FOSOSolver wraps the concatenated graph through the scheme of CCZ4"""

    def construct_output(
        self, exahype2_project, peano_project, time_step, add_global_tasks=True
    ):
        solver = self._solver
        stages = getattr(self._ccz4.numerical_scheme, "rk_order", 1)
        self.stage_graphs = []
        for stage in range(stages):
            graph = solver.create_task_graph(
                exahype2_project, peano_project, stage, stages
            )
            time_step.merge(graph)
            self.stage_graphs.append(graph)
        title = f"Time stepping task graph for {solver.name}"
        if peano4.api.Logger().is_visualisation:
            time_step.visualise(title=title)
        if peano4.api.Logger().is_text:
            time_step.log_as_text(title=title)


class FOSOSolver:
    """!

    CCZ4 in the first-/second-order reconstructed formulation as a coupled pair

    The object hosts two solvers, which take turns in the time stepping task
    graph (@ref page_exagrype_fo_so_ccz4, @ref page_exahype_programmers_guide_coupling):

    - ccz4 (exagrype.api.FOSOCCZ4Solver) evolves the 25 primary unknowns of
      CCZ4. It holds the 33 derivatives as auxiliary variables. It is the
      the one which owns the global tasks: Both solvers are
      realised as intermediate steps, and after concatenation and coupling, the
      host wraps the whole graph into the load balancing and dynamic mesh
      refinement tasks through the scheme of ccz4
      (wrap_task_graph_in_load_balancing_and_amr_tasks()).
    - reconstruction (exagrype.api.FOSOReconstructionSolver) evolves the
      derivatives, i.e. it recomputes them from the primary unknowns in
      front of each Runge-Kutta stage of ccz4.

    The host holds the numerical scheme: Build it for ccz4 and assign it to
    the host. The reconstruction solver receives a scheme that forwards to it. Configure ccz4 otherwise as any other solver (grid
    construction, plotter, constraint enforcement, ...), and then add the
    host to the project with exahype2.api.Project.add_solver(). The
    reconstruction needs no configuration: it takes patch size and halo from
    the scheme.

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    solver = exagrype.api.FOSOSolver("SO")
    solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(solver.ccz4, patch_size, rk_order=4, ...)
    ...
    project.add_solver(solver)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ## Task graph

    Both solvers build a task graph of their own, and report it: the scheme of the host
    builds the graph of ccz4, and asks the reconstruction solver for its graph
    once per stage (exagrype.api.FOSOReconstructionSolver.create_task_graph()). The reconstruction
    solver's graph is a chain per stage (see @ref exagrype.api.FOSOReconstructionSolver),
    and the chains of different stages are not connected.
    The host concatenates the graphs and wires them (_couple()): The first tasks of
    each stage's chain take over the dependencies of the CCZ4 solver's reconstruction
    of this stage, and the latter depends on the last task of the chain. The host
    reports the outcome.

    """

    def __init__(
        self,
        name: str,
        parameters: Dict[str, Union[int, float]] = None,
        initial_condition: exahype2.api.InitPointWisely = None,
        reconstruction_name: str = None,
    ):
        """!

        @param name: str
          Name of the CCZ4 solver, @see exahype2.api.Solver.

        @param parameters: Dict[str, Union[int, float]]
          CCZ4 parameters, @see exagrype.api.FOSOCCZ4Solver.

        @param initial_condition: exahype2.api.InitPointWisely
          Initial condition, @see exagrype.api.FOSOCCZ4Solver.

        @param reconstruction_name: str
          Name of the reconstruction solver. Defaults to name plus
          "Reconstruction".

        """
        self.ccz4 = FOSOCCZ4Solver(name, parameters, initial_condition)
        """! The solver that evolves the primary unknowns """
        self.reconstruction = FOSOReconstructionSolver(
            reconstruction_name
            if reconstruction_name is not None
            else name + "Reconstruction",
            self.ccz4,
        )
        """! The solver that reconstructs the derivatives """
        self._name = name
        self._numerical_scheme = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def solvers(self):
        """! The solvers in the order of their realisation: CCZ4 first, as the reconstruction needs its data """
        return [self.ccz4, self.reconstruction]

    @property
    def numerical_scheme(self) -> exahype2.api.Solver.NumericalScheme:
        """! The scheme of ccz4. Assigning it hands a forwarding scheme to the reconstruction solver """
        return self._numerical_scheme

    @numerical_scheme.setter
    def numerical_scheme(self, value: exahype2.api.Solver.NumericalScheme):
        self.ccz4.numerical_scheme = value
        self._numerical_scheme = value
        self.reconstruction.numerical_scheme = _HostedScheme(
            self.reconstruction, self.ccz4
        )

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

        Realise both solvers, couple their time stepping graphs, and add the outcome to the project's graphs

        The arguments are those of exahype2.api.Solver.generate_solver_realisation().
        Both solvers are realised without global tasks. The load balancing and the dynamic
        mesh refinement wrap the coupled time step graph afterwards, through the scheme of CCZ4.

        """
        peano4.api.Logger().new_section("Host " + self.name)
        if self._numerical_scheme is None:
            raise ValueError(
                f"Host {self.name} has no numerical_scheme. Assign one to the host."
            )
        coupled_time_step = peano4.api.taskgraph.Graph()
        for solver in self.solvers:
            own_time_step = peano4.api.taskgraph.Graph()
            solver.generate_solver_realisation(
                exahype2_project,
                peano_project,
                setup_simulation,
                own_time_step,
                plot,
                mesh_adaptation,
                add_global_tasks=False,
            )
            coupled_time_step.merge(own_time_step)
        self._couple(exahype2_project, peano_project, coupled_time_step)

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

    def _couple(self, exahype2_project, peano_project, time_step):
        """!

        Let each stage of the CCZ4 solver use the freshly reconstructed derivatives

        In each stage, the CCZ4 solver reconstructs the halo of its unknowns
        and the halo of the derivatives (both tasks interpolate the hanging
        faces themselves, see
        src/exahype2/api/adrs/0001-interpolation-and-restriction-ride-along-with-face-users.md),
        and then evaluates the right-hand side. The reconstruction solver
        recomputes the derivatives from the unknowns in between, i.e.

        - the first reconstruction tasks of the stage wait for whatever the
          CCZ4 solver's reconstruction of the derivatives waits for, and
        - the CCZ4 solver's reconstruction of the derivatives and its
          reconstruction of the unknowns wait for the last reconstruction task.

        The chain is hence a block within the stage: The maximum concurrency of
        the merged graph does not exceed the one of the two graphs.

        @param exahype2_project: exahype2.api.Project
          The project that hosts the pair (unused).

        @param peano_project: peano4.api.Project
          The Peano project that the exahype2 project assembles (unused).

        @param time_step: peano4.api.taskgraph.Graph
          The concatenated time stepping graphs of both solvers. It receives the dependencies.

        """

        def find(task_name, stage):
            found = [t for t in time_step.tasks if t.name == task_name]
            if len(found) != 1:
                raise ValueError(
                    f"Host {self.name}: cannot find the task {task_name} "
                    f"of stage {stage} of solver {self.ccz4.name} in its time stepping task graph"
                )
            return found[0]

        ## The reconstruction solver returned one graph per stage (HostedScheme)
        for stage, graph in enumerate(
            self.reconstruction.numerical_scheme.stage_graphs
        ):
            first_tasks = graph.source_tasks()
            (last_task,) = graph.sink_tasks()
            (
                unknowns_reconstruction_name,
                derivatives_reconstruction_name,
            ) = self.reconstruction.partner_task_names(stage)
            unknowns_reconstruction = find(unknowns_reconstruction_name, stage)
            derivatives_reconstruction = find(derivatives_reconstruction_name, stage)
            for dependency in derivatives_reconstruction.dependencies:
                for first_task in first_tasks:
                    first_task.add_dependency(dependency)
            derivatives_reconstruction.add_dependency(last_task)
            unknowns_reconstruction.add_dependency(last_task)
