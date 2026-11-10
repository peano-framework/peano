""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

from typing import Dict, Union

import exahype2.api

from .FOSOCCZ4Solver import FOSOCCZ4Solver
from .FOSOReconstructionSolver import FOSOReconstructionSolver


class FOSOSolver(exahype2.api.SolverHost):
    """!

    CCZ4 in the first-/second-order reconstructed formulation as a coupled pair

    The object hosts two solvers, which take turns in the time stepping task
    graph (@ref page_exagrype_fo_so_ccz4, @ref page_exahype_programmers_guide_coupling):

    - ccz4 (exagrype.api.FOSOCCZ4Solver) evolves the 25 primary unknowns of
      CCZ4. It holds the 34 derivatives as auxiliary variables. It is the
      global_tasks_solver (@ref exahype2.api.SolverHost): Both solvers are
      realised as intermediate steps, and after concatenation and coupling, the
      host wraps the whole graph into the load balancing and dynamic mesh
      refinement tasks through the scheme of ccz4
      (wrap_task_graph_in_load_balancing_and_amr_tasks()).
    - reconstruction (exagrype.api.FOSOReconstructionSolver) evolves the
      derivatives, i.e. it recomputes them from the primary unknowns in
      front of each Runge-Kutta stage of ccz4.

    The host holds the numerical scheme (@ref exahype2.api.SolverHost):
    Build it for ccz4 and assign it to the host. Neither solver has a scheme
    of its own. Configure ccz4 otherwise as any other solver (grid
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
    The host concatenates the graphs and wires them (couple()): The first tasks of
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
        # The CCZ4 solver is realised first, as the reconstruction needs its data
        super().__init__(name, [self.ccz4, self.reconstruction])

    def couple(self, exahype2_project, peano_project, own_graphs, host_graphs):
        """!

        Let each stage of the CCZ4 solver use the freshly reconstructed derivatives

        In each stage, the CCZ4 solver interpolates the unknowns and the
        derivatives onto the hanging faces, reconstructs their halos, and
        evaluates the right-hand side. The reconstruction solver recomputes
        the derivatives from the unknowns in between, i.e.

        - the first reconstruction tasks of the stage wait for whatever the
          CCZ4 solver's interpolation of the derivatives waits for,
        - the interpolation of the primary variables also waits for the
          CCZ4 solver's interpolation of the unknowns, as both write the same
          faces, and
        - the CCZ4 solver's interpolation of the derivatives and its
          reconstruction of the unknowns wait for the last reconstruction task.

        The chain is hence a block within the stage: The maximum concurrency of
        the merged graph does not exceed the one of the two graphs.

        @param exahype2_project: exahype2.api.Project
          The project that hosts the pair (unused).

        @param peano_project: peano4.api.Project
          The Peano project that the exahype2 project assembles (unused).

        @param own_graphs: Dict[exahype2.api.Solver, exahype2.api.PhaseGraphs]
          The graphs of the two solvers (unused).

        @param host_graphs: exahype2.api.PhaseGraphs
          The concatenated graphs. The time step graph receives the dependencies.

        """
        time_step = host_graphs.time_step

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
                unknowns_name,
                derivatives_name,
                reconstruction_name,
            ) = self.reconstruction.partner_task_names(stage)
            unknowns_interpolation = find(unknowns_name, stage)
            derivatives_interpolation = find(derivatives_name, stage)
            for dependency in derivatives_interpolation.dependencies:
                for first_task in first_tasks:
                    first_task.add_dependency(dependency)
            first_tasks[0].add_dependency(unknowns_interpolation)
            derivatives_interpolation.add_dependency(last_task)
            find(reconstruction_name, stage).add_dependency(last_task)
