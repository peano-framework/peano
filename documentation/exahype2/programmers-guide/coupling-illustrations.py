# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Create the task graphs of the page "Coupling solvers"

Run from anywhere via

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 documentation/exahype2/programmers-guide/coupling-illustrations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script writes four pictures next to itself. They show the time stepping task graph of

- Euler with Finite Volumes (coupling-graph-euler.png),
- CCZ4 with fourth-order Finite Differences and Runge-Kutta 2, without any
  additional step such as lopsided advection (coupling-graph-ccz4.png), and
- the two solvers coupled in an outer additive way
  (coupling-graph-outer-additive.png) or in an outer multiplicative way
  (coupling-graph-outer-multiplicative.png).

The generated project goes into a temporary directory. The script uses a
matplotlib backend without windows, i.e. nothing pops up.

"""
import os
import sys
import tempfile

Here = os.path.dirname(os.path.abspath(__file__))
Root = os.path.abspath(os.path.join(Here, "..", "..", ".."))
sys.path.insert(0, os.path.join(Root, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import peano4.api
import exahype2.api
import exagrype.api


def plot_graph(graph, title, file_name):
    plt.close("all")
    graph.visualise(title=title)
    output = os.path.join(Here, file_name)
    plt.gcf().savefig(output, dpi=90)
    plt.close("all")
    print("Wrote", output)


def create_host(host_class, ccz4_patch_size=8):
    coupled = host_class()
    ccz4, euler = coupled.ccz4, coupled.euler
    ccz4.add_pde_term(exagrype.api.initialvalues.gauge_wave())
    ccz4.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        ccz4, ccz4_patch_size, rk_order=2
    )
    euler.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(
        euler, 2 * ccz4_patch_size + 1
    )
    ccz4.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        ccz4, volume_size=1.01 / 3 / ccz4_patch_size
    )
    ccz4.plotter = exahype2.api.rkfd4.PlotAsVTKFile(
        ccz4, first_plot_time_stamp=0.0, time_in_between_plots=0.1
    )
    euler.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
        euler, volume_size=1.01 / 3 / (2 * ccz4_patch_size + 1)
    )
    euler.plotter = exahype2.api.fv.PlotAsVTKFile(
        euler, first_plot_time_stamp=0.0, time_in_between_plots=0.1
    )
    return coupled


def plot_host(host_class, file_name, title, single_solver_graphs):
    peano4.api.Logger().file_name = os.path.join(
        tempfile.gettempdir(), "coupling-illustrations-log.md"
    )
    coupled = create_host(host_class)
    single_solver_pictures = {
        coupled.ccz4: (
            "Time stepping task graph of CCZ4 (Runge-Kutta 2)",
            "coupling-graph-ccz4.png",
        ),
        coupled.euler: (
            "Time stepping task graph of Euler (Finite Volumes)",
            "coupling-graph-euler.png",
        ),
    }

    # Each solver builds its time step graph on its own. Draw it before the
    # coupling adds its dependencies.
    for solver in coupled.solvers:

        def realise(*args, _solver=solver, _original=solver.generate_solver_realisation, **kwargs):
            outcome = _original(*args, **kwargs)
            if single_solver_graphs:
                plot_graph(args[3], *single_solver_pictures[_solver])
            return outcome

        solver.generate_solver_realisation = realise

    # The load balancing and AMR tasks are added after the coupling, but they
    # wrap the graph, i.e. the picture shows the coupling itself
    scheme = coupled.ccz4.numerical_scheme
    original_wrap = scheme.wrap_task_graph_in_load_balancing_and_amr_tasks

    def wrap(coupled_time_step):
        plot_graph(coupled_time_step, title, file_name)
        original_wrap(coupled_time_step)

    scheme.wrap_task_graph_in_load_balancing_and_amr_tasks = wrap

    with tempfile.TemporaryDirectory() as directory:
        project = exahype2.api.Project(
            namespace=["illustration"],
            project_name="illustration",
            directory=directory,
            executable="illustration",
        )
        project.add_solver(coupled)
        project.compiler = peano4.api.taskgraph.compiler.FuseSteps.translate
        project.set_global_simulation_parameters(
            dimensions=3,
            offset=[0.0, 0.0, 0.0],
            size=[1.0, 1.0, 1.0],
            min_end_time=0.02,
            max_end_time=0.02,
            periodic_BC=[True, True, True],
        )
        project.set_Peano4_installation(Root, peano4.api.output.CompileMode.RELEASE)
        project.generate_Peano4_project(verbosity=peano4.api.LoggerVerbosity.TEXT)


if __name__ == "__main__":
    plot_host(
        exagrype.api.euler.CCZ4WithEuler_OuterAdditiveCoupling,
        "coupling-graph-outer-additive.png",
        "Outer additive coupling of CCZ4 (RK2) and Euler (FV)",
        single_solver_graphs=True,
    )
    plot_host(
        exagrype.api.euler.CCZ4WithEuler_OuterMultiplicativeCoupling,
        "coupling-graph-outer-multiplicative.png",
        "Outer multiplicative coupling of CCZ4 (RK2) and Euler (FV)",
        single_solver_graphs=False,
    )
