# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Create the coupled task graph of the page "Coupling solvers"

Run from anywhere via

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 documentation/exahype2/programmers-guide/coupling-illustrations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script writes coupling-additive-task-graph-coupled.png next to it: the
time stepping task graph of CCZ4 coupled to Euler, as it looks after
CCZ4WithEuler.couple() has inserted the KickOff and WrapUp tasks.
It is built from the benchmark benchmarks/exagrype/gauge-wave-with-Euler with
the same solver setup. The generated files go into a temporary directory.

"""
import os
import sys
import tempfile

Here = os.path.dirname(os.path.abspath(__file__))
Root = os.path.abspath(os.path.join(Here, "..", "..", ".."))
sys.path.insert(0, os.path.join(Root, "src"))
sys.path.insert(0, os.path.join(Root, "benchmarks", "exagrype", "gauge-wave-with-Euler"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import peano4.api
import exahype2.api
import exagrype.api

from CCZ4WithEuler import CCZ4WithEuler


def plot_coupled_time_step(file_name):
    peano4.api.Logger().file_name = os.path.join(
        tempfile.gettempdir(), "coupling-illustrations-log.md"
    )
    coupled = CCZ4WithEuler()
    ccz4, euler = coupled.ccz4, coupled.euler
    Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
    coupled.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        ccz4,
        9,
        rk_order=4,
        right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
        advection_velocity=exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY,
        advected_quantities=ccz4.lopsided_advection_quantities(),
    )
    coupled.set_numerical_scheme(
        euler,
        exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(euler, 9, rk_order=4),
    )
    for solver in [ccz4, euler]:
        solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
            solver, volume_size=1.01 / 27
        )
        solver.plotter = exahype2.api.rkfd4.PlotAsVTKFile(
            solver, first_plot_time_stamp=0.0, time_in_between_plots=0.1
        )

    # Grab the time step graph once the coupling has completed it
    graphs = []
    original = coupled.couple

    def couple(exahype2_project, peano_project, own_graphs, host_graphs):
        original(exahype2_project, peano_project, own_graphs, host_graphs)
        graphs.append(host_graphs.time_step)

    coupled.couple = couple

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
    plt.close("all")

    graphs[-1].visualise(title="Time stepping task graph for CCZ4 coupled to Euler")
    output = os.path.join(Here, file_name)
    plt.gcf().savefig(output, dpi=90)
    plt.close(plt.gcf())
    print("Wrote", output)


if __name__ == "__main__":
    plot_coupled_time_step("coupling-additive-task-graph-coupled.png")
