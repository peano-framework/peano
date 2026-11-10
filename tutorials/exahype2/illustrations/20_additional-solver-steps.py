# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Create the PNG illustrations of the tutorial 20_additional-solver-steps.ipynb

The script writes the following pictures into the directory of this script:

- 20_additional-solver-steps-fv-task-graph.png: the time stepping task graph
  of the Finite Volumes with a density floor as additional solver step;
- 20_additional-solver-steps-rk-task-graph.png: the same for the
  fourth-order Finite Differences with a Runge-Kutta scheme of order two,
  where the floor follows each Runge-Kutta stage;
- 20_additional-solver-steps-profile.png: the density along y=0 after one
  period with and without the density floor.

Call it from anywhere:

     python3 tutorials/exahype2/illustrations/20_additional-solver-steps.py

All builds and outputs go into a temporary directory (or --work-directory).
A run whose directory already holds the output is not repeated. The script
builds four executables, i.e. it runs for a few minutes.

"""
import argparse
import os
import subprocess
import sys
import tempfile
import types

Here = os.path.dirname(os.path.abspath(__file__))
PeanoRoot = os.path.abspath(os.path.join(Here, "..", "..", ".."))
sys.path.insert(0, os.path.join(PeanoRoot, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import peano4.api
import peano4.api.taskgraph
import exahype2.api
import toolbox.blockstructured.api
import tarch.plotter.api.notebook


Flux = """
  const double irho  = 1.0 / Q[0];
  constexpr double gamma = 1.4;
  const double p = (gamma - 1.0) * (Q[3] - 0.5 * irho * (Q[1] * Q[1] + Q[2] * Q[2]));
  switch (normal) {
    case 0:
      F[0] = Q[1];
      F[1] = irho * Q[1] * Q[1] + p;
      F[2] = irho * Q[1] * Q[2];
      F[3] = irho * Q[1] * (Q[3] + p);
      break;
    case 1:
      F[0] = Q[2];
      F[1] = irho * Q[2] * Q[1];
      F[2] = irho * Q[2] * Q[2] + p;
      F[3] = irho * Q[2] * (Q[3] + p);
      break;
  }
"""

MaxEigenvalue = """
  const double irho  = 1.0 / Q[0];
  constexpr double gamma = 1.4;
  const double p = (gamma - 1.0) * (Q[3] - 0.5 * irho * (Q[1] * Q[1] + Q[2] * Q[2]));
  const double c = std::sqrt(gamma * p * irho);
  const double u = Q[1 + normal] * irho;
  lambdaMax[0] = std::max(std::abs(u + c), std::abs(u - c));
"""

InitPointWisely = """
  constexpr double gamma = 1.4;
  constexpr double p     = 1.0;
  constexpr double u     = 1.0;
  const double     rho   = 1.0 + 0.2 * std::sin(2.0 * M_PI * (x[0] + 0.1) / 0.2);
  Q[0] = rho;
  Q[1] = rho * u;
  Q[2] = 0.0;
  Q[3] = p / (gamma - 1) + 0.5 * rho * u * u;
"""

DensityFloor = """
  constexpr double gamma      = 1.4;
  constexpr double minDensity = 0.9;
  if (Q[0] < minDensity) {
    const double u = Q[1] / Q[0];
    const double v = Q[2] / Q[0];
    const double p = (gamma - 1.0) * (Q[3] - 0.5 * Q[0] * (u * u + v * v));
    Q[0] = minDensity;
    Q[1] = minDensity * u;
    Q[2] = minDensity * v;
    Q[3] = p / (gamma - 1.0) + 0.5 * minDensity * (u * u + v * v);
  }
"""



def insert_density_floor(solver, time_step, tag, patch):
    """!

    Insert the density floor after each task of solver which carries tag

    Mirrors the routine of the tutorial: The numerical scheme knows which
    patch its tagged tasks write and passes it in.

    """
    for anchor in time_step.find_tasks(tag):
        if not anchor.name.startswith(solver.name + "_"):
            continue
        task = peano4.api.taskgraph.Task(
            name=f"{solver.name}_Density floor after {anchor.name[len(solver.name) + 1:]}",
            entity=peano4.api.taskgraph.Task.GridEntity.CELL,
        )
        task.use_cell(patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        task.add_action_set(
            toolbox.blockstructured.api.ApplyPointWiselyToPatch(
                patch=patch,
                implementation=DensityFloor,
                guard=solver.grid_selector.createCellGuard(),
                additional_includes=solver._get_default_includes()
                + solver.user_solver_includes,
            )
        )
        time_step.insert_after(anchor, task)


class RusanovWithDensityFloor(exahype2.api.fv.RusanovWithExplicitEuler):
    """! Finite Volumes which insert the density floor after each update"""

    def construct_output(self, exahype2_project, peano_project, time_step, add_global_tasks=True):
        super().construct_output(exahype2_project, peano_project, time_step, add_global_tasks)
        insert_density_floor(
            self._solver,
            time_step,
            exahype2.api.Solver.SolutionUpdateTaskTag,
            self._solver._patch,
        )


class RungeKuttaWithDensityFloor(exahype2.api.rkfd4.CentralDifferencesWithRungeKutta):
    """! Runge-Kutta Finite Differences which insert the density floor after each stage"""

    def construct_output(self, exahype2_project, peano_project, time_step, add_global_tasks=True):
        super().construct_output(exahype2_project, peano_project, time_step, add_global_tasks)
        insert_density_floor(
            self._solver,
            time_step,
            exahype2.api.Solver.SolutionUpdateTaskTag,
            self._solver._patch,
        )
        insert_density_floor(
            self._solver,
            time_step,
            exahype2.api.Solver.IntermediateSolutionUpdateTag,
            self._solver._stage_input,
        )


DomainOffset = -0.1
DomainWidth = 0.2
VolumeSize = 0.0075
EndTime = 0.2


def exact_density(x, t):
    return 1.0 + 0.2 * np.sin(
        2.0 * np.pi * (np.asarray(x) - t - DomainOffset) / DomainWidth
    )


def create_solver(name, rk_order, density_floor):
    """!

    Create the Euler solver of the tutorial

    @param rk_order: int or None
      None selects the Finite Volumes, an integer the fourth-order Finite
      Differences with the Runge-Kutta scheme of this order.

    """
    solver = exahype2.api.Solver(name, 4)
    solver.add_user_solver_includes("#include <cmath>\n#include <algorithm>\n")
    solver.add_pde_term(exahype2.api.Flux(implementation=Flux, stateless=False))
    solver.add_pde_term(
        exahype2.api.MaxEigenvalue(implementation=MaxEigenvalue, stateless=False)
    )
    solver.add_pde_term(
        exahype2.api.InitPointWisely(implementation=InitPointWisely, stateless=False)
    )
    if rk_order is None:
        scheme = (
            RusanovWithDensityFloor
            if density_floor
            else exahype2.api.fv.RusanovWithExplicitEuler
        )
        solver.numerical_scheme = scheme(solver, 9)
        solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
            solver, volume_size=VolumeSize
        )
        solver.plotter = exahype2.api.fv.PlotAsVTKFile(
            solver, first_plot_time_stamp=0.0, time_in_between_plots=0.05
        )
    else:
        scheme = (
            RungeKuttaWithDensityFloor
            if density_floor
            else exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
        )
        solver.numerical_scheme = scheme(solver, patch_size=9, rk_order=rk_order)
        solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
            solver, volume_size=VolumeSize
        )
        solver.plotter = exahype2.api.rkfd4.PlotAsVTKFile(
            solver, first_plot_time_stamp=0.0, time_in_between_plots=0.05
        )
    return solver


def plot_task_graph(rk_order, file_name, title):
    """! Construct the time stepping task graph including the density floor and draw it"""
    peano4.api.Logger().file_name = os.path.join(
        tempfile.gettempdir(), "20-additional-solver-steps-log.md"
    )
    with tempfile.TemporaryDirectory() as directory:
        peano_project = peano4.api.Project(
            ["illustration"], "illustration", directory=directory
        )
        solver = create_solver("Euler", rk_order, density_floor=True)
        exahype2_project = types.SimpleNamespace(periodic_BC=[True, True])
        time_step = peano4.api.taskgraph.Graph()
        solver.numerical_scheme.construct_output(
            exahype2_project, peano_project, time_step
        )
    time_step.visualise(title=title)
    output = os.path.join(Here, file_name)
    plt.gcf().savefig(output, dpi=90)
    plt.close(plt.gcf())
    print("Wrote", output)


def build_and_run(rk_order, density_floor, directory, peano_root):
    name = "EulerFV" if rk_order is None else "EulerFD4"
    patch_file = os.path.join(directory, "solution-{}.peano-patch-file".format(name))
    if os.path.exists(patch_file):
        print("reuse {}".format(directory))
        return patch_file
    print("build and run {}".format(directory))
    os.makedirs(directory, exist_ok=True)
    project = exahype2.api.Project(
        namespace=["tutorials", "exahype2", "additionalsteps"],
        project_name="AdditionalSteps",
        directory=directory,
        executable="additional-steps",
    )
    project.add_solver(create_solver(name, rk_order, density_floor))
    project.set_global_simulation_parameters(
        dimensions=2,
        offset=[DomainOffset, DomainOffset],
        size=[DomainWidth, DomainWidth],
        min_end_time=EndTime,
        max_end_time=EndTime,
        periodic_BC=[True, True, True],
    )
    project.set_Peano4_installation(peano_root, peano4.api.output.CompileMode.RELEASE)
    peano4_project = project.generate_Peano4_project(
        verbosity=peano4.api.LoggerVerbosity.TEXT
    )
    peano4_project.generate()
    subprocess.run(
        ["make", "-j4"], cwd=directory, check=True, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["./additional-steps"], cwd=directory, check=True, stdout=subprocess.DEVNULL
    )
    return patch_file


def density_along_x_axis(patch_file):
    """! Return time stamp plus grid cell centres and densities along y=0 of the last snapshot"""
    time_stamp, files = tarch.plotter.api.notebook.read_snapshots(patch_file)[-1]
    x, rho = [], []
    for file_name in files:
        for offset, size, values in tarch.plotter.api.notebook.read_patches(file_name):
            size = np.array([DomainWidth / round(DomainWidth / s) for s in size])
            offset = (
                np.round((np.array(offset) - DomainOffset) / size) * size + DomainOffset
            )
            dofs = values.shape[0]
            if offset[1] <= 0.0 < offset[1] + size[1]:
                row = int((0.0 - offset[1]) / (size[1] / dofs))
                x += list(offset[0] + (np.arange(dofs) + 0.5) * size[0] / dofs)
                rho += list(values[row, :, 0])
    order = np.argsort(x)
    return time_stamp, np.array(x)[order], np.array(rho)[order]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the PNGs of the tutorial 20_additional-solver-steps.ipynb"
    )
    parser.add_argument(
        "--work-directory",
        default=None,
        help="Directory for builds and outputs (default: temporary directory)",
    )
    parser.add_argument(
        "--peano-root",
        default=PeanoRoot,
        help="Configured and built Peano installation to link against",
    )
    args = parser.parse_args()

    plot_task_graph(
        None,
        "20_additional-solver-steps-fv-task-graph.png",
        "Finite Volumes with a density floor",
    )
    plot_task_graph(
        2,
        "20_additional-solver-steps-rk-task-graph.png",
        "Runge-Kutta of order two with a density floor",
    )

    import contextlib

    with contextlib.ExitStack() as stack:
        work_directory = os.path.abspath(
            args.work_directory or stack.enter_context(tempfile.TemporaryDirectory())
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, rk_order, title in [
            (axes[0], None, "Finite Volumes"),
            (axes[1], 4, "Finite Differences, Runge-Kutta 4"),
        ]:
            x_fine = np.linspace(DomainOffset, DomainOffset + DomainWidth, 400)
            ax.plot(
                x_fine, exact_density(x_fine, EndTime), "k-", linewidth=1, label="exact"
            )
            for density_floor, marker in [(False, "o"), (True, "s")]:
                directory = os.path.join(
                    work_directory,
                    "{}-{}".format(
                        "fv" if rk_order is None else "rk4",
                        "floor" if density_floor else "plain",
                    ),
                )
                patch_file = build_and_run(
                    rk_order, density_floor, directory, os.path.abspath(args.peano_root)
                )
                t, x, rho = density_along_x_axis(patch_file)
                print(
                    "{}: t={:.4f}, min density={:.4f}, max density={:.4f}".format(
                        directory, t, rho.min(), rho.max()
                    )
                )
                ax.plot(
                    x,
                    rho,
                    marker,
                    markersize=3,
                    label="with density floor" if density_floor else "without",
                )
            ax.axhline(0.9, color="tab:red", linestyle="--", linewidth=1, label="floor")
            ax.set_title(title)
            ax.set_xlabel("x (along y=0)")
            ax.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
        axes[0].set_ylabel("density after one period")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        output = os.path.join(Here, "20_additional-solver-steps-profile.png")
        fig.savefig(output, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("Wrote", output)
