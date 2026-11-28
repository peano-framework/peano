# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Create the PNG illustrations of the tutorial
45_coupling-sharing-auxiliary-variables.ipynb

The script builds the two tsunamis which share their sea floor, runs them and
writes the initial conditions and the difference of their arrival times into
the directory of this script. It repeats the setup of the notebook, i.e.
the PDE terms are the ones of tutorial 11_tsunami.ipynb for the surface
elevation eta as unknown. Call the script from anywhere:

     python3 tutorials/exahype2/illustrations/45_coupling.py

All builds and outputs go into a temporary directory (or --work-directory).

"""
import argparse
import os
import subprocess
import sys
import tempfile

Here = os.path.dirname(os.path.abspath(__file__))
PeanoRoot = os.path.abspath(os.path.join(Here, "..", "..", ".."))
sys.path.insert(0, os.path.join(PeanoRoot, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import exahype2
import peano4.api
import tarch.plotter.api.notebook

Flux = """
  constexpr double g = 9.81;
  const double h  = Q[Variables::eta] - bathymetry[Bathymetry::b];
  const double un = Q[Variables::hu + normal] / h;
  F[Variables::eta]         = Q[Variables::hu + normal];
  F[Variables::hu]          = Q[Variables::hu] * un;
  F[Variables::hv]          = Q[Variables::hv] * un;
  F[Variables::hu + normal] += 0.5 * g * h * h;
"""

MaxEigenvalue = """
  constexpr double g = 9.81;
  const double h  = Q[Variables::eta] - bathymetry[Bathymetry::b];
  const double un = Q[Variables::hu + normal] / h;
  lambdaMax[0] = std::abs(un) + std::sqrt(g * h);
"""

NonconservativeProduct = """
  constexpr double g = 9.81;
  const double h = Q[Variables::eta] - bathymetry[Bathymetry::b];
  BTimesDeltaQ[Variables::eta]         = 0.0;
  BTimesDeltaQ[Variables::hu]          = 0.0;
  BTimesDeltaQ[Variables::hv]          = 0.0;
  BTimesDeltaQ[Variables::hu + normal] = g * h * deltaBathymetry[Bathymetry::b];
"""

InitialSeaFloor = """
  const int column = std::clamp(static_cast<int>((x[0] + 1.0) / 2.0 * DurhamLogoSize), 0, DurhamLogoSize - 1);
  const int row    = std::clamp(static_cast<int>((1.0 - x[1]) / 2.0 * DurhamLogoSize), 0, DurhamLogoSize - 1);
  bathymetry[Bathymetry::b] = DurhamLogo[row][column] == '1' ? -4.0 : -1.0;
"""

InitialWater = """
  const double dx = x[0] - {centre};
  Q[Variables::eta] = 0.1 * std::exp(-100.0 * (dx * dx + x[1] * x[1]));
  Q[Variables::hu]  = 0.0;
  Q[Variables::hv]  = 0.0;
"""

Bathymetry = os.path.join(Here, "..", "bathymetry")

# Centre of the hump along the x axis
CentreOfFirstTsunami = 0.0
CentreOfSecondTsunami = 0.1


def build_and_run(directory):
    """! Generate, compile and run the coupled setup within directory """
    project = exahype2.api.Project(
        namespace=["tutorials", "exahype2", "tsunami"],
        project_name="Coupling",
        directory=directory,
        executable="coupling",
    )
    bathymetry = exahype2.api.AuxiliaryVariables("Bathymetry", {"b": 1})
    for name, centre, owns_sea_floor in [
        ("TsunamiA", CentreOfFirstTsunami, True),
        ("TsunamiB", CentreOfSecondTsunami, False),
    ]:
        solver = exahype2.api.Solver(name, {"eta": 1, "hu": 1, "hv": 1})
        solver.add_auxiliary_variables(bathymetry)
        solver.add_user_solver_includes(
            '#include <cmath>\n#include <algorithm>\n#include "DurhamLogo.h"\n'
        )
        solver.add_pde_term(exahype2.api.Flux(implementation=Flux, stateless=True))
        solver.add_pde_term(
            exahype2.api.MaxEigenvalue(implementation=MaxEigenvalue, stateless=True)
        )
        solver.add_pde_term(
            exahype2.api.NonconservativeProduct(
                implementation=NonconservativeProduct, stateless=True
            )
        )
        solver.add_pde_term(
            exahype2.api.InitPointWisely(
                implementation=InitialWater.format(centre=centre),
                stateless=False,
            )
        )
        if owns_sea_floor:
            solver.add_pde_term(
                exahype2.api.InitAuxiliaryVariablesPointWisely(
                    implementation=InitialSeaFloor
                )
            )
        solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(solver, 9)
        solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
            solver, volume_size=0.01
        )
        solver.plotter = exahype2.api.fv.PlotAsVTKFile(
            solver, first_plot_time_stamp=0.0, time_in_between_plots=0.01
        )
        project.add_solver(solver)
    project.set_global_simulation_parameters(
        dimensions=2,
        offset=[-1.0, -1.0],
        size=[2.0, 2.0],
        min_end_time=0.4,
        max_end_time=0.4,
        periodic_BC=[True, True, True],
    )
    project.set_Peano4_installation(PeanoRoot, peano4.api.output.CompileMode.RELEASE)
    peano4_project = project.generate_Peano4_project()
    makefile = peano4_project.output.makefile
    makefile.add_cpp_file(os.path.join(Bathymetry, "DurhamLogo.cpp"))
    makefile.add_h_file(os.path.join(Bathymetry, "DurhamLogo.h"))
    makefile.add_header_search_path(Bathymetry)
    peano4_project.generate()
    subprocess.run(["make", "-j"], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["./coupling"], cwd=directory, check=True, stdout=subprocess.DEVNULL)


def read_field(file_name, snapshot, unknown=0):
    """! One unknown of one snapshot as array field[y,x] over the whole domain """
    _, patch_files = tarch.plotter.api.notebook.read_snapshots(file_name)[snapshot]
    patches = [
        patch
        for patch_file in patch_files
        for patch in tarch.plotter.api.notebook.read_patches(patch_file)
    ]
    volume_size = patches[0][1][0] / patches[0][2].shape[1]
    volumes = round(2.0 / volume_size)
    field = np.zeros((volumes, volumes))
    for offset, _, values in patches:
        x = round((offset[0] + 1.0) / volume_size)
        y = round((offset[1] + 1.0) / volume_size)
        n = values.shape[0]
        field[y : y + n, x : x + n] = values[..., unknown]
    return field


def read_time_series(file_name, unknown=0):
    """! Time stamps and an array series[snapshot,y,x] of one unknown """
    snapshots = tarch.plotter.api.notebook.read_snapshots(file_name)
    times = np.array([time for time, _ in snapshots])
    series = np.array([read_field(file_name, s, unknown) for s in range(len(snapshots))])
    return times, series


def arrival_times(times, series, threshold):
    """!

    First time at which the surface elevation exceeds the threshold

    The time is interpolated linearly between the two snapshots which enclose
    the crossing. Volumes which the wave never reaches hold NaN.

    """
    exceeded = series >= threshold
    first = np.argmax(exceeded, axis=0)
    reached = exceeded.any(axis=0)
    after = np.take_along_axis(series, first[None], axis=0)[0]
    before = np.take_along_axis(series, np.maximum(first - 1, 0)[None], axis=0)[0]
    weight = np.where(first > 0, (threshold - before) / np.maximum(after - before, 1e-300), 1.0)
    arrival = times[np.maximum(first - 1, 0)] + weight * (times[first] - times[np.maximum(first - 1, 0)])
    return np.where(reached, arrival, np.nan)


def plot_initial_conditions(first_file, second_file, prefix):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, file_name, title in zip(
        axes, [first_file, second_file], ["first tsunami (hump at x=0)", "second tsunami (hump at x=0.1)"]
    ):
        tarch.plotter.api.notebook.plot_snapshot(
            file_name, 0, 0, ax=ax, label="surface elevation eta"
        )
        ax.set_title(title)
    fig.savefig(os.path.join(Here, prefix + "initial.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_arrival_difference(first_file, second_file, prefix, threshold=0.002):
    times, first = read_time_series(first_file)
    _, second = read_time_series(second_file)
    arrival_first = arrival_times(times, first, threshold)
    arrival_second = arrival_times(times, second, threshold)
    difference = arrival_second - arrival_first
    fig, ax = plt.subplots(figsize=(6.5, 5))
    bound = np.nanmax(np.abs(difference))
    image = ax.imshow(
        difference,
        origin="lower",
        extent=(-1, 1, -1, 1),
        cmap="coolwarm",
        vmin=-bound,
        vmax=bound,
    )
    fig.colorbar(image, label="arrival time second - arrival time first")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Arrival of eta >= {threshold}")
    fig.savefig(os.path.join(Here, prefix + "arrival-difference.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the PNGs of the tutorial 45_coupling-sharing-auxiliary-variables.ipynb"
    )
    parser.add_argument(
        "--work-directory",
        default=None,
        help="Directory for builds and outputs (default: temporary directory)",
    )
    args = parser.parse_args()

    work_directory = args.work_directory or tempfile.mkdtemp(prefix="45_coupling-")
    print("build and run in {}".format(work_directory))
    build_and_run(work_directory)
    first_file = os.path.join(work_directory, "solution-TsunamiA.peano-patch-file")
    second_file = os.path.join(work_directory, "solution-TsunamiB.peano-patch-file")
    plot_initial_conditions(first_file, second_file, "45_coupling-")
    plot_arrival_difference(first_file, second_file, "45_coupling-")
