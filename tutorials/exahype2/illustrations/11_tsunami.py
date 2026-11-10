# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Create the PNG illustrations of the tutorial 11_tsunami.ipynb

The script builds the tutorial's shallow water setup, runs it and writes
snapshots of the water height into the directory of this script. It does
so twice: for the flat sea floor and for the sea floor shaped like the logo
of Durham University (section "Adding bathymetry and external files", which
reads the logo from tutorials/exahype2/bathymetry). The tutorial derives
flux and eigenvalue of the latter with SymPy, whereas we use the handwritten
versions here. Both give the same results. Finally, it runs the setup of
section "Letting the bathymetry shape the flow", which uses the surface
elevation eta as unknown, and plots it at t=0, 0.1, 0.2 and 0.4. Call the
script from anywhere:

     python3 tutorials/exahype2/illustrations/11_tsunami.py

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

import peano4.api
import exahype2
import tarch.plotter.api.notebook


Flux = """
  constexpr double g = 9.81;
  const double h  = Q[Variables::h];
  const double un = Q[Variables::hu + normal] / h;
  F[Variables::h]           = Q[Variables::hu + normal];
  F[Variables::hu]          = Q[Variables::hu] * un;
  F[Variables::hv]          = Q[Variables::hv] * un;
  F[Variables::hu + normal] += 0.5 * g * h * h;
"""

MaxEigenvalue = """
  constexpr double g = 9.81;
  const double h  = Q[Variables::h];
  const double un = Q[Variables::hu + normal] / h;
  lambdaMax[0] = std::abs(un) + std::sqrt(g * h);
"""

InitPointWisely = """
  Q[Variables::h]  = 1.0 + 0.1 * std::exp(-100.0 * (x[0] * x[0] + x[1] * x[1]));
  Q[Variables::hu] = 0.0;
  Q[Variables::hv] = 0.0;
"""

InitPointWiselyWithLogo = """
  const int column = std::clamp(static_cast<int>((x[0] + 1.0) / 2.0 * DurhamLogoSize), 0, DurhamLogoSize - 1);
  const int row    = std::clamp(static_cast<int>((1.0 - x[1]) / 2.0 * DurhamLogoSize), 0, DurhamLogoSize - 1);
  const double depth = DurhamLogo[row][column] == '1' ? 4.0 : 1.0;

  Q[Variables::h]  = depth + 0.1 * std::exp(-100.0 * (x[0] * x[0] + x[1] * x[1]));
  Q[Variables::hu] = 0.0;
  Q[Variables::hv] = 0.0;
  bathymetry[Bathymetry::b] = -depth;
"""

FluxEta = """
  constexpr double g = 9.81;
  const double h  = Q[Variables::eta] - bathymetry[Bathymetry::b];
  const double un = Q[Variables::hu + normal] / h;
  F[Variables::eta]         = Q[Variables::hu + normal];
  F[Variables::hu]          = Q[Variables::hu] * un;
  F[Variables::hv]          = Q[Variables::hv] * un;
  F[Variables::hu + normal] += 0.5 * g * h * h;
"""

MaxEigenvalueEta = """
  constexpr double g = 9.81;
  const double h  = Q[Variables::eta] - bathymetry[Bathymetry::b];
  const double un = Q[Variables::hu + normal] / h;
  lambdaMax[0] = std::abs(un) + std::sqrt(g * h);
"""

NonconservativeProductEta = """
  constexpr double g = 9.81;
  const double h = Q[Variables::eta] - bathymetry[Bathymetry::b];
  BTimesDeltaQ[Variables::eta]         = 0.0;
  BTimesDeltaQ[Variables::hu]          = 0.0;
  BTimesDeltaQ[Variables::hv]          = 0.0;
  BTimesDeltaQ[Variables::hu + normal] = g * h * deltaBathymetry[Bathymetry::b];
"""

InitPointWiselyEta = """
  const int column = std::clamp(static_cast<int>((x[0] + 1.0) / 2.0 * DurhamLogoSize), 0, DurhamLogoSize - 1);
  const int row    = std::clamp(static_cast<int>((1.0 - x[1]) / 2.0 * DurhamLogoSize), 0, DurhamLogoSize - 1);
  const double depth = DurhamLogo[row][column] == '1' ? 4.0 : 1.0;

  Q[Variables::eta] = 0.1 * std::exp(-100.0 * (x[0] * x[0] + x[1] * x[1]));
  Q[Variables::hu]  = 0.0;
  Q[Variables::hv]  = 0.0;
  bathymetry[Bathymetry::b] = -depth;
"""

Bathymetry = os.path.join(Here, "..", "bathymetry")

PatchFile = "solution-SWE.peano-patch-file"


def build_and_run(
    directory,
    init=InitPointWisely,
    with_logo=False,
    first_unknown="h",
    flux=Flux,
    max_eigenvalue=MaxEigenvalue,
    nonconservative_product=None,
):
    """!

    Generate, compile and run the tutorial setup within directory

    with_logo adds the depth map DurhamLogo from Bathymetry to the build.
    first_unknown is the name of the first unknown, i.e. h or eta.

    """
    project = exahype2.api.Project(
        namespace=["tutorials", "exahype2", "tsunami"],
        project_name="Tsunami",
        directory=directory,
        executable="tsunami",
    )
    solver = exahype2.api.Solver("SWE", {first_unknown: 1, "hu": 1, "hv": 1})
    if with_logo:
        solver.add_auxiliary_variables(
            exahype2.api.AuxiliaryVariables(
                "Bathymetry",
                {"b": 1},
                mode=exahype2.api.AuxiliaryVariables.Mode.STATIONARY_ON_FACES,
            )
        )
    solver.add_user_solver_includes("#include <cmath>\n#include <algorithm>\n")
    if with_logo:
        solver.add_user_solver_includes('#include "DurhamLogo.h"\n')
    solver.add_pde_term(exahype2.api.Flux(implementation=flux, stateless=True))
    solver.add_pde_term(
        exahype2.api.MaxEigenvalue(implementation=max_eigenvalue, stateless=True)
    )
    if nonconservative_product is not None:
        solver.add_pde_term(
            exahype2.api.NonconservativeProduct(
                implementation=nonconservative_product, stateless=True
            )
        )
    solver.add_pde_term(
        exahype2.api.InitPointWisely(implementation=init, stateless=False)
    )
    solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(solver, 9)
    solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
        solver, volume_size=0.01
    )
    solver.plotter = exahype2.api.fv.PlotAsVTKFile(
        solver, first_plot_time_stamp=0.0, time_in_between_plots=0.05
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
    if with_logo:
        makefile = peano4_project.output.makefile
        makefile.add_cpp_file(os.path.join(Bathymetry, "DurhamLogo.cpp"))
        makefile.add_h_file(os.path.join(Bathymetry, "DurhamLogo.h"))
        makefile.add_header_search_path(Bathymetry)
    peano4_project.generate()
    subprocess.run(["make", "-j"], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["./tsunami"], cwd=directory, check=True, stdout=subprocess.DEVNULL)


def plot_snapshots(directory, prefix, label="water height h", with_early=False):
    """!

    First unknown at the start, half way through and at the end

    with_early adds the snapshot at a quarter of the simulated time.

    """
    file_name = os.path.join(directory, PatchFile)
    snapshots = len(tarch.plotter.api.notebook.read_snapshots(file_name))
    selection = [
        (0, "start"),
        ((snapshots - 1) // 2, "middle"),
        (snapshots - 1, "end"),
    ]
    if with_early:
        selection.insert(1, ((snapshots - 1) // 4, "early"))
    for snapshot, suffix in selection:
        fig, ax = plt.subplots(figsize=(6, 5))
        tarch.plotter.api.notebook.plot_snapshot(
            file_name, snapshot, 0, ax=ax, label=label
        )
        fig.savefig(
            os.path.join(Here, prefix + suffix + ".png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_sea_floor(directory, prefix):
    """! Sea floor b, which does not change over time"""
    fig, ax = plt.subplots(figsize=(6, 5))
    tarch.plotter.api.notebook.plot_snapshot(
        os.path.join(directory, "solution-SWE-Bathymetry.peano-patch-file"), 0, 0, ax=ax, cmap="Purples_r", label="sea floor b"
    )
    fig.savefig(os.path.join(Here, prefix + "b.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the PNGs of the tutorial 11_tsunami.ipynb"
    )
    parser.add_argument(
        "--work-directory",
        default=None,
        help="Directory for builds and outputs (default: temporary directory)",
    )
    args = parser.parse_args()

    work_directory = args.work_directory or tempfile.mkdtemp(prefix="11_tsunami-")
    print("build and run in {}".format(work_directory))
    flat = os.path.join(work_directory, "flat")
    build_and_run(flat)
    plot_snapshots(flat, "11_tsunami-")

    bathymetry = os.path.join(work_directory, "bathymetry")
    build_and_run(bathymetry, InitPointWiselyWithLogo, with_logo=True)
    plot_snapshots(bathymetry, "11_tsunami-bathymetry-")
    plot_sea_floor(bathymetry, "11_tsunami-bathymetry-")

    ncp = os.path.join(work_directory, "ncp")
    build_and_run(
        ncp,
        InitPointWiselyEta,
        with_logo=True,
        first_unknown="eta",
        flux=FluxEta,
        max_eigenvalue=MaxEigenvalueEta,
        nonconservative_product=NonconservativeProductEta,
    )
    plot_snapshots(
        ncp, "11_tsunami-ncp-", label="surface elevation eta", with_early=True
    )
    print("wrote PNGs into {}".format(Here))
