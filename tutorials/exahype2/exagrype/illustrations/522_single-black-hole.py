# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Create the PNG illustrations of the tutorial 522_single-black-hole.ipynb

The script builds the single black hole setup of the tutorial, runs it,
extracts the lapse and the conformal factor along the x-axis with Peano's
PointProbe, and writes all pictures into the directory of this script. It
runs three setups, one after the other, so their runtimes can be compared:

- plain: the main run of the tutorial, i.e. exahype2.api.fv.RusanovWithExplicitEuler
  with 6x6x6 patches and without load balancing;
- decomposed: the run of the section "Speeding up", i.e. 10x10x10 patches,
  a domain decomposition and four OpenMP threads;
- decomposed-constrained: the same with the CCZ4 constraint enforcement
  after each time step (section "Improving the physics").

The pictures of the initial data and of the lapse stem from the plain run.
522_single-black-hole-speeding-up.png compares the plain and the decomposed
run, 522_single-black-hole-constraints.png the two decomposed runs. Call it
from anywhere:

     python3 tutorials/exahype2/exagrype/illustrations/522_single-black-hole.py

Peano has to be configured with --with-gsl, as the initial data stems from
TwoPunctures. All builds and outputs go into a temporary directory (or
--work-directory). If you pass a work directory, a run which has completed
before is reused, i.e. you can rerun the script to redraw the pictures
without recomputing.

"""
import argparse
import contextlib
import csv
import io
import os
import subprocess
import sys
import tempfile
import time

Here = os.path.dirname(os.path.abspath(__file__))
PeanoRoot = os.path.abspath(os.path.join(Here, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(PeanoRoot, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import peano4.api
import exahype2.api
import exagrype.api
import tarch.plotter.api.output


PatchFile = "solution-CCZ4.peano-patch-file"
Mass = 1.0
DomainOffset = -9.0
DomainSize = 18.0
CellsPerAxis = 9
PatchSize = 6
VolumesPerAxis = CellsPerAxis * PatchSize
TimeInBetweenPlots = 0.5

## y and z coordinate of the line along which we probe: the centre of the
## first volume next to the origin
LineOffset = DomainSize / VolumesPerAxis / 2.0


def psi(r):
    """! Conformal factor of the Schwarzschild black hole in isotropic coordinates"""
    return 1.0 + Mass / 2.0 / r


## The run of the section "Speeding up": the plain Finite Volumes with
## 10x10x10 patches over the same 9x9x9 mesh cells, the domain decomposed by
## a load balancing strategy, and four OpenMP threads
DecomposedPatchSize = 10
DecomposedThreads = 4

## The runs: name, patch size, load balancing, threads, constraint enforcement
Runs = [
    ("plain", PatchSize, False, None, exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint.No),
    (
        "decomposed",
        DecomposedPatchSize,
        True,
        DecomposedThreads,
        exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint.No,
    ),
    (
        "decomposed-constrained",
        DecomposedPatchSize,
        True,
        DecomposedThreads,
        exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint.OnlyFinalSolution,
    ),
]


def build_and_run(
    directory,
    peano_installation,
    end_time,
    scheme,
    enforce,
    patch_size=PatchSize,
    load_balancing=False,
    threads=None,
):
    """!

    Generate, compile and run the tutorial setup within directory

    @param patch_size: int
      Number of volumes per axis within one patch. The mesh always has
      CellsPerAxis cells per axis.

    @param load_balancing: bool
      If True, decompose the domain with
      toolbox::loadbalancing::strategies::SpreadOutOnceGridStagnates.

    @param threads: int or None
      If not None, run with OMP_NUM_THREADS set to this value.

    @param scheme: type
      Numerical scheme, i.e. a Finite Volume class from exahype2.api.fv.

    @param enforce: exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint
      Constraint enforcement.

    Returns the runtime of the executable in seconds. If the directory holds
    the result of a previous run, we do not rerun but return the stored
    runtime.

    """
    runtime_file = os.path.join(directory, "runtime.txt")
    if os.path.exists(runtime_file):
        with open(runtime_file) as f:
            return float(f.read())

    os.makedirs(directory, exist_ok=True)
    print(
        "build and run single black hole until t={} in {}".format(end_time, directory)
    )
    project = exahype2.api.Project(
        namespace=["tutorials", "exagrype", "singleblackhole"],
        project_name="SingleBlackHole",
        directory=directory,
        executable="single-black-hole",
    )
    solver = exagrype.api.FOCCZ4Solver(
        "CCZ4",
        parameters={"CCZ4b": 1, "CCZ4s": 1},
        initial_condition=exagrype.api.initialvalues.single_black_hole(mass=Mass),
    )
    solver.add_pde_term(
        exahype2.api.DirichletBoundaryConditions(
            implementation="::exagrype::initialvalues::singleBlackHole(Q, x, t, {});".format(
                Mass
            )
        )
    )
    solver.numerical_scheme = scheme(solver, patch_size)
    solver.enforce_ccz4_constraint(enforce)
    solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
        solver, volume_size=1.01 * DomainSize / CellsPerAxis / patch_size
    )
    solver.plotter = exahype2.api.fv.PlotAsVTKFile(
        solver,
        first_plot_time_stamp=0.0,
        time_in_between_plots=TimeInBetweenPlots,
        plotter_precision=10,
    )
    solver.plotter.add_plot_filter(
        offset=[DomainOffset, 0.1, 0.1], size=[DomainSize, 0.1, 0.1]
    )
    project.add_solver(solver)
    project.set_global_simulation_parameters(
        dimensions=3,
        offset=[DomainOffset] * 3,
        size=[DomainSize] * 3,
        min_end_time=end_time,
        max_end_time=end_time,
        periodic_BC=[False, False, False],
    )
    if load_balancing:
        project.set_load_balancing(
            "toolbox::loadbalancing::strategies::SpreadOutOnceGridStagnates",
            "new ::exahype2::LoadBalancingConfiguration()",
        )
    project.set_Peano4_installation(
        peano_installation, peano4.api.output.CompileMode.RELEASE
    )
    with contextlib.redirect_stdout(io.StringIO()):
        peano4_project = project.generate_Peano4_project(
            verbosity=peano4.api.LoggerVerbosity.TEXT
        )
        peano4_project.generate()

    subprocess.run(
        ["make", "-j"],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    start = time.time()
    with open(os.path.join(directory, "run.log"), "w") as log:
        environment = dict(os.environ)
        if threads is not None:
            environment["OMP_NUM_THREADS"] = str(threads)
        subprocess.run(
            ["./single-black-hole"],
            cwd=directory,
            check=True,
            stdout=log,
            env=environment,
        )
    runtime = time.time() - start
    with open(runtime_file, "w") as f:
        f.write(str(runtime))
    return runtime


def probe(directory, point, output_file):
    """!

    Probe all snapshots at one point

    Returns the time stamps, the lapse and the conformal factor phi.

    """
    point_probe = tarch.plotter.api.output.PointProbe(
        file_name=os.path.join(directory, PatchFile),
        point=point,
        output_file=os.path.join(directory, output_file),
        start=0,
        end=sys.maxsize,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        point_probe.display()
    with open(os.path.join(directory, output_file)) as csv_file:
        rows = list(csv.DictReader(csv_file))
    return (
        np.array([float(row["time"]) for row in rows]),
        np.array([float(row["lapse"]) for row in rows]),
        np.array([float(row["phi"]) for row in rows]),
    )


def profile(directory, volumes_per_axis=VolumesPerAxis):
    """!

    Probe lapse and phi in the centre of each volume along the line through the
    centres of the volumes next to the puncture, i.e. along y=z=h/2

    Returns the x coordinates, the time stamps and two 2d arrays
    lapse[snapshot,x] and phi[snapshot,x].

    """
    h = DomainSize / volumes_per_axis
    x = DomainOffset + (np.arange(volumes_per_axis) + 0.5) * h
    lapse, phi = [], []
    for i, x_i in enumerate(x):
        t, lapse_over_time, phi_over_time = probe(
            directory, [x_i, h / 2.0, h / 2.0], "profile-{}.csv".format(i)
        )
        lapse.append(lapse_over_time)
        phi.append(phi_over_time)
    return x, t, np.array(lapse).T, np.array(phi).T


def radius(x, line_offset=LineOffset):
    return np.sqrt(np.asarray(x) ** 2 + 2.0 * line_offset**2)


def time_steps(directory):
    """!

    Number of time steps and average wall clock time per time step

    Both stem from the final report of peano4::DefaultStepReport in run.log,
    i.e. from the line "TimeStep: ...s (avg=...,#measurements=...".

    """
    with open(os.path.join(directory, "run.log")) as log:
        for line in log:
            if "reportFinalResults" in line and "TimeStep:" in line:
                statistics = line.split("(avg=")[1]
                average = float(statistics.split(",")[0])
                count = int(statistics.split("#measurements=")[1].split(",")[0])
                return count, average
    raise RuntimeError(f"no time step statistics in {directory}/run.log")


def plot_speeding_up(coarse, fine):
    """! Lapse next to the puncture over time, and along the x-axis at the end, for both meshes"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (label, (x, t, lapse, volumes_per_axis)), marker in zip(
        [("54 volumes per axis", coarse), ("90 volumes per axis", fine)], ["o", "s"]
    ):
        h = DomainSize / volumes_per_axis
        centre = volumes_per_axis // 2
        axes[0].plot(
            t,
            lapse[:, centre],
            marker + "-",
            markersize=3,
            linewidth=1,
            label="{}, r={:.2f}".format(label, radius(x[centre], h / 2.0)),
        )
        axes[1].plot(x, lapse[-1], marker + "-", markersize=2, linewidth=1, label=label)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("lapse")
    axes[0].set_title("Lapse next to the puncture")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("lapse")
    axes[1].set_title("Lapse along the x-axis at t={:.0f}".format(coarse[1][-1]))
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower right")
    fig.savefig(
        os.path.join(Here, "522_single-black-hole-speeding-up.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_initial_data(x, lapse, phi):
    """! Initial lapse and conformal factor along the x-axis against the closed form"""
    h = DomainSize / VolumesPerAxis
    x_fine = np.linspace(DomainOffset, DomainOffset + DomainSize, 2000)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        x_fine,
        1.0 / psi(radius(x_fine)),
        "-",
        color="tab:blue",
        linewidth=1,
        label=r"$1/\psi$",
    )
    ax.plot(
        x_fine,
        psi(radius(x_fine)) ** (-2),
        "-",
        color="tab:orange",
        linewidth=1,
        label=r"$\psi^{-2}$",
    )
    ax.plot(x, lapse[0], "o", color="tab:blue", markersize=3, label="lapse (ExaGRyPE)")
    ax.plot(x, phi[0], "s", color="tab:orange", markersize=3, label="phi (ExaGRyPE)")
    for cell in range(CellsPerAxis + 1):
        ax.axvline(
            DomainOffset + cell * PatchSize * h,
            color="tab:red",
            linewidth=0.5,
            linestyle="--",
        )
    ax.plot(
        [], [], color="tab:red", linewidth=0.5, linestyle="--", label="patch boundaries"
    )
    ax.set_xlabel("x (along y=z={:.3f})".format(LineOffset))
    ax.set_ylabel("initial value")
    ax.set_title("Initial data of a single black hole with M={}".format(Mass))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(
        os.path.join(Here, "522_single-black-hole-initial-data.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_lapse_profile(x, t, lapse):
    """! Lapse along the x-axis for several snapshots"""
    fig, ax = plt.subplots(figsize=(8, 4))
    snapshots = sorted(set([0, len(t) // 4, len(t) // 2, 3 * len(t) // 4, len(t) - 1]))
    colours = plt.cm.viridis(np.linspace(0.0, 0.9, len(snapshots)))
    for snapshot, colour in zip(snapshots, colours):
        ax.plot(
            x,
            lapse[snapshot],
            "o-",
            color=colour,
            markersize=2,
            linewidth=1,
            label="t={:.2f}".format(t[snapshot]),
        )
    ax.set_xlabel("x (along y=z={:.3f})".format(LineOffset))
    ax.set_ylabel("lapse")
    ax.set_title("Lapse along the x-axis over time")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(
        os.path.join(Here, "522_single-black-hole-lapse-profile.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_lapse_over_time(x, t, lapse):
    """! Lapse over time at a few distances from the puncture"""
    fig, ax = plt.subplots(figsize=(8, 4))
    for x_probe in [LineOffset, 1.0 + LineOffset, 3.0 + LineOffset, 8.0 + LineOffset]:
        i = np.argmin(np.abs(x - x_probe))
        ax.plot(
            t,
            lapse[:, i],
            "o-",
            markersize=2,
            linewidth=1,
            label="r={:.2f}".format(radius(x[i])),
        )
    ax.set_xlabel("t")
    ax.set_ylabel("lapse")
    ax.set_title("Lapse over time at several distances from the puncture")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(
        os.path.join(Here, "522_single-black-hole-lapse-over-time.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_constraints(unconstrained, constrained):
    """! Lapse and phi next to the puncture over time with and without constraint enforcement"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (label, run), marker in zip(
        [("without enforcement", unconstrained), ("with enforcement", constrained)],
        ["o", "s"],
    ):
        centre = run["volumes_per_axis"] // 2
        axes[0].plot(
            run["t"],
            run["lapse"][:, centre],
            marker + "-",
            markersize=3,
            linewidth=1,
            label=label,
        )
        axes[1].plot(
            run["t"],
            run["phi"][:, centre],
            marker + "-",
            markersize=3,
            linewidth=1,
            label=label,
        )
    centre = unconstrained["volumes_per_axis"] // 2
    r = radius(unconstrained["x"][centre], unconstrained["h"] / 2.0)
    for ax, quantity in zip(axes, ["lapse", "phi"]):
        ax.set_xlabel("t")
        ax.set_ylabel(quantity)
        ax.set_title("{} next to the puncture (r={:.2f})".format(quantity, r))
    axes[1].legend(loc="lower right")
    fig.savefig(
        os.path.join(Here, "522_single-black-hole-constraints.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the PNGs of the tutorial 522_single-black-hole.ipynb"
    )
    parser.add_argument(
        "--work-directory",
        default=None,
        help="Directory for builds and outputs (default: temporary directory)",
    )
    parser.add_argument(
        "--peano-root",
        default=PeanoRoot,
        help="Configured (--with-gsl) and built Peano installation to link against (default: the repository holding this script)",
    )
    parser.add_argument(
        "--end-time", type=float, default=10.0, help="Simulated time span (default: 10)"
    )
    args = parser.parse_args()

    with contextlib.ExitStack() as stack:
        work_directory = args.work_directory or stack.enter_context(
            tempfile.TemporaryDirectory()
        )
        runs = {}
        for name, patch_size, load_balancing, threads, enforce in Runs:
            directory = os.path.join(os.path.abspath(work_directory), name)
            runtime = build_and_run(
                directory,
                os.path.abspath(args.peano_root),
                args.end_time,
                exahype2.api.fv.RusanovWithExplicitEuler,
                enforce,
                patch_size=patch_size,
                load_balancing=load_balancing,
                threads=threads,
            )
            volumes_per_axis = CellsPerAxis * patch_size
            x, t, lapse, phi = profile(directory, volumes_per_axis)
            count, average = time_steps(directory)
            h = DomainSize / volumes_per_axis
            runs[name] = dict(
                runtime=runtime,
                x=x,
                t=t,
                lapse=lapse,
                phi=phi,
                h=h,
                volumes_per_axis=volumes_per_axis,
            )
            centre = volumes_per_axis // 2
            print(
                "{}: runtime {:.1f}s, {} time steps of {:.3f}s, {} volumes, {:.3e}s per volume update".format(
                    name,
                    runtime,
                    count,
                    average,
                    volumes_per_axis**3,
                    average / volumes_per_axis**3,
                )
            )
            print(
                "{}: next to the puncture (r={:.3f}) lapse {:.4f} -> {:.4f}, phi {:.4f} -> {:.4f}, NaN: {}".format(
                    name,
                    radius(x[centre], h / 2.0),
                    lapse[0, centre],
                    lapse[-1, centre],
                    phi[0, centre],
                    phi[-1, centre],
                    np.isnan(lapse).any(),
                )
            )
        print(
            "max deviation of decomposed-constrained from decomposed: lapse {:.3e}, phi {:.3e}".format(
                np.max(
                    np.abs(
                        runs["decomposed-constrained"]["lapse"]
                        - runs["decomposed"]["lapse"]
                    )
                ),
                np.max(
                    np.abs(
                        runs["decomposed-constrained"]["phi"]
                        - runs["decomposed"]["phi"]
                    )
                ),
            )
        )
        plot_speeding_up(
            (
                runs["plain"]["x"],
                runs["plain"]["t"],
                runs["plain"]["lapse"],
                VolumesPerAxis,
            ),
            (
                runs["decomposed"]["x"],
                runs["decomposed"]["t"],
                runs["decomposed"]["lapse"],
                runs["decomposed"]["volumes_per_axis"],
            ),
        )
        plot_constraints(runs["decomposed"], runs["decomposed-constrained"])

        x, t, lapse, phi = (runs["plain"][key] for key in ["x", "t", "lapse", "phi"])
        print("snapshots at t =", np.round(t, 3))
        for i in [
            VolumesPerAxis // 2,
            VolumesPerAxis // 2 + 3,
            VolumesPerAxis // 2 + 9,
            VolumesPerAxis - 1,
        ]:
            print(
                "r={:.3f}: lapse(t=0)={:.4f} (1/psi={:.4f}), lapse(t={:.2f})={:.4f}, phi(t=0)={:.4f} (psi^-2={:.4f})".format(
                    radius(x[i]),
                    lapse[0, i],
                    1.0 / psi(radius(x[i])),
                    t[-1],
                    lapse[-1, i],
                    phi[0, i],
                    psi(radius(x[i])) ** (-2),
                )
            )
        print("NaN in lapse:", np.isnan(lapse).any())

        plot_initial_data(x, lapse, phi)
        plot_lapse_profile(x, t, lapse)
        plot_lapse_over_time(x, t, lapse)
