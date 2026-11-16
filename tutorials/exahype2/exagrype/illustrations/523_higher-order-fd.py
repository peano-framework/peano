# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Create the PNG illustrations of the tutorial 523_higher-order-fd.ipynb

The script builds the single black hole setup of the tutorial with several
numerical schemes, runs them one after the other, extracts the lapse and
the conformal factor along the x-axis with Peano's PointProbe, and writes
all pictures into the directory of this script. All runs use Sommerfeld
boundary conditions, a domain decomposition and four OpenMP threads, i.e.
the final setup of 522_single-black-hole.ipynb, over 9x9x9 mesh cells with
10x10x10 volumes or grid cells each:

- fv: exahype2.api.fv.RusanovWithExplicitEuler;
- fd4: exahype2.api.rkfd4.CentralDifferencesWithRungeKutta with the classic
  fourth-order Runge-Kutta scheme and without dissipation;
- fd4-ko-{sigma}: exagrype.api.FD4WithKODissipation with the Kreiss-Oliger
  dissipation scaled by sigma, for sigma 0.3, 1 and 8;
- fd4-lop: the fourth-order Finite Differences with the kernel
  CentralDifferencesWithLopsidedAdvection, i.e. with the advection terms
  which exagrype::ccz4::ncp() omits;
- fd4-lop-ko-8.0: the same with the Kreiss-Oliger dissipation (sigma=8) on
  all unknowns;
- fd4-lop-ko-8.0-subset: the same with the dissipation on Q0-Q24 only (DissipationSubset).

The dictionary Runs holds the configurations.

Call it from anywhere:

     python3 tutorials/exahype2/exagrype/illustrations/523_higher-order-fd.py

Peano has to be configured with --with-gsl, as the initial data stems from
TwoPunctures. All builds and outputs go into a temporary directory (or
--work-directory). If you pass a work directory, a run which has completed
before is reused, i.e. you can rerun the script to redraw the pictures
without recomputing. A run which crashes or produces NaNs is reported, and
its output up to that point is used.

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
PatchSize = 10
VolumesPerAxis = CellsPerAxis * PatchSize
H = DomainSize / VolumesPerAxis
Threads = 4
TimeInBetweenPlots = 0.5

## Unknowns which the restricted Kreiss-Oliger dissipation damps: Q0-Q24
## (G, K, theta, Z, lapse, shift, b, Kex, phi)
DissipationSubset = list(range(0, 25))

## The runs: name -> (Kreiss-Oliger sigma or None, lopsided advection,
## quantities of the dissipation or None for all). The fv run is special.
Runs = {
    "fv": (None, False, None),
    "fd4": (None, False, None),
    "fd4-ko-0.3": (0.3, False, None),
    "fd4-ko-1.0": (1.0, False, None),
    "fd4-ko-8.0": (8.0, False, None),
    "fd4-lop": (None, True, None),
    "fd4-lop-ko-8.0": (8.0, True, None),
    "fd4-lop-ko-8.0-subset": (8.0, True, DissipationSubset),
}

## Colours, markers and labels per run, so all pictures use the same ones
Style = {
    "fv": ("tab:blue", "o", "Finite Volumes"),
    "fd4": ("tab:red", "s", "FD4, no dissipation"),
    "fd4-ko-0.3": ("tab:green", "^", r"FD4, KO $\sigma=0.3$"),
    "fd4-ko-1.0": ("tab:purple", "v", r"FD4, KO $\sigma=1$"),
    "fd4-ko-8.0": ("tab:orange", "D", r"FD4, KO $\sigma=8$"),
    "fd4-lop": ("tab:brown", "P", "FD4, lopsided, no dissipation"),
    "fd4-lop-ko-8.0": ("tab:pink", "X", r"FD4, lopsided, KO $\sigma=8$ (all)"),
    "fd4-lop-ko-8.0-subset": (
        "tab:olive",
        "*",
        r"FD4, lopsided, KO $\sigma=8$ (Q0-Q24)",
    ),
}


def create_scheme(name, solver):
    """! The numerical scheme, grid construction and plotter of one run"""
    if name == "fv":
        package = exahype2.api.fv
        solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(
            solver, PatchSize
        )
    else:
        package = exahype2.api.rkfd4
        sigma, lopsided, dissipative_quantities = Runs[name]
        Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
        arguments = {}
        if lopsided:
            arguments = dict(
                right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
                advection_velocity=exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY,
                advected_quantities=solver.lopsided_advection_quantities(),
            )
        if sigma is None:
            solver.numerical_scheme = (
                exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
                    solver, PatchSize, rk_order=4, **arguments
                )
            )
        else:
            solver.numerical_scheme = exagrype.api.FD4WithKODissipation(
                solver,
                PatchSize,
                rk_order=4,
                kreiss_oliger_sigma=sigma,
                quantities=dissipative_quantities,
                **arguments,
            )
    solver.grid_construction = package.SetupSimulationOverRegularGrid(
        solver, volume_size=1.01 * H
    )
    solver.plotter = package.PlotAsVTKFile(
        solver,
        first_plot_time_stamp=0.0,
        time_in_between_plots=TimeInBetweenPlots,
        plotter_precision=10,
    )
    solver.plotter.add_plot_filter(
        offset=[DomainOffset, 0.1, 0.1], size=[DomainSize, 0.1, 0.1]
    )


def build_and_run(name, directory, peano_installation, end_time):
    """!

    Generate, compile and run one setup within directory

    Returns the runtime of the executable in seconds and whether it
    terminated successfully. If the directory holds the result of a
    previous run, we do not rerun but return the stored values.

    """
    runtime_file = os.path.join(directory, "runtime.txt")
    if os.path.exists(runtime_file):
        with open(runtime_file) as f:
            runtime, success = f.read().split()
            return float(runtime), success == "True"

    os.makedirs(directory, exist_ok=True)
    print("build and run {} until t={} in {}".format(name, end_time, directory))
    project = exahype2.api.Project(
        namespace=["tutorials", "exagrype", "higherorderfd"],
        project_name="HigherOrderFD",
        directory=directory,
        executable="single-black-hole",
    )
    solver = exagrype.api.FOCCZ4Solver(
        "CCZ4",
        parameters={"CCZ4b": 1, "CCZ4s": 1},
        initial_condition=exagrype.api.initialvalues.single_black_hole(mass=Mass),
    )
    solver.add_pde_term(
        exahype2.api.SommerfeldBoundaryConditions(
            implementation="::exagrype::initialvalues::flat(Q, x, 0.0);"
        )
    )
    create_scheme(name, solver)
    project.add_solver(solver)
    project.set_global_simulation_parameters(
        dimensions=3,
        offset=[DomainOffset] * 3,
        size=[DomainSize] * 3,
        min_end_time=end_time,
        max_end_time=end_time,
        periodic_BC=[False, False, False],
    )
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
        ["make", "-j{}".format(Threads)],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    start = time.time()
    with open(os.path.join(directory, "run.log"), "w") as log:
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = str(Threads)
        result = subprocess.run(
            ["./single-black-hole"],
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    runtime = time.time() - start
    success = result.returncode == 0
    with open(runtime_file, "w") as f:
        f.write("{} {}".format(runtime, success))
    return runtime, success


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


def profile(directory):
    """!

    Probe lapse and phi in the centre of each volume or grid cell along the
    line y=z=h/2, i.e. through the centres next to the puncture

    Returns the x coordinates, the time stamps and two 2d arrays
    lapse[snapshot,x] and phi[snapshot,x].

    """
    x = DomainOffset + (np.arange(VolumesPerAxis) + 0.5) * H
    lapse, phi = [], []
    for i, x_i in enumerate(x):
        t, lapse_over_time, phi_over_time = probe(
            directory, [x_i, H / 2.0, H / 2.0], "profile-{}.csv".format(i)
        )
        lapse.append(lapse_over_time)
        phi.append(phi_over_time)
    snapshots = min(len(entry) for entry in lapse)
    return (
        x,
        t[:snapshots],
        np.array([entry[:snapshots] for entry in lapse]).T,
        np.array([entry[:snapshots] for entry in phi]).T,
    )


def radius(x):
    return np.sqrt(np.asarray(x) ** 2 + 2.0 * (H / 2.0) ** 2)


def time_steps(directory):
    """!

    Number of time steps and average wall clock time per time step

    Both stem from the final report of peano4::DefaultStepReport in run.log,
    i.e. from the line "TimeStep: ...s (avg=...,#measurements=...". Returns
    None if the run did not terminate regularly.

    """
    with open(os.path.join(directory, "run.log")) as log:
        for line in log:
            if "reportFinalResults" in line and "TimeStep:" in line:
                statistics = line.split("(avg=")[1]
                average = float(statistics.split(",")[0])
                count = int(statistics.split("#measurements=")[1].split(",")[0])
                return count, average
    return None


def plot_over_time_and_profile(runs, names, file_name):
    """!

    Lapse next to the puncture over time (left) and along the x-axis at the
    last snapshot all runs share (right)

    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    centre = VolumesPerAxis // 2
    last = min(len(runs[name]["t"]) for name in names) - 1
    for name in names:
        colour, marker, label = Style[name]
        run = runs[name]
        axes[0].plot(
            run["t"],
            run["lapse"][:, centre],
            marker + "-",
            color=colour,
            markersize=3,
            linewidth=1,
            label=label,
        )
        axes[1].plot(
            run["x"],
            run["lapse"][last],
            marker + "-",
            color=colour,
            markersize=2,
            linewidth=1,
            label=label,
        )
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("lapse")
    axes[0].set_title(
        "Lapse next to the puncture (r={:.2f})".format(
            radius(runs[names[0]]["x"][centre])
        )
    )
    axes[1].set_xlabel("x (along y=z={:.2f})".format(H / 2.0))
    axes[1].set_ylabel("lapse")
    axes[1].set_title(
        "Lapse along the x-axis at t={:.1f}".format(runs[names[0]]["t"][last])
    )
    # Runs which collapse write nonsensical values into their last snapshot
    for ax in axes:
        ax.set_ylim(0.0, 1.0)
    axes[0].set_xlim(0.0, max(runs[name]["t"][-1] for name in names))
    axes[1].legend(loc="lower right")
    fig.savefig(os.path.join(Here, file_name), dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_fourier_analysis():
    """!

    Fourier analysis of the fourth-order central differences and of the
    Kreiss-Oliger dissipation

    A Fourier mode exp(i k x) on a grid with spacing h is characterised by
    theta = k h between 0 (smooth) and pi (changes sign from grid cell to
    grid cell). The left picture shows the wave number which the
    fourth-order central difference "sees", i.e. theta_eff = (8 sin theta -
    sin 2 theta) / 6, against the exact one. The right picture shows the
    rate h/sigma * lambda with which the Kreiss-Oliger term
    sigma/(64 h) * (Q_{i-3} - 6 Q_{i-2} + ... + Q_{i+3}) damps the mode
    along one axis, i.e. (4 sin^2(theta/2))^3 / 64.

    """
    theta = np.linspace(0.0, np.pi, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(theta, theta, "--", color="black", linewidth=1, label="exact")
    axes[0].plot(
        theta,
        (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0,
        "-",
        color="tab:red",
        linewidth=2,
        label="fourth-order central difference",
    )
    axes[0].set_xlabel(r"$\theta = k h$")
    axes[0].set_ylabel("effective wave number times h")
    axes[0].set_title("How the derivative sees a Fourier mode")
    axes[0].legend(loc="upper left")
    axes[1].semilogy(
        theta[1:],
        (4.0 * np.sin(theta[1:] / 2.0) ** 2) ** 3 / 64.0,
        "-",
        color="tab:green",
        linewidth=2,
    )
    axes[1].set_xlabel(r"$\theta = k h$")
    axes[1].set_ylabel(r"damping rate $\times\, h / \sigma$")
    axes[1].set_title("Kreiss-Oliger damping per axis")
    axes[1].set_ylim(1e-6, 2.0)
    for ax in axes:
        ax.set_xticks([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    fig.savefig(
        os.path.join(Here, "523_higher-order-fd-fourier.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the PNGs of the tutorial 523_higher-order-fd.ipynb"
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
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(Runs),
        help="Runs to perform (default: all, i.e. {})".format(" ".join(Runs)),
    )
    args = parser.parse_args()

    plot_fourier_analysis()

    with contextlib.ExitStack() as stack:
        work_directory = args.work_directory or stack.enter_context(
            tempfile.TemporaryDirectory()
        )
        runs = {}
        for name in args.runs:
            directory = os.path.join(os.path.abspath(work_directory), name)
            runtime, success = build_and_run(
                name, directory, os.path.abspath(args.peano_root), args.end_time
            )
            x, t, lapse, phi = profile(directory)
            runs[name] = dict(x=x, t=t, lapse=lapse, phi=phi)
            centre = VolumesPerAxis // 2
            statistics = time_steps(directory)
            print(
                "{}: {}, runtime {:.1f}s, {}".format(
                    name,
                    "terminated" if success else "FAILED",
                    runtime,
                    (
                        "{} time steps of {:.3f}s".format(*statistics)
                        if statistics
                        else "no time step statistics"
                    ),
                )
            )
            print(
                "{}: snapshots until t={:.2f}, next to the puncture (r={:.3f}) lapse {:.4f} -> {:.4f}, phi {:.4f} -> {:.4f}, min lapse at t_end {:.4f}, NaN: {}".format(
                    name,
                    t[-1],
                    radius(x[centre]),
                    lapse[0, centre],
                    lapse[-1, centre],
                    phi[0, centre],
                    phi[-1, centre],
                    np.nanmin(lapse[-1]),
                    np.isnan(lapse).any(),
                )
            )
            for i in [centre, centre + 5, centre + 15, VolumesPerAxis - 1]:
                print(
                    "{}:   r={:.2f}: lapse {}".format(
                        name,
                        radius(x[i]),
                        " ".join("{:.3f}".format(value) for value in lapse[::4, i]),
                    )
                )

        if "fv" in runs and "fd4" in runs:
            plot_over_time_and_profile(
                runs, ["fv", "fd4"], "523_higher-order-fd-fv-vs-fd4.png"
            )
        ko_runs = [name for name in Style if name.startswith("fd4-ko") and name in runs]
        if "fv" in runs and "fd4" in runs and ko_runs:
            plot_over_time_and_profile(
                runs,
                ["fv", "fd4"] + ko_runs,
                "523_higher-order-fd-kreiss-oliger.png",
            )
        lopsided_runs = [
            name for name in Style if name.startswith("fd4-lop") and name in runs
        ]
        if "fv" in runs and "fd4" in runs and lopsided_runs:
            plot_over_time_and_profile(
                runs,
                ["fv", "fd4"] + lopsided_runs,
                "523_higher-order-fd-lopsided.png",
            )
