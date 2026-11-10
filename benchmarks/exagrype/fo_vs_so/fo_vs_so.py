r"""!

First-order vs. second-order reconstructed CCZ4: gauge wave benchmark

This is the gauge wave benchmark (@ref benchmarks_exagrype_gauge_wave) with the
fourth-order Finite Differences (exahype2.api.rkfd4), but we instantiate two
solvers in one setup: exagrype.api.FOCCZ4Solver, which evolves all 59 CCZ4
unknowns, and exagrype.api.FOSOCCZ4Solver, which evolves 25 of them and obtains
the 34 first derivatives from the exagrype.api.FOSOReconstructionSolver next to it
(both are hosted by an exagrype.api.FOSOSolver). Both solvers
use the same Runge-Kutta scheme, the same lopsided advection and no
dissipation. The two do not yield identical results: The first-order solver
evolves the derivatives as unknowns, whereas the second-order solver
re-computes them from the 25 unknowns in each Runge-Kutta stage. The script
builds and runs the code, and then quantifies the difference of the 25
evolved unknowns of both solvers, and the deviation of both from the
analytical solution of the gauge wave.

Two exahype2.api.rkfd4 solvers cannot live in one project yet, as both
generate the solver-independent kernel rungekutta_linear_combination into the
same file (@ref page_exagrype_fo_so_project_plan). The script therefore
builds and runs each solver on its own, in a directory of its own
(run-FO and run-SO), and then compares the plot files.

@todo This post-run comparison of plot files is a stop-gap. We have to return
  to it and replace it by an on-the-fly check, i.e. a check within the
  simulation that compares both solvers' data while the code runs.

The gauge wave is a pure gauge perturbation of flat spacetime, i.e. an
analytical solution of the Einstein equations that travels along the x-axis
with speed one. It is periodic over the unit cube and hence a standard first
test of any numerical relativity code. This script uses exagrype.api, i.e.
the CCZ4 solvers of the new Python API. It requires a Peano build with

    --enable-exahype --enable-blockstructured --enable-loadbalancing --enable-exagrype

and is run from this directory via

    python3 fo_vs_so.py

## Schemes, formulations and grid sizes

The benchmark covers two numerical schemes:

- RK4FD4: fourth-order Finite Differences with classic Runge-Kutta 4 and lopsided
  advection (exahype2.api.rkfd4.CentralDifferencesWithRungeKutta). The
  second-order solver reconstructs the derivatives with the fourth-order
  stencil in each Runge-Kutta stage.
- RK1FD4: the same spatial discretisation, but with first-order Runge-Kutta
  (explicit Euler, rk_order=1), i.e. one stage per time step.
- FV: Finite Volumes with a Rusanov flux and explicit Euler
  (exahype2.api.fv.RusanovWithExplicitEuler). The second-order solver
  reconstructs the derivatives with second-order central differences
  once per time step. Neither advection nor dissipation are used, which is
  fine as the shift of the gauge wave is zero.

Each of the two comes in the formulations FO (exagrype.api.FOCCZ4Solver) and
SO (exagrype.api.FOSOCCZ4Solver). The grid has cells x cells x cells grid
points (patch size 9 and 3^k patches per axis, i.e. cells is 27, 81, ... (Peano refines to at least three patches per axis)). The grid size is
compiled into the executable, so each combination is built in a directory of
its own (run-scheme-formulation-cells).

    python3 fo_vs_so.py                         # same as --mode=run-and-compare
    python3 fo_vs_so.py --mode=build            # generate and compile all combinations
    python3 fo_vs_so.py --mode=run              # generate, compile and run all combinations
    python3 fo_vs_so.py --mode=run-and-compare  # run, then print the table
    python3 fo_vs_so.py --mode=compare-only     # re-print the table from existing runs

The script loops sequentially, within one Python process, over all schemes,
grid sizes and formulations.

The option --build-mode=debug|release selects the build (default release; debug
also enables the generator logging). --schemes (default RK4FD4,RK1FD4,FV), --formulations (default FO,SO; with just one,
the comparison table lists the lapse error and wall time of that formulation only), --sizes (default 27,81), --end-time (default
0.1) and --cfl (CFL constant of the Finite Volume scheme, default 0.9) override
the remaining settings.

@page benchmarks_exagrype_fo_vs_so First-order vs hybrid First-order second-order CCZ4 formulation

\tableofcontents

This page has yet to be written.


See the benchmark benchmarks/exagrype/fo_vs_so. It runs the gauge wave
with both solvers and prints the maximum and relative L2 difference of the 25
evolved unknowns at the end of the run.

Setup (`python3 fo_vs_so.py`, i.e. release build, end time 0.1): gauge wave on the
periodic unit cube, 27 grid points per axis (patch size 9, 3 patches per axis), fourth-order
Finite Differences with classic Runge-Kutta 4 and lopsided advection, no dissipation,
adaptive time step size (CFL), end time 0.1 (about 33 time steps). Both solvers
use the same numerical scheme. We compare the 25 evolved unknowns at the end of the
run. The last column shows how far the first-order solution moved away from its
initial condition, i.e. the scale of the dynamics.

| unknown | max abs. difference SO vs. FO | relative L2 difference | max abs. change of FO in the run |
|---|---|---|---|
| G (xx, yy, zz) | 7.6e-6 / 4.1e-6 / 4.1e-6 | 3.4e-6 / 1.8e-6 / 1.8e-6 | 4.2e-2 / 2.1e-2 / 2.1e-2 |
| K (xx, yy, zz) | 7.7e-5 / 4.1e-5 / 4.1e-5 | 2.3e-4 | 1.4e-1 / 7.9e-2 / 7.9e-2 |
| lapse | 2.5e-6 | 1.0e-6 | 3.1e-2 |
| phi | 2.0e-6 | 8.7e-7 | 1.1e-2 |
| traceK | 1.3e-4 | 2.4e-4 | 2.3e-1 |
| Z (x) | 5.0e-5 | 7.7e-5 | 3.1e-1 |
| theta | 3.6e-5 | (not meaningful) | 1.9e-5 |
| all others (off-diagonals, shift, b, Z_y, Z_z) | 0 | 0 | 0 |

**Reading the table.** The two solvers agree to 1e-6 to 1e-4 in absolute terms,
which is two to four orders of magnitude below the change of the solution within
the run. The metric, the lapse and phi agree best (relative 1e-6), the extrinsic
curvature and its trace worst (relative 2e-4). The remaining entries are
identically zero for both (the gauge wave does not excite them). Theta (a measure of the
Hamilton constraint violation) is tiny in the first-order run (it changes by 1.9e-5
at most), and the two solvers differ by 3.6e-5, i.e. by more than the first-order
solver's theta moves. The relative difference is therefore not meaningful. We have
no reference solution for theta yet, so we cannot say which solver is closer to the
truth.

**Caveats.** One resolution, one end time, one test case. The comparison measures
how close SO is to FO, not how accurate either of them is. A convergence study
against the analytical gauge wave (@ref benchmarks_exagrype_gauge_wave), a
longer run, and a black hole run are still missing.

**Cost.** In the release build with 8 OpenMP threads (machine shared with other
jobs, so treat with care), one reported TimeStep phase took about 0.15 s for the
first-order solver and about 0.21 s for the second-order solver on this mesh,
i.e. SO is about 40% slower per step. This is expected: SO reconstructs the
derivatives in an extra mesh sweep per Runge-Kutta stage, exchanges an extra
halo of 34 values per stage, and gathered/scattered the 59 entries in every PDE term
evaluation (it does not anymore, the kernels take the primary and the secondary
entries as two pointers). We did not profile it. SO stores fewer values per stage (25 instead of
59 for the stage input, the right-hand-side estimates and the halo patch). A
first, finer run (729 patches of 9^3 points) was killed by the operating system
for the first-order solver (out of memory, 15 GB machine shared with other
processes), while the second-order solver ran through with a resident set of
about 3.5 GB. This is an observation, not a measurement.


## Grid sizes and Finite Volumes

The benchmark script now takes the scheme (RK4FD4, RK1FD4 or FV), the formulation (FO or SO) and the grid
size (`python3 fo_vs_so.py`, `--build-mode=debug` for debug builds). Peano refines to at least three patches
per axis and then in powers of three, and a patch has 9 points per axis, i.e. valid grids are 27^3 and 81^3
points (a first attempt with 9, 18 and 45 points silently ran on the next larger grid). Setup otherwise as above: gauge wave, periodic unit cube,
RK4 with lopsided advection, end time 0.1 (the last step overshoots, the time stamp is given). We also compare the lapse of
both solvers against the analytical lapse
\f$ \sqrt{1-0.1\sin(4\pi(x-t))} \f$. The wall time is the one of the whole run (setup, plots and about 33 (27^3) or
100 (81^3) time steps; machine shared with other jobs, 8 threads, treat with care).

| scheme | grid | t | max abs. diff. SO vs. FO (25 unknowns) | max abs. diff. of the lapse | rel. L2 lapse error FO | rel. L2 lapse error SO | wall time FO | wall time SO |
|---|---|---|---|---|---|---|---|---|
| RK4FD4 | 27^3 | 0.102 | 1.3e-4 | 2.5e-6 | 1.1e-6 | 1.6e-6 | 4.8 s | 5.6 s |
| RK4FD4 | 81^3 | 0.101 | 1.6e-6 | 3.1e-8 | 1.4e-8 | 2.2e-8 | 300 s | 377 s |

**Reading the table.** The 27^3 row reproduces the numbers of the first table
(the largest difference is the one of traceK). Refining the grid by a factor of three reduces the
difference between the two solvers by a factor of about 80, and the error of each solver against the analytical lapse
by about 80 (FO) and 75 (SO), i.e. \f$ 3^4 \f$: both formulations converge with fourth order, and the SO solver stays within a factor of
1.5 of the first-order solver's error. On the finer grid, SO is about 25% slower than FO per run (about 15% on the 27^3 grid, where setup and plots matter more).

### Finite Volumes

The FV variant (second-order central differences, one reconstruction task per step) builds,
generates and runs, but we could **not** obtain a meaningful FV comparison. The reason is not
the reconstruction: the Finite Volume time step kernel in 3d that exahype2.api.fv.RusanovWithExplicitEuler generates
contains neither the non-conservative product, the source, nor the maximum eigenvalue. The kernel template guards these blocks with
`if nonconservativeProduct:`, `if volumetricSource:`, `if maxEigenvalue:` (and `if flux:`), but
the DSL parser compares the guard with the class names of the PDE terms
(NonconservativeProduct, VolumetricSource, MaxEigenvalue, Flux, see exahype2.api.dslhype2.Parser._parse_if()), so the blocks are dropped. The generated kernel
then only applies the Rusanov diffusion to the unknowns, and it never writes the maximum eigenvalue of the cell,
so the global time step size is derived from uninitialised memory (time step sizes of 1e-280 or infinity after a few steps,
for the first-order solver as well). This is a generic Finite Volume issue, independent of CCZ4, and it
is why the FV runs of both formulations "worked" with an error of about 2e-2 against the analytical lapse (diffusion only) or
broke down after a few steps. The fix is a rename in the kernel template (2d and 3d); it has not been applied, as it is outside
of the files of this project. Once done, rerun `python3 fo_vs_so.py` and add the FV rows
here.


"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import peano4.api
import exahype2.api
import exagrype.api


parser = argparse.ArgumentParser(description="First-order vs. second-order CCZ4")
parser.add_argument(
    "--mode",
    choices=["compare-only", "run-and-compare", "run", "build"],
    default="run-and-compare",
    help="build: generate and compile; run: build and run; "
    "run-and-compare: build, run, print table; compare-only: print table from existing runs",
)
parser.add_argument(
    "--build-mode",
    choices=["debug", "release"],
    default="release",
    help="Build mode of the generated code. Debug also enables the generator logging.",
)
parser.add_argument(
    "--sizes",
    type=lambda text: [int(size) for size in text.split(",")],
    default=[27, 81],
    help="Comma-separated grid sizes (points per axis)",
)
parser.add_argument(
    "--schemes",
    type=lambda text: text.split(","),
    default=["RK4FD4", "RK1FD4", "FV"],
    help="Comma-separated numerical schemes to build, run and compare (RK4FD4, RK1FD4, FV)",
)
parser.add_argument(
    "--formulations",
    type=lambda text: text.split(","),
    default=["FO", "SO"],
    help="Comma-separated formulations to build, run and compare (FO, SO)",
)
parser.add_argument(
    "--compiler",
    choices=["sequential", "fusion"],
    default="fusion",
    help="Task graph compiler: one mesh traversal per task (sequential) or fused traversals (fusion)",
)
parser.add_argument("--end-time", type=float, default=0.1, help="Simulation end time")
parser.add_argument(
    "--cfl", type=float, default=0.9, help="CFL constant of the FV scheme"
)
arguments = parser.parse_args()
schemes = arguments.schemes
formulations = arguments.formulations
for formulation in formulations:
    if formulation not in ["FO", "SO"]:
        parser.error(f"unknown formulation {formulation}, use FO and/or SO")
for scheme in schemes:
    if scheme not in ["RK4FD4", "RK1FD4", "FV"]:
        parser.error(f"unknown scheme {scheme}, use RK4FD4, RK1FD4 and/or FV")

if arguments.build_mode == "release":
    build_mode = peano4.api.output.CompileMode.RELEASE
    verbosity = peano4.api.LoggerVerbosity.OFF
else:
    build_mode = peano4.api.output.CompileMode.DEBUG
    verbosity = peano4.api.LoggerVerbosity.VISUALISATION
grid_sizes = arguments.sizes
end_time = arguments.end_time
time_in_between_plots = end_time / 2.0
patch_size = 9


def directory_of(scheme, formulation, cells):
    return f"run-{scheme}-{formulation}-{cells}"


def build(scheme, formulation, cells):
    """!

    Generate and build one solver on one grid in its own directory

    @param scheme: str
      RK4FD4, RK1FD4 or FV.

    @param formulation: str
      FO or SO.

    @param cells: int
      Grid points per axis. Has to be a multiple of the patch size.

    """
    # Peano's regular grid has 3^k patches per axis
    assert cells in [patch_size * 3**k for k in range(1, 6)], cells
    name = f"{scheme}{formulation}"
    directory = directory_of(scheme, formulation, cells)
    executable = f"fo-vs-so-{scheme}-{formulation}-{cells}"
    # The generator appends to kernels of an earlier run, so start from scratch
    shutil.rmtree(directory, ignore_errors=True)
    project = exahype2.api.Project(
        namespace=["benchmarks", "exagrype", "fo_vs_so", name],
        project_name=executable,
        directory=directory,
        executable=executable,
    )
    initial_condition = exagrype.api.initialvalues.gauge_wave()
    if formulation == "FO":
        solver = exagrype.api.FOCCZ4Solver(name, initial_condition=initial_condition)
        scheme_owner = solver
    else:
        # The pair of CCZ4 and reconstruction solver. The pair holds the scheme.
        pair = exagrype.api.FOSOSolver(name, initial_condition=initial_condition)
        solver = pair.ccz4
        scheme_owner = pair
    # The grid cells have to be strictly smaller than volume_size times the patch size
    volume_size = 1.01 / cells
    match scheme:
        case "RK4FD4" | "RK1FD4":
            Kernel = (
                exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
            )
            scheme_owner.numerical_scheme = (
                exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
                    solver,
                    patch_size,
                    rk_order=4 if scheme == "RK4FD4" else 1,
                    right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
                    advection_velocity=exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY,
                    advected_quantities=solver.lopsided_advection_quantities(),
                )
            )
            solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
                solver, volume_size=volume_size
            )
            solver.plotter = exahype2.api.rkfd4.PlotAsVTKFile(
                solver,
                first_plot_time_stamp=0.0,
                time_in_between_plots=time_in_between_plots,
            )
        case "FV":
            # No advection, no dissipation: the shift of the gauge wave is zero
            scheme_owner.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(
                solver, patch_size, cfl_constant=arguments.cfl
            )
            solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
                solver, volume_size=volume_size
            )
            solver.plotter = exahype2.api.fv.PlotAsVTKFile(
                solver,
                first_plot_time_stamp=0.0,
                time_in_between_plots=time_in_between_plots,
            )
    if formulation == "FO":
        project.add_solver(solver)
    else:
        project.add_solver(pair)

    project.compiler = (
        peano4.api.taskgraph.compiler.Sequential.translate
        if arguments.compiler == "sequential"
        else peano4.api.taskgraph.compiler.FuseSteps.translate
    )
    project.set_global_simulation_parameters(
        dimensions=3,
        offset=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 1.0],
        min_end_time=end_time,
        max_end_time=end_time,
        periodic_BC=[True, True, True],
    )
    # Absolute, as the run directories are one level deeper than this script
    project.set_Peano4_installation(
        os.path.abspath("../../../"),
        build_mode,
    )
    peano4_project = project.generate_Peano4_project(verbosity=verbosity)
    peano4_project.generate()
    peano4_project.build(make_clean_first=True)


def run(scheme, formulation, cells):
    """!

    Run the executable built by build() and store its wall time

    @param scheme: str
      RK4FD4, RK1FD4 or FV.

    @param formulation: str
      FO or SO.

    @param cells: int
      Grid points per axis.

    """
    directory = directory_of(scheme, formulation, cells)
    executable = f"fo-vs-so-{scheme}-{formulation}-{cells}"
    start = time.time()
    if subprocess.call([f"./{executable}"], cwd=directory) != 0:
        sys.exit(f"{scheme} {formulation} {cells} run failed")
    wall_time = time.time() - start
    with open(os.path.join(directory, "wall-time.json"), "w") as file:
        json.dump({"wall_time": wall_time}, file)


def snapshot(scheme, formulation, cells, unknowns, snapshot_index=-1):
    """!

    Return the unknowns of one snapshot of one solver

    @param scheme: str
      RK4FD4, RK1FD4 or FV.

    @param formulation: str
      FO or SO.

    @param cells: int
      Grid points per axis.

    @param unknowns: int
      Number of unknowns per grid point in the plot file

    @param snapshot_index: int
      Index of the snapshot, -1 is the last one.

    @return time stamp, values, positions
      The values and the x coordinates of the grid points, both sorted by the
      offset of the patches. Values has one row per grid point.

    """
    from tarch.plotter.api.input.PatchFileParser import PatchFileParser
    from tarch.plotter.api.notebook import read_snapshots

    name = f"{scheme}{formulation}"
    directory = directory_of(scheme, formulation, cells)
    time_stamp, patch_files = read_snapshots(
        f"{directory}/solution-{name}.peano-patch-file"
    )[snapshot_index]
    result = {}
    for patch_file in patch_files:
        parser = PatchFileParser(patch_file, "", 0)
        parser.parse_file()
        for attributes in parser.unknown_attributes.values():
            for patch in attributes.cell_data:
                values = np.array(patch.values).reshape(-1, unknowns)
                n = round(len(values) ** (1.0 / 3.0))
                # x runs fastest within a patch
                x = patch.offset[0] + (np.arange(len(values)) % n + 0.5) * patch.size[0] / n
                result[tuple(patch.offset)] = (values, x)
    keys = sorted(result)
    return (
        time_stamp,
        np.concatenate([result[key][0] for key in keys]) if keys else None,
        np.concatenate([result[key][1] for key in keys]) if keys else None,
        keys,
    )


def gauge_wave_lapse(x, t):
    """! Analytical lapse, see exagrype::initialvalues::gaugeWave() """
    return np.sqrt(1.0 - 0.1 * np.sin(2.0 * np.pi * (x - t)))


def compare_one(scheme, cells, verbose):
    """!

    Compare FO and SO on one grid

    @return dict
      Summary numbers, or None if one of the runs is missing.

    """
    evolved = sum(exagrype.api.FOSOCCZ4Solver.Unknowns.values())
    try:
        t, fo, x, keys_fo = snapshot(scheme, "FO", cells, 59)
        _, so, _, keys_so = snapshot(scheme, "SO", cells, evolved)
        _, initial, _, _ = snapshot(scheme, "FO", cells, 59, 0)
    except (FileNotFoundError, IndexError):
        return None
    if fo is None or keys_fo != keys_so:
        sys.exit("FAILED: the plots of FO and SO hold different or no patches")
    fo = fo[:, :evolved]
    initial = initial[:, :evolved]
    difference = so - fo
    if verbose:
        names = exagrype.api.FOSOCCZ4Solver("names").variable_names_per_entry()
        print(f"\n{scheme}, {cells}^3 grid points, t={t}")
        print(
            f"{'unknown':>10} {'max |SO-FO|':>13} {'rel. L2 |SO-FO|':>16} {'max |FO(t)-FO(0)|':>18}"
        )
        for k, name in enumerate(names):
            norm = np.linalg.norm(fo[:, k])
            relative = np.linalg.norm(difference[:, k]) / norm if norm > 0 else 0.0
            movement = np.max(np.abs(fo[:, k] - initial[:, k]))
            print(
                f"{name:>10} {np.max(np.abs(difference[:, k])):13.3e} {relative:16.3e} {movement:18.3e}"
            )
    exact = gauge_wave_lapse(x, t)
    lapse = 16
    summary = {
        "time": t,
        "max_difference": float(np.max(np.abs(difference))),
        "lapse_difference": float(np.max(np.abs(difference[:, lapse]))),
        "lapse_error_FO": float(np.linalg.norm(fo[:, lapse] - exact) / np.linalg.norm(exact)),
        "lapse_error_SO": float(np.linalg.norm(so[:, lapse] - exact) / np.linalg.norm(exact)),
    }
    for formulation in formulations:
        with open(os.path.join(directory_of(scheme, formulation, cells), "wall-time.json")) as file:
            summary["wall_time_" + formulation] = json.load(file)["wall_time"]
    return summary


def compare_single(scheme, formulation, cells):
    """!

    Summarise one formulation on one grid against the analytical lapse. Used
    if only one of FO and SO is built, i.e. there is nothing to compare it to.

    @return dict
      Summary numbers, or None if the run is missing.

    """
    evolved = sum(exagrype.api.FOSOCCZ4Solver.Unknowns.values())
    try:
        t, values, x, _ = snapshot(
            scheme, formulation, cells, 59 if formulation == "FO" else evolved
        )
        with open(os.path.join(directory_of(scheme, formulation, cells), "wall-time.json")) as file:
            wall_time = json.load(file)["wall_time"]
    except (FileNotFoundError, IndexError):
        return None
    if values is None:
        return None
    exact = gauge_wave_lapse(x, t)
    return {
        "time": t,
        "lapse_error": float(np.linalg.norm(values[:, 16] - exact) / np.linalg.norm(exact)),
        "wall_time": wall_time,
    }


def compare():
    if len(formulations) == 1:
        formulation = formulations[0]
        print(f"\n{'scheme':>7} {'cells':>6} {'form.':>6} {'t':>7} {'lapse err':>10} {'wall [s]':>9}")
        for scheme in schemes:
            for cells in grid_sizes:
                s = compare_single(scheme, formulation, cells)
                if s is not None:
                    print(
                        f"{scheme:>7} {cells:>6} {formulation:>6} {s['time']:7.3f} "
                        f"{s['lapse_error']:10.2e} {s['wall_time']:9.2f}"
                    )
        return
    rows = []
    for scheme in schemes:
        for cells in grid_sizes:
            summary = compare_one(scheme, cells, verbose=len(grid_sizes) == 1)
            if summary is not None:
                rows.append((scheme, cells, summary))
    print(
        f"\n{'scheme':>7} {'cells':>6} {'t':>7} {'max|SO-FO|':>11} {'max|SO-FO| lapse':>17} "
        f"{'lapse err FO':>13} {'lapse err SO':>13} {'wall FO [s]':>12} {'wall SO [s]':>12}"
    )
    for scheme, cells, s in rows:
        print(
            f"{scheme:>7} {cells:>6} {s['time']:7.3f} {s['max_difference']:11.2e} {s['lapse_difference']:17.2e} "
            f"{s['lapse_error_FO']:13.2e} {s['lapse_error_SO']:13.2e} "
            f"{s['wall_time_FO']:12.2f} {s['wall_time_SO']:12.2f}"
        )


if arguments.mode != "compare-only":
    for scheme in schemes:
        for cells in grid_sizes:
            for formulation in formulations:
                build(scheme, formulation, cells)
                if arguments.mode != "build":
                    run(scheme, formulation, cells)
if arguments.mode in ["compare-only", "run-and-compare"]:
    compare()
