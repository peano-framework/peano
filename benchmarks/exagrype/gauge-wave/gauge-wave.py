r"""!

@page benchmarks_exagrype_gauge_wave Gauge wave

\tableofcontents

@image html benchmarks/exagrype/gauge-wave/example-output/convergence.png

The gauge wave is a pure gauge perturbation of flat spacetime: an analytical
solution of the Einstein equations which travels along the x-axis with speed one.
It is periodic over the unit cube and hence a standard first test of any numerical
relativity code. The solution is known exactly, so we can always tell what the
numerical error is. If the wave shape or its amplitude changes over time,
this is due to numerical effects, and the setup is convenient to assess and compare
numerical schemes. The picture above shows the error of the lapse over the mesh size
for five schemes. It is the outcome of the convergence study that doubles as the
integration test of the numerical schemes (see below).

The script uses exagrype.api, i.e. the first-order CCZ4 solver
exagrype.api.FOCCZ4Solver, and discretises it with


- Finite Volumes with explicit Euler time stepping (`--scheme fv`),
- the same with optimistic time stepping (`--scheme fv-optimistic`), or
- fourth-order Finite Differences with a Runge-Kutta scheme of order one to four (`--scheme
  rk1fd4`, ..., `rk4fd4`).


## Prepare Peano

Peano has to be built with ExaHyPE and ExaGRyPE, i.e. with

~~~~~~~~~~~~~~~~~~~~~~~
--enable-exahype --enable-blockstructured --enable-loadbalancing --enable-exagrype
~~~~~~~~~~~~~~~~~~~~~~~

The generated code is compiled with the compiler of your choice (we use the
Intel compiler with its path in `PATH`), and the Python packages have to be
on the Python path (`export PYTHONPATH=$PYTHONPATH:../../../src`).


## Run a simulation

The script has to be told what to do. It requires a build mode and a scheme:

~~~~~~~~~~~~~~~~~~~~~~~
python3 gauge-wave.py --mode release --scheme rk4fd4
python3 gauge-wave.py --mode debug --scheme fv --mesh-size 0.1
~~~~~~~~~~~~~~~~~~~~~~~

Use `--help` to see all options. `--mode release` builds an optimised executable
and runs until time 1.0; `--mode debug` builds with assertions and visualises the
task graphs. `--mesh-size` is the edge length h of one grid cell. The default is 1/27, i.e. 3x3x3
mesh cells with patches of 9x9x9 grid cells. The mesh is regular and refined by factors of
three, i.e. h is rounded to patch size over 3^k (the patch size can be set via
`--patch-size`, but the root cell is refined at least once). `--compiler` switches
between the task graph compilers (sequential, fusion which is the default, and the
producer-consumer variants of both), `--end-time`, `--threads` and `--directory` do what
they say.

The script generates the C++ code, the Makefile, and compiles the executable. Run the
executable `gauge-wave` afterwards. To get rid of all the generated files, remove the
directories `celldata facedata globaldata observers tasks vertexdata repositories`, the
`Abstract*` files, the `Makefile` and all `*.o` files. You can recreate them any time by
rerunning the Python script.


## Convergence study (integration test)

The script `convergence-study.py` builds and runs the benchmark for all schemes `fv`,
`rk1fd4`, ..., `rk4fd4` and for the volume sizes h=1/9, 1/27 and 1/81 (patch size 9; the
coarsest mesh needs patches of size 3, as Peano refines the root cell at least once) until time
t=0.1. It records the maximum error of the lapse along the line y=z=0.5 in the last
snapshot against the exact solution

@f[ \alpha (x,t) = \sqrt{1 - A \sin (2 \pi (x-t))}, \quad A=0.1, @f]

estimates the convergence order p ~ log(e(3h)/e(h))/log(3), and exits with a non-zero
status if the order between the two finest meshes of a scheme falls below a threshold. It
is run via the skill `exahype2:integration-test` (see
documentation/exahype2/integration-test/SKILL.md). Whenever you change a numerical scheme, the
task graph, the kernels or the code generation, this is the test to rerun.

| scheme | h = 1/9 | h = 1/27 | order | h = 1/81 | order |
|---|---|---|---|---|---|
| fv     | 1.58e-2 | 6.59e-3 | 0.79 | 2.34e-3 | 0.94 |
| rk1fd4 | 1.09e-3 | 3.37e-4 | 1.06 | 1.10e-4 | 1.02 |
| rk2fd4 | 1.46e-4 | 3.01e-6 | 3.53 | 2.39e-7 | 2.31 |
| rk3fd4 | 1.39e-4 | 2.70e-6 | 3.59 | 3.34e-8 | 4.00 |
| rk4fd4 | 1.39e-4 | 2.71e-6 | 3.58 | 3.39e-8 | 3.99 |

What we see:

- The Finite Volume scheme is first order, as expected. Its error is orders of magnitude
  above the one of the Finite Difference schemes, which discretise in space with fourth order.
- The order of the Finite Difference schemes is the smaller one of the spatial order (four)
  and the order of the time integration. With one Runge-Kutta stage (explicit Euler),
  the time integration error dominates from the coarsest mesh on, and the scheme is
  first order. As the time step size shrinks with the mesh size (CFL condition), the time
  error is tiny for higher orders on coarse meshes: rk2fd4, rk3fd4 and rk4fd4 yield the same
  error on the two coarsest meshes, and rk2fd4 leaves the fourth-order regime as soon as its
  second-order time error becomes visible (order 2.3 between 1/27 and 1/81). Rk3fd4 and
  rk4fd4 show the full fourth order.
- The estimated orders on the coarsest meshes are not asymptotic: The wave is barely
  resolved with 9 volumes per wave length.
- The error is measured at the last snapshot, whose time stamp overshoots the end time by less
  than one time step and differs between meshes. This biases the orders by about 0.01.
- The study ran on an eight-core workstation. The finest meshes are the limit of what
  fits into 16 GB of memory: A Finite Difference run with 81 volumes per axis needs 2 to 3 GB,
  and the memory grows with the number of volumes, i.e. h=1/243 needs 27 times more and does not fit
  (the study tried and the runs were killed). The finest mesh, h=1/81, took between
  3 and 11 minutes with two threads. These numbers are recorded in the table of the
  script, but they are not an acceptance criterion; the (to-be-written) skill
  `exahype2:performance-test` is supposed to cover that.


## Postprocess/visualise the results

The convergence study uses `--plot-along-x-axis`, which restricts the output to the patches along
the line y=z=0.5 and writes them with ten digits. Without this option, the code writes
the whole solution as Peano patch files every 0.1 time units. You can convert them into vtk and
visualise them via Paraview:

~~~~~~~~~~~~~~~~~~~~~~~
pvpython ../../../python/peano4/visualisation/render.py --filter-fine-grid solution-CCZ4.peano-patch-file
~~~~~~~~~~~~~~~~~~~~~~~

"""

import argparse
import os

import peano4.api
import exahype2.api
import exagrype.api


parser = argparse.ArgumentParser(description="ExaGRyPE gauge wave benchmark")
parser.add_argument(
    "--mode",
    choices=["release", "debug"],
    required=True,
    help="Build mode of the generated code. Release uses a longer simulation "
    "and text logging; debug a very short one with visualised logging.",
)
parser.add_argument(
    "--mesh-size",
    type=float,
    default=1.0 / 27.0,
    help="Edge length h of one Finite Volume cell or Finite Difference grid cell "
    "(default: 1/27, i.e. 3x3x3 mesh cells with patches of 9x9x9 cells). The mesh is "
    "regular and is refined by factors of three from the root cell, i.e. h is "
    "rounded to patch-size/3^k.",
)
parser.add_argument(
    "--patch-size",
    type=int,
    default=9,
    help="Number of grid cells per patch and axis (default: 9). Peano refines the root "
    "cell at least once, i.e. h can be at most patch-size/3.",
)
parser.add_argument(
    "--end-time",
    type=float,
    default=None,
    help="Final simulation time (default: 1.0 for release, 0.02 for debug)",
)
parser.add_argument(
    "--directory",
    default=".",
    help="Directory for the generated code, its build and the outputs (default: the current directory)",
)
parser.add_argument(
    "--threads",
    type=int,
    default=None,
    help="Number of threads of the generated executable (default: all cores)",
)
parser.add_argument(
    "--plot-along-x-axis",
    action="store_true",
    help="Restrict the plots to the patches along the line y=z=0.5 and write them "
    "with ten digits. Needed to measure the error of the lapse, see convergence-study.py. "
    "Four plots per simulation are written.",
)
parser.add_argument(
    "--scheme",
    choices=["fv", "fv-optimistic", "rk1fd4", "rk2fd4", "rk3fd4", "rk4fd4"],
    required=True,
    help="Numerical scheme: Finite Volumes with explicit Euler (fv), the same "
    "with optimistic time stepping (fv-optimistic), or fourth-order Finite "
    "Differences with Runge-Kutta of order 1 to 4 (rk1fd4, ..., rk4fd4).",
)
parser.add_argument(
    "--compiler",
    choices=["sequential", "fusion", "producer-consumer", "producer-consumer-fusion"],
    default="fusion",
    help="Task graph compiler: one traversal per task (sequential), tasks fused "
    "into as few traversals as possible (fusion, default), or the same two "
    "variants with cell tasks spawned as producer-consumer tasks.",
)
args = parser.parse_args()

compilers = peano4.api.taskgraph.compiler
task_graph_compiler = {
    "sequential": compilers.Sequential.translate,
    "fusion": compilers.FuseSteps.translate,
    "producer-consumer": compilers.CellTaskProducerConsumer.translate_with_sequential,
    "producer-consumer-fusion": compilers.CellTaskProducerConsumer.translate_with_fusion,
}[args.compiler]

## The grid construction refines every cell whose edge length is greater than
## or equal to volume_size times the patch size, i.e. the grid cells end up
## strictly smaller than volume_size. We pass in a value slightly bigger than the
## mesh size h to obtain grid cells of size h.
volume_size = 1.01 * args.mesh_size

if args.mode == "release":
    build_mode = peano4.api.output.CompileMode.RELEASE
    end_time = 1.0
    time_in_between_plots = 0.1
    verbosity = peano4.api.LoggerVerbosity.TEXT
else:
    build_mode = peano4.api.output.CompileMode.DEBUG
    end_time = 0.02
    time_in_between_plots = 0.01
    verbosity = peano4.api.LoggerVerbosity.VISUALISATION

if args.end_time is not None:
    end_time = args.end_time
if args.plot_along_x_axis:
    time_in_between_plots = end_time / 4.0

patch_size = args.patch_size
plotter_arguments = {}
if args.plot_along_x_axis:
    plotter_arguments["plotter_precision"] = 10


project = exahype2.api.Project(
    namespace=["benchmarks", "exagrype", "gaugewave"],
    project_name="gauge-wave",
    directory=args.directory,
    executable="gauge-wave",
)

solver = exagrype.api.FOCCZ4Solver(
    "CCZ4",
    initial_condition=exagrype.api.initialvalues.gauge_wave(),
)

if args.scheme in ["fv", "fv-optimistic"]:
    if args.scheme == "fv":
        scheme_class = exahype2.api.fv.RusanovWithExplicitEuler
    else:
        scheme_class = exahype2.api.fv.RusanovWithOptimisticExplicitEuler
    solver.numerical_scheme = scheme_class(solver, patch_size)
    solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
        solver, volume_size=volume_size
    )
    solver.plotter = exahype2.api.fv.PlotAsVTKFile(
        solver,
        first_plot_time_stamp=0.0,
        time_in_between_plots=time_in_between_plots,
        **plotter_arguments,
    )
    if args.plot_along_x_axis:
        solver.plotter.add_plot_filter([0.0, 0.49, 0.49], [1.0, 0.02, 0.02])
else:
    Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
    solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver,
        patch_size,
        rk_order=int(args.scheme[2]),
        right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
        advection_velocity=exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY,
        advected_quantities=solver.lopsided_advection_quantities(),
    )
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        solver, volume_size=volume_size
    )
    solver.plotter = exahype2.api.rkfd4.PlotAsVTKFile(
        solver,
        first_plot_time_stamp=0.0,
        time_in_between_plots=time_in_between_plots,
        **plotter_arguments,
    )
    if args.plot_along_x_axis:
        solver.plotter.add_plot_filter([0.0, 0.49, 0.49], [1.0, 0.02, 0.02])

project.add_solver(solver)
project.compiler = task_graph_compiler

project.set_global_simulation_parameters(
    dimensions=3,
    offset=[0.0, 0.0, 0.0],
    size=[1.0, 1.0, 1.0],
    min_end_time=end_time,
    max_end_time=end_time,
    periodic_BC=[True, True, True],
)

if args.threads is not None:
    project.set_number_of_threads(args.threads)

PeanoRoot = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
project.set_Peano4_installation(PeanoRoot, build_mode)
peano4_project = project.generate_Peano4_project(verbosity=verbosity)
peano4_project.generate()
peano4_project.build(make_clean_first=True)
