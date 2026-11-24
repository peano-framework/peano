r"""!

@page benchmarks_exagrype_gauge_wave_with_scalar_field Gauge wave with a scalar field

\tableofcontents

The gauge wave is a pure gauge perturbation of flat spacetime, i.e. an analytical solution of the
Einstein equations that travels along the x-axis with speed one. It is periodic over the unit cube.
This script evolves it with the first-order CCZ4 solver of exagrype.api (exagrype.api.FOCCZ4Solver).
Next to CCZ4, it hosts four scalar fields which the spacetime advects
(exagrype.api.scalarfield.ScalarField)

@f$ \partial_t \phi + \alpha^2 \partial_x \phi = 0, @f$

with the lapse @f$ \alpha @f$ of CCZ4 (the metric component g_xx of the gauge wave is @f$ \alpha^2 @f$).
The coupling is a fake coupling: the spacetime advects the field, but the field does not feed back.
The equation has a closed-form solution on the gauge wave, so the benchmark measures the
error of the scalar field against the exact solution (see convergence-study.py).

## Couplings

The coupling decides how the time integrators of the two solvers interact:

- `--coupling outer-multiplicative`: each solver has its numerical scheme. CCZ4 completes its time
  step, then the scalar field performs its step with the frozen spacetime of the end of the step
  (exagrype.api.scalarfield.CCZ4WithScalarField_OuterMultiplicativeCoupling).
  The scalar field is first order in the time step size, whatever the order of the time integrators is.
- `--coupling inner-additive` and `--coupling inner-multiplicative`: one numerical scheme for both
  solvers (a "generator" which one assigns to the coupled solver
  exagrype.api.scalarfield.CCZ4WithScalarField_InnerCoupling). The solvers exchange data in
  each stage of the time integrator, i.e. the scalar field sees the spacetime of the time level of its stage. The scalar field has the
  order of the time integrator (four for `rk4fd4`). Additive: the stages of the solvers run in parallel.
  Multiplicative: in each stage, the scalar field follows CCZ4.

The numerical scheme (`--scheme`) is the same for both solvers: `fv` (Finite Volumes, explicit Euler)
or `rk1fd4`, ..., `rk4fd4` (fourth-order Finite Differences with Runge-Kutta of order one to four).
benchmarks/exagrype/gauge-wave-with-scalar-field/DESIGN.md documents the design.

## Usage

The script requires a Peano build with

    --enable-exahype --enable-blockstructured --enable-loadbalancing --enable-exagrype

and is run from this directory via

    python3 gauge-wave-with-scalar-field.py --mode release --scheme rk4fd4 --coupling inner-additive
    python3 gauge-wave-with-scalar-field.py --mode debug --scheme fv --coupling outer-multiplicative --mesh-size 0.1

The debug mode visualises the task graphs. The script convergence-study.py runs all couplings
over several mesh sizes, and estimates the convergence orders.

"""

import argparse
import os

import peano4.api
import peano4.api.taskgraph.compiler.Compilers
import exahype2.api
import exagrype.api


parser = argparse.ArgumentParser(description="ExaGRyPE gauge wave with a scalar field")
parser.add_argument(
    "--mode",
    choices=["release", "debug"],
    required=True,
    help="Build mode of the generated code. Release uses a longer simulation "
    "and text logging; debug a very short one with visualised logging.",
)
parser.add_argument(
    "--scheme",
    choices=["fv", "rk1fd4", "rk2fd4", "rk3fd4", "rk4fd4"],
    required=True,
    help="Numerical scheme of both solvers: Finite Volumes with explicit Euler (fv), or "
    "fourth-order Finite Differences with Runge-Kutta of order 1 to 4 (rk1fd4, ..., rk4fd4).",
)
parser.add_argument(
    "--coupling",
    choices=["outer-multiplicative", "inner-additive", "inner-multiplicative"],
    required=True,
    help="How the time integrators of CCZ4 and the scalar field interact.",
)
parser.add_argument(
    "--mesh-size",
    type=float,
    default=1.0 / 27.0,
    help="Edge length h of one volume or grid cell (default: 1/27, i.e. 3x3x3 mesh cells "
    "with patches of 9x9x9 cells). The mesh is regular and is refined by factors of three "
    "from the root cell, i.e. h is rounded to patch-size/3^k.",
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
    "--compiler",
    choices=list(peano4.api.taskgraph.compiler.Compilers.COMPILERS),
    default="fusion",
    help="Task graph compiler (default: fusion)",
)
parser.add_argument(
    "--plot-along-x-axis",
    action="store_true",
    help="Restrict the plots to the patches along the line y=z=0.5 and write them "
    "with ten digits. Needed to measure the error of the scalar field, see convergence-study.py. "
    "Four plots per simulation are written.",
)
args = parser.parse_args()

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
finite_volumes = args.scheme == "fv"
rk_order = None if finite_volumes else int(args.scheme[2])
flavour = exahype2.api.fv if finite_volumes else exahype2.api.rkfd4
scalarfield = exagrype.api.scalarfield

project = exahype2.api.Project(
    namespace=["benchmarks", "exagrype", "gaugewavewithscalarfield"],
    project_name="gauge-wave-with-scalar-field",
    directory=args.directory,
    executable="gauge-wave-with-scalar-field",
)

## The coupling hosts both solvers. Only the coupling is added to the project.
if args.coupling == "outer-multiplicative":
    coupled = scalarfield.CCZ4WithScalarField_OuterMultiplicativeCoupling()
    for solver in coupled.solvers:
        ## Each solver brings a numerical scheme of its own
        if finite_volumes:
            solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(
                solver, patch_size
            )
        else:
            Scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
            Kernel = Scheme.RightHandSideKernel
            if solver is coupled.ccz4:
                solver.numerical_scheme = Scheme(
                    solver,
                    patch_size,
                    rk_order=rk_order,
                    right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
                    advection_velocity=exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY,
                    advected_quantities=solver.lopsided_advection_quantities(),
                )
            else:
                solver.numerical_scheme = Scheme(solver, patch_size, rk_order=rk_order)
else:
    coupled = scalarfield.CCZ4WithScalarField_InnerCoupling()
    ## One numerical scheme (generator) for the whole coupled system, none for the solvers
    multiplicative = args.coupling == "inner-multiplicative"
    if finite_volumes:
        Generator = (
            scalarfield.FiniteVolumesInnerMultiplicative
            if multiplicative
            else scalarfield.FiniteVolumesInnerAdditive
        )
        coupled.numerical_scheme = Generator(patch_size)
    else:
        Generator = (
            scalarfield.RungeKuttaFD4InnerMultiplicative
            if multiplicative
            else scalarfield.RungeKuttaFD4InnerAdditive
        )
        coupled.numerical_scheme = Generator(patch_size, rk_order)

ccz4 = coupled.ccz4
scalar_field = coupled.scalar_field
ccz4.add_pde_term(exagrype.api.initialvalues.gauge_wave())

plotter_arguments = {}
if args.plot_along_x_axis:
    plotter_arguments["plotter_precision"] = 10
for solver in coupled.solvers:
    solver.grid_construction = flavour.SetupSimulationOverRegularGrid(
        solver, volume_size=volume_size
    )
    solver.plotter = flavour.PlotAsVTKFile(
        solver,
        first_plot_time_stamp=0.0,
        time_in_between_plots=time_in_between_plots,
        **plotter_arguments,
    )
    if args.plot_along_x_axis:
        solver.plotter.add_plot_filter([0.0, 0.49, 0.49], [1.0, 0.02, 0.02])

project.add_solver(coupled)

project.compiler = peano4.api.taskgraph.compiler.Compilers.COMPILERS[args.compiler]

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
