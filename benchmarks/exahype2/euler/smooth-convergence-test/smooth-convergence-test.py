# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Entropy wave benchmark of the Euler equations

Generates and builds one executable of the entropy wave benchmark
(@ref benchmarks_exahype2_euler_smooth_setup) for one numerical scheme and
one mesh width. Run the script once per mesh width, each time with a
different output directory, and compare the outcomes with compute-errors.py
(or let convergence-study.py do all of this).

"""
import argparse
import os

import peano4.api
import exahype2.api


## CFL constants per numerical scheme. They are hard-coded as they are a
## property of the scheme, not of the benchmark: 0.9 is the default of the
## Finite Volume schemes, 0.5 the conservative default of the fourth-order
## Finite Differences.
CFL_CONSTANTS = {
    "fv": 0.9,
    "fv-optimistic": 0.9,
    "rk1fd4": 0.5,
    "rk2fd4": 0.5,
    "rk3fd4": 0.5,
    "rk4fd4": 0.5,
}

parser = argparse.ArgumentParser(
    description="ExaHyPE 2 smooth convergence test (Euler entropy wave)"
)
parser.add_argument(
    "--scheme",
    choices=list(CFL_CONSTANTS.keys()),
    default="rk4fd4",
    help="Numerical scheme: Finite Volumes with explicit Euler (fv), the same with "
    "optimistic time stepping (fv-optimistic), or fourth-order Finite Differences with "
    "Runge-Kutta of order 1 to 4 (rk1fd4, ..., rk4fd4) (default: rk4fd4)",
)
parser.add_argument(
    "--compiler",
    choices=["sequential", "fusion", "producer-consumer", "producer-consumer-fusion"],
    default="fusion",
    help="Task graph compiler (default: fusion)",
)
parser.add_argument(
    "--cxx",
    default=None,
    help="C++ compiler to build the generated code with, e.g. clang++, g++ or icpx "
    "(default: the one Peano was configured with)",
)
parser.add_argument(
    "-j",
    "--build-jobs",
    type=int,
    default=0,
    help="Number of parallel make jobs used to build; 0 or less uses all available cores (default: all cores)",
)
parser.add_argument(
    "-t",
    "--threads",
    type=int,
    default=None,
    help="Number of threads of the generated executable (default: all available cores)",
)
parser.add_argument(
    "--output-directory",
    default=".",
    help="Directory the code is generated and built in (default: .)",
)
parser.add_argument(
    "--volume-size",
    type=float,
    default=0.0075,
    help="Maximum grid cell size. 0.0075, 0.0025 and 0.00083 yield 27, 81 and 243 grid cells per axis (default: 0.0075)",
)
parser.add_argument(
    "--patch-size", type=int, default=9, help="Grid cells per patch and axis (default: 9)"
)
parser.add_argument(
    "--end-time", type=float, default=0.05, help="Simulated time (default: 0.05)"
)
parser.add_argument(
    "--plot-interval",
    type=float,
    default=None,
    help="Time between plots (default: the end time, i.e. the initial and the final state)",
)
parser.add_argument(
    "--time-step-size",
    type=float,
    default=None,
    help="Fixed time step size (rkNfd4 only). Without this argument, the time step size is adaptive",
)
parser.add_argument(
    "--periodic",
    action="store_true",
    help="Use periodic boundary conditions instead of Dirichlet conditions with the exact solution",
)
parser.add_argument(
    "--dimensions", type=int, default=2, choices=[2, 3], help="Dimension (default: 2)"
)
parser.add_argument(
    "-m",
    "--build-mode",
    default="release",
    choices=["release", "debug"],
    help="Peano build flavour to link against (default: release)",
)
parser.add_argument(
    "--peano-root",
    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."
    ),
    help="Configured and built Peano directory (default: the repository this script resides in)",
)
parser.add_argument(
    "--verbosity",
    default=None,
    choices=["visualisation", "text"],
    help="Show the task graphs in a window or print them as text (default: text for release, visualisation for debug)",
)
parser.add_argument(
    "--no-clean", action="store_true", help="Do not run make clean before building"
)
parser.add_argument(
    "--no-build", action="store_true", help="Only generate the code, do not build it"
)
args = parser.parse_args()

is_finite_volume = args.scheme in ["fv", "fv-optimistic"]
if args.time_step_size is not None and is_finite_volume:
    parser.error("--time-step-size is only supported by the rkNfd4 schemes")
if args.verbosity is None:
    args.verbosity = "text" if args.build_mode == "release" else "visualisation"
plot_interval = args.plot_interval if args.plot_interval is not None else args.end_time

compilers = peano4.api.taskgraph.compiler
task_graph_compiler = {
    "sequential": compilers.Sequential.translate,
    "fusion": compilers.FuseSteps.translate,
    "producer-consumer": compilers.CellTaskProducerConsumer.translate_with_sequential,
    "producer-consumer-fusion": compilers.CellTaskProducerConsumer.translate_with_fusion,
}[args.compiler]


project = exahype2.api.Project(
    namespace=["benchmarks", "exahype2", "euler", "smooth_convergence_test"],
    project_name="smooth-convergence-test",
    directory=args.output_directory,
    executable="smooth-convergence-test",
)

solver = exahype2.api.Solver("EulerSmooth", {"rho": 1, "j": args.dimensions, "E": 1})
solver.add_user_solver_includes(
    """
#include <algorithm>
#include <cmath>
"""
)

# Pressure from the conserved quantities. The momentum has Dimensions
# components, the energy is the last unknown.
PRESSURE = """
      constexpr double gamma = 1.4;
      const double irho = 1.0 / Q[0];
      double kineticEnergy = 0.0;
      for (int d=0; d<Dimensions; d++) {
        kineticEnergy += 0.5 * irho * Q[1+d] * Q[1+d];
      }
      const double p = (gamma - 1.0) * (Q[Dimensions+1] - kineticEnergy);
"""

solver.add_pde_term(
    exahype2.api.Flux(
        implementation=PRESSURE
        + """
      const double velocity = Q[1+normal] * irho;
      F[0] = Q[1+normal];
      for (int d=0; d<Dimensions; d++) {
        F[1+d] = velocity * Q[1+d];
      }
      F[1+normal] += p;
      F[Dimensions+1] = velocity * (Q[Dimensions+1] + p);
"""
    )
)
solver.add_pde_term(
    exahype2.api.MaxEigenvalue(
        implementation=PRESSURE
        + """
      const double c = std::sqrt(gamma * p * irho);
      const double u = Q[1+normal] * irho;
      lambdaMax[0] = std::max(std::abs(u + c), std::abs(u - c));
"""
    )
)

# Entropy wave: one period of a sine density profile over the domain
# [-0.1,0.1]^d, constant velocity u=1 along x and constant pressure p=1. The
# exact solution is rho(x-ut), i.e. the wave is back where it started after
# t = 0.2.
EXACT_SOLUTION = """
      constexpr double gamma       = 1.4;
      constexpr double p           = 1.0;
      constexpr double u           = 1.0;
      constexpr double domainLeft  = -0.1;
      constexpr double domainWidth = 0.2;
      const double     rho         = 1.0 + 0.2 * std::sin(2.0 * M_PI * (x(0) - u * t - domainLeft) / domainWidth);
      Q[0] = rho;
      Q[1] = rho * u;
      for (int d=1; d<Dimensions; d++) {
        Q[1+d] = 0.0;
      }
      Q[Dimensions+1] = p / (gamma - 1.0) + 0.5 * rho * u * u;
"""
solver.add_pde_term(
    exahype2.api.InitPointWisely(
        implementation="      constexpr double t = 0.0;\n" + EXACT_SOLUTION
    )
)
if not args.periodic:
    solver.add_pde_term(
        exahype2.api.DirichletBoundaryConditions(implementation=EXACT_SOLUTION)
    )

cfl_constant = CFL_CONSTANTS[args.scheme]
if is_finite_volume:
    scheme_class = {
        "fv": exahype2.api.fv.RusanovWithExplicitEuler,
        "fv-optimistic": exahype2.api.fv.RusanovWithOptimisticExplicitEuler,
    }[args.scheme]
    solver.numerical_scheme = scheme_class(
        solver, args.patch_size, cfl_constant=cfl_constant
    )
    solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
        solver, volume_size=args.volume_size
    )
    # The plot files carry 14 significant digits, as the errors of the finer
    # meshes are far below the default precision
    solver.plotter = exahype2.api.fv.PlotAsVTKFile(
        solver,
        first_plot_time_stamp=0.0,
        time_in_between_plots=plot_interval,
        plotter_precision=14,
    )
else:
    solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver,
        patch_size=args.patch_size,
        rk_order=int(args.scheme[2]),
        time_step_size=args.time_step_size,
        cfl_constant=cfl_constant,
    )
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        solver, volume_size=args.volume_size
    )
    solver.plotter = exahype2.api.rkfd4.PlotAsVTKFile(
        solver,
        first_plot_time_stamp=0.0,
        time_in_between_plots=plot_interval,
        plotter_precision=14,
    )
project.add_solver(solver)

project.set_global_simulation_parameters(
    dimensions=args.dimensions,
    offset=[-0.1] * args.dimensions,
    size=[0.2] * args.dimensions,
    min_end_time=args.end_time,
    max_end_time=args.end_time,
    periodic_BC=[args.periodic] * 3,
)
project.set_load_balancing(
    "toolbox::loadbalancing::strategies::SpreadOutOnceGridStagnates",
    "new ::exahype2::LoadBalancingConfiguration()",
)
project.compiler = task_graph_compiler
if args.threads is not None:
    project.set_number_of_threads(args.threads)

project.set_Peano4_installation(
    args.peano_root,
    peano4.api.output.CompileMode.RELEASE
    if args.build_mode == "release"
    else peano4.api.output.CompileMode.DEBUG,
)
peano4_project = project.generate_Peano4_project(
    verbosity=peano4.api.LoggerVerbosity.TEXT
    if args.verbosity == "text"
    else peano4.api.LoggerVerbosity.VISUALISATION
)
if args.cxx is not None:
    peano4_project.output.makefile.set_CXX_compiler(args.cxx)
peano4_project.generate()
if not args.no_build:
    peano4_project.build(
        make_clean_first=not args.no_clean,
        number_of_parallel_builds=args.build_jobs,
    )
