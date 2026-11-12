import argparse
import os

import peano4.api
import exahype2


parser = argparse.ArgumentParser(description="ExaHyPE 2 smooth convergence test (Euler entropy wave)")
parser.add_argument("-m", "--build-mode", choices=["release", "debug"], default="release",
                    help="Release uses fine meshes and text logging; debug a coarse mesh and visualised logging (default: release)")
parser.add_argument("--scheme", choices=["fv", "fv-optimistic"], default="fv-optimistic",
                    help="Numerical scheme: Finite Volumes with explicit Euler (fv) or with optimistic time stepping (fv-optimistic, default)")
parser.add_argument("--compiler", choices=["sequential", "fusion", "producer-consumer", "producer-consumer-fusion"],
                    default="sequential", help="Task graph compiler (default: sequential)")
parser.add_argument("--cxx", default=None,
                    help="C++ compiler to build the generated code with, e.g. clang++, g++ or icpx (default: the one Peano was configured with)")
parser.add_argument("-j", "--build-jobs", type=int, default=0,
                    help="Number of parallel make jobs used to build; 0 or less uses all available cores (default: all cores)")
parser.add_argument("-t", "--threads", type=int, default=None,
                    help="Number of threads of the generated executable (default: all available cores)")
parser.add_argument("--dimensions", type=int, choices=[2, 3], default=2, help="Dimension (default: 2)")
parser.add_argument("--patch-size", type=int, default=9, help="Cells per patch and axis (default: 9)")
parser.add_argument("--volume-size", type=float, default=None,
                    help="Maximum grid cell size (default: 0.0001 for release, 0.001 for debug)")
parser.add_argument("--end-time", type=float, default=None,
                    help="Final simulation time (default: 0.001 for release, 0.01 for debug)")
parser.add_argument("--plot-interval", type=float, default=0.001, help="Time between plots (default: 0.001)")
parser.add_argument("--directory", default=".", help="Directory to generate and build in (default: .)")
parser.add_argument("--no-clean", action="store_true", help="Do not run make clean before building")
parser.add_argument("--no-build", action="store_true", help="Only generate the code, do not build it")
args = parser.parse_args()

create_release_build = args.build_mode == "release"
if create_release_build:
    build_mode = peano4.api.output.CompileMode.RELEASE
    volume_size = 0.0001
    end_time    = 0.001
    verbosity   = peano4.api.LoggerVerbosity.TEXT
else:
    build_mode = peano4.api.output.CompileMode.DEBUG
    volume_size = 0.001
    end_time    = 0.01
    verbosity   = peano4.api.LoggerVerbosity.VISUALISATION
if args.volume_size is not None:
    volume_size = args.volume_size
if args.end_time is not None:
    end_time = args.end_time
time_in_between_plots = args.plot_interval

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
    directory=args.directory,
    executable="smooth-convergence-test",
)

solver_name = "EulerFVSmooth"
unknowns = 4
patch_size = args.patch_size
dimensions = args.dimensions

thesolver = exahype2.api.Solver(
    solver_name,
    unknowns,
)

thesolver.add_pde_term( exahype2.api.Flux(stateless=False,))
thesolver.add_pde_term(exahype2.api.MaxEigenvalue(stateless=False,))
thesolver.add_pde_term(exahype2.api.InitPointWisely(stateless=False,))
thesolver.add_pde_term(exahype2.api.DirichletBoundaryConditions(stateless=False,))

thesolver.numerical_scheme = {"fv": exahype2.api.fv.RusanovWithExplicitEuler,
                             "fv-optimistic": exahype2.api.fv.RusanovWithOptimisticExplicitEuler}[args.scheme](thesolver, patch_size)
thesolver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
    thesolver, volume_size=volume_size
)
thesolver.plotter = exahype2.api.fv.PlotAsVTKFile(
    thesolver, first_plot_time_stamp=0.0, time_in_between_plots=time_in_between_plots
)
project.add_solver(thesolver)



project.compiler = task_graph_compiler


project.set_global_simulation_parameters(
    dimensions=dimensions,
    offset=[-0.1] * dimensions,
    size=[0.2] * dimensions,
    min_end_time=end_time,
    max_end_time=end_time,
    periodic_BC=[False, False, False],
#    periodic_BC=[True, True, True],
)

project.set_load_balancing("toolbox::loadbalancing::strategies::SpreadOutOnceGridStagnates", "new ::exahype2::LoadBalancingConfiguration()")
if args.threads is not None:
    project.set_number_of_threads(args.threads)
project.set_Peano4_installation(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."), build_mode)
peano4_project = project.generate_Peano4_project(verbosity=verbosity)
if args.cxx is not None:
    peano4_project.output.makefile.set_CXX_compiler(args.cxx)
peano4_project.generate()
if not args.no_build:
    peano4_project.build(make_clean_first=not args.no_clean, number_of_parallel_builds=args.build_jobs)
