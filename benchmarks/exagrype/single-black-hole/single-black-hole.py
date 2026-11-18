"""

Single black hole benchmark for the ExaGRyPE Python API

A single Schwarzschild black hole at the origin of the domain [-243,243]^3,
as described in README.dox. The initial data stems from TwoPunctures, the
boundaries are Sommerfeld boundaries. The script uses exagrype.api, i.e. the
CCZ4 solver of the new Python API. Everything that matters for the benchmark
is configurable on the command line:

- the numerical scheme (--scheme, --rk-order, --ko-sigma, --lopsided),
- the adaptive mesh refinement (--amr-levels, --amr-radius),
- the task graph compiler (--compiler),
- the build variant (--variant), which picks the compile mode and the size of
  the setup, and
- the final time (--end-time).

The build variants are

- debug: Peano's debug build with a tiny setup to check that the code works,
- profile: Peano's build with statistics counters and a reduced setup
  (domain 172, volumes of size 1) which is small enough to profile on one
  node, and
- production: the release build with the setup of README.dox (domain 486,
  volumes of size 0.4, t=100). This one is far too big for a workstation.

Each variant fixes the defaults of --domain-size, --volume-size and
--end-time. Pass them explicitly to override them.

Adaptive mesh refinement respects the constraints of README.dox: the
maximum volume size (--volume-size) is a hard bound, i.e. the mesh is never
coarser than the regular mesh of this size. The refinement only ever
subdivides cells around the black hole further, by --amr-levels levels of
factor three each within --amr-radius of the puncture. It is available for
both schemes. The fd4 scheme starts on the regular mesh and refines it while
it time steps; it cannot combine AMR with the Kreiss-Oliger dissipation yet.

It requires a Peano build with

    --enable-exahype --enable-blockstructured --enable-loadbalancing --enable-exagrype --with-gsl

A run of the fourth-order Finite Differences with lopsided advection and
the Kreiss-Oliger dissipation looks like

    python3 single-black-hole.py --scheme fd4 --rk-order 4 --lopsided --ko-sigma 8.0 \
      --compiler producer-consumer-fusion --variant profile

and Finite Volumes with two additional levels of adaptive mesh refinement
around the black hole are

    python3 single-black-hole.py --scheme fv --amr-levels 2 --variant profile

The script writes two kinds of output into the output directory: binary
checkpoints (--checkpoint-interval, every 10 time units by default) holding
the whole solver state, and visualisation snapshots in Peano's patch file
format (--plot-interval, every 1 time unit by default), i.e.
solution-CCZ4.peano-patch-file and its companion files. The snapshots
contain all unknowns. probe-single-black-hole.py extracts profiles from
them. Pass 0 to switch either kind off.

"""

import argparse
import os

import peano4.api
import exahype2.api
import exagrype.api
import toolbox.loadbalancing.api.input.RegularGridGenerator as RegularGridGenerator
import toolbox.loadbalancing.api.strategies.SplitTrees as SplitTrees
import toolbox.loadbalancing.api.strategies.Decomposition as Decomposition
import toolbox.loadbalancing.api.output.Hardcoded as Hardcoded


Compilers = {
    "sequential": peano4.api.taskgraph.compiler.Sequential.translate,
    "fusion-only": peano4.api.taskgraph.compiler.FuseSteps.translate,
    "producer-consumer-sequential": peano4.api.taskgraph.compiler.CellTaskProducerConsumer.translate_with_sequential,
    "producer-consumer-fusion": peano4.api.taskgraph.compiler.CellTaskProducerConsumer.translate_with_fusion,
}

EnforceModes = {
    mode.name.lower(): mode
    for mode in exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint
}

## What a build variant fixes: the compile mode and the default setup size
Variants = {
    "debug": dict(
        mode=peano4.api.output.CompileMode.DEBUG,
        domain_size=18.0,
        volume_size=1.2,
        end_time=0.5,
    ),
    "profile": dict(
        mode=peano4.api.output.CompileMode.STATS,
        domain_size=172.0,
        volume_size=1.2,
        end_time=1.0,
    ),
    "production": dict(
        mode=peano4.api.output.CompileMode.RELEASE,
        domain_size=486.0,
        volume_size=0.4,
        end_time=100.0,
    ),
}

SplitModes = {
    "bottom-up": Hardcoded.BottomUp,
    "top-down": Hardcoded.TopDown,
    "inferred": None,
}

Verbosities = {
    mode.name.lower(): mode for mode in peano4.api.LoggerVerbosity
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Single Schwarzschild black hole with ExaGRyPE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    scheme = parser.add_argument_group("numerical scheme")
    scheme.add_argument(
        "--scheme",
        choices=["fv", "fd4"],
        default="fd4",
        help="fv: Rusanov Finite Volumes with explicit Euler. fd4: fourth-order Finite Differences with Runge-Kutta.",
    )
    scheme.add_argument(
        "--rk-order",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Order of the Runge-Kutta time integrator (fd4 only).",
    )
    scheme.add_argument(
        "--ko-sigma",
        type=float,
        default=0.0,
        help="Scaling of the Kreiss-Oliger dissipation (fd4 only). 0 switches the dissipation off.",
    )
    scheme.add_argument(
        "--ko-quantities",
        type=int,
        nargs="+",
        default=None,
        metavar="Q",
        help="Indices of the unknowns which are damped by the Kreiss-Oliger dissipation. Default: all.",
    )
    scheme.add_argument(
        "--lopsided",
        action="store_true",
        help="Add the lop-sided advection terms along the shift (fd4 only).",
    )
    scheme.add_argument(
        "--enforce-constraints",
        choices=list(EnforceModes),
        default="no",
        help="When to push the solution back onto the algebraic CCZ4 constraints.",
    )

    mesh = parser.add_argument_group("domain and mesh")
    mesh.add_argument(
        "--domain-size",
        type=float,
        default=None,
        help="Edge length of the domain. Default: depends on --variant.",
    )
    mesh.add_argument(
        "--volume-size",
        type=float,
        default=None,
        help="Maximum size of a volume or grid cell. The mesh is a regular 3-adic grid, i.e. the actual size is smaller. Default: depends on --variant.",
    )
    mesh.add_argument(
        "--amr-levels",
        type=int,
        default=0,
        help="Number of additional mesh refinement levels (factor 3 each) around the black hole. 0 yields the regular mesh.",
    )
    mesh.add_argument(
        "--amr-radius",
        type=float,
        default=12.0,
        help="Radius around the puncture within which the mesh is refined to the finest level.",
    )
    mesh.add_argument(
        "--patch-size",
        type=int,
        default=6,
        help="Volumes/grid cells per axis per mesh cell. Has to be even, so no volume centre hits the puncture.",
    )
    mesh.add_argument("--mass", type=float, default=1.0, help="Mass of the black hole.")

    time = parser.add_argument_group("time and checkpoints")
    time.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Final time. Default: depends on --variant.",
    )
    time.add_argument(
        "--checkpoint-interval",
        type=float,
        default=10.0,
        help="Time between two binary checkpoints. 0 switches checkpointing off.",
    )
    time.add_argument(
        "--plot-interval",
        type=float,
        default=1.0,
        help="Time between two visualisation snapshots. 0 switches the snapshots off.",
    )
    time.add_argument(
        "--first-checkpoint",
        type=float,
        default=0.0,
        help="Time stamp of the first checkpoint and of the first snapshot.",
    )
    time.add_argument(
        "--checkpoint-precision",
        type=int,
        default=10,
        help="Significant digits per value in the visualisation snapshots. Checkpoints are lossless.",
    )

    balancing = parser.add_argument_group("load balancing")
    balancing.add_argument(
        "--load-balancing",
        choices=["hardcoded", "spread-out"],
        default="hardcoded",
        help="hardcoded: decomposition of the regular mesh, computed offline. spread-out: SpreadOutOnceGridStagnates decides at runtime.",
    )
    balancing.add_argument(
        "--decomposition",
        choices=["equidistant", "quantised"],
        default="equidistant",
        help="Chunk sizes of the hardcoded load balancing. equidistant: every tree gets the same number of cells. quantised: chunk sizes are multiples of 3^(k*d) cells, i.e. trees hold (almost) complete subtrees and few virtual cells, but they might be imbalanced.",
    )
    balancing.add_argument(
        "--split-mode",
        choices=["bottom-up", "top-down", "inferred"],
        default="bottom-up",
        help="How Peano picks the cells of a split (hardcoded load balancing only).",
    )
    balancing.add_argument(
        "--trees",
        type=int,
        default=os.cpu_count(),
        help="Number of trees (subdomains) of the hardcoded load balancing. They all live on rank 0, i.e. one tree per thread.",
    )

    build = parser.add_argument_group("build")
    build.add_argument(
        "--compiler",
        choices=list(Compilers),
        default="producer-consumer-fusion",
        help="Task graph compiler, i.e. how time step tasks are mapped onto mesh traversals.",
    )
    build.add_argument(
        "--variant",
        choices=list(Variants),
        default="production",
        help="Build variant: compile mode and default size of the setup.",
    )
    build.add_argument(
        "--verbosity", choices=list(Verbosities), default="text"
    )
    build.add_argument(
        "--output-directory",
        default=".",
        help="Directory the code is generated into and the checkpoints are written to.",
    )
    build.add_argument(
        "--peano-directory",
        default="../../../",
        help="Root of the Peano installation.",
    )
    build.add_argument(
        "--no-build",
        action="store_true",
        help="Generate the code but do not compile it.",
    )
    build.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not run make clean before the build.",
    )

    arguments = parser.parse_args()

    for key, value in Variants[arguments.variant].items():
        if key != "mode" and getattr(arguments, key) is None:
            setattr(arguments, key, value)

    if arguments.patch_size % 2 != 0:
        parser.error("--patch-size has to be even")
    if arguments.scheme == "fv" and (arguments.ko_sigma > 0.0 or arguments.lopsided):
        parser.error("--ko-sigma and --lopsided require --scheme fd4")
    if arguments.ko_quantities is not None and arguments.ko_sigma <= 0.0:
        parser.error("--ko-quantities requires --ko-sigma > 0")
    if arguments.amr_levels < 0:
        parser.error("--amr-levels must not be negative")
    if arguments.amr_levels > 0 and arguments.scheme == "fd4" and arguments.ko_sigma > 0.0:
        parser.error("--amr-levels with --scheme fd4 does not support --ko-sigma yet")
    if arguments.load_balancing == "hardcoded" and arguments.trees < 2:
        parser.error("--trees has to be at least 2 for the hardcoded load balancing")
    if arguments.checkpoint_interval < 0.0 or arguments.plot_interval < 0.0:
        parser.error("--checkpoint-interval and --plot-interval must not be negative")
    return arguments


def create_numerical_scheme(solver, arguments):
    """! Assign numerical scheme, grid construction and checkpoint plotter to solver"""
    if arguments.scheme == "fv":
        package = exahype2.api.fv
        solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(
            solver,
            arguments.patch_size,
            min_volume_size=min_volume_size(arguments),
            max_volume_size=arguments.volume_size,
        )
    else:
        package = exahype2.api.rkfd4
        Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
        options = {}
        if arguments.lopsided:
            options = dict(
                right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
                advection_velocity=exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY,
                advected_quantities=solver.lopsided_advection_quantities(),
            )
        if arguments.ko_sigma > 0.0:
            solver.numerical_scheme = exagrype.api.FD4WithKODissipation(
                solver,
                arguments.patch_size,
                rk_order=arguments.rk_order,
                kreiss_oliger_sigma=arguments.ko_sigma,
                quantities=arguments.ko_quantities,
                **options,
            )
        else:
            solver.numerical_scheme = (
                exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
                    solver,
                    arguments.patch_size,
                    rk_order=arguments.rk_order,
                    min_volume_size=min_volume_size(arguments),
                    max_volume_size=arguments.volume_size,
                    **options,
                )
            )

    if arguments.scheme == "fv":
        solver.grid_construction = package.SetupSimulationWithUserCodedRefinementCriterion(
            solver,
            volume_size=arguments.volume_size,
            min_volume_size=min_volume_size(arguments),
        )
    else:
        # fd4 has no refinement in the setup: It starts on the regular mesh
        # and the time stepping refines it
        solver.grid_construction = package.SetupSimulationOverRegularGrid(
            solver, volume_size=arguments.volume_size
        )
    if arguments.checkpoint_interval > 0.0:
        solver.add_plotter(
            package.WriteCheckpoint(
                solver,
                first_plot_time_stamp=arguments.first_checkpoint,
                time_in_between_plots=arguments.checkpoint_interval,
                output_path=arguments.output_directory,
            )
        )
    if arguments.plot_interval > 0.0:
        solver.add_plotter(
            package.PlotAsVTKFile(
                solver,
                first_plot_time_stamp=arguments.first_checkpoint,
                time_in_between_plots=arguments.plot_interval,
                plotter_precision=arguments.checkpoint_precision,
                output_path=arguments.output_directory,
            )
        )


def actual_volume_size(arguments):
    """! Size of a volume/grid cell the regular 3-adic mesh yields"""
    cell_size = arguments.domain_size
    while cell_size >= arguments.volume_size * arguments.patch_size:
        cell_size /= 3.0
    return cell_size / arguments.patch_size


def min_volume_size(arguments):
    """! Volumes of this size or bigger may still be refined.

    The regular mesh of --volume-size is the coarsest mesh the benchmark
    allows (README.dox), so the AMR levels sit on top of its actual volume
    size. Refining a cell yields volumes strictly smaller than the bound, so
    the bound is the volume size of the second-finest level, slightly
    reduced to be robust against rounding. Without AMR levels, the bound
    would exceed the --volume-size. It is capped there, which is still above the
    volume size of the regular mesh, i.e. the mesh stays regular.
    """
    return min(
        actual_volume_size(arguments) / 3.0 ** (arguments.amr_levels - 1) * 0.99,
        arguments.volume_size,
    )


def regular_mesh_levels(arguments):
    """! Number of refinement levels of the regular 3-adic mesh (see actual_volume_size())"""
    levels = 0
    cell_size = arguments.domain_size
    while cell_size >= arguments.volume_size * arguments.patch_size:
        cell_size /= 3.0
        levels += 1
    return levels


def create_load_balancing(arguments):
    """! Name and constructor arguments of the load balancing strategy

    The hardcoded strategy splits the regular mesh into arguments.trees trees
    of the same number of cells, i.e. the outcome SpreadOutOnceGridStagnates
    aims for. The splits are computed offline on a synthetic regular tree
    along the space-filling curve, and all trees live on rank 0. AMR levels
    do not enter, as the splits fire once the mesh has reached the regular
    mesh and are bottom-up.
    """
    if arguments.load_balancing == "spread-out":
        return (
            "toolbox::loadbalancing::strategies::SpreadOutOnceGridStagnates",
            "new ::exahype2::LoadBalancingConfiguration()",
        )

    tree = RegularGridGenerator.RegularGridGenerator(
        max_level=regular_mesh_levels(arguments), dimensions=3
    )
    if arguments.decomposition == "quantised":
        decomposition = Decomposition.quantised_splitting(
            peano_tree=tree, number_of_subtrees=arguments.trees, dimension=3
        )
    else:
        decomposition = Decomposition.equidistant_splitting(
            peano_tree=tree, number_of_subtrees=arguments.trees
        )
    SplitTrees.split_tree(
        tree, decomposition=decomposition, periodic_BC=[False, False, False]
    )
    destination_ranks = Hardcoded.assign_trees_to_ranks_round_Robin(
        number_of_subtrees=tree.number_of_subtrees, max_ranks=1
    )
    configuration, _ = Hardcoded.create_constructor_initialisation_string(
        tree, destination_ranks, modes=SplitModes[arguments.split_mode]
    )
    return "toolbox::loadbalancing::strategies::Hardcoded", configuration


def main():
    arguments = parse_arguments()

    project = exahype2.api.Project(
        namespace=["benchmarks", "exagrype", "singleblackhole"],
        project_name="single-black-hole",
        directory=arguments.output_directory,
        executable="single-black-hole",
    )

    solver = exagrype.api.FOCCZ4Solver(
        "CCZ4",
        parameters={"CCZ4b": 1, "CCZ4s": 1},
    )
    solver.add_pde_term(
        exagrype.api.initialvalues.single_black_hole(mass=arguments.mass)
    )
    solver.enforce_ccz4_constraint(EnforceModes[arguments.enforce_constraints])
    solver.add_pde_term(
        exahype2.api.SommerfeldBoundaryConditions(
            implementation="::exagrype::initialvalues::flat(Q, x, 0.0);"
        )
    )
    # The criterion never asks for an erase. The mesh is a function of the
    # position only, i.e. it is static, and it never coarsens below the
    # regular mesh anyway. Without AMR levels, the volume size bound keeps
    # the mesh regular.
    solver.add_pde_term(
        exahype2.api.RefineErasePointwisely(
            implementation=f"""
  const double r2 = tarch::la::norm2Squared(x);
  if (r2 < {arguments.amr_radius ** 2}) *refinementCommand = ::exahype2::RefinementCommand::Refine;
"""
        )
    )
    create_numerical_scheme(solver, arguments)
    project.add_solver(solver)

    project.compiler = Compilers[arguments.compiler]

    half_size = arguments.domain_size / 2.0
    project.set_global_simulation_parameters(
        dimensions=3,
        offset=[-half_size] * 3,
        size=[arguments.domain_size] * 3,
        min_end_time=arguments.end_time,
        max_end_time=arguments.end_time,
        periodic_BC=[False, False, False],
    )

    project.set_load_balancing(*create_load_balancing(arguments))

    project.set_Peano4_installation(
        arguments.peano_directory, Variants[arguments.variant]["mode"]
    )
    peano4_project = project.generate_Peano4_project(
        verbosity=Verbosities[arguments.verbosity]
    )
    peano4_project.generate()

    h = actual_volume_size(arguments)
    print(
        f"Volume size {h:.6g} ({round(arguments.domain_size / h)} volumes per axis), "
        f"scheme {arguments.scheme}, {arguments.amr_levels} AMR levels, "
        f"variant {arguments.variant}, compiler {arguments.compiler}, "
        f"end time {arguments.end_time}"
    )

    if not arguments.no_build:
        peano4_project.build(make_clean_first=not arguments.no_clean)


if __name__ == "__main__":
    main()
