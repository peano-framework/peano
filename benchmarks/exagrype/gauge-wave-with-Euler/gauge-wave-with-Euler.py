"""!

Gauge wave benchmark for the ExaGRyPE Python API with an additional Euler fluid

The gauge wave is a pure gauge perturbation of flat spacetime, i.e. an
analytical solution of the Einstein equations that travels along the x-axis
with speed one. It is periodic over the unit cube. This script evolves it with
the first-order CCZ4 solver of exagrype.api (exagrype.api.FOCCZ4Solver),
discretised with fourth-order Finite Differences and classic Runge-Kutta 4
(exahype2.api.rkfd4).

Next to CCZ4, the compound hosts a compressible Euler solver
(exagrype.api.euler.Euler), discretised with Finite Volumes and explicit Euler time
stepping (exahype2.api.fv.RusanovWithExplicitEuler). Both solvers are configured
right here, and both run on the same mesh with periodic boundary
conditions. They are hosted by exagrype.api.CCZ4WithEuler_AdditiveCoupling,
which copies the fields of each solver into auxiliary variables of the other
one at the begin of each time step (piecewise linear interpolation between the
two resolutions), and which synchronises the time step sizes at the end.

Each mesh cell hosts a patch of p x p x p grid cells for CCZ4 and a patch of
(2p+1) x (2p+1) x (2p+1) volumes for Euler. With p=8, Euler has 17 x 17 x 17
volumes per cell. The Euler volumes are hence a bit smaller than half the
size of the CCZ4 grid cells.

The script requires a Peano build with

    --enable-exahype --enable-blockstructured --enable-loadbalancing --enable-exagrype

and is run from this directory via

    python3 gauge-wave-with-Euler.py --mode release
    python3 gauge-wave-with-Euler.py --mode debug --cell-size 0.1111 --patch-size 8

"""

import argparse
import os

import peano4.api
import exahype2.api
import exagrype.api


parser = argparse.ArgumentParser(description="ExaGRyPE gauge wave with Euler")
parser.add_argument(
    "--mode",
    choices=["release", "debug"],
    required=True,
    help="Build mode of the generated code. Release uses a longer simulation "
    "and text logging; debug a very short one with visualised logging.",
)
parser.add_argument(
    "--cell-size",
    type=float,
    default=1.0 / 3.0,
    help="Edge length of one mesh cell (default: 1/3). The mesh is regular and "
    "refined by factors of three from the root cell, i.e. the value is rounded "
    "to 3^-k (k>=1).",
)
parser.add_argument(
    "--patch-size",
    type=int,
    default=8,
    help="Number of CCZ4 grid cells per mesh cell and axis (default: 8). "
    "Euler hosts 2*patch-size+1 volumes per axis.",
)
args = parser.parse_args()

if args.patch_size < 1:
    raise ValueError("The patch size has to be positive")
if args.cell_size <= 0.0 or args.cell_size >= 1.0:
    raise ValueError("The cell size has to be in (0,1)")

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

ccz4_patch_size = args.patch_size
euler_patch_size = 2 * args.patch_size + 1

## The grid construction refines every cell whose edge length is greater than
## or equal to volume_size times the patch size, i.e. the grid cells end up
## strictly smaller than volume_size. We pass in values slightly bigger than the
## intended sizes. Both solvers have to end up with the same mesh cells.
ccz4_volume_size = 1.01 * args.cell_size / ccz4_patch_size
euler_volume_size = 1.01 * args.cell_size / euler_patch_size


project = exahype2.api.Project(
    namespace=["benchmarks", "exagrype", "gaugewavewitheuler"],
    project_name="gauge-wave-with-Euler",
    directory=".",
    executable="gauge-wave-with-Euler",
)

## The compound hosts CCZ4 and Euler. Only the compound is added to the mesh.
## It couples the solvers at the begin and the end of each time step, i.e. each
## solver brings the numerical scheme of its choice.
coupled = exagrype.api.CCZ4WithEuler_AdditiveCoupling()
ccz4 = coupled.ccz4
euler = coupled.euler

## CCZ4: fourth-order Finite Differences with RK4
Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
coupled.set_numerical_scheme(
    ccz4,
    exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        ccz4,
        ccz4_patch_size,
        rk_order=4,
        right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
        advection_velocity=exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY,
        advected_quantities=ccz4.lopsided_advection_quantities(),
    ),
)
ccz4.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
    ccz4, volume_size=ccz4_volume_size
)
ccz4.plotter = exahype2.api.rkfd4.PlotAsVTKFile(
    ccz4, first_plot_time_stamp=0.0, time_in_between_plots=time_in_between_plots
)

## Euler: Finite Volumes with a Rusanov flux and explicit Euler time stepping
coupled.set_numerical_scheme(
    euler, exahype2.api.fv.RusanovWithExplicitEuler(euler, euler_patch_size)
)
euler.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
    euler, volume_size=euler_volume_size
)
euler.plotter = exahype2.api.fv.PlotAsVTKFile(
    euler, first_plot_time_stamp=0.0, time_in_between_plots=time_in_between_plots
)

project.add_solver(coupled)

project.compiler = peano4.api.taskgraph.compiler.FuseSteps.translate

project.set_global_simulation_parameters(
    dimensions=3,
    offset=[0.0, 0.0, 0.0],
    size=[1.0, 1.0, 1.0],
    min_end_time=end_time,
    max_end_time=end_time,
    periodic_BC=[True, True, True],
)

PeanoRoot = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
project.set_Peano4_installation(PeanoRoot, build_mode)
peano4_project = project.generate_Peano4_project(verbosity=verbosity)
peano4_project.generate()
peano4_project.build(make_clean_first=True)
