"""

Gauge wave benchmark for the ExaGRyPE Python API, coupled to an Euler fluid

The gauge wave is a pure gauge perturbation of flat spacetime, i.e. an
analytical solution of the Einstein equations that travels along the x-axis
with speed one. It is periodic over the unit cube. This script evolves it with
the first-order CCZ4 solver of exagrype.api, discretised with fourth-order
Finite Differences and classic Runge-Kutta 4 (exahype2.api.rkfd4) with the
fused task graph compiler. No other solver and no other compiler are offered.

Besides the gauge wave, the script runs a compressible Euler fluid
(exagrype.api.Euler) on the same mesh: a density wave which the fluid advects
along the x-axis. The two solvers are coupled (@ref CCZ4WithEuler.py). All
ingredients, i.e. the two solvers, are part of exagrype.api. The assembly is
done here in the benchmark directory:

- The coupling is additive: Both solvers run in parallel within the same mesh
  sweeps.
- It is volumetric: The Euler solver holds a snapshot of the lapse of CCZ4 as
  an auxiliary variable, and scales its fluxes and its maximum eigenvalue
  with it. The snapshot is refreshed in front of each Runge-Kutta stage by a
  KickOff task, so the fluid never reads data that CCZ4 writes at the same
  time.
- A WrapUp task after each stage waits for all tasks that nothing else
  depends on. It calls the functor synchronise() of the generated abstract
  Euler solver class, which is empty unless you set it in the main routine.
- It is uni-directional: The fluid does not feed back into the Einstein
  equations, i.e. CCZ4 still evolves the vacuum gauge wave. The Euler
  equations are those of flat space with a lapse-scaled clock, not general
  relativistic hydrodynamics. This is a trial of the coupling machinery.

The script requires a Peano build with

    --enable-exahype --enable-blockstructured --enable-loadbalancing --enable-exagrype

and is run from this directory via

    python3 gauge-wave-with-Euler.py

"""

import peano4.api
import exahype2.api
import exagrype.api

from CCZ4WithEuler import CCZ4WithEuler


create_release_build = False

if create_release_build:
    build_mode = peano4.api.output.CompileMode.RELEASE
    cells = 81
    end_time = 1.0
    time_in_between_plots = 0.1
    verbosity = peano4.api.LoggerVerbosity.TEXT
else:
    build_mode = peano4.api.output.CompileMode.DEBUG
    cells = 27
    end_time = 0.02
    time_in_between_plots = 0.01
    verbosity = peano4.api.LoggerVerbosity.VISUALISATION

patch_size = 9
# The grid cells have to be strictly smaller than volume_size times the patch size
volume_size = 1.01 / cells


project = exahype2.api.Project(
    namespace=["benchmarks", "exagrype", "gaugewavewitheuler"],
    project_name="gauge-wave-with-Euler",
    directory=".",
    executable="gauge-wave-with-Euler",
)

coupled = CCZ4WithEuler()

## CCZ4 with lopsided advection of the gauge
ccz4 = coupled.ccz4
Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
ccz4.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
    ccz4,
    patch_size,
    rk_order=4,
    right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
    advection_velocity=exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY,
    advected_quantities=ccz4.lopsided_advection_quantities(),
)

## The Euler solver runs on the same mesh, with the same Runge-Kutta scheme
euler = coupled.euler
euler.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
    euler, patch_size, rk_order=4
)

for solver in [ccz4, euler]:
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        solver, volume_size=volume_size
    )
    solver.plotter = exahype2.api.rkfd4.PlotAsVTKFile(
        solver, first_plot_time_stamp=0.0, time_in_between_plots=time_in_between_plots
    )
coupled.add_to_project(project)

project.compiler = peano4.api.taskgraph.compiler.FuseSteps.translate

project.set_global_simulation_parameters(
    dimensions=3,
    offset=[0.0, 0.0, 0.0],
    size=[1.0, 1.0, 1.0],
    min_end_time=end_time,
    max_end_time=end_time,
    periodic_BC=[True, True, True],
)

project.set_Peano4_installation("../../../", build_mode)
peano4_project = project.generate_Peano4_project(verbosity=verbosity)
peano4_project.generate()
peano4_project.build(make_clean_first=True)
