# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
r"""!

@page benchmarks_exahype2_euler_point_source Point source (Euler equations)

\tableofcontents

A gas at rest (density one, pressure one) over the unit square is heated by an energy
source at the origin. The source is a narrow Gaussian that is injected through the
volumetric source term (exahype2.api.VolumetricSource), as no numerical scheme
evaluates exahype2.api.PointSource yet. A blast wave runs out of the origin. There is no
exact solution, so the benchmark is a test of the mesh handling and of the AMR, not of the
accuracy of the scheme.

## Mesh and memory

Unless --volume-size is given, the script picks the finest regular start grid that fits
into the available main memory (--memory-fraction of MemAvailable). A regular grid has
p*3^k volumes per axis (p volumes per patch, 3^k patches). As p is free between 6 and 32,
the script fills the memory much more closely than 3^d steps in k would allow. The
footprint of the Finite Volume scheme without AMR was measured as follows (2d, release
build, peak resident set size):

| volumes   | resident set size | per volume |
|-----------|-------------------|------------|
| 59 049    | 22 MB             |  375 B     |
| 531 441   | 102 MB            |  191 B     |
| 4 782 969 | 683 MB            |  143 B     |

The offset of about 12 MB dominates for small meshes; the slope is 140 B per volume. The
script assumes 160 B in 2d. For 3d, no measurement is available yet, and the script assumes
240 B.

## AMR

--amr-levels l:

- l=0: the whole simulation runs on the regular start grid, i.e. the smallest mesh size is
  --volume-size everywhere.
- l>0: the simulation starts from the very same regular grid, but the gradient-based
  refinement criterion (exahype2.api.GradientRefinement) is on throughout. As the start grid
  is the finest permitted grid, the mesh first coarsens wherever the solution is smooth
  and refines around the blast wave afterwards. Cells may become l levels (factor 3 each)
  coarser than the start grid. The criterion looks at the one-sided gradients of density
  plus energy: it asks for a refinement above --refine-threshold, and for an erase below
  --erase-threshold.

Result of the runs below (2d, one 8-core workstation, --volume-size 0.001373 --patch-size 9
--end-time 0.1, i.e. 729^2 volumes, 6561 mesh cells of 9x9 volumes, 343 time steps).

### Volumes coarsened and refined over time

A patch hosts 9x9=81 volumes; the table counts volumes of the leaf cells of the mesh.
"Coarsened" is the number of volumes that the mesh loses in the interval, "refined" the
number that it gains. The mesh changes in a few mesh adaptation phases only.

| time interval | l=1 cells | l=1 coarsened | l=1 refined | l=2 cells | l=2 coarsened | l=2 refined |
|---------------|-----------|---------------|-------------|-----------|---------------|-------------|
| (0, 0.01]     | 6561 -> 1937 | 374 544    | 0           | 6561 -> 1905 | 377 136    | 0           |
| (0.01, 0.02]  | 1937 -> 2385 | 0          | 36 288      | 1905 -> 2353 | 0          | 36 288      |
| (0.02, 0.06]  | 2385 -> 2385 | 0          | 0           | 2353 -> 2353 | 0          | 0           |
| (0.06, 0.08]  | 2385 -> 2801 | 0          | 33 696      | 2353 -> 2801 | 0          | 36 288      |
| (0.08, 0.10]  | 2801 -> 2801 | 0          | 0           | 2801 -> 2801 | 0          | 0           |

l=0 keeps 6561 cells (531 441 volumes) throughout. l=1 ends with 43% of the cells. The
mesh sizes 0.0123 (start grid) and 0.037 (one level coarser) occur; l=2 allows 0.111 as
well, but only four such cells exist at t=0.02, and none later. The second level of
coarsening is hence not exploited yet, and l=2 yields the same final mesh and solution as l=1.

### Quality

Difference of the final solution (t=0.1) between the AMR runs and l=0, evaluated on the
start grid (a coarse patch is injected piecewise constant into the fine volumes):

| quantity | range of the l=0 solution | maximum difference | root mean square difference |
|----------|---------------------------|--------------------|-----------------------------|
| density  | 0.162 .. 1.114            | 1.4e-3             | 2.6e-5                      |
| energy   | 2.500 .. 2.920            | 5.0e-3             | 9.2e-5                      |

AMR hence changes the density by at most 0.15% and the energy by at most 1.2% of their ranges with 43% of the cells. amr-study.py runs this comparison (tolerance 2%) and checks that the mesh has coarsened.

## Run

Configure and build Peano with --enable-exahype --enable-blockstructured
--enable-loadbalancing, set PYTHONPATH to the repository's src directory, and put the Intel
compiler on the path. Then

~~~~~~~~~~~~~~~~~~~~~~~
python3 point-source.py --amr-levels 1 --verbosity text
~~~~~~~~~~~~~~~~~~~~~~~

generates, builds and compiles the code in the current directory; run ./point-source
afterwards. The plots are written as Peano patch files. The default end time is short
(0.01), as the memory-filling mesh takes minutes per time step.

The script uses peano4.api.taskgraph.compiler.FuseSteps rather than the producer-consumer
task compiler: On meshes of this size, the latter deadlocks in
tarch::multicore::waitForTasks(), which waits for tasks while it holds the read lock of
the dependency tracking.

"""
import argparse
import os

import peano4.api
import exahype2.api


parser = argparse.ArgumentParser(
    description="Euler equations with a point source (narrow Gaussian volumetric source)"
)
parser.add_argument(
    "--output-directory",
    default=".",
    help="Directory the code is generated and built in (default: .)",
)
parser.add_argument(
    "--amr-levels",
    type=int,
    default=0,
    help="Number of adaptive mesh refinement levels (default: 0). With 0, the "
    "simulation runs on the regular grid of the smallest volume size. With "
    "l>0, it starts from the same regular grid, but the gradient-based "
    "refinement criterion is on and coarsens the mesh by up to l levels (a "
    "factor of 3 each)",
)
parser.add_argument(
    "--volume-size",
    type=float,
    default=None,
    help="Smallest volume size, i.e. the size on the regular start grid. By "
    "default, the script picks the finest regular grid that fits into the "
    "main memory (see --memory-fraction)",
)
parser.add_argument(
    "--memory-fraction",
    type=float,
    default=0.8,
    help="Fraction of the available main memory that the start grid may "
    "occupy if --volume-size is not given (default: 0.8)",
)
parser.add_argument(
    "--patch-size",
    type=int,
    default=None,
    help="Volumes per mesh cell and axis. By default, the script picks the "
    "patch size such that the regular start grid fills the memory best",
)
parser.add_argument(
    "--refine-threshold",
    type=float,
    default=2.0,
    help="Mesh cells ask for refinement if the one-sided gradient of density "
    "plus energy exceeds this value (default: 2)",
)
parser.add_argument(
    "--erase-threshold",
    type=float,
    default=0.2,
    help="Mesh cells ask for coarsening if the one-sided gradient of density "
    "plus energy falls below this value (default: 0.2)",
)
parser.add_argument(
    "--end-time", type=float, default=0.01, help="Simulated time (default: 0.01)"
)
parser.add_argument(
    "--plots", type=int, default=20, help="Number of snapshots (default: 20)"
)
parser.add_argument(
    "--source-width",
    type=float,
    default=0.02,
    help="Standard deviation of the Gaussian that models the point (default: 0.02)",
)
parser.add_argument(
    "--source-strength",
    type=float,
    default=100.0,
    help="Energy per time and volume at the origin (default: 100)",
)
parser.add_argument(
    "--dimensions", type=int, default=2, choices=[2, 3], help="Dimension (default: 2)"
)
parser.add_argument(
    "--optimistic",
    action="store_true",
    help="Use the optimistic Finite Volume scheme",
)
parser.add_argument(
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
    default="visualisation",
    choices=["visualisation", "text"],
    help="Show the task graphs in a window or print them as text (default: visualisation)",
)
args = parser.parse_args()


## Estimated memory footprint of one volume in bytes (per dimension count).
#
# Measured peak resident set size of the 2d run without AMR: 22 MB for 59k
# volumes, 102 MB for 0.53M volumes, 683 MB for 4.8M volumes, i.e. about
# 140 bytes per volume (patch data with halo, face data, tree), plus a margin.
# The 3d value is an estimate (five instead of four unknowns, larger halos).
BYTES_PER_VOLUME = {2: 160.0, 3: 240.0}


def available_memory_in_bytes() -> float:
    """!
    Main memory that is available right now (MemAvailable in /proc/meminfo)
    """
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) * 1024.0
    raise RuntimeError("cannot read MemAvailable from /proc/meminfo")


def pick_regular_grid(dimensions: int, budget_in_bytes: float, patch_sizes):
    """!
    Pick the finest regular start grid that fits into the budget

    A regular grid over the unit cube consists of 3^k cells per axis, each
    hosting a patch of p volumes per axis. This yields p*3^k volumes per
    axis. The patch size p is free within a range, which makes the grid
    granularity much finer than the factor 3^d between subsequent k.

    @return (patch_size, k)
    """
    best = None
    for patch_size in patch_sizes:
        k = 0
        while (patch_size * 3 ** (k + 1)) ** dimensions * BYTES_PER_VOLUME[
            dimensions
        ] <= budget_in_bytes:
            k += 1
        volumes = (patch_size * 3**k) ** dimensions
        if volumes * BYTES_PER_VOLUME[dimensions] <= budget_in_bytes and (
            best is None or volumes > best[0]
        ):
            best = (volumes, patch_size, k)
    return best[1], best[2]


if args.volume_size is None:
    patch_size, k = pick_regular_grid(
        args.dimensions,
        args.memory_fraction * available_memory_in_bytes(),
        [args.patch_size] if args.patch_size is not None else range(6, 33),
    )
    # The setup refines every cell whose size is not smaller than
    # volume_size * patch_size. The factor makes the cell 3^-k the final one.
    volume_size = 1.001 / (patch_size * 3**k)
else:
    patch_size = args.patch_size if args.patch_size is not None else 9
    volume_size = args.volume_size
print(
    f"Start grid: {patch_size} volumes per cell and axis, volume size {volume_size}, "
    f"AMR levels {args.amr_levels}"
)


project = exahype2.api.Project(
    namespace=["benchmarks", "exahype2", "euler", "point_source"],
    project_name="point-source",
    directory=args.output_directory,
    executable="point-source",
)

solver = exahype2.api.Solver(
    "EulerPointSource", {"rho": 1, "j": args.dimensions, "E": 1}
)
solver.add_user_solver_includes(
    """
#include <algorithm>
#include <cmath>
"""
)

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

# The point is smeared over a few grid cells, as a delta distribution cannot be
# represented by a Finite Volume scheme. Only the energy equation is fed.
solver.add_pde_term(
    exahype2.api.VolumetricSource(
        implementation=f"""
      constexpr double sigma    = {args.source_width};
      constexpr double strength = {args.source_strength};
      for (int i=0; i<NumberOfUnknowns; i++) {{
        S[i] = 0.0;
      }}
      S[Dimensions+1] = strength * std::exp(-0.5 * tarch::la::norm2Squared(x) / (sigma * sigma));
"""
    )
)

# Ambient state: gas at rest with density 1 and pressure 1
AMBIENT_STATE = """
      constexpr double gamma = 1.4;
      Q[0] = 1.0;
      for (int d=0; d<Dimensions; d++) {
        Q[1+d] = 0.0;
      }
      Q[Dimensions+1] = 1.0 / (gamma - 1.0);
"""
solver.add_pde_term(exahype2.api.InitPointWisely(implementation=AMBIENT_STATE))
solver.add_pde_term(
    exahype2.api.DirichletBoundaryConditions(implementation=AMBIENT_STATE)
)

# With AMR, the simulation starts from the finest regular grid, and the
# gradient criterion is always on. It may only coarsen: the start grid
# is the finest grid that we allow, and the coarsest volumes are those
# amr_levels levels (factor 3 each) above the start volumes.
if args.amr_levels > 0:
    solver.add_pde_term(
        exahype2.api.GradientRefinement(
            implementation=f"""
      const double indicator = std::abs(gradient[0]) + std::abs(gradient[Dimensions+1]);
      if (indicator > {args.refine_threshold}) {{
        *refinementCommand = ::exahype2::RefinementCommand::Refine;
      }}
      else if (indicator < {args.erase_threshold}) {{
        *refinementCommand = ::exahype2::RefinementCommand::Erase;
      }}
"""
        )
    )
amr_bounds = dict(
    min_volume_size=volume_size,
    max_volume_size=volume_size * 3**args.amr_levels,
)
if args.optimistic:
    solver.numerical_scheme = exahype2.api.fv.RusanovWithOptimisticExplicitEuler(
        solver, patch_size, **amr_bounds
    )
else:
    solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(
        solver, patch_size, **amr_bounds
    )
solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
    solver, volume_size=volume_size
)
solver.plotter = exahype2.api.fv.PlotAsVTKFile(
    solver,
    first_plot_time_stamp=0.0,
    time_in_between_plots=args.end_time / args.plots,
)
project.add_solver(solver)

project.set_global_simulation_parameters(
    dimensions=args.dimensions,
    offset=[-0.5] * args.dimensions,
    size=[1.0] * args.dimensions,
    min_end_time=args.end_time,
    max_end_time=args.end_time,
    periodic_BC=[False] * 3,
)
project.set_load_balancing(
    "toolbox::loadbalancing::strategies::SpreadOutOnceGridStagnates",
    "new ::exahype2::LoadBalancingConfiguration()",
)
# No spawned background tasks: On meshes that fill the main memory, the
# producer-consumer compiler deadlocks in tarch::multicore::waitForTasks(),
# which waits for tasks while holding the read lock of the dependency tracking.
project.compiler = peano4.api.taskgraph.compiler.FuseSteps.translate

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
peano4_project.generate()
peano4_project.build(make_clean_first=True)
