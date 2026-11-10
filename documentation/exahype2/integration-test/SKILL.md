---
name: exahype2:integration-test
description: Re-run the ExaHyPE 2 / ExaGRyPE integration test, i.e. the convergence study of the gauge wave benchmark (benchmarks/exagrype/gauge-wave) over the schemes fv, rk1fd4, rk2fd4, rk3fd4 and rk4fd4 and over several mesh sizes, plus the Euler entropy wave convergence study of the fourth-order Finite Differences with adaptive time stepping (benchmarks/exahype2/euler/smooth-convergence-test-fd4), plus the AMR check of the Euler point source (benchmarks/exahype2/euler/point-source). Use when the user asks to run/re-run the integration test or the gauge wave convergence study, or to check that changes to the numerical schemes, task graph, kernels or code generation have not broken the numerical correctness. Runtimes are recorded, but they are not an acceptance criterion (that is the job of exahype2:performance-test).
allowed-tools: [Bash, Read]
---

# Integration test: convergence studies

The integration test consists of two studies: the gauge wave (ExaGRyPE, CCZ4) and the Euler
entropy wave (ExaHyPE 2, see the last section). Run both.

## Gauge wave convergence study

The gauge wave has a known exact solution. `benchmarks/exagrype/gauge-wave/convergence-study.py`
builds and runs `gauge-wave.py` for every scheme and mesh, measures the maximum error of the
lapse along the line y=z=0.5 at the end time against the exact solution, estimates the
convergence orders, writes a plot and a table, and exits non-zero if the order of any scheme
between its two finest meshes falls below `ExpectedOrders` in the script. The results of the
last reference run are documented in the page on top of `gauge-wave.py`.

## Prerequisites

- Peano is configured and built with `--enable-exahype --enable-blockstructured
  --enable-loadbalancing --enable-exagrype` (release libraries are enough).
- The generated code is compiled with the Intel compiler:
  `export PATH=/opt/intel/oneapi/compiler/2024.2/bin:$PATH`.
- `export PYTHONPATH=<repo>/src`.
- The machine is otherwise idle. The runs use all cores, and the coarse runs are packed next
  to each other; background load pollutes the recorded runtimes and the memory watchdog.

## Steps

1. Pick a work directory outside the repository (builds and outputs take a few GB) and run
   the study in the background (about an hour on an eight-core workstation):
   ```sh
   python3 benchmarks/exagrype/gauge-wave/convergence-study.py \
       --work-directory <work> --output-directory <work>/out
   ```
   Useful options: `--schemes fv rk4fd4` and `--meshes 9 27` restrict the study (the latter is
   a quick smoke test of a few minutes); `--cores N` limits the cores. The default meshes
   are `9 27 81`; `243` fits only into a machine with far more than 64 GB of memory.
   Completed runs in the work directory are reused (`results.json`), so an interrupted study
   continues where it stopped, and runs that were skipped for lack of memory are retried.
   Delete the run's entry in `results.json` to force a rerun (a rerun starts from a clean
   directory). Do not kill the study with `pkill -f` from the same shell that runs it.
2. Watch the log (`build ...`, `run ...`, `skip ...`) with the Monitor tool instead of
   polling. All schemes run at the same time, each walking its meshes from coarse to
   fine. The coarse runs use one thread; at most two large runs go at once.
3. The script prints the table and `PASSED` or `FAILED` plus the schemes whose order is too
   low. On `FAILED`, look at the table first: a single scheme dropping in order points to that
   scheme's kernels or task graph; all schemes dropping points to the common parts (CCZ4
   solver, grid, plotting, time step size).
4. Report the table and the verdict. If the user wants the documentation refreshed, copy
   `<work>/out/convergence.png` to `benchmarks/exagrype/gauge-wave/example-output/` and update
   the table in the page description on top of `gauge-wave.py`, then run the skill
   `peano:regen-docs`. Do not commit run directories or generated sources.

## What to expect

- Reference orders between the two finest meshes (h=1/27 to 1/81, end time 0.1, patch size 9),
  as documented in the page description of `gauge-wave.py`: fv 0.94, rk1fd4 1.02, rk2fd4 2.31,
  rk3fd4 4.00, rk4fd4 3.99. The thresholds in `ExpectedOrders` leave a margin
  (0.8, 0.8, 1.8, 3.2, 3.2). The Finite Difference schemes are limited by the smaller of
  their spatial order (four) and the order of their Runge-Kutta scheme. On coarse
  meshes the time step is so small that the time error is invisible: rk2fd4, rk3fd4
  and rk4fd4 produce nearly identical errors there (an order around 3.5 between h=1/9
  and 1/27 is pre-asymptotic and not a quantity to test).
- The error is measured at the last snapshot, whose time stamp overshoots the end time by
  less than one time step and differs a bit between meshes. This biases the orders by
  about 0.01.
- Memory is the limit. A run with 81 volumes per axis needs 0.8 GB (fv) to 3 GB
  (Runge-Kutta), and the footprint grows with the number of volumes: h=1/243 needs
  more than 16 GB and is not part of the study on a workstation (it was tried: all five
  schemes were killed). The study runs at most two large runs at once
  (`--concurrent-large-runs`). A watchdog kills the run with the biggest footprint if less
  than 1.5 GB remain available on the machine; a killed run is reported as skipped and
  retried on the next invocation. Close other programs if runs get skipped.
- Memory that grows with the simulated time (not just with the mesh) indicates data that
  piles up on Peano's stacks, i.e. an inconsistent access qualifier of a task's cell
  data (once found for a write-only first access to the Runge-Kutta estimates).
  Compare the peak memory of two runs with different end times.
- Compile errors in the generated code usually show up in `<work>/<scheme>-<volumes>/generate.log`;
  the output of the simulation is in `run.log` of the same directory.

## Performance

The script records runtime, thread count and peak memory per run. They are part of the table
but never decide on pass or fail. Performance regressions are to be checked with the
`exahype2:performance-test` skill.

## Euler entropy wave convergence study

`benchmarks/exahype2/euler/smooth-convergence-test-fd4/convergence-study.py` generates, builds
and runs the entropy wave of the Euler equations (2d, fourth-order Finite Differences with
fourth-order Runge-Kutta and adaptive time stepping) on 27, 81 and 243 grid cells per axis. It
compares the density of the last snapshot with the exact solution and exits non-zero if the order
of the maximum error between the two finest meshes is below 3.5. It needs the same prerequisites
as above (the Intel compiler on the `PATH`, `PYTHONPATH=<repo>/src`) and about ten minutes:

```sh
python3 benchmarks/exahype2/euler/smooth-convergence-test-fd4/convergence-study.py \
    --work-directory <work2>
```

Finished runs in the work directory are reused; delete `<work2>/run-<size>` to rerun one.
Expected: 64, 190 and 566 time steps, max errors 3.8e-05, 4.8e-07 and 5.9e-09, orders 3.99 and
4.00 (documented in `README.dox` next to the script), and `PASSED`.

If it fails, look at the log of the run: `run.log` in the run directory. Typical symptoms:

- `new dt=inf` after a few steps and `nan` errors: the solution turned NaN. Once found because the
  eigenvalue task, which the producer-consumer compiler splits, pushed the roll over of the
  faces into the next mesh sweep, away from the temporary face data of the projection (see
  `src/exahype2/api/rkfd4/adrs/0002-eigenvalue-follows-projection.md`). Check which traversal
  holds the project and the roll over task in the compiled task graph (`generate` output).
- The generated code does not compile: look at the compiler errors in the generator output.
  Task names become C++ identifiers.
- The same study with a fixed time step (`--time-step-size 0.0005`, one run) separates problems of
  the time step size computation from problems of the scheme.

## Euler point source: AMR check

`benchmarks/exahype2/euler/point-source/amr-study.py` generates, builds and runs the point source
benchmark (2d, Finite Volumes) twice: on the regular grid of 81x81 mesh cells with 9x9 volumes
each (`--amr-levels 0`), and from the same grid with the gradient-based refinement criterion
switched on (`--amr-levels 1`). It checks that (a) the dynamic AMR works, i.e. the mesh coarsens,
and (b) the quality does not suffer, i.e. the density and the energy of the last snapshot differ
from the regular-grid run by at most 2% of their ranges. It prints the table of coarsened and
refined volumes over time, and `PASSED` or `FAILED`; it exits non-zero on `FAILED`. It needs the
same prerequisites as above and about ten minutes:

```sh
python3 benchmarks/exahype2/euler/point-source/amr-study.py --work-directory <work3>
```

Finished runs in the work directory are reused; delete `<work3>/levels-<l>` to rerun one.
Expected (documented in the page on top of `point-source.py`): the mesh drops from 6561 to 1937
cells in the first time steps, grows to 2801 cells at the end (43% of the regular grid), 374 544
volumes are coarsened and 69 984 refined, the maximum differences are 1.4e-3 (density) and
5.0e-3 (energy).

If it fails:

- `new dt=inf` and NaN: see the symptoms of the entropy wave study. The point source run found that
  the roll over of the faces of the explicit Finite Volume scheme was a task of its own, which the
  producer-consumer compiler fused into a later traversal than the projection (ADR 0003 in
  `src/exahype2/api/fv/adrs`). The benchmark itself uses `FuseSteps` rather than the
  producer-consumer compiler.
- A run that makes no progress and burns CPU time on all threads: in `tarch::multicore::waitForTasks()`,
  the producer-consumer compiler deadlocked on large meshes (all threads wait for the read lock of the
  dependency tracking). Attach `gdb -p` to see it. Not an issue of the benchmark.
- "The mesh never coarsened": check the thresholds `--refine-threshold` and `--erase-threshold`
  of `point-source.py`, and that `min_volume_size` and `max_volume_size` of the scheme allow
  coarsening.
- `--amr-levels 2` currently yields the same mesh as `--amr-levels 1` (only four cells reach the
  second level and are refined again), i.e. it is not a test of multi-level coarsening.
