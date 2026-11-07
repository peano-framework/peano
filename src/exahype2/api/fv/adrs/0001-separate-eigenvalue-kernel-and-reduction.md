# ADR 0001: Compute, store and reduce the maximum eigenvalue in separate tasks

Status: accepted (implementation in progress)
Date: 2026-11-07

## Context

The Finite Volume kernels (`timeStepWithRusanov2D/3D`, and the optimistic
interior/shell variants) returned the maximum eigenvalue of the patch as a
by-product of the update. `UpdateCell` stored it per cell, and
`ComputeGlobalCFLCondition` reduced it over all cells in a mesh traversal of its
own and derived the next time step size in `unprepareTraversal()`.
RKFD4 had to follow a different route, as its Runge-Kutta stages do not yield an
eigenvalue as a by-product.

## Decision

FV and RKFD4 follow the same compute-store-in-cell / reduce pattern, with three tasks:

1. **Compute max eigenvalue** (CELL task): a kernel of its own,
   `<Solver>_maxEigenvalueOverPatch`, evaluates the updated patch (without halo)
   and stores the result per cell (`_max_eigenvalue`).
   The update kernel does not return any eigenvalue anymore.
2. **Reduce max eigenvalue** (CELL task): hands the per-cell value to
   the solver's `reduceMaxEigenvalue()` (maximum over cells and threads, hosted in the
   abstract solver, see `MaxEigenvalueReduction.py`).
3. **Compute global CFL condition** (`GLOBAL_END_OF_STEP` task): acts in
   `unprepareTraversal()` only, i.e. reads and resets the reduced value, does the MPI
   reduction and feeds the new time step size into the time step bookkeeping.

For RKFD4, the eigenvalue task follows the final linear combination. For FV, it follows the
cell update. The task graph compiler fuses the steps into one traversal
(`peano4.api.taskgraph.compiler.FuseSteps`), so the split costs no extra sweeps.

### Optimistic variant

`RusanovWithOptimisticExplicitEuler` gets the same task pair directly after
the update (traditional branch) or the insertion of the patch interior
(optimistic branch). It evaluates the whole patch. The interior/shell kernels do not
return eigenvalues either. The pair runs in parallel to the task that
spawns the next time step's interior update.

## Consequences

- Eigenvalue evaluation is one kernel call per patch on the updated solution, i.e.
  numerically the same value as before (which was evaluated on the output).
- The eigenvalue kernel always runs on the host, also if the update runs on an
  accelerator.
- Kernels exist in both DSL backends (`dslhype`, `dslhype2`), generated into the same
  `kernels/<Solver>_rusanov{2,3}d` files as the time step kernel.
- `ComputeGlobalCFLCondition` loses its guard and static state;
  `UpdateCell`/`UpdateCellGPU` lose the `max_eigenvalue` argument.
- The optimistic cell data no longer needs the eigenvalue entries for shell
  and interior.
- The dynamic mesh refinement task depends on the reduction task rather than on the
  (global) CFL task, as nothing may depend on a `GLOBAL_END_OF_STEP` task.
