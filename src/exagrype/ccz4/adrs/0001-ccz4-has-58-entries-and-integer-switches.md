# ADR 0001: CCZ4 has 58 entries, and its switches are integers

Status: proposed
Date: 2026-11-13

## Context

The kernels in `exagrype::ccz4` worked on 59 entries: 25 primary and 34 secondary ones.
The last secondary entry was `K0`, an offset of the trace of the extrinsic curvature in
the slicing condition, `dt alpha = -alpha^2 g(alpha) (K - K0 - 2 c Theta)`. `K0` is no
unknown of CCZ4. No initial condition of ExaGRyPE sets it, and its right-hand side is
zero, i.e. it was a field of zeros that every solver stored, exchanged over the faces
and plotted.

The kernels furthermore took the parameters `CCZ4ds` and `CCZ4xi` (double), which every
setup left at one, and `CCZ4bs` and `CCZ4sk` (double), which are switches: the code only
makes sense for zero and one.

The unknown that holds the trace of the extrinsic curvature was called `traceK`,
whereas the kernels call this quantity `Kex`.

## Decision

- CCZ4 has 58 entries, 25 primary and 33 secondary ones
  (`exagrype.api.CCZ4SolverBase.NUMBER_OF_CCZ4_ENTRIES`). `K0` is gone, and the slicing
  condition reads `K - 2 c Theta`.
- `CCZ4ds` and `CCZ4xi` are no parameters anymore. The formulas use one.
- `CCZ4b` (the former `CCZ4bs`) and `CCZ4s` (the former `CCZ4sk`) are `int`, like
  `CCZ4LapseType`. The Python solvers reject any value but the integers 0 and 1
  (`exagrype.api.CCZ4SolverBase.Switches`).
- The unknown `traceK` is called `Kex`.
- There is no switch between the first-order and the second-order formulation. They
  are different kernels (`ncpFirstOrder()`, `ncpSecondOrder()`, ...) and different solver
  classes.

## Consequences

- Snapshots and checkpoints with 59 entries per degree of freedom cannot be read by the
  new solvers, and scripts which address `traceK`, `K0`, `CCZ4ds`, `CCZ4xi`, `CCZ4sk` or
  `CCZ4bs` have to be adapted. The solvers raise a `KeyError` for the removed and the
  renamed parameters.
- Whoever needs a slicing condition with an offset reintroduces it as a parameter
  of the kernels, not as an entry of the solution.
- `applications/exahype2/ccz4` and `benchmarks/exagrype/kernel-benchmarks` still call the
  former kernels `ncp()` and `source()` with 59 entries. These kernels had been replaced
  before this change already, i.e. both were broken, and they have not been touched. The
  same holds for the tutorial pages which describe this legacy script (`advanced.dox`,
  `visualize.dox` and `binary-black-hole.dox` in `tutorials/exahype2/exagrype`): They keep
  the 59 entries and the former order of the entries on purpose.
- The postprocessing routines (`admconstraints()`, `Psi4Calc()`, ...) use `traceK` for the
  trace and `Kex[3][3]` for the full extrinsic curvature as local names. This clashes
  with the name of the unknown and is left for a later cleanup.

## How the change was validated

The physics must not change, as `K0` was zero and the removed parameters were one.
This was checked bit-wise rather than by looking at errors:

- A harness compiles the former `Kernels.cpp` into a namespace of its own, next to the new
  one, and compares `ncp`, `source` (first and second order) and `maxEigenvalue` on random
  states for all combinations of the switches: 5.36 million values are identical.
- The gauge wave with 27 volumes per axis (`fv` and `rk4fd4`) yields snapshots which are
  identical to the ones of the former code in all 58 entries.

A first attempt was made on a branch that was 163 commits behind, where the kernels
still had one `ncp()` with a parameter `CCZ4SO`. It was redone from scratch after the
merge, as the split into first-order and second-order kernels had replaced that parameter.
