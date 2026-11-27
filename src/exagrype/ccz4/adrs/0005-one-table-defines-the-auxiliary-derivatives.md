# ADR 0005: One table defines the auxiliary derivatives of the second-order formulation

Status: proposed
Date: 2026-11-26

## Context

The second-order formulation (`exagrype.api.FOSOCCZ4Solver`) evolves 25 unknowns and
holds the other 33 entries of CCZ4 as auxiliary variables. Each of them is a first
derivative of an unknown along one axis, possibly with a factor: `dLapse[k]` is the
derivative of the lapse, `dxShift` the one of the shift, `dxG` holds
`D_xij = 1/2 d_x G_ij` and `P[k]` is the derivative of phi.

Which entry differentiates which unknown was spelled out four times in C++, in the loop
bodies of `recomputeAuxiliaryVariablesFD4_4thOrder()`, `..._centralDifferences()`,
`..._leftDifferences()` and `..._rightDifferences()`, as entry numbers
(`25+normal`, `28+3*normal+i`, `37+6*normal+k`, `55+normal`). Only the fourth-order version
applied the factor one half to `D`. The second-order central differences, which the Finite
Volume variant uses, computed `D = d G`, i.e. twice the derivative CCZ4 expects. The
C++ test of that routine checked the derivatives of the lapse and phi only.

The compact second derivatives in the Finite Difference kernel (next change) need the
same relation once more: to replace the derivative of `dxG` along x, the kernel has to
know that `dxG` is one half times the derivative of `G` along x.

## Decision

- `FOSOCCZ4Solver.auxiliary_derivatives()` is the only definition of the derivatives: a
  table of (derivative, unknown, axis, factor), by the names of the variables.
  `FOSOCCZ4Solver.derivative_table()` expands it per entry and rejects tables which name
  unknown variables, pair variables of different sizes or do not cover each derivative
  exactly once.
- `exagrype::ccz4::reconstructFirstDerivativesFD4()` and
  `reconstructFirstDerivativesCentralDifferences()` take the table as arguments and know
  nothing about CCZ4, as `exahype2::fd::addCentralAdvection()` takes the indices of the
  advected quantities (ADR 0003). `FOSOReconstructionSolver` generates the call with the
  table. The routines compute the derivatives directly from the 25 unknowns with halo into
  the 33 derivatives, without a buffer of 58 entries.
- The four `recomputeAuxiliaryVariablesFD4_...()` routines and their loop bodies are
  removed. Nothing but the two wrappers called them.
- Every further use of the derivatives, such as the compact second derivatives, reads the
  same table.

## Consequences

- A change of the derivatives (another factor, another entry) is one line in Python, and
  the reconstruction and the kernels follow.
- The Finite Differences do not change: the fourth-order weights are formed as before, and
  the factors one and one half make the products exact, i.e. the results are bit-wise
  identical.
- The Finite Volume variant now computes `D = 1/2 d G`. Its results change. There is no
  reference to compare with: the Finite Volume kernel drops the non-conservative product
  (`benchmarks/exagrype/fo_vs_so`), and the second-order Finite Volume runs did not
  converge before.
- The C++ tests check the routines with a table of their own, which mixes variables, axes
  and factors. That the table of the solver is the one the CCZ4 kernels expect is shown by
  the bit-wise identical Finite Difference results and the agreement of the second-order
  with the first-order solver.
- The routines are physics-free and could move to `exahype2::fd`. They stay in
  `exagrype::ccz4` until a second solver needs them.

## How the change was validated

- The 33 rows of `derivative_table()` equal the entry numbers and factors of the former
  fourth-order loop body.
- Gauge wave of `benchmarks/exagrype/fo_vs_so`, RK4FD4, 27^3 grid points, t = 0.102: the
  snapshots of the second-order solver are bit-wise identical to the ones before the
  change, with lopsided advection (halo three) and with plain central differences
  (halo two).
- `FlatSpacetimeTest` (asserts and release builds) applies both routines to polynomials
  with a table of nine rows, including the factors one half and two. The test of the
  central differences now covers the metric and the shift.
- The Python unit tests check that the table covers each derivative once, that the
  generated call carries it, and that inconsistent tables are rejected.
