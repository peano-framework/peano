# ADR 0003: Auxiliary variables which are derivatives get compact second derivatives

Status: proposed
Date: 2026-11-26

## Context

The right-hand side kernels (`FD4Kernels.rightHandSideWithFD42D/3D`) hand the
non-conservative product, along each axis k, the fourth-order first derivative of every
entry of the input times the mesh width (`delta_k`), auxiliary variables included.

Some solvers hold first derivatives of their unknowns as auxiliary variables and
recompute them in front of each stage, e.g. the second-order CCZ4 formulation
(`exagrype.api.FOSOCCZ4Solver`), where `dLapse[0]` is the derivative of the lapse along
x. The derivative of such an entry a = c d_k u along its own axis k is a second
derivative of u. The kernels compute it as the first derivative of the first derivative,
i.e. with the nine-point stencil (1, -16, 64, 16, -130, 16, 64, -16, 1)/(144 h^2):

- Its truncation error is h^4/15 f^(6), six times the one of the compact five-point
  stencil (-1, 16, -30, 16, -1)/(12 h^2), h^4/90 f^(6).
- It returns zero for the highest frequency of the grid, (-1)^i = cos(pi x/h), whose
  second derivative is -(pi/h)^2 times the mode; the compact stencil returns -(16/3)/h^2
  times the mode. With the nine-point stencil, noise on the scale of the grid is
  invisible to the principal part of the equations.

The legacy ExaGRyPE kernel (`FD4DSLKernelExaGRyPE.py`, removed in 2026-10) overwrote these
entries with the compact stencil. It held the CCZ4 entry numbers in the generic kernel
and did so for the first-order formulation, too, where the entries are evolved unknowns.

## Decision

- `CentralDifferencesWithRungeKutta` takes the optional argument `auxiliary_derivatives`:
  rows (derivative, unknown, axis, factor) by the names of the variables, meaning that the
  auxiliary variable equals factor times the derivative of the unknown along axis,
  entry by entry. `exahype2.api.Quantities.resolve_auxiliary_derivatives()` translates
  the names into indices of the kernel input.
- With such a table, the scheme generates the right-hand side kernels from the source of
  `FD4Kernels` with, behind each assignment of `delta_k`, statements which overwrite the
  rows of the derivatives along k with factor times the compact second derivative of the
  unknown, times the mesh width (`CompactSecondDerivatives.right_hand_side_source()`).
  Consecutive entries with the same factor form one statement.
- Rows of derivatives along other axes, i.e. mixed second derivatives, keep the first
  derivative of the first derivative: All stencils stay along the axes and never read the
  diagonal blocks of the halo, which Peano does not fill.
- The kernels are physics-free. The table comes from the solver, by names, as
  `advected_quantities` does.
- DSLHyPE (`exahype2.api.dslhype.Parser.parse()`) accepts the source text of a function,
  besides the function itself, so a scheme can hand over a kernel it has assembled.

## Consequences

- Without the argument, the generated kernels are the same as before, i.e. the change
  does not affect any solver which does not ask for it.
- The generated source relies on how DSLHyPE indexes temporaries: `delta_k` inherits the
  index range of its first operand, -2:NC-2 along k. The overwrites therefore address
  `delta_k[..., -2:NC-2, ...]` along k. With 0:NC they write two grid cells off, partly
  beyond the end of the array, and nothing reports it. A unit test checks the generated
  C++ for this.
- The compact stencil reaches two grid cells, like the first derivative, i.e. the halo
  does not grow.
- The overwritten first derivatives are still computed. Skipping them would split the one
  statement which computes `delta_k` into several and save little, as the
  non-conservative product dominates the cost.
- Whoever holds derivatives as auxiliary variables has to keep the table consistent with
  how they are computed. The second-order CCZ4 solver computes them from the very same
  table (ccz4 ADR 0005).

## How the change was validated

- Unit tests (`tests/unit/test_rkfd4_compact_second_derivatives.py`): the stencil is exact
  for polynomials up to degree five and returns -64/12 for the highest frequency, where
  the first derivative of the first derivative returns zero; names resolve into kernel
  indices; the generated source and the generated C++ write grid cell c of the patch.
  Writing the left-hand side with 0:NC instead of -2:NC-2 fails both tests.
- The generated kernels of the first-order CCZ4 solver are byte-identical to the ones
  before the change, with lopsided advection and with plain central differences.
- The second-order CCZ4 solver uses the option (ccz4 ADR 0006), with the results given
  there.
