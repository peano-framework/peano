# ADR 0004: The FOSO reconstruction computes the derivatives of its cell only and takes the halo of the scheme

Status: proposed
Date: 2026-11-26

## Context

The second-order formulation (`exagrype.api.FOSOSolver`) evolves 25 unknowns and
recomputes their 33 first derivatives in front of each Runge-Kutta stage
(`exagrype.api.FOSOReconstructionSolver`, `exagrype::ccz4::reconstructFirstDerivativesFD4()`).
The CCZ4 solver then reconstructs the halo of the derivatives from the faces and
differentiates them once more for the second derivatives.

`recomputeAuxiliaryVariablesFD4_4thOrder()` evaluated the derivatives on the grid cells
of the cell and on one layer outside of it, along each normal. With the fourth-order
stencil, which reaches two layers, this needed a halo of three, and
`FOSOReconstructionSolver` rejected Finite Difference schemes with a smaller halo. The
argument was the one of the Finite Volumes: the fluxes and non-conservative products
on the faces of a cell need the derivatives left and right of each face.

The argument does not hold for `exahype2.api.rkfd4`:

- The Finite Differences evaluate all terms at grid cells, none on faces. The
  right-hand side of a grid cell differentiates the derivatives with the same stencil,
  i.e. it needs them two grid cells to the left and right.
- Where these grid cells lie in the halo, they belong to the neighbouring cell. That
  cell computes their derivatives as its own and projects them onto the shared face.
  The CCZ4 solver reconstructs the halo of the derivatives from the faces. The Finite
  Volume variant relies on the same exchange (`reconstructFirstDerivativesCentralDifferences()`).
- The wrapper `reconstructFirstDerivativesFD4()` copied only the cell's own grid cells
  back, i.e. the layer outside was computed and discarded. In addition, Peano holds no
  diagonal blocks of the halo, so only the derivatives along the normal of that layer
  were correct.

The plain `CentralDifferencesWithRungeKutta`, whose halo is two, could thus not be
combined with the second-order solver, although ADR 0003 makes it a complete
configuration (the non-conservative product adds the advection). The first-order and
the second-order solver followed different halo rules.

## Decision

- `recomputeAuxiliaryVariablesFD4_4thOrder()` evaluates the stencil on the grid cells
  of the cell only. It uses the template that the second-order variants already use,
  instead of a copy of its loops, and needs a halo of two.
- `FOSOReconstructionSolver` accepts any halo of the Finite Difference scheme. The
  second-order solver takes the halo of the scheme like the first-order solver: two for
  the plain scheme, three with lopsided advection or `exagrype.api.FD4WithKODissipation`.

## Consequences

- The plain Finite Differences can be used with the second-order solver.
- Runs with a halo of three do not change: the values of the cell's own grid cells are
  computed by the same loop body, which reads the 25 unknowns and writes the
  derivatives only.
- The reconstruction evaluates the stencil on n^3 instead of about (n+2) n^2 grid cells
  per normal.
- Domain boundaries other than periodic ones still lack the derivatives in the ghost
  grid cells (see the shortcomings of `FOSOReconstructionSolver`). A wider evaluation
  would not provide them either, as the unknowns of the ghost grid cells reach only as
  far as the halo.
- `FlatSpacetimeTest` checks the reconstruction for halos of two, three and four. The
  Python unit test which expected a rejection of the halo of two now expects the
  generated reconstruction to use it.

## How the change was validated

Gauge wave of `benchmarks/exagrype/fo_vs_so`, RK4FD4, 27^3 grid points, t = 0.102:

- Lopsided advection (halo three): the snapshots of the second-order solver are
  bit-wise identical to the ones before the change.
- Plain central differences (halo two), newly possible: the snapshots are bit-wise
  identical to the ones of the same scheme with its halo forced to three (a subclass
  with `HALO_SIZE = 3`), i.e. the third layer is never read. Relative L2 error of
  the lapse 1.64e-6 (second order) versus 1.13e-6 (first order), the numbers of
  lopsided advection, as the shift of the gauge wave vanishes.
- `FlatSpacetimeTest` (asserts build) passes for the halo of two.
