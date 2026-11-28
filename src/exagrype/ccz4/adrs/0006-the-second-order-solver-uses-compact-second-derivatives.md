# ADR 0006: The second-order solver computes pure second derivatives with the compact stencil

Status: proposed
Date: 2026-11-26

## Context

The second-order formulation (`exagrype.api.FOSOCCZ4Solver`) holds the first derivatives
of the lapse, the shift, the conformal metric and phi as auxiliary variables (ADR 0005).
Its non-conservative product needs their second derivatives: the Ricci tensor (dD, dP),
the second covariant derivative of the lapse (dA) and Ghat (dB).

Since the legacy ExaGRyPE Finite Difference kernel was removed (2026-10), the generic
kernels of `exahype2.api.rkfd4` compute all of them as the fourth-order first
derivative of the reconstructed first derivatives, the pure ones (d_xx, d_yy, d_zz)
included. Baojiu Li asked whether this differs from the old second-order code, which
computed the second derivatives directly, and whether the difference has been checked.
It had not. For the pure second derivatives, the first derivative of the first
derivative has six times the truncation error of the compact stencil and does not see
the highest frequency of the grid (rkfd4 ADR 0003).

## Decision

- With Finite Differences, the solver computes the pure second derivatives with the
  compact stencil of the unknown: the scheme gets the table
  `FOSOCCZ4Solver.auxiliary_derivatives()` as its argument `auxiliary_derivatives`
  (rkfd4 ADR 0003). It is the table the reconstruction uses (ADR 0005), i.e. the
  reconstruction and the second derivatives cannot disagree on which entry is the
  derivative of which unknown.
- Mixed second derivatives remain first derivatives of first derivatives. A direct mixed
  stencil reads the diagonal blocks of the halo, which Peano does not hold.
- The user hands the table to the scheme, as `advected_quantities` (ADR 0003). The solver
  checks it when it is realised and refuses a Finite Difference scheme without this
  table or with another one (strict), rather than filling it into the scheme: The solver
  does not change the scheme the user has created, the script states what the kernels
  do, and nobody falls back to the less accurate stencil without noticing.

## Consequences

- The results of the second-order solver with Finite Differences change (see below).
  Scripts which create its scheme have to pass `auxiliary_derivatives`; the benchmark
  `benchmarks/exagrype/fo_vs_so` and the unit tests do.
- The first-order solver and all other solvers do not change: their kernels are the same
  as before.
- The Finite Volume variant is not affected; it has no such option.
- The legacy kernel overwrote the entries for the first-order formulation, too, where they
  are evolved unknowns. Here, only a scheme with the table changes its kernels, and only
  the second-order solver hands one over.

## How the change was validated

Gauge wave of `benchmarks/exagrype/fo_vs_so`, RK4FD4 with lopsided advection, t = 0.1,
relative L2 error of the lapse:

| grid | first order | second order before | second order now |
| --- | --- | --- | --- |
| 27^3 | 1.13e-6 | 1.64e-6 | 2.81e-7 |
| 81^3 | 1.41e-8 | 2.2e-8 (fo_vs_so docs) | 7.76e-9 |

- The first-order error and the difference between the two solvers (2.28e-6 and 2.83e-8)
  both shrink by a factor of 80, i.e. with fourth order. The second-order error, which
  is their sum, shrinks by 36 only: the leading terms of the two cancel, more so on the
  coarse grid. A finer grid has not been run.
- The kernel generated from the table is identical to the one of a hand-made prototype,
  and so are the snapshots. The first-order kernels and snapshots do not change.
- Flat spacetime with noise of 1e-10 on all entries, periodic, 27^3, no dissipation,
  t = 10: no exponential growth, neither before nor after the change. Both second-order
  variants grow linearly, driven by the frozen noisy shift (CCZ4s = 0) in the equation
  of Ghat: max |Q - flat| 1.27e-6 before and 8.7e-7 now, against 2.6e-9 for the first
  order. With the compact stencil, Ghat and the metric deviate less, K, theta and Kex
  more, as the stencil passes noise on the scale of the grid on.
- Python unit tests: strict refusal without or with another table.
