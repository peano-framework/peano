# ADR 0003: The solver picks whether the scheme or the non-conservative product adds the advection

Status: proposed
Date: 2026-11-17, revised 2026-11-20

## Context

CCZ4 holds advection terms `beta^k d_k Q` along the shift. In the first-order formulation,
the advection of the metric, the lapse, the shift and phi is part of the source, as it is
written with the secondary entries (`beta^k D_kij`, `beta^k A_k`, `beta^k B_k^i`,
`beta^k P_k`). The advection of all other entries needs the derivative of the entry
itself. The non-conservative product (`ncpImplementation`, a
`template <bool SecondOrder, bool Advection>`) held these terms, but only compiled them if
`Advection` was set. All kernels set it to `false`, i.e. the branch had never been compiled.
They left the terms to the numerical scheme, which adds them with lopsided stencils
(`exahype2.api.rkfd4.CentralDifferencesWithRungeKutta` with the kernel
`CentralDifferencesWithLopsidedAdvection` and `advected_quantities()`).

This left three gaps:

- The default kernel of the Finite Differences, `CentralDifferences`, adds no advection.
  CCZ4 then lacked the advection of 47 of the 58 entries (first order) or of all 25
  (second order), without any warning. The single black hole benchmark without `--lopsided`
  and the plain Finite Difference runs of tutorial 523 ran like this.
- Setting `Advection` to `true` in the kernels would have fixed that, but the schemes with
  lopsided advection would then have added the terms twice. One switch cannot serve both.
- The second-order source multiplies the advection of the metric, the lapse, the shift
  and phi with `CCZ4FO`, i.e. with zero, as it has no secondary entries to write them
  with. The `Advection` branch of the second-order kernel did not add them either.

## Decision

- **One definition.** `advected_quantities()` (which entries) and
  `ADVECTION_VELOCITY` (which velocity) of the solver are the only place that defines the
  advection terms which need the derivative of the advected entry itself. The kernels of
  `exagrype::ccz4` never hold such a term. The `Advection` template argument and its
  never-compiled branches are removed from `ncpImplementation`.
- **Two ways to add them, from the same definition.** Either the Finite Differences add
  them with lopsided stencils (`exahype2::fd::addLopsidedAdvection()`), or the solver
  appends them to the non-conservative product, discretised with the derivative the
  scheme hands over, i.e. its central differences (`exahype2::fd::addCentralAdvection()`).
  Both routines take the same indices of the velocity and of the advected entries.
  These terms are linear in the gradient, i.e. a non-conservative product
  `B = -beta^n` on the diagonal of the advected entries.
- The user chooses the formulation (`FOCCZ4Solver` or `FOSOCCZ4Solver`) and whether the
  Finite Differences use lopsided advection. The solver decides when it is realised, as
  the numerical scheme is assigned after the solver has been created
  (`CCZ4SolverBase.ncp_adds_advection()`):

  | Numerical scheme | Non-conservative product |
  |---|---|
  | `rkfd4`, kernel `CentralDifferences` (also `FD4WithKODissipation`) | with advection |
  | `rkfd4`, lopsided advection of exactly `advected_quantities()` with `ADVECTION_VELOCITY` | without advection |
  | `rkfd4`, lopsided advection of other quantities or with another velocity | `ValueError` |
  | `fv` (both Rusanov schemes) | without advection, as before |
  | anything else | `TypeError` |

- The solver registers its non-conservative product only then, not when it is created,
  and refuses one that somebody else has added with `add_pde_term()`
  (`CCZ4SolverBase._register_non_conservative_product()`). Thus exactly one place
  decides what the generated non-conservative product calls.
- The non-shifting shift (`CCZ4b=0`, ADR 0002) is the one exception to "no advection in
  the kernels": the advection of Ghat in the equation of b is no advection of b itself,
  and the kernel adds it, for both ways.

### Discarded first version

The first version of this decision (2026-11-17, never merged) added two kernels,
`ncpFirstOrderWithAdvection()` and `ncpSecondOrderWithAdvection()`, which instantiated the
template with `Advection` set and added the advection of the metric, the lapse, the shift
and phi to the second-order branch. A C++ test (`AdvectionTest`) checked that they differ
from the kernels without advection by exactly `-c_q beta^n d_n Q_q` for the entries of
`advected_quantities()`. It worked, but the entries were then defined twice: as
`if constexpr (Advection)` lines in the kernel, and as the Python list for the lopsided
advection. Only the two tests kept them consistent. As the difference is exactly the
advection of the listed entries, the second definition is not needed. The first version
also registered the non-conservative product without advection when the solver was
created and replaced it when the solver was realised; code reading the PDE terms in
between saw the wrong term.

## Consequences

- Plain central Finite Differences now solve CCZ4 with all advection terms, discretised
  with the central differences of the scheme. Their results change wherever the shift is
  not zero, e.g. the single black hole without `--lopsided` and the Finite Difference runs
  without lopsided advection in tutorial 523, whose text and figures still describe the
  runs without these terms. Runs with a vanishing shift, such as the gauge wave, do not
  change. The kernels are bit-wise the same as before, i.e. runs with lopsided advection
  do not change either.
- A kernel must never hold an advection term of an entry of
  `advected_quantities()`, or it would be added twice. `Kernels.h` states this
  rule. It replaces the former rule that two definitions have to agree.
- Central differences do not upwind the advection. Whether they are stable for moving
  punctures has not been tested; lopsided advection remains the choice of codes such as
  GRChombo.
- With central differences and the non-shifting shift (`CCZ4b=0`), the advection of Ghat
  in the equation of b and in the equation of Ghat uses the same stencil. The two stencils
  of ADR 0002 remain for lopsided advection only.
- Scripts which hand other quantities to the lopsided advection than
  `advected_quantities()` now fail when the solver is realised, instead of
  silently adding some advection terms twice or missing some.
- The Finite Volumes still solve CCZ4 without these advection terms. Whether they should
  get `addCentralAdvection()` as well is open; the decision is a single condition in
  `ncp_adds_advection()`, but it needs a stability and convergence check.
- The solver has no non-conservative product until it is realised. Code which reads the
  PDE terms before `generate_solver_realisation()` does not find one.
- A non-conservative product that somebody added with `add_pde_term()` makes the
  realisation fail with a `ValueError` instead of being replaced silently. Whoever
  needs another non-conservative product, e.g. for the terms of a coupled matter field,
  derives a solver class and overrides `_ncp_implementation()`. A project can still be
  generated several times: the solver only replaces the term it registered itself.
- `lopsided_advection_quantities()` is renamed to `advected_quantities()`, as it serves
  both ways, and its name now matches the parameter `advected_quantities` of the scheme.
  There is no alias: scripts which call the former name fail and have to be adapted.
  All scripts, tutorials and tests of the repository are.
- The logic which decides who adds the advection lives in `CCZ4SolverBase`, but it only
  reads the list and the velocity. A scalar field solver, whose advection velocity is an
  auxiliary variable (the shift of a coupled CCZ4 solver), could reuse it once it lives
  in a component of its own and once `resolve_quantities()` and the generated call accept
  auxiliary variables. Neither is done yet.
- The right-hand side kernels call the non-conservative product once per normal.
  `addCentralAdvection()` handles one normal per call. If the non-conservative product
  receives all normals at once one day, the routine has to sum over them.

## How the change was validated

- `ncpFirstOrder()`, `ncpSecondOrder()`, both sources and `maxEigenvalue()` were compared
  with upstream on random states for all combinations of `CCZ4LapseType`, `CCZ4s` and
  `CCZ4b`: 5.36 million values are bit-wise identical.
- The kernel without advection plus `addCentralAdvection()` with the list of the solver was
  compared with the discarded `...WithAdvection` kernels of the first version on the same
  states, for all normals, switches and both formulations: of 3.98 million values,
  3.63 million are bit-wise identical, the others differ by at most 8.9e-16 (rounding).
  Dropping a single entry from the list makes them differ by 8.8e-2, i.e. the comparison
  detects a missing term.
- The Python unit tests check that the generated call covers exactly the indices and the
  velocity which the lopsided advection gets, for both formulations and all switches, the
  choice for all schemes, the registration when the solver is realised, and the rejection
  of a foreign non-conservative product.
