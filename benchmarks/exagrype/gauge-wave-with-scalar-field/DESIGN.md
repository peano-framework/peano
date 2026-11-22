# Design plan: CCZ4 with a scalar field, outer and inner couplings

Status: written before the implementation and kept up to date while implementing (see "Deviations and findings" at the end).

## Goal

The benchmark `gauge-wave-with-scalar-field` couples first-order CCZ4 (gauge wave, exact solution known) to an advected scalar field.
It exists to study the numerical order of different couplings of two solvers:

| coupling | wiring | expected order of the scalar field (rk4fd4) |
|---|---|---|
| outer multiplicative | CCZ4 step, copy, scalar field step | 1 |
| inner additive | per Runge-Kutta stage: copy, then both stages in parallel | 4 |
| inner multiplicative | per Runge-Kutta stage: copy, CCZ4 stage, scalar field stage | 4 |

The outer additive coupling for the scalar field is removed (it remains available for Euler).

## Decisions taken without the user (the user is away)

1. **"rkdg" is read as `rkfd4`.** The new Python API (`exahype2.api`) has no Runge-Kutta DG scheme. The legacy `rkdg` lives in `python/exahype2/solvers`, uses the old solver classes
   and cannot be coupled to CCZ4 through the new API. The benchmark names its schemes `rk1fd4` ... `rk4fd4`, and the task asks for "rk4fd4" in the same breath.
   The four generators are: Finite Volumes (Rusanov, explicit Euler) x {multiplicative, additive}, and Runge-Kutta with fourth-order Finite Differences x {multiplicative, additive}.
2. **The "generator" is the numerical scheme.** In the code, what the task calls a generator is `Solver.NumericalScheme` (`solver.numerical_scheme`; a legacy attribute is still called `_generator`).
   The inner coupling class has a `numerical_scheme` attribute of its own. The four generators are what one assigns to it.
3. **Advection through the lapse.** Scalar field: `dq/dt + alpha^2 dq/dx = 0` for all four unknowns, with the lapse `alpha` read from CCZ4 (auxiliary variable).
   `alpha^2 = H = 1 - A sin(2 pi (x-t))` is the metric component `g_xx` of the gauge wave. The characteristics in `xi = x - t` obey `d xi/dt = -A sin(2 pi xi)`, which has a closed-form solution,
   so the error of the scalar field can be measured against an exact solution. The coupling is a fake coupling: nothing feeds back into CCZ4, so only one direction exists.
4. **One direction, one copy task ("prep") per stage.** CCZ4 -> scalar field. No auxiliary set on CCZ4's side (the minimal skeleton of the coupling skill).
5. **Both solvers use the same numerical scheme and the same patch size** (the `--scheme` of the benchmark), in every coupling. The data exchange is then a patch copy
   (`CopyPatchWithLinearInterpolation` degenerates to a copy), and the couplings can be compared.
6. **Both inner flavours exchange the same, stage-consistent data.** The order requirement forces this (see below). The flavours differ in the schedule, not in the numbers, as long as only one direction exists.

## Exact solution

Gauge wave: `H = 1 - A sin(2 pi xi)`, `xi = x - t`, `A = 0.1`, lapse `alpha = sqrt(H)` (`src/exagrype/initialvalues/InitialValues.cpp`).
Scalar field: `q_t + H q_x = 0`. In `xi`, `d xi/dt = H - 1 = -A sin(2 pi xi)`. With `theta = pi xi`: `tan(theta(t)) = tan(theta_0) exp(-2 pi A t)`,
so the foot point of the characteristic is `theta_0 = atan2(exp(2 pi A t) sin(theta), cos(theta))`. Everything below depends on `2 theta` only, so it is periodic with period one in `xi`.
Initial data `q_m(x) = sin(2 pi x + m pi/4)`, `m = 0..3`, and hence `q_m(x,t) = sin(2 theta_0 + m pi/4)`. (First attempt: harmonics `sin(2 pi (m+1) x)`. The fourth harmonic is not resolved on the coarsest mesh of nine volumes per axis, which gave an O(1) error that has nothing to do with the coupling.)
The error is measured with a point probe along `y = z = 0.5`, as in `benchmarks/exagrype/gauge-wave/convergence-study.py`.

## Why outer coupling gives order one and inner coupling order four

Outer: the scalar field integrates a whole time step (all Runge-Kutta stages) with the lapse frozen at one time level (multiplicative: `t+dt`).
The frozen coefficient is the error: first order in `dt`, independent of the Runge-Kutta order.

Inner: the lapse that the scalar field sees in stage `k` is the lapse of CCZ4's stage input `Y_k`, i.e. at time `t + c_k dt`. The scalar field's Runge-Kutta scheme then sees the correct
non-autonomous problem and keeps its order. If it saw the stage output `Y_{k+1}` (time level `c_{k+1}`), the order condition `sum_k b_k c_k = 1/2` would fail
(classic RK4 with shifted levels yields `3/4`), and the scheme would drop to first order. Therefore:
**both inner flavours copy the stage input before the stage consumes it.**
CCZ4's reconstruction discards the stage input for `k > 0`, so the copy has to run in front of `Reconstruct stage k`
(the problem the removed KickOff task of the old `CCZ4WithEuler` host solved).

Time step sizes: the scalar field's advection speed `alpha^2` is about one, so its CFL limit is similar to CCZ4's. Both solvers advance with the minimum of both,
which `SynchroniseTimeStepSizes` enforces in a global end-of-step task, as in the outer couplings.

## Architecture (files)

```
src/exagrype/scalarfield/ScalarFieldKernels.h/.cpp                (+ Makefile.am; CMake globs)  advection velocity, non-conservative product, eigenvalue, initial values
src/exagrype/api/scalarfield/ScalarField.py                       solver with four unknowns, reads the CCZ4 fields as auxiliary variables
src/exagrype/api/scalarfield/CCZ4WithScalarField_OuterMultiplicativeCoupling.py
src/exagrype/api/scalarfield/CCZ4WithScalarField_InnerCoupling.py   host "CCZ4WithScalarField": two solvers, auxiliary set, numerical_scheme slot
src/exagrype/api/scalarfield/InnerCouplingGenerator.py            abstract base of the four generators
src/exagrype/api/scalarfield/FiniteVolumesInner{Multiplicative,Additive}.py
src/exagrype/api/scalarfield/RungeKuttaFD4Inner{Multiplicative,Additive}.py
benchmarks/exagrype/gauge-wave-with-scalar-field/gauge-wave-with-scalar-field.py   --coupling, --scheme
benchmarks/exagrype/gauge-wave-with-scalar-field/convergence-study.py
```

### The solver

`ScalarField` stays a plain `exahype2.api.Solver` with hooks, like `Euler`. PDE terms: a `NonconservativeProduct` (advective form; a flux form would add `q d_x(alpha^2)`
and spoil the closed-form solution), a `MaxEigenvalue`, an `InitPointWisely`. No flux, no source. All stateless, all calling `exagrype::scalarfield::*`.
The 58 CCZ4 entries are an auxiliary set that the coupling adds.

### Outer multiplicative coupling

Plain class `CCZ4WithScalarField_OuterMultiplicativeCoupling`, sibling of the Euler one: `graph(CCZ4) -> prep -> graph(ScalarField) -> synchronise`.
The set is `STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES`. Each solver has a numerical scheme of its own, assigned by the user.

### Inner coupling: host and generators

The host `CCZ4WithScalarField` is plain, too (`solvers`, `name`, `generate_solver_realisation()`), plus a `numerical_scheme` property:

- The **user** attaches a generator to the host: `coupled.numerical_scheme = RungeKuttaFD4InnerAdditive(coupled, patch_size=9, rk_order=4)`.
- The host **rejects** aggregated solvers which hold a numerical scheme of their own (`ValueError`, when the generator is attached and again when the project is realised).
  The user never assigns `ccz4.numerical_scheme` or `scalar_field.numerical_scheme` in an inner coupling.
- Grid construction and plotters stay per solver (they are not generators).
- The auxiliary set is `CHANGING_ON_FACES`: its halo is reconstructed in every stage.

The generator is **not** a copy of the 1400 lines of `CentralDifferencesWithRungeKutta`:

1. Its constructor installs, on each aggregated solver, the component scheme of its kind (`CentralDifferencesWithRungeKutta` or `RusanovWithExplicitEuler`; same patch size, same order).
   These are private building blocks: they create the data and kernels of their solver and the per-solver stage tasks. The user does not see them.
2. `construct_output()` realises both solvers (`add_global_tasks=False`) into graphs of their own and merges them.
3. Per stage `k` (Finite Volumes: one stage), it adds the copy task and wires the stage (see below).
4. It adds the final `Synchronise time step sizes` task and wraps the graph into load balancing and AMR (through CCZ4's scheme, as in the outer couplings).

So the generator builds the task graph of the combination, but it composes component graphs rather than rewriting the schemes. A literal from-scratch rewrite would duplicate the data
structures, kernels and boundary treatment of both schemes, which contradicts "as simple as possible"; the interface (`construct_output()` of the generator) is the same either way.

#### Per-stage wiring

`first[S,k]`: the tasks of solver `S` that start stage `k` (stage 0: the sources of S's graph; stage `k>0`: all tasks that depend on `end[S,k-1]`).
`end[S,k]`: the tasks that finish stage `k` (Runge-Kutta, not the last stage: the roll over of the stage's faces, found by name; last stage and Finite Volumes: all non-global sinks of S's graph).

```
copy[k]:         depends on end[CCZ4,k-1] and end[SF,k-1]; reads CCZ4's stage input (patch for k=0, stage input otherwise; READ_AND_WRITE, so it survives),
                 writes the scalar field's auxiliary patch, projects it onto the faces, restricts, rolls over
additive:        first[CCZ4,k] and first[SF,k] depend on copy[k]
multiplicative:  first[CCZ4,k] depends on copy[k];  first[SF,k] depends on copy[k] and end[CCZ4,k]
```

Finite Volumes have one stage. The additive wiring is then the one of an outer additive coupling. The multiplicative one reads "copy, CCZ4, scalar field". In the outer multiplicative coupling the scalar field sees the
*updated* CCZ4; here it sees the stage input, for the reasons above (for a first-order scheme this is immaterial).

#### Four generators

`InnerCouplingGenerator` holds `__init__(host, patch_size)`, `construct_output()`, `_create_copy_task()`, `_wire_stage()` and the class attribute `MULTIPLICATIVE`.
Subclasses provide `_create_scheme(solver)`, `_number_of_stages()` and `_end_of_stage(solver, graph, stage)`. The four concrete generators are a few lines each.

## Benchmark and checks

`gauge-wave-with-scalar-field.py --mode {release,debug} --scheme {fv,rk1fd4,...,rk4fd4} --coupling {outer-multiplicative,inner-additive,inner-multiplicative}`.
`convergence-study.py` runs `rk4fd4` for the three couplings on meshes of 9, 27 and 81 volumes per axis, measures the error of the scalar field against the exact solution and
checks the estimated order (outer ~1, inner ~4 with the safety margins of the gauge-wave study).

Checks during implementation: (1) every coupling generates and compiles; (2) the task graphs (text log) match the wiring above; (3) the convergence study.

## Documentation

`coupling.dox`: write up the inner multiplicative and inner additive sections (currently `@todo`), the scalar field example, the decision "component schemes instead of from scratch".
`coupling/SKILL.md`: replace the two inner placeholders by a recipe (host with a `numerical_scheme` slot, generator base class, stage wiring, stage-consistent copies) and update the questions in "Capture Intent".

## Deviations and findings

- Generation, compilation and a run work for `rk4fd4` with the inner additive coupling (mesh of nine volumes per axis). All six combinations of scheme and coupling generate their task graphs.
  The wiring of the multiplicative flavour was checked in the logged graph (the scalar field's stage 0 waits for CCZ4's roll over of stage 1).
- Smoke runs on the coarsest mesh (build and run, no error measurement): `fv` inner-multiplicative and `rk4fd4` outer-multiplicative finish without errors.
- The initial data are phase-shifted sine waves of one harmonic (see "Exact solution"): the first choice of four harmonics was not resolved on the coarsest mesh.
- With that first choice, the run showed that the exact solution and the coupling are right: the first component (harmonic one) differs by about 5e-3 on nine volumes per axis.
- **Not verified:** the convergence orders (outer ~1, inner ~4). The study was stopped by the system's memory guard before the 27-volume run, and has not been repeated.
  Run `python3 convergence-study.py --work-directory <dir>` (needs the release libraries, see below) and check the table; adjust `ExpectedOrders` if the measured orders differ.
- The study needs a configured Peano with `--enable-exahype --enable-blockstructured --enable-loadbalancing --enable-exagrype`; the release libraries are enough.
- Neither the debug mode (visualised task graphs) nor the doxygen documentation was regenerated.
