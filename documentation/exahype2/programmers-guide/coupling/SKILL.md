---
name: peano:couple-exahype-solvers
description: Take several solvers from ExaHyPE and couple them together, so they work as a unit.
---

#Couple ExaHyPE solvers

## Capture Intent

Start by understanding the user's intent.

1. Which solvers should be coupled?
2. Do you favour an additive or a multiplicative coupling?
3. What will the solver be called and where to place the resulting files?
4. Will the solvers exchange data only via volumetric coupling, or will they also globally synchronise with each other?

There is no preferred version of additive vs multiplicative.

## Preparatory steps

Create a new host class which derives from `exahype2.api.SolverHost`. It is what the user
adds to the project (`project.add_solver(host)`), and it brings its solvers along. 
The constructor creates one solver instance per solver to be coupled, and hands them over
to the superclass in the order in which they are realised. A solver can rely on the data
of its predecessors. Do **not** let the solvers call each other's realisation, and do not
add the individual solvers to the project yourself.

### Who holds the numerical scheme?

The host holds the generator of the numerical scheme (`host.numerical_scheme`, a 
`Solver.NumericalScheme` built for `host.global_tasks_solver`). The hosted solvers have
**none** of their own: `SolverHost` raises a `ValueError` in its constructor, when the scheme
is assigned and when it realises the solvers, if a solver has a `numerical_scheme`. Users 
therefore write `host.numerical_scheme = Scheme(host.solver, ...)`, never
`host.solver.numerical_scheme = ...`. A solver that runs a complete scheme of its own next to the
others (additive coupling of two full solvers) gets it via `host.set_numerical_scheme(solver, scheme)`;
the host stays the one that holds and hands out every scheme. The mapping of each solver to C++ (`SolverToCPPFiles`,
grid construction, plotters) remains solver-specific. Solvers query patch size and halo
through `solver.numerical_scheme`, which the host fills with the scheme itself (for 
`global_tasks_solver`) or with a `SolverHost.HostedScheme` that forwards to it (all others).

### Where the task graphs are composed

Composition happens in the host, nowhere else:

1. The host's scheme builds the task graph of `global_tasks_solver` as it would for a
   non-hosted solver.
2. Every other solver implements `Solver.create_task_graph(exahype2_project, peano_project,
   stage, number_of_stages)` and returns a fresh `peano4.api.taskgraph.Graph` with the tasks of
   one stage. It must not depend on the other solvers' tasks. `SolverHost.HostedScheme` calls it 
   once per stage (`SolverHost.number_of_stages` is the Runge-Kutta order of the scheme, or one),
   merges the graphs into the solver's own graph and keeps them in `stage_graphs`. If a solver's 
   graph occurs in several stages, `create_task_graph()` is simply called several times; 
   share data (scratch patches) between the calls through the solver, create tasks per call.
3. The host's `couple()` composes the graphs through **dependencies**. It does not copy tasks.
   Use `Graph.source_tasks()` / `Graph.sink_tasks()` of a stage graph to find where to hook in, 
   `Graph.append()` to run a whole graph after another one, `Graph.merge()` to put graphs next to each other, and
   `SolverHost.run_in_parallel()` for the additive scheme.
4. The superclass wraps the outcome into load balancing and AMR tasks.

The host does not override `generate_solver_realisation()`. The superclass does the following, 
and the host only implements `couple()`:

1. Each solver is realised into task graphs of its own (`exahype2.api.PhaseGraphs`), and 
   each solver reports its own time stepping graph (text or visualisation, depending on the logger).
2. The superclass concatenates the graphs of all solvers into graphs of the host 
   (`peano4.api.taskgraph.Graph.merge()`).
3. `couple()` adds tasks and dependencies between the chains of the solvers. 
4. The superclass reports the concatenated, coupled time stepping graph once more and 
   appends it to the graphs of the project.

If possible, display a rendering of each of these graphs to the user.

### Global tasks

Some tasks belong to the mesh rather than to a solver: the task that switches the load balancing
off, the last task which evaluates the adaptivity criterion and switches the load balancing on again,
and the mesh adaptation phase. They must exist once. Each solver's 
`generate_solver_realisation()` takes a flag `add_global_tasks`; the superclass passes True to
the solver in `global_tasks_solver` (default: the first one) and False to all others. 
If your host needs a different solver to own them, override the property. 
Ask the user which solver's adaptivity criterion should drive the mesh.
If a solver uses a custom numerical scheme, check that its `construct_output()` accepts and 
honours the flag.


## Volumetric coupling and data sets

If the coupling is volumetric, ask the user if auxiliary variables of one solver are
actually primary (evolved) variables of another one, or if solvers share auxiliary
variables accordingly. After that, change the data association accordingly.


## Algorithmic coupling

### Multiplicative solver coupling

`couple()` makes tasks of one solver wait for tasks of the other one. The graphs are already
concatenated (or `Graph.append()` them yourself if a solver's stage graph simply follows the
stage of another one), so there is nothing to copy: look up the tasks to be tied together (ideally 
through tags that the solver's scheme puts on them - names only as a fallback, and then ask 
the solver's scheme for the names rather than hard-coding them), take over the dependencies 
of the consuming task, and make the consuming task depend on the producer. The first task of 
the second solver has to depend on the last task of the first solver, or, for 
a coupling per time step stage, each stage of the consumer on the matching stage of the 
producer. We create a chain of subtasks.
`exagrype.api.FOSOSolver` is a worked example. 

If possible, display a rendering of the arising task graph to the user.


### Additive solver coupling

The solvers' graphs run in parallel. There are two variants. Ask the user which one they want,
and explain the difference:

| | A. Coupling at the begin and the end of the time step | B. One numerical scheme over the compound |
|---|---|---|
| Inside the time step | solvers are independent | solvers run in lockstep, stage by stage |
| Numerical schemes | any, one per solver (`host.set_numerical_scheme(solver, scheme)`); time stepping, patch size and halo may differ | one scheme, held by the host (`host.numerical_scheme`) |
| Exchange | once per time step: copy, project, run, synchronise | once per stage (KickOff/WrapUp per Runge-Kutta stage) |
| Typical use | different discretisations, e.g. FD4 with RK4 for CCZ4 next to Finite Volumes for a fluid | same scheme and patch size for all solvers |
| Worked example | `exagrype.api.CCZ4WithEuler_AdditiveCoupling` | `CCZ4WithEuler` in `benchmarks/exagrype/gauge-wave-with-Euler` (predates `SolverHost`) |

#### A. Coupling at the begin and the end of the time step

The host creates the solvers and, for each solver, a set of `exahype2.api.AuxiliaryVariables`
that holds the fields of the other solver (`Mode.CHANGING_ON_FACES`: read-only for the
solver, on the faces, rewritten every time step). The set lives on the patch of the
solver that reads it. Each solver realises its complete graph with its own scheme;
`couple()` only adds tasks in front of and behind the concatenated graphs:

1. **Copy tasks** (CELL, one per direction, running in parallel) copy the persistent patch of the
   producer into the auxiliary variables of the consumer. If the patches have a different
   number of volumes, the copy interpolates; `exagrype.api.actionsets.CopyPatchWithLinearInterpolation`
   does it piecewise (tri)linearly. Ask the user which interpolation they want.
2. **Face projection** of the new auxiliary variables (see the check below): project, restrict hanging
   faces, roll over. Copy them from `FOSOReconstructionSolver` if in doubt.
3. The first tasks of **both** solvers (`Graph.source_tasks()` of each solver's own graph) depend on the
   projection tasks of both directions. As the copies are complete when any solver starts, there is no
   data race, although the solvers read and write the same cells.
4. The solvers run.
5. A `GLOBAL_END_OF_STEP` task depends on all sinks of the concatenated graph (excluding
   `GLOBAL` tasks) and synchronises the time step sizes
   (`exagrype.api.actionsets.SynchroniseTimeStepSizes` enforces the minimum on all solvers).
   Add further global exchange here (see the end-of-step section below).

The global tasks (load balancing, AMR) wrap the whole outcome as usual.

#### B. One numerical scheme over the compound

The host's scheme asks each solver for its graph per Runge-Kutta
stage (`create_task_graph()`), i.e. there is one instance of each graph per stage. In `couple()`,
call `SolverHost.run_in_parallel(host_graph, [graph_of_solver_a, graph_of_solver_b])` per stage: it
adds a KickOff task which the first task of each solver's stage graph depends on. Make the 
KickOff of stage `k+1` depend on the sinks of stage `k`.
`CCZ4WithEuler` in `benchmarks/exagrype/gauge-wave-with-Euler` is a worked example 
(it predates `SolverHost`).

If possible, display a rendering of the arising task graph to the user.

#### Data races (both variants)

Ask from the user if any of the mesh data are read and written by two solvers.
In this case, there could be a data race. If this is the case, add to KickOff
a temporary copy of the respective data field and add an action set which befills
this copy. One of the solvers has to redirect their access to the copy. After the
task graph has finished, the copy can be discarded, too.

#### Check: do the copied data have to be projected onto the faces?

Whenever an exchange writes data into auxiliary variables, decide whether the faces hold stale
values afterwards:

- Auxiliary variables with `Mode.CELL_WISE` do not sit on the faces. No projection is needed, but the
  schemes currently reject such sets.
- Auxiliary variables with `Mode.STATIONARY_ON_FACES` are projected once in the setup. If the exchange
  rewrites them, they are not stationary: switch to `CHANGING_ON_FACES`.
- Auxiliary variables with `Mode.CHANGING_ON_FACES` **have to be projected onto the faces right after
  each exchange**, before anybody reconstructs a halo from them: the reconstruction reads
  the halo layers of neighbours from the faces, not from the cell. Add the chain
  project -> restrict hanging faces -> roll over as tasks behind the copy, and make the consumers
  wait for it. Interpolation onto hanging faces needs no task, as it rides along with the
  reconstruction (`exahype2/api/adrs/0001-...`).

Report to the user for each exchanged set which of the three cases applies and whether you added the
projection. A missing projection does not fail: the halos silently contain the values of the previous exchange.

### End of time step coupling (global coupling)

If two solvers furthermore have to synchronise each other globally, then we have 
to insert a WrapUp task after the whole task graph which waits for the last tasks
of all the created subtasks. Note that these might fork in their own right, i.e. we 
have to find all tasks that are not predecessor of any other task. 
`peano4.api.taskgraph.Graph.sink_tasks()` does that, for the host's concatenated graph.

The WrapUp tasks only does a global action in unprepare. The Abstract solver base
class that's generated get a new functor the says synchronise, and the user is told
to supply the missing information. 


### Support for Runge-Kutta time stepping

As we always want to support multi-step time stepping, too, we have to reread all
of the descriptions above re time steps as "subgraph per time step". The host
instantiates the subgraph of each hosted solver once per stage (`create_task_graph(..., stage, ...)`), 
so the same task chain exists several times, once per stage, and the multiplicative scheme ties each instance to the 
matching stage of the consumer (`FOSOSolver.couple()`). For the additive scheme, WrapUp and KickOff 
might have to be inserted multiple times at the corresponding markers. Tasks at the stages are 
looked up in the concatenated graph in `couple()`.


## Optimise (additive coupling only)

This holds for both variants.


If we do temporary copies, do a C++ source code analysis and try to find out which 
unknowns of the copied data are actually read. In principle, the developer could now
narrow down the auxiliary variables. This required a code rewrite. It is easier to 
only copy those unknowns which are actually used. The analysis needs a confirmation 
by the user if this is all correct. 

## Wrap up question

Analyse if both solvers use some input data from the other solver. If not, we work
with uni-directional coupling. Ask the user if they want to insert or append certain
coupling tasks which introduce bi-directional coupling between fields, or if some
key operations need alteration.

Often, the outcome (primary variables) of one task feed into the right-hand sides 
of coupled solvers. If this is the case and if we employ additive coupling, then
we might need to rerun the dependency analysis and introduce temporary copies.


