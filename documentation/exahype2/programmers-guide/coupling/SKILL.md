---
name: peano:couple-exahype-solvers
description: Take two solvers from ExaHyPE and couple them together, so they work as a unit, with a minimalist plain-class coupling (outer additive or outer multiplicative).
---

#Couple ExaHyPE solvers

## Capture Intent

Couplings in ExaGRyPE are deliberately **minimalist**, and so are the ones you write. The two reference couplings,
`exagrype.api.CCZ4WithEuler_OuterAdditiveCoupling` and `exagrype.api.CCZ4WithEuler_OuterMultiplicativeCoupling`, are plain classes:
no base class, two named solvers, no loops, no generic helpers, and next to the docstrings only a few dozen lines of code. 
Whatever the user does not ask for is not built. Ask **at most** the following questions, propose the default in brackets, and 
move on as soon as the user accepts it:

1. Which two solvers should be coupled? (Euler + CCZ4: use the existing classes in `exagrype.api` and stop here, unless the user wants a variation.)
2. Do the solvers run **in parallel** (additive) or **one after the other** (multiplicative)? [additive, the benchmark]
   Both are **outer** couplings: each solver keeps a complete numerical scheme of its own and the solvers exchange data once per time step. 
   Inner couplings (one shared scheme, exchange per Runge-Kutta stage) are only placeholders below; if the user wants one, say so and offer an outer coupling.
   If multiplicative: which solver runs first? [the one whose data the other one consumes]
3. In which direction do data flow? [both directions: one prep task per direction. If only one solver consumes data of the other one, you need **one** prep task
   and the coupling gets even smaller: see the minimal skeleton below.]
4. What is the class called, and where do the files go? [`<A>With<B>_<Outer…>Coupling` in `src/exagrype/api`, i.e. next to its siblings]

Do **not** ask about, and do not build, the following unless the user brings it up: data-race analyses, restricting the copied unknowns,
global user-defined synchronisation, bi-directional physics terms, Runge-Kutta stage coupling. They are listed as optional extras at the end.
Defaults you do not ask about: copy **all** unknowns of the producer, interpolate piecewise linearly, enforce the **minimum** time step size, 
the first solver owns load balancing and AMR.

## Preparatory steps

Keep it simple. There is no generic framework class to derive from: write one plain
Python class per coupling, with a few dozen lines of code, and copy the pattern from
`exagrype.api.CCZ4WithEuler_OuterAdditiveCoupling` (outer coupling) rather than inventing
new machinery. Do not add base classes, registries or hooks. The class is what the user adds
to the project (`project.add_solver(coupled)`). It offers

- `solvers`: the list of solvers, in the order in which they are realised (a solver can rely on the
  data of its predecessors). `Project.add_solver()` recognises a coupling through this attribute,
  registers the solvers, and lets the coupling realise them. Do **not** add the individual solvers
  to the project yourself, and do not let the solvers call each other's realisation.
- `name`,
- `generate_solver_realisation(exahype2_project, peano_project, setup_simulation, time_step, plot, mesh_adaptation)`,
  which is the same in every coupling except for how the graphs of the solvers are put together (copy it from an
  existing coupling, ~40 lines): realise each solver with `add_global_tasks=False` into a time step graph of its own,
  put these graphs together into one coupled graph (see below), wrap the coupled graph through the scheme of the solver which owns the global tasks
  (`solver.numerical_scheme.wrap_task_graph_in_load_balancing_and_amr_tasks()`), let this solver's
  `mesh_adaptation` fill the mesh adaptation graph, report the coupled graph (visualise/log), and merge it into
  the project's `time_step` graph.

How the graphs are put together is the only part that differs between couplings. Write it out in
`generate_solver_realisation()` (**spaghetti code is fine**: there are two solvers, so no loop, no helper hierarchy):

- **Multiplicative:** build the graphs one after the other with `Graph.append()` (the sources of the new graph wait for all sinks of the graph so far).
  There is no further routine: the coupling is the order of the `append()` calls.
- **Additive:** `Graph.merge()` the solver graphs, then call a private routine `_couple(exahype2_project, peano_project, time_step)`
  which adds tasks and dependencies to the merged graph. `time_step.source_tasks()` and `time_step.sink_tasks()` of the merged graph are all
  it asks for. It does not need the graphs of the individual solvers.


### The minimal skeleton

This is the whole coupling of a one-directional multiplicative coupling (CCZ4 runs first, Euler sees its result). It generates a complete Peano project
(checked for Euler + CCZ4 with text logging). Everything else in the reference couplings is a second direction, comments, or the second solver's copy of the same code:

```python
import exahype2.api, peano4.api, peano4.api.taskgraph
from exagrype.api import FOCCZ4Solver
from exagrype.api.euler import Euler
from exagrype.api.actionsets import SynchroniseTimeStepSizes
GridEntity = peano4.api.taskgraph.Task.GridEntity

class CCZ4IntoEuler:
    def __init__(self):
        self.ccz4 = FOCCZ4Solver("CCZ4")
        self.euler = Euler("Euler")
        self.ccz4_copies = exahype2.api.AuxiliaryVariables(
            "CCZ4Fields", dict(FOCCZ4Solver.Unknowns),
            mode=exahype2.api.AuxiliaryVariables.Mode.STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES,
            data_name="CCZ4FieldsForEuler")
        self.euler.add_auxiliary_variables(self.ccz4_copies)
        self.name = "CCZ4IntoEuler"
        self.solvers = [self.ccz4, self.euler]          # CCZ4 first: it owns the global tasks

    def generate_solver_realisation(self, exahype2_project, peano_project, setup_simulation, time_step, plot, mesh_adaptation):
        graphs = []
        for solver in self.solvers:                      # one graph per solver, without global tasks
            graph = peano4.api.taskgraph.Graph()
            solver.generate_solver_realisation(exahype2_project, peano_project, setup_simulation, graph, plot,
                                               mesh_adaptation, add_global_tasks=False)
            graphs.append(graph)
        prep = peano4.api.taskgraph.Graph()
        self._create_prep_task(prep, self.ccz4, self.euler, self.ccz4_copies)   # copied from the reference coupling
        coupled = peano4.api.taskgraph.Graph()           # the coupling: a chain
        coupled.append(graphs[0]); coupled.append(prep); coupled.append(graphs[1])
        synchronise = peano4.api.taskgraph.Task(name="Synchronise time step sizes", entity=GridEntity.GLOBAL_END_OF_STEP)
        for task in coupled.sink_tasks():
            synchronise.add_dependency(task)
        synchronise.add_action_set(SynchroniseTimeStepSizes(self.solvers))
        coupled.add_task(synchronise)
        self.ccz4.numerical_scheme.wrap_task_graph_in_load_balancing_and_amr_tasks(coupled)
        if self.ccz4.mesh_adaptation is not None:
            self.ccz4.mesh_adaptation.construct_output(exahype2_project, peano_project, mesh_adaptation)
        if peano4.api.Logger().is_visualisation: coupled.visualise(title=self.name)
        if peano4.api.Logger().is_text: coupled.log_as_text(title=self.name)
        time_step.merge(coupled)
```

`_create_prep_task()` (~45 lines: one cell task with four action sets, see Step 2) is the only larger piece. Copy it **unchanged** from a reference coupling into the new class, 
together with the constant `PREP_TASK_NAME`, rather than importing it or moving it into a shared base: each coupling stays readable on its own.
For a second direction, add a second auxiliary set and a second prep task. For the additive flavour, replace the three `append()` lines by
`Graph.merge()` of the solver graphs plus the dozen lines of `_couple()` from the reference.

### Numerical schemes

In an **outer** coupling every solver has a complete numerical scheme of its own, assigned as for any
other solver (`coupled.ccz4.numerical_scheme = ...`, `coupled.euler.numerical_scheme = ...`).
Only an **inner** coupling (one scheme per Runge-Kutta stage) needs a scheme that is shared; see the
placeholders below and `exagrype.api.FOSOSolver`, which hands the reconstruction solver a forwarding scheme.
The mapping of each solver to C++ (`SolverToCPPFiles`, grid construction, plotters) remains solver-specific.

### Where the task graphs are composed

Composition happens through **dependencies**. It does not copy tasks. Use
`Graph.source_tasks()` / `Graph.sink_tasks()` to find where to hook in, `Graph.append()` to run a whole
graph after another one, and `Graph.merge()` to put graphs next to each other.
If possible, display a rendering of each of these graphs to the user.

### Global tasks

Some tasks belong to the mesh rather than to a solver: the task that switches the load balancing
off, the last task which evaluates the adaptivity criterion and switches the load balancing on again,
and the mesh adaptation phase. They must exist once. Each solver's
`generate_solver_realisation()` takes a flag `add_global_tasks`; the coupling passes False to all solvers
and wraps the coupled graph itself (see above). Ask the user which solver's adaptivity criterion should drive the mesh
(by default the first one).


## Volumetric coupling and data sets

If the coupling is volumetric, ask the user if auxiliary variables of one solver are
actually primary (evolved) variables of another one, or if solvers share auxiliary
variables accordingly. After that, change the data association accordingly.


## Algorithmic coupling

The outer types are complete. Outer additive is the one in `benchmarks/exagrype/gauge-wave-with-Euler`. Outer multiplicative has a host in
`exagrype.api` which was validated by generating the Euler + CCZ4 project (no C++ build or run yet); it has no benchmark.
The inner types are placeholders, which keep the notes we had before the split into outer and inner. Do not claim that they are validated.

### Outer coupling (multiplicative and additive)

Both outer flavours are complete and share everything but one step. In both, each solver brings its **own numerical scheme**
(time stepping, patch size and halo may differ), and the solvers exchange data only **once per time step**, never per Runge-Kutta stage.
Ask the user which flavour they want (Capture Intent) and then follow Steps 1 to 4. Only Step 2 branches.

| | outer multiplicative | outer additive |
|---|---|---|
| Update | `Q(t+dt) = (U_A o U_B) Q(t)` | `Q(t+dt) = (id + U_A + U_B) Q(t)` |
| Wiring of the graphs | in sequence: A, copy, B | in parallel: copy, A and B together |
| What B sees of A | A's **updated** state | A's state at the **begin** of the time step |
| Prep tasks | two, one in front of A and one between A and B | two, both in front of A and B, run in parallel |
| Host in the repository | `exagrype.api.CCZ4WithEuler_OuterMultiplicativeCoupling` | `exagrype.api.CCZ4WithEuler_OuterAdditiveCoupling` |
| Benchmark | none yet (same driver as the additive one, other host class) | `benchmarks/exagrype/gauge-wave-with-Euler` |

For Euler + CCZ4, the driver of the benchmark is the template for both. The two host classes are independent of each other and differ in how they wire the graphs only; everything else
(constructor, `generate_solver_realisation()`, `_create_prep_task()`) is deliberately repeated, so that each class reads on its own. If the user wants exactly this pair,
use the host class from `exagrype.api` directly and do not copy it. For other solver pairs, write a host along the same lines.
`coupling.dox` (running example: Euler with Finite Volumes and CCZ4 with Runge-Kutta 2) shows the graphs that each flavour produces;
`documentation/exahype2/programmers-guide/coupling-illustrations.py` regenerates those pictures.

#### Step 1: the coupling class (both flavours)

Derive from nothing: one class per coupling, even if it repeats the constructor and the prep tasks of another one. Name the two solvers explicitly (`self.ccz4`, `self.euler`), no loops or lists of solvers over which you iterate generically. The constructor does this, **in this order**:

1. Create the solvers (CCZ4 first, as it owns the global tasks (load balancing, AMR) and Euler depends on its data). The order of `solvers`
   is the order of realisation; it does not fix the order of execution, which the wiring of the graphs does.
2. For each solver, create one `exahype2.api.AuxiliaryVariables` set that holds **all** unknowns of the other solver
   (`dict(OtherSolver.Unknowns)`), and add it with `add_auxiliary_variables()` to the solver that reads it
   (`EulerFieldsForCCZ4` into CCZ4, `CCZ4FieldsForEuler` into Euler; name the data explicitly through `data_name`).
   Use `Mode.STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES`: the exchange overwrites the data, but never within the time step of the consuming
   solver, so a Runge-Kutta scheme reconstructs the halo once, in its first stage. Not `CHANGING_ON_FACES`
   (reconstructs the halo in every stage for nothing), not `STATIONARY_ON_FACES` (the faces would never see the exchange).
   The Euler solver checks at generation time that it holds a set with the 58 entries of CCZ4.
3. Store the name (`f"{a}With{b}"`) and provide `solvers`, `name` and `generate_solver_realisation()` (plus `_couple()` for the additive flavour) as described above.

#### Step 2: wiring the graphs

Each solver ends in a `GLOBAL_END_OF_STEP` task that computes the next time step size. Other tasks may depend on such a task, but the
task graph compiler then does not fuse them into the mesh sweep of that task: they run in a sweep of their own.

**Common to both flavours.** The final synchronisation depends on the sinks of the coupled graph (`coupled.sink_tasks()`; the additive reference filters out
`GLOBAL` tasks, the multiplicative one need not).

Create **two prep tasks** (`GridEntity.CELL`, `"Prepare auxiliary variables from <producer>"`), one per direction. A prep task is the one place that writes new face data:

| data | access |
|---|---|
| `producer._patch` (persistent patch) | `READ_AND_WRITE` |
| `auxiliary_copies.patch` (auxiliary set of the consumer) | `READ_AND_WRITE` |
| `auxiliary_copies.patch_overlap_update` | `TEMPORARY` |
| `auxiliary_copies.patch_overlap_current_time_step` | `READ_AND_WRITE` |

Action sets in this order, **guarded by the consumer's** `grid_selector.createCellGuard()` (face sets: `createFaceGuard()`), with the consumer's
`_get_default_includes()` + `user_solver_includes`:
1. `exagrype.api.actionsets.CopyPatchWithLinearInterpolation(source_patch, target_patch, guard, additional_includes)`: copy, piecewise (tri)linear if the patch sizes differ. Ask the user if they want a different interpolation.
2. `exahype2.api.rkfd4.actionsets.ProjectPatchOntoFaces(auxiliary_copies.patch, auxiliary_copies.patch_overlap_update, ...)`,
3. `toolbox.blockstructured.api.RestrictHangingFaces(auxiliary_copies.patch_overlap_update, scheme=consumer.numerical_scheme.restriction, clear_guard=face guard, ...)`.
   Restriction belongs to the writer of face data, i.e. it sits here, even on a regular mesh.
4. `exahype2.api.rkfd4.actionsets.RollOverUpdatedFace(auxiliary_copies.patch_overlap_current_time_step, auxiliary_copies.patch_overlap_update, ...)`.

Interpolation onto hanging faces needs no task: it rides along with the solvers' reconstruction.

Finally add one `GridEntity.GLOBAL_END_OF_STEP` task `"Synchronise time step sizes"` that depends on all of `final` and holds
`exagrype.api.actionsets.SynchroniseTimeStepSizes([A, B])`. It enforces the minimum of the time step sizes on all solvers, the conservative choice.
If the user wants further global exchange, put it into this task (see the end-of-step section below).

**Branch: outer multiplicative (A first, B second).** Build the chain with `Graph.append()`, each graph after the one before:

```
coupled = Graph()
coupled.append(prep_from_B_graph)   # A sees B(t)
coupled.append(graph_of_A)
coupled.append(prep_from_A_graph)   # copies the updated A, i.e. waits for A's time step size task, too
coupled.append(graph_of_B)
```

(Each prep task sits in a graph of its own, created through `_create_prep_task(graph, producer, consumer, auxiliary_copies)`.)

~~~~~~~~~~~~~~~~~~~~~~~~
 prep(from B) -> graph of A -> prep(from A) -> graph of B -> Synchronise (GLOBAL_END_OF_STEP)
~~~~~~~~~~~~~~~~~~~~~~~~

Ask the user which solver runs first. For CCZ4 + Euler, CCZ4 first is the default (the fluid then sees the new spacetime).
A's graph is complete before the copy, and B starts after it, so there is no data race.
Both solvers advance by the same (minimum) step, but B sees A at the end of that step.

**Branch: outer additive.** Both prep tasks precede **every** first task of **both** solvers:

~~~~~~~~~~~~~~~~~~~~~~~~
 prep(from A)  \            / graph of A \
                >-- first --<             >-- final --> Synchronise (GLOBAL_END_OF_STEP)
 prep(from B)  /            \ graph of B /
~~~~~~~~~~~~~~~~~~~~~~~~

```
for first_task in first[A] + first[B]:
    for prep in preps: first_task.add_dependency(prep)
```

The two prep tasks do not depend on each other and run in parallel. As both copies are complete when any solver starts,
no solver writes a patch that a copy still reads: there is no data race, and no temporary copy is needed.
The solvers only see the state of the other one at the begin of the time step.
For the additive flavour, the end of `_couple()` is exactly `CCZ4WithEuler_OuterAdditiveCoupling._couple()`.
Collect `first = time_step.source_tasks()` **before** you create the prep tasks, as those are sources, too.

Show the graph of each solver alone, the concatenation and the coupled graph to the user: run the driver with the visualised logger
(`peano4.api.LoggerVerbosity.VISUALISATION`, `--mode debug` in the benchmark), which renders each graph. The coupled graph has to show
the chain (multiplicative) or the diamond (additive) above. Compare it with the pictures in `coupling.dox`.

Load balancing and AMR wrap the outcome. They come from the solver that owns the global tasks.

#### Step 3: the driver script (both flavours)

Take `benchmarks/exagrype/gauge-wave-with-Euler/gauge-wave-with-Euler.py`. Only the line that creates the host changes between the flavours.

1. Only the coupling goes into `project.add_solver(coupled)`; never the individual solvers. The coupling sets no initial conditions: add them to the solvers as to any other solver
   (`ccz4.add_pde_term(exagrype.api.initialvalues.gauge_wave())`; Euler brings a smooth wave by default).
2. Schemes per solver: `ccz4.numerical_scheme = ...`, `euler.numerical_scheme = ...`.
   - CCZ4: `rkfd4.CentralDifferencesWithRungeKutta(ccz4, p, rk_order=4, right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection, advection_velocity=exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY, advected_quantities=ccz4.advected_quantities())`
   - Euler: `fv.RusanovWithExplicitEuler(euler, 2*p+1)`
3. Grid construction and plotter per solver in the solver's flavour (`rkfd4.SetupSimulationOverRegularGrid` + `rkfd4.PlotAsVTKFile`,
   `fv.SetupSimulationOverRegularGrid` + `fv.PlotAsVTKFile`). **Both solvers have to end up on the same mesh cells**: pass
   `volume_size = 1.01 * cell_size / patch_size` per solver, a bit bigger than intended, as the grid construction refines every cell whose edge length is
   greater than or equal to volume size times patch size. Use `2p+1` volumes for the finite volume solver if CCZ4 uses `p`.
4. `project.compiler = peano4.api.taskgraph.compiler.FuseSteps.translate`, global parameters (periodic boundary conditions for the gauge wave),
   `set_Peano4_installation`, `generate_Peano4_project`, `generate()`, `build()`. State the configure flags in the header
   (`--enable-exahype --enable-blockstructured --enable-loadbalancing --enable-exagrype`).
5. The debug mode visualises the task graphs (windows the user has to close). The release mode logs them as text.

#### Step 4: checks before you hand over (both flavours; the list is short on purpose)

- The two prep tasks exist and are wired as in the diagram of the chosen flavour.
- Each auxiliary set has `STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES`, and the prep task that fills it projects, restricts and rolls over.
  A missing projection does not fail: the halos silently contain the values of the previous exchange. Report to the user which mode each set has and that the projection is in place.
- Generation does not raise "depends on GLOBAL_END_OF_STEP task" (see Step 2), and the last task is a global end-of-step task.
- Out of the box, PDE terms ignore the auxiliary variables. Say so, ask which terms of which solver should read which entries
  (see how `exagrype.api.euler.Euler` hands auxiliary values to its PDE terms), and whether data should flow in both directions.
  With a purely uni-directional coupling, drop the prep task that is not needed. For the multiplicative flavour, the producer then runs first.
- Compare the generated code against the benchmark (task names, data uses, action-set order) and list the deviations.
  The additive flavour should match the benchmark. For the multiplicative one no benchmark exists yet (its host has been validated by generating the
  project only): offer to add a sibling of `gauge-wave-with-Euler` and to run the convergence check of `exahype2:integration-test` on it.

#### Check: do the copied data have to be projected onto the faces?

- `Mode.CELL_WISE` does not sit on the faces. No projection is needed, but the schemes currently reject such sets.
- `Mode.STATIONARY_ON_FACES` is projected once in the setup. If the exchange rewrites the data, they are not stationary:
  switch to `STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES` (outer coupling) or `CHANGING_ON_FACES` (data change per stage).
- `Mode.STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES` and `Mode.CHANGING_ON_FACES` **have to be projected onto the faces right after
  each exchange**, before anybody reconstructs a halo from them (project -> restrict hanging faces -> roll over).

#### Optional: data races

If both solvers read and write the same mesh data other than the exchanged sets, there might be a race. In the outer flavours, the persistent patches are only
read by the prep tasks, which finish before the consumer starts, so the answer is usually no. Ask the user anyway. If yes, add a temporary copy to a prep task and redirect one solver's access to it.


### Inner multiplicative coupling (placeholder)

@todo To be written. The host holds the numerical scheme and calls the solvers one after the other per
(Runge-Kutta) stage; see `coupling.dox`. 
The notes below were written before the split into outer and inner. 
`exagrype.api.FOSOSolver` is a worked example of this type
(the reconstruction tasks are inserted per Runge-Kutta stage in front of the CCZ4 stage which consumes them).

`_couple()` makes tasks of one solver wait for tasks of the other one. The graphs are already
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


### Inner additive coupling (placeholder)

@todo To be revised. The notes below predate the distinction between outer and inner. They describe a coupling per
Runge-Kutta stage with one numerical scheme in the host.

The coupling's shared scheme asks each solver for its graph per Runge-Kutta
stage (a method of the hosted solver, as `FOSOReconstructionSolver.create_task_graph()`; `Solver` itself has no such hook), i.e. there is one instance of each graph per stage. In `_couple()`,
add a KickOff task (`GridEntity.GLOBAL_START_OF_STEP`) per stage which the first task of each solver's stage graph depends on. Make the 
KickOff of stage `k+1` depend on the sinks of stage `k`.
No worked example exists for this type at the moment: the former `CCZ4WithEuler` host with KickOff/WrapUp tasks per stage has been removed (see git history). Write the host from scratch.

If possible, display a rendering of the arising task graph to the user.

### Optional: end of time step coupling (global coupling)

For outer additive coupling, `SynchroniseTimeStepSizes` in the final `GLOBAL_END_OF_STEP` task already is this global
coupling (see above). The generic recipe below is for further, user-defined exchange.

If two solvers furthermore have to synchronise each other globally, then we have 
to insert a WrapUp task after the whole task graph which waits for the last tasks
of all the created subtasks. Note that these might fork in their own right, i.e. we 
have to find all tasks that are not predecessor of any other task. 
`peano4.api.taskgraph.Graph.sink_tasks()` does that, for the host's concatenated graph.

The WrapUp tasks only does a global action in unprepare. The Abstract solver base
class that's generated get a new functor the says synchronise, and the user is told
to supply the missing information. 


### Support for Runge-Kutta time stepping

(Outer additive coupling: nothing to do. Each solver runs its stages internally, and the host's tasks sit
in front of the first and behind the last task of the whole time step. The paragraph below concerns the other types.)

As we always want to support multi-step time stepping, too, we have to reread all
of the descriptions above re time steps as "subgraph per time step". The host
instantiates the subgraph of each hosted solver once per stage (through a method of the solver, not through a `Solver` hook), 
so the same task chain exists several times, once per stage, and the multiplicative scheme ties each instance to the 
matching stage of the consumer (`FOSOSolver._couple()`). For the additive scheme, WrapUp and KickOff 
might have to be inserted multiple times at the corresponding markers. Tasks at the stages are 
looked up in the concatenated graph in `_couple()`.


## Optional extras (only on request)

The sections below go beyond a minimal coupling. Do not start them on your own initiative.

### Optimise (outer additive coupling)


If we do temporary copies, do a C++ source code analysis and try to find out which 
unknowns of the copied data are actually read. In principle, the developer could now
narrow down the auxiliary variables. This required a code rewrite. It is easier to 
only copy those unknowns which are actually used. The analysis needs a confirmation 
by the user if this is all correct. 

### Wrap up question

Analyse if both solvers use some input data from the other solver. If not, we work
with uni-directional coupling. Ask the user if they want to insert or append certain
coupling tasks which introduce bi-directional coupling between fields, or if some
key operations need alteration.

Often, the outcome (primary variables) of one task feed into the right-hand sides 
of coupled solvers. If this is the case and if we employ additive coupling, then
we might need to rerun the dependency analysis and introduce temporary copies.


