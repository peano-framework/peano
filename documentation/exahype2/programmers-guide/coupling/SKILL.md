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

Create a new ExaHyPE solver with the given name. For each solver that it brings 
together, it aggregates an instance of this solver in the constructor.


## Volumetric coupling and data sets

If the coupling is volumetric, ask the user if auxiliary variables of one solver are
actually primary (evolved) variables of another one, or if solvers share auxiliary
variables accordingly. After that, change the data association accordingly.


## Algorithmic coupling

### Multiplicative solver coupling

generate_solver_realisation() now delegates this call to each aggregate. Ensure 
that the second solver appends its tasks to the task graph of the first one and 
does not replace the tasks. The first task of the second solver has to depend on 
the last task of the first solver. We create a chain of subtasks.

Ensure 
that the second solver appends its tasks to the task graph of the first one and 
does not replace the tasks. The first task of the second solver has to depend on 
the last task of the first solver. We create a chain of subtasks.

If possible, display a rendering of the arising task graph to the user.


### Additive solver coupling

generate_solver_realisation() now delegates this call to each aggregate. Each one
creates its own task graph. Introduce a new KickOff task as the very first one,
i.e. the first task of each solver's task graph depends on KickOff to have terminated.

If possible, display a rendering of the arising task graph to the user.

Ask from the user if any of the mesh data are read and written by two solvers.
In this case, there could be a data race. If this is the case, add to KickOff
a temporary copy of the respective data field and add an action set which befills
this copy. One of the solvers has to redirect their access to the copy. After the
task graph has finished, the copy can be discarded, too.


### End of time step coupling (global coupling)

If two solvers furthermore have to synchronise each other globally, then we have 
to insert a WrapUp task after the whole task graph which waits for the last tasks
of all the created subtasks. Note that these might fork in their own right, i.e. we 
have to find all tasks that are not predecessor of any other task. There should be a
peano.api.taskgraph routine for that.

The WrapUp tasks only does a global action in unprepare. The Abstract solver base
class that's generated get a new functor the says synchronise, and the user is told
to supply the missing information. 


### Support for Runge-Kutta time stepping

As we always want to support multi-step time stepping, too, we have to reread all
of the descriptions above re time steps as "subgraph per time step". This is only 
relevant for the additive scheme. It implies that WrapUp and KickOff might have to 
be inserted multiple times at the corresponding markers.


## Optimise (additive coupling only)

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


