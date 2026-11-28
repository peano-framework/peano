# ADR 0007: The setup of the second-order solver reconstructs the derivatives

Status: accepted
Date: 2026-11-28

## Context

`exagrype.api.FOSOCCZ4Solver` evolves 25 unknowns and holds the 33 first derivatives
as auxiliary variables, which `exagrype.api.FOSOReconstructionSolver` recomputes in front
of each stage. The initial condition used to fill all 58 entries: A wrapper ran a
58-entry routine on a local array and split the outcome into unknowns and derivatives.
For this, `initPointWisely()` received a pointer to the auxiliary variables.

The Finite Volume setup now evaluates the initial condition within a generated kernel
which knows the unknowns only. With auxiliary variables, the generated code did not
compile.

## Decision

- `initPointWisely(x, Q)` sets the unknowns, and receives no auxiliary variables, in all
  of ExaHyPE. Auxiliary variables whose values come from the user get a term of their own,
  `exahype2.api.InitAuxiliaryVariablesPointWisely`, which the setup calls in a task
  behind the initial condition.
- The derivatives need no such term, as they follow from the unknowns, but from those of
  the neighbouring grid cells, too. `exagrype.api.FOSOSolver` appends three tasks to the
  setup, behind the projection of the unknowns onto the faces: reconstruct the halos of
  the unknowns, apply the stencil of the scheme, and project the derivatives onto the
  faces (with restriction and roll over). It is the chain of a time step without the
  reconstruction of the old derivatives, which the stencil does not read.
- The initial condition of the second-order solver fills 25 entries. The routines in
  `exagrype::initialvalues` come in two variants, `...EvolvedVariables()` with 25 entries
  and the first-order one, which calls the former and adds the analytical derivatives.
- The solver rejects the 58-entry routines of `exagrype.api.initialvalues`, which would
  write beyond the end of Q. It cannot check initial conditions in C++ of the user.

## Consequences

- At t=0, the derivatives are the discrete ones of the scheme, no longer the analytical
  ones. The solution does not change, as each stage reconstructs the derivatives before
  it reads them.
- The setup needs additional mesh traversals.
- A grid construction of the CCZ4 solver has to provide the task which projects onto the
  faces (`Solver.GridConstruction.PROJECT_ON_FACES_TASK_NAME`).
- The mesh adaptation during the time stepping does not reconstruct the derivatives of
  new cells. The next stage does so before the solver reads them, but a plot or a
  refinement criterion in between would see undefined values.
- Setups which wrote auxiliary variables in `initPointWisely()` have to move this code
  into `InitAuxiliaryVariablesPointWisely`. The generated C++ does not compile
  otherwise, as the pointers are not declared anymore.
