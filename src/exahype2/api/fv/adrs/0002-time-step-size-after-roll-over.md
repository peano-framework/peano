# ADR 0002: Compute the global CFL condition only after the roll over of the faces

Status: accepted
Date: 2026-11-09

## Context

The task `Compute global CFL condition` (`GLOBAL_END_OF_STEP`) turns the reduced maximum
eigenvalue into the next time step size and advances the time stamp. Its only data
dependency is the reduction of the maximum eigenvalue (ADR 0001).

The roll over of the faces, however, may need the current time step size. The Sommerfeld
boundary conditions advance the ghost volumes within the roll over (they call
`getMinTimeStepSize()`). This is not visible in the data the tasks declare. The
Runge-Kutta Finite Differences resolve the same issue in
`exahype2/api/rkfd4/adrs/0001-time-step-size-after-roll-over.md`; its text announced that
the same holds for Finite Volumes, but the FV task graphs did not model it.

## Decision

In both `RusanovWithExplicitEuler` and `RusanovWithOptimisticExplicitEuler`, the roll over
is part of `Project on faces` (ADR 0003), and `Compute global CFL condition` depends on
that task. It never runs before the roll over has
terminated, in addition to its dependency on the eigenvalue reduction.

## Consequences

- Whoever relaxes this dependency has to ensure that no roll over reads the time step size
  or time stamp afterwards. The task graph does not detect this; the roll over then
  silently uses the new step.
- The task acts in `unprepareTraversal()` only, i.e. the dependency costs no extra mesh
  sweep: the task fuses with the traversal that holds the roll over.
