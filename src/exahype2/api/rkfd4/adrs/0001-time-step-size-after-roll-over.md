# ADR 0001: Compute the time step size only after the final roll over

Status: accepted
Date: 2026-11-08

## Context

The task `Compute time step size` (`GLOBAL_END_OF_STEP`) turns the reduced maximum
eigenvalue into the next time step size and advances the time stamp. Its only data
dependency is the reduction of the maximum eigenvalue, i.e. the task could run as
soon as the reduction has finished, in parallel to the projection onto the faces,
the restriction and the roll over of the last stage.

The roll over tasks, however, may need the current time step size. The Sommerfeld
boundary conditions advance the ghost cells within the roll over from the time stamp of
one stage to the one of the next stage (they call `getMinTimeStepSize()`).
This is not visible in the data the tasks declare.

## Decision

`Compute time step size` depends on the roll over of the last stage
(`previous_stage_task`) and, with adaptive time stepping, additionally on the
reduction of the maximum eigenvalue. It never runs before the final roll over.

## Consequences

- Some roll over schemes require the valid current time step size. Whoever relaxes this
  dependency (to shorten the task graph) has to ensure that no roll over reads
  the time step size or time stamp afterwards. The task graph does not detect
  this; the roll over then silently uses the new step.
- The task acts in `unprepareTraversal()` only, so the dependency costs no extra
  mesh sweep: the task fuses with the traversal that holds the final roll over.
- The same holds for the Finite Volume schemes, where the roll over is part of
  the projection onto the faces.
