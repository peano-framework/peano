# ADR 0002: The maximum eigenvalue is computed after the projection, not parallel to it

Status: accepted
Date: 2026-11-09

## Context

The task `Compute max eigenvalue` writes cell data and is therefore split by
`CellTaskProducerConsumer` into a producer, which spawns the kernel as a task, and a
consumer, which waits for it. The task graph compiler never fuses producer and consumer,
and `FuseSteps` treats all tasks of one dependency layer as one node.

If the eigenvalue task depends on `Complete time step` only, i.e. runs in parallel to
`Project solution onto faces`, its consumer sits in the same layer as
`Roll over solution faces`. The node holding the consumer cannot join the traversal of the
producer, and the roll over moves into the next mesh sweep with it. The projection writes
the `TEMPORARY` face data `QUpdate`; the roll over reads it one sweep later, when it is
uninitialised. With adaptive time stepping, the solution turned NaN after the second
time step and the next time step size was infinite.

## Decision

`Compute max eigenvalue` depends on `Project solution onto faces`. Producer and roll
over then share a layer, the consumer is one layer further, and projection and roll over
stay in one traversal as they do for the intermediate stages.

## Consequences

- The eigenvalue kernel starts after the projection of the cell has finished instead of
  next to it. Both only read the new solution, so nothing is lost but a little
  parallelism.
- Whoever adds a task that consumes a producer's outcome has to check that no
  `TEMPORARY` face data spans the traversals the producer-consumer split creates. The
  task graph compiler does not detect this.
