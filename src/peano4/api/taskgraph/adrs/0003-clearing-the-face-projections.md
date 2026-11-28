# ADR 0003: The clearing of the face projections is a task right in front of the projection

Status: accepted
Date: 2026-11-24

## Context

`RestrictHangingFaces` with averaging accumulates the fine data onto the coarse faces. The faces
have to be cleared before the projection writes them. `RestrictHangingFaces` did this in `touchFaceFirstTime()`
itself. With ADR 0002 (fusion follows the first and the last event), a task with a `touchFaceFirstTime()`
cannot follow a task with a `touchCellFirstTime()` in the same traversal, i.e. the projection could not follow the cell update.

## Decision

`ClearFaceProjections` (`toolbox.blockstructured.api.ClearFaceProjections`) clears the faces in a task of
its own, which `add_clearing_tasks()` adds in front of each projection with averaging. It runs before the compilation
(`exahype2.api.Project.generate_Peano4_project`), i.e. no scheme has to model it. The task waits for the same tasks
as the projection, and the projection waits for it.

The update patch which the projection writes is `TEMPORARY`: it lives within one mesh traversal.
Clearing and projection thus have to share a traversal.

## Failed attempts

- The clearing task as sibling of the projection's dependencies (waiting for their dependencies), to fuse
  it into the traversal of the cell update. In Runge-Kutta, the task between them is split into producer and consumer
  by the producer-consumer compiler: clearing ended up in the traversal of the producer, the projection after the consumer,
  and the projection accumulated onto uninitialised memory (`dt=inf` after the first step).
- Merging clearing and projection into the same task would have worked, but the clearing acts in `touchFaceFirstTime()`
  and the projection in `touchCellFirstTime()`: it would force each task which acts in a face event to lead
  its traversal.

## Consequences

- The compiler does not detect that a `TEMPORARY` patch is used in two traversals. Whoever adds a task which uses the
  update patch has to make sure that the compiler cannot separate it from the projection. All tasks which write or
  read the patch are action sets of the projection (projection, restriction, roll over) or wait for the same tasks
  (clearing).
- Tasks which are independent of the projection (the reduction of the eigenvalue) fuse with it whatever their events
  (ADR 0002), i.e. the explicit Finite Volume step needs two traversals.
