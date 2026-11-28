# ADR 0003: The roll over of the faces is part of the projection onto the faces

Status: accepted
Date: 2026-11-09

## Context

The projection onto the faces writes the update of the face data
(`_patch_overlap_update`), and the roll over copies it into the persistent face
data of the current time step. The update is declared `TEMPORARY`: Peano creates
it at the beginning of a mesh traversal and destroys it at the end, i.e. it is
neither loaded nor stored. This is only valid if projection and roll over
run in the same traversal.

`RusanovWithExplicitEuler` modelled both as tasks of their own (`Project on
faces`, `Roll over faces`), relying on the task graph compiler to fuse them. The
producer-consumer compiler (`CellTaskProducerConsumer.translate_with_fusion`) does
not: It fuses by dependency layers, and the roll over falls into the layer of the
eigenvalue consumer, which must not share a traversal with the eigenvalue
producer. The roll over then ran one traversal after the projection and copied a
freshly allocated, uninitialised update into the face data. The first time step read
garbage in all halos, and the time step size became infinite after a few steps
(valgrind: "uninitialised value created by a heap allocation in
`EulerPointSourceQUpdate()`").

## Decision

The roll over is an action set of the task `Project on faces`, as it has been
all along in `RusanovWithOptimisticExplicitEuler` and in the setup. It acts after
the projection and the restriction of the hanging faces, i.e. it copies the final coarse faces.
`Compute global CFL condition` depends on `Project on faces`
(ADR 0002 holds, with this task standing in for `Roll over faces`).

## Consequences

- Whoever splits the roll over off again has to make the update non-temporary,
  i.e. persist it between the traversals, or fuse both tasks reliably. Otherwise
  the roll over copies garbage and nothing detects it in a release build.
- The explicit scheme has no task `Roll over faces` anymore.
- Runge-Kutta Finite Differences follow the same pattern
  (`src/exahype2/api/rkfd4/adrs/0004-roll-over-is-part-of-the-projection.md`), and so do the FOSO
  derivatives of ExaGRyPE. Their separate roll over tasks also produced `dt=inf` as soon as the compiler
  honoured the events of the tasks (`src/peano4/api/taskgraph/adrs/0002-fusion-follows-first-and-last-event.md`):
  the roll over acts in `touchFaceLastTime()`, and the node of the max eigenvalue task and of the roll over
  could not follow the projection.
