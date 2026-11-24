# ADR 0002: Two tasks fuse if the last event of the first is not after the first event of the second

Status: proposed (implemented on branch peano-event-spans, not merged)
Date: 2026-11-24

## Context

ADR 0001 synthesises the entity marks of a task. FuseSteps ordered tasks by their primary
entity (the highest mark), i.e. a task which acts in `touchFaceLastTime()` and in
`touchCellFirstTime()` counted as a cell task. The compiler did not know when a task acts.

## Decision

A mesh traversal invokes the events in the order `prepareTraversal`, `beginTraversal`,
`touchVertexFirstTime`, `touchFaceFirstTime`, `touchCellFirstTime`, `touchCellLastTime`,
`touchFaceLastTime`, `touchVertexLastTime`, `endTraversal`, `unprepareTraversal`
(`Task.TraversalEvent`). Creating a vertex, face or cell counts as its first touch, destroying it
as its last touch (`ValidateTaskGraph.EVENTS_OF_OPERATIONS`).

`Task.events` holds the events in which the action sets of a task are not empty. `Task.first_event` and
`Task.last_event` are the smallest and largest of them. `Task.event_label` names the pair, e.g.
`VERTEX_FIRST_CELL_LAST`, or `CELL_FIRST` if both coincide. A task without events acts in the first
event of its declared entity.

Two tasks A and B can share a traversal if B does not depend on A (directly or through other tasks), or if the last event of A is not after the first event of B. Tasks which a scheme models as parallel are thus always fusable with respect to the event order.
A conflict on face (vertex) data between two fused tasks is ignored if the earlier task acts in face
(vertex) events only.

## Consequences

- The compiler fuses tasks which the entity order kept apart (a task with `touchFaceLastTime` followed by another one with
  `touchFaceLastTime`), and it separates tasks which the entity order wrongly fused: a task which acts in
  `touchFaceFirstTime()` (the clearing of the coarse faces by `RestrictHangingFaces`) cannot follow a task which acts in `touchCellFirstTime()`.
  The explicit Finite Volume time step then needs three instead of two mesh traversals.
- The note on the assumption "a VERTEX or FACE task acts in the first-touch events" is obsolete.
