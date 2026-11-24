# ADR 0001: The entity of a task is synthesised from the events its action sets use

Status: accepted
Date: 2026-11-24

## Context

A task is labelled with a grid entity, but its action sets can plug into any event. A task
labelled CELL could hold an action set acting in `touchFaceLastTime()` or in
`unprepareTraversal()`, i.e. the label did not tell what the task does, and the
producer-consumer compiler (which maps cell events onto background tasks) had to
guess whether a task is safe to split.

## Decision

The entity is a set of marks (`Task.GridEntity` is an `enum.Flag`): VERTEX, FACE, CELL and
GLOBAL. They combine freely (2x2x2x2 = 16 variants, e.g. FACE_CELL).
Nobody picks the combination: `Task.entity` is the union of the marks of all events which the
task's action sets yield code for (`ValidateTaskGraph.MARKS_OF_EVENTS`)
and of the entity the task is declared with, which is a lower bound.

| Mark   | Events                                                                                    |
| ------ | ----------------------------------------------------------------------------------------- |
| CELL   | `touchCellFirstTime/LastTime`, interpolation: `createCell`, `createHangingFace`, `createHangingVertex` |
| FACE   | `touchFaceFirstTime/LastTime`, restriction: `destroyHangingFace`, `destroyCell`           |
| VERTEX | `touchVertexFirstTime/LastTime`, `destroyHangingVertex`                                   |
| GLOBAL | `beginTraversal`, `endTraversal`, `prepareTraversal`, `unprepareTraversal`                |

The persistent create and destroy events of faces and vertices count as FACE and VERTEX.
Code which only consists of comments (the default of every event) does not count.
The table implements ADR `exahype2/api/adrs/0001`: A task which uses cell data carries
interpolation, a task writing face data carries restriction. The projection of the patches
onto the faces thus becomes a FACE_CELL task.

GLOBAL_START_OF_STEP and GLOBAL_END_OF_STEP say when a task runs. This cannot be derived and has to be
declared. `ValidateTaskGraph.validate()` runs first in every task graph compiler and
rejects such a task if its action sets use mesh events.

Compilers use the entity as follows:

- FuseSteps orders tasks by their primary entity, i.e. their highest mesh mark.
  A FACE_CELL task is a cell task. A task which only has the GLOBAL mark is appended to the
  traversal of its last dependency. A mesh task with the GLOBAL mark (e.g. CELL|GLOBAL)
  acts in the prepare/unprepare events of its own traversal.
- The producer-consumer compiler splits tasks which are CELL only. It spawns the
  `touchCellFirstTime()` and `touchCellLastTime()` of their action sets as tasks:
  The last task depends on the first task of the same cell and on the last tasks of all children.
  Each cell holds two task numbers per action set and an array of 3^d numbers (`childrenTasks`).
  A cell reserves its last number in `touchCellFirstTime()` and writes it into its
  parent's array, as the parent's `touchCellLastTime()` runs after all children have been visited.
  The first task of a cell depends on the first task of the parent. A last task whose guard does not hold is spawned
  as empty task, as the parent waits for it.

## Consequences

- Adding an event to an action set can change the entity of every task holding it,
  and with it the fusion and the splitting into producer and consumer.
- The entity is not available as a plain attribute: it is recomputed whenever the action sets change.
- The split of the load balancing into global tasks of its own (as in the schemes and setups) is not
  required anymore, but keeps the global events out of the producer-consumer pairs.
- Within the FuseSteps entity order, FACE_CELL counts as CELL, i.e. the face events of a task are not
  ordered against the cell events of the tasks it is fused with.
