# ADR 0001: The entity of a task determines the events its action sets may use

Status: accepted (implementation in progress: validator and producer-consumer done; the roll over of the Finite Volume and Runge-Kutta schemes still violates the rule)
Date: 2026-11-24

## Context

A task is labelled with a grid entity, but its action sets could plug into any event.
A task labelled CELL could hold an action set acting in `touchFaceLastTime()` or in
`unprepareTraversal()`. The producer-consumer compiler turns the cell events of a task into
background tasks. Events of other entities, and the traversal events, have no place in this
mapping, i.e. the compiler had to refuse to split such tasks
(`_has_events_after_touch_cell_first_time()`).

## Decision

`compiler/ValidateTaskGraph.validate()` runs first in every task graph compiler
(Sequential, FuseSteps and all producer-consumer variants) and aborts if

- a CELL task uses an event other than `touchCellFirstTime()` and `touchCellLastTime()`.
  As ADR `exahype2/api/adrs/0001` requires every cell task to carry interpolation and
  restriction, CELL tasks may also use `createHanging*`, `destroyHanging*`, `createCell` and
  `destroyCell`. No other entity may.
- a FACE (VERTEX) task uses an event other than `touchFace*Time()` (`touchVertex*Time()`);
- an action set uses `beginTraversal()`, `endTraversal()`, `prepareTraversal()` or
  `unprepareTraversal()` in a task which is not GLOBAL, GLOBAL_START_OF_STEP or
  GLOBAL_END_OF_STEP, or a global task uses any other event.

Code which only consists of comments (the default of every event) does not count.
Numerical schemes split tasks accordingly. The load balancing, for example, is a global task
of its own.

The producer-consumer compiler uses the rule:

- `touchCellFirstTime()` and `touchCellLastTime()` of an action set become tasks. The last
  task depends on the first task of the same cell and on the last tasks of all children.
- Each cell holds two task numbers per action set and an array of 3^d numbers
  (`childrenTasks`). A cell reserves its last number in `touchCellFirstTime()` and writes it into
  its parent's array, as the parent's `touchCellLastTime()` runs after all children have been
  visited. The first task of a cell depends on the first task of the parent.
- A last task whose guard does not hold is spawned as empty task, as the parent waits for it.

## Consequences

- Whoever adds an event to an action set has to check the entity of all tasks that hold the
  action set. The compilers report violations with the task, action set and event.
- A scheme with a face event inside a cell task (the roll over, which acts in `touchFaceLastTime()`)
  has to split the task. The face data written by the cell task then has to survive until the
  face task runs: ADR `exahype2/api/fv/adrs/0003` is affected.
- A repeating task which is followed by a global task (the load balancing in the setup) keeps the
  global task in its mesh sweep with FuseSteps, but not with Sequential, where the global task runs
  in a traversal of its own, after the repetitions.
