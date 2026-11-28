# ADR 0004: The roll over of the faces is part of the projection onto the faces

Status: accepted
Date: 2026-11-27

## Context

In each stage, the projection onto the faces writes the update of the face data
(`_patch_overlap_update`), and the roll over copies it into the persistent face
data of the current time step. The update is declared `TEMPORARY`: it lives
within one mesh traversal only. Projection and roll over belong together.

The Runge-Kutta scheme modelled them as two tasks (`Project ... onto faces`,
`Roll over ... faces`) and relied on the task graph compiler to fuse them. To make
the compiler do so, the projection declared that it writes the face data of
the current time step, which only the roll over did, and the roll over declared
these data as temporary.

The `Sequential` compiler, the default of `exahype2.api.Project`, maps each task
onto a traversal of its own. The roll over then copied an uninitialised update,
and its traversal did not store the outcome. With `rk2fd4` and adaptive time
stepping, the Euler entropy wave (27 grid cells per axis) reached an infinite
time step size in its third computation of the step size. Finite Volumes had
the same problem with a producer-consumer compiler
(`src/exahype2/api/fv/adrs/0003-roll-over-is-part-of-the-projection.md`).

## Decision

The roll over is an action set of the task which projects onto the faces
(`CentralDifferencesWithRungeKutta.PROJECTION_TASK_NAME`). It follows the
restriction of the hanging faces, i.e. it copies the final coarse faces. The
task declares that it writes the face data of the current time step, and the
update as temporary.

## Consequences

- The scheme has no task `Roll over ... faces` anymore. What waited for it waits
  for the projection, such as `Compute time step size` (ADR 0001) and the end of a
  stage of the scalar field coupling (`exagrype.api.scalarfield`).
- The validity of the temporary update no longer depends on the compiler.
- With `FuseSteps` and the producer-consumer compilers, the number of mesh
  traversals, the data access of each traversal and its action sets are the same
  as before (compared for one to four stages, fixed and adaptive time step sizes,
  Dirichlet and Sommerfeld boundary conditions). `Sequential` saves one
  traversal per stage.
- A task with several action sets is never part of a parallel node of
  `FuseSteps`. The roll over hence does not run in parallel to action sets of
  other tasks anymore, such as the roll over of a coupled solver. With adaptive
  time step sizes, it lost the parallel flag it shared with the eigenvalue task of
  the same solver. Both act in different events, so they never ran concurrently.
- No other task can be inserted between the projection and the roll over. Such a
  task could put them into different traversals.
