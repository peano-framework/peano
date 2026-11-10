# ADR 0001: Interpolation and restriction are combined with the tasks that use face data

Status: accepted (implementation in progress: interpolation done, done for FV and Runge-Kutta; open for the ExaGRyPE FOSO coupling)
Date: 2026-11-09

## Context

All numerical schemes of `exahype2.api` (Finite Volumes, optimistic Finite Volumes,
Finite Differences with Runge-Kutta, and whatever follows) exchange data between
neighbouring patches through faces. On adaptive meshes, a face next to a coarser
neighbour is hanging: before anybody reads it, it has to be filled by interpolating the
coarse face data (`toolbox.blockstructured.api.InterpolateHangingFaces`). After
someone has written fine face data, it has to be restricted onto the coarse face
(`toolbox.blockstructured.api.RestrictHangingFaces`).
Interpolation and restriction as tasks of their own (as, for example, the
`Interpolate hanging faces stage k` tasks of the Runge-Kutta scheme) cost a mesh
traversal each, unless the task graph compiler happens to fuse them, and they
spread the knowledge on which faces are valid when over the schemes.

## Decision

**Interpolation and restriction never exist as stand-alone tasks.**

Whenever the action set of a task uses face data, the task also carries the
interpolate and restrict action sets for these face data:

- A task which reads face data (halo reconstruction, Riemann solve, ...) carries
  `InterpolateHangingFaces`, so the hanging faces are valid by the time the action set
  runs.
- A task which writes face data (projection of the solution onto the faces, ...) carries
  `RestrictHangingFaces`, so the coarse faces are up to date once the task has
  finished.

The rule applies to all numerical schemes, i.e. a scheme's task graph builder must not
add interpolation or restriction tasks. If a task needs the face data of several
data sets, it carries the action sets of all of them.

### Auxiliary variables

The rule holds for the unknowns and for the auxiliary variables, with a different
nuance, as auxiliary variables are not necessarily held by the faces:

- Whenever a task **reconstructs a patch** from auxiliary variables, it uses
  `InterpolateHangingFaces` for them. (Only sets that are
  not cell-wise have faces in the first place.)
- Whenever a task **writes to a patch** of auxiliary variables, it uses
  `RestrictHangingFaces` for them, if and only if the auxiliary data are changing and held by
  the faces (`AuxiliaryVariables.Mode.CHANGING_ON_FACES` and
  `STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES`, whose writer acts between the time steps). Auxiliary variables without face data need no restriction.

## Consequences

- Interpolation is folded into the reconstruction tasks of FV, of Runge-Kutta Finite
  Differences and of the auxiliary variables (done). Task graphs get shorter, the
  explicit FV time step needs two instead of three mesh traversals, and dependencies
  between interpolation and reconstruction vanish.
- Finite Volumes and Runge-Kutta host the restriction in the projection task (done).
- Open: the ExaGRyPE FOSO coupling (`exagrype.api.FOSOSolver.couple()`) still looks up
  the interpolation tasks by name, and the FOSO reconstruction solver still has
  stand-alone interpolation and restriction tasks.
- The interpolation scheme and the restriction scheme remain attributes of the
  numerical scheme and are handed to the action sets of the tasks.
- A new scheme or task using face data does not decide on its own how hanging faces are
  treated: it follows the rule. Reviewers can check the rule by looking for tasks that
  hold nothing but an interpolate or restrict action set.
