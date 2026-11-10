# ADR 0001: Unit tests query names from the code under test

Status: accepted
Date: 2026-11-08

## Context

Many unit tests inspect generated task graphs and look tasks up by their name, e.g.
`tasks["Probe_Dynamic mesh refinement"]`. The names are defined in the Python API
(`peano4.api.taskgraph.Task(name=...)`), and a test that repeats the string
duplicates this definition. When the task `Dynamic mesh refinement` became
`Dynamic mesh refinement and load balancing`, ten tests broke although the
behaviour they check was unchanged.

## Decision

Whenever a name (task name, tag, file name, ...) is fixed and used by tests, the
class that defines it holds it as a static (class) field, the code under test builds
the name from this field, and the unit tests read the field instead of
repeating the string. Example:

    # src/exahype2/api/Solver.py, class Solver.NumericalScheme
    DYNAMIC_MESH_REFINEMENT_TASK_NAME = "Dynamic mesh refinement and load balancing"

    # tests
    tasks[f"Probe_{exahype2.api.Solver.NumericalScheme.DYNAMIC_MESH_REFINEMENT_TASK_NAME}"]

The field holds the part of the name which does not depend on the instance (the
solver name prefix is added by the caller).

This applies to every unit test we add from now on. We do not retrofit the
existing unit tests, i.e. they keep their literal names until we touch them for
another reason.

## Consequences

- A rename touches one line. The tests keep checking structure (dependencies,
  fusion, access) instead of spelling.
- The tests still pin behaviour: a test that wants to assert that a name did
  not change on purpose (an interface to the outside world, such as a file format
  or a name which users reference) keeps the literal.
- The class that defines a name acquires a public field, which is part of its
  interface. Classes that only share a name by convention (such as the generator
  classes next to the schemes) refer to the field of the class that owns it.
