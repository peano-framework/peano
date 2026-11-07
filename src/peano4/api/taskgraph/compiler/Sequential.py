# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from typing import List

import re
import peano4.api
import peano4.api.solversteps

from ..Graph import Graph
from ..Task import Task
from .ReportCompiledStepSequence import report_compiled_step_sequence


def translate(graph: Graph, identifier: str, peano_project: "peano4.api.Project") -> List["peano4.api.solversteps.Step"]:
    """!

    Simplest possible task graph compiler

    Linearises the task graph into one mesh traversal (Observer) per task, in
    an order that respects every dependency: a task's Observer always comes
    after the Observers of all tasks it depends on. This throws away any
    parallelism the task graph might encode - it is the "no fusion, no
    reordering" baseline compiler that smarter compilers can be measured
    against.

    ## Implementation

    A straightforward, direct mapping: graph.ordered_tasks() linearises the
    graph's tasks into a valid topological order (a task always comes after
    every task it depends on - @see Graph.ordered_tasks() for how), and
    every task in that order becomes exactly one Observer, populated with
    that task's action sets and the cell/face/vertex data it declared via
    use_cell()/use_face()/use_vertex().

    Note that this function no longer derives any *further* use_cell()/
    use_face()/use_vertex() declarations itself - that is the job of
    @ref peano4.api.taskgraph.compiler.PropagateUseStatements.propagate_use_statements(),
    which has to run *before* translate() (it reasons about tasks, not the
    Steps translate() produces here, and about the composability of
    several graphs at once, which a single graph's own translate() call
    cannot see). By the time translate() runs, every task already carries
    every use_cell()/use_face()/use_vertex() declaration it will ever need.

    A peano4.api.Logger().new_section() call brackets the translation, and
    @ref peano4.api.taskgraph.compiler.ReportCompiledStepSequence.report_compiled_step_sequence()
    reports its outcome right after - a textual listing of the resulting
    sequence and its action sets under LoggerVerbosity.TEXT, or a flow chart
    in the same style as Graph.visualise() under
    LoggerVerbosity.VISUALISATION.

    @param graph: peano4.api.taskgraph.Graph
      The task graph to linearise. Assumed to be acyclic - translate() does
      not check for cycles and will recurse indefinitely on one.

    @param identifier: str
      Identifies which of the surrounding project's algorithmic phases this
      graph belongs to (e.g. "SetupSimulation", "TimeStep", "Plot"). Every
      generated Observer's name is prefixed with this identifier, joined by
      an underscore, so Observers coming from different phases never clash
      even if their tasks happen to share a name.

    @param peano_project: peano4.api.Project
      Not used by this compiler. @ref peano4.api.taskgraph.compiler

    @return [peano4.api.solversteps.Step]
      One Observer per task, ordered so that every task's Observer comes
      after the Observers of all tasks it depends on, carrying that task's
      action_sets (in the order they were added to the Task) plus the
      cell/face/vertex data use_cell()/use_face()/use_vertex() determined.
      Each Step's cell_data_access/face_data_access/vertex_data_access is
      also populated as a copy of the originating Task's, purely for
      reporting (@see
      peano4.api.taskgraph.compiler.ReportCompiledStepSequence) - nothing
      about the generated Observer depends on it.

    """
    order = graph.ordered_tasks()

    steps = []
    for task in order:
        # Turn every whitespace-separated word into its own name segment,
        # capitalising only its first letter - a blind .title() would also
        # lower-case the rest of the word, which mangles existing camel
        # case and acronyms (e.g. "EulerFVSmooth" would become
        # "Eulerfvsmooth", "CFL" would become "Cfl").
        # Words are separated by anything that C++ identifiers do not allow, too
        task_name = "_".join(
            word[:1].upper() + word[1:]
            for word in re.split(r"[^0-9A-Za-z_]+", task.name)
            if word
        )
        step = peano4.api.solversteps.Step(
            name=f"{identifier}_{task_name}",
            add_user_defined_actions=False,
        )
        step.has_to_repeat = task.has_to_repeat
        for cell_data in task.cell_data:
            step.use_cell(cell_data)
            step.cell_data_access[cell_data] = task.cell_data_access[cell_data]
        for face_data in task.face_data:
            step.use_face(face_data)
            step.face_data_access[face_data] = task.face_data_access[face_data]
        for vertex_data in task.vertex_data:
            step.use_vertex(vertex_data)
            step.vertex_data_access[vertex_data] = task.vertex_data_access[vertex_data]
        for action_set in task.action_sets:
            step.add_action_set(action_set)
        steps.append(step)

    peano4.api.Logger().new_section(f"Compile task graph '{identifier}'")
    report_compiled_step_sequence(steps, title=f"Compiled task graph for {identifier}")

    return steps
