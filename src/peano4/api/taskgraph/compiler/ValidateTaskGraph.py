# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from typing import Dict, List, Set

from peano4.api.solversteps.ActionSet import ActionSet

from ..Graph import Graph
from ..Task import Task

OPERATION_PREPARE_TRAVERSAL = "prepareTraversal"
OPERATION_UNPREPARE_TRAVERSAL = "unprepareTraversal"

MARKS_OF_EVENTS: Dict[str, "Task.GridEntity"] = {
    ActionSet.OPERATION_BEGIN_TRAVERSAL: Task.GridEntity.GLOBAL,
    ActionSet.OPERATION_END_TRAVERSAL: Task.GridEntity.GLOBAL,
    OPERATION_PREPARE_TRAVERSAL: Task.GridEntity.GLOBAL,
    OPERATION_UNPREPARE_TRAVERSAL: Task.GridEntity.GLOBAL,
    ActionSet.OPERATION_TOUCH_VERTEX_FIRST_TIME: Task.GridEntity.VERTEX,
    ActionSet.OPERATION_TOUCH_VERTEX_LAST_TIME: Task.GridEntity.VERTEX,
    ActionSet.OPERATION_CREATE_PERSISTENT_VERTEX: Task.GridEntity.VERTEX,
    ActionSet.OPERATION_DESTROY_PERSISTENT_VERTEX: Task.GridEntity.VERTEX,
    ActionSet.OPERATION_DESTROY_HANGING_VERTEX: Task.GridEntity.VERTEX,
    ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME: Task.GridEntity.FACE,
    ActionSet.OPERATION_TOUCH_FACE_LAST_TIME: Task.GridEntity.FACE,
    ActionSet.OPERATION_CREATE_PERSISTENT_FACE: Task.GridEntity.FACE,
    ActionSet.OPERATION_DESTROY_PERSISTENT_FACE: Task.GridEntity.FACE,
    ActionSet.OPERATION_DESTROY_HANGING_FACE: Task.GridEntity.FACE,
    ActionSet.OPERATION_DESTROY_CELL: Task.GridEntity.FACE,
    ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME: Task.GridEntity.CELL,
    ActionSet.OPERATION_TOUCH_CELL_LAST_TIME: Task.GridEntity.CELL,
    ActionSet.OPERATION_CREATE_CELL: Task.GridEntity.CELL,
    ActionSet.OPERATION_CREATE_HANGING_FACE: Task.GridEntity.CELL,
    ActionSet.OPERATION_CREATE_HANGING_VERTEX: Task.GridEntity.CELL,
}
"""!

The mark an event contributes to the entity of the task which holds the action set

Cells interpolate (create the newly created cells, hanging faces and hanging
vertices), faces restrict (destroy cells and hanging faces), and a vertex
destroys its hanging vertices, besides the touch events.

"""

EVENTS_OF_OPERATIONS: Dict[str, "Task.TraversalEvent"] = {
    OPERATION_PREPARE_TRAVERSAL: Task.TraversalEvent.PREPARE_TRAVERSAL,
    ActionSet.OPERATION_BEGIN_TRAVERSAL: Task.TraversalEvent.BEGIN_TRAVERSAL,
    ActionSet.OPERATION_TOUCH_VERTEX_FIRST_TIME: Task.TraversalEvent.VERTEX_FIRST,
    ActionSet.OPERATION_CREATE_PERSISTENT_VERTEX: Task.TraversalEvent.VERTEX_FIRST,
    ActionSet.OPERATION_CREATE_HANGING_VERTEX: Task.TraversalEvent.VERTEX_FIRST,
    ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME: Task.TraversalEvent.FACE_FIRST,
    ActionSet.OPERATION_CREATE_PERSISTENT_FACE: Task.TraversalEvent.FACE_FIRST,
    ActionSet.OPERATION_CREATE_HANGING_FACE: Task.TraversalEvent.FACE_FIRST,
    ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME: Task.TraversalEvent.CELL_FIRST,
    ActionSet.OPERATION_CREATE_CELL: Task.TraversalEvent.CELL_FIRST,
    ActionSet.OPERATION_TOUCH_CELL_LAST_TIME: Task.TraversalEvent.CELL_LAST,
    ActionSet.OPERATION_DESTROY_CELL: Task.TraversalEvent.CELL_LAST,
    ActionSet.OPERATION_TOUCH_FACE_LAST_TIME: Task.TraversalEvent.FACE_LAST,
    ActionSet.OPERATION_DESTROY_PERSISTENT_FACE: Task.TraversalEvent.FACE_LAST,
    ActionSet.OPERATION_DESTROY_HANGING_FACE: Task.TraversalEvent.FACE_LAST,
    ActionSet.OPERATION_TOUCH_VERTEX_LAST_TIME: Task.TraversalEvent.VERTEX_LAST,
    ActionSet.OPERATION_DESTROY_PERSISTENT_VERTEX: Task.TraversalEvent.VERTEX_LAST,
    ActionSet.OPERATION_DESTROY_HANGING_VERTEX: Task.TraversalEvent.VERTEX_LAST,
    ActionSet.OPERATION_END_TRAVERSAL: Task.TraversalEvent.END_TRAVERSAL,
    OPERATION_UNPREPARE_TRAVERSAL: Task.TraversalEvent.UNPREPARE_TRAVERSAL,
}
"""!

Where an operation occurs within a mesh traversal. Creating something belongs to its first touch, destroying it to its last touch.

"""

ALL_EVENT_OPERATIONS: List[str] = [
    value for name, value in vars(ActionSet).items() if name.startswith("OPERATION_")
]


def _is_not_empty(code) -> bool:
    """!

    True if the C++ snippet holds more than whitespace and comments. The
    default implementation of an action set's events is a to-do comment.

    """
    import re

    if not isinstance(code, str):
        return False
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    return code.strip() != ""


def events_of_action_set(action_set: ActionSet) -> List[str]:
    """!

    Names of the events for which the action set yields code

    @param action_set: peano4.api.solversteps.ActionSet

    @return [str]
      Event names (the OPERATION_ constants, prepareTraversal and
      unprepareTraversal) in a stable order.

    """
    result = [
        operation
        for operation in ALL_EVENT_OPERATIONS
        if _is_not_empty(action_set.get_implementation_of_operation(operation))
    ]
    for name, getter in (
        (OPERATION_PREPARE_TRAVERSAL, "get_implementation_of_prepareTraversal"),
        (OPERATION_UNPREPARE_TRAVERSAL, "get_implementation_of_unprepareTraversal"),
    ):
        if name not in result and _is_not_empty(getattr(action_set, getter)()):
            result.append(name)
    return result


def derive_marks_and_events(action_sets):
    """!

    The marks (VERTEX, FACE, CELL, GLOBAL) and the traversal events of all operations the action sets use

    @param action_sets: [peano4.api.solversteps.ActionSet]

    @return (peano4.api.taskgraph.Task.Task.GridEntity, frozenset)
      The union of the marks (might be empty, a Flag with value 0), and the
      set of Task.TraversalEvent.

    """
    marks = Task.GridEntity(0)
    events = set()
    for action_set in action_sets:
        for operation in events_of_action_set(action_set):
            marks |= MARKS_OF_EVENTS[operation]
            events.add(EVENTS_OF_OPERATIONS[operation])
    return marks, frozenset(events)


def violations_of_task(task: Task) -> List[str]:
    """!

    Describe all events of a start-of-step or end-of-step task which are mesh events

    These tasks are global and do not work on the mesh. The other tasks
    derive their entity from their action sets
    (@ref peano4.api.taskgraph.Task.Task.entity).

    @param task: peano4.api.taskgraph.Task.Task

    @return [str]
      One entry "<action set name>::<event>" per violation. Empty if the
      task is valid.

    """
    if task.declared_entity not in (
        Task.GridEntity.GLOBAL_START_OF_STEP,
        Task.GridEntity.GLOBAL_END_OF_STEP,
    ):
        return []
    result = []
    for action_set in task.action_sets:
        for event in events_of_action_set(action_set):
            if MARKS_OF_EVENTS[event] != Task.GridEntity.GLOBAL:
                result.append(f"{action_set.get_action_set_name()}::{event}")
    return result


def validate(graph: Graph):
    """!

    Validate the task graph

    All task graph compilers call this routine first. The entity of a task is
    derived from the events its action sets use
    (@ref peano4.api.taskgraph.Task.Task.entity). The only thing which can go
    wrong is a task declared as GLOBAL_START_OF_STEP or GLOBAL_END_OF_STEP:
    These tasks may only use the traversal events, as they do not work on the
    mesh and have no traversal of their own.

    @param graph: peano4.api.taskgraph.Graph

    @exception AssertionError
      If at least one task violates the rule. The message lists all
      violations.

    """
    messages = []
    for task in graph.tasks:
        violations = violations_of_task(task)
        if violations:
            messages.append(f"  Task '{task.name}' ({task.declared_entity.label}): " + ", ".join(violations))
    assert not messages, (
        "Task graph is invalid: Tasks of the start or the end of the step use mesh events. "
        "Split the task into a global task and a task for the mesh.\n" + "\n".join(messages)
    )
