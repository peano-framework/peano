# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import re
from typing import Dict, List, Set

from peano4.api.solversteps.ActionSet import ActionSet

from ..Graph import Graph
from ..Task import Task

OPERATION_PREPARE_TRAVERSAL = "prepareTraversal"
OPERATION_UNPREPARE_TRAVERSAL = "unprepareTraversal"

GLOBAL_OPERATIONS: Set[str] = {
    ActionSet.OPERATION_BEGIN_TRAVERSAL,
    ActionSet.OPERATION_END_TRAVERSAL,
    OPERATION_PREPARE_TRAVERSAL,
    OPERATION_UNPREPARE_TRAVERSAL,
}
"""! Events which only tasks labelled with one of the GLOBAL entities may use"""

ALLOWED_OPERATIONS: Dict["Task.GridEntity", Set[str]] = {
    Task.GridEntity.VERTEX: {
        ActionSet.OPERATION_TOUCH_VERTEX_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_VERTEX_LAST_TIME,
    },
    Task.GridEntity.FACE: {
        ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_FACE_LAST_TIME,
    },
    Task.GridEntity.CELL: {
        ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_CELL_LAST_TIME,
        ActionSet.OPERATION_CREATE_HANGING_FACE,
        ActionSet.OPERATION_DESTROY_HANGING_FACE,
        ActionSet.OPERATION_CREATE_HANGING_VERTEX,
        ActionSet.OPERATION_DESTROY_HANGING_VERTEX,
        ActionSet.OPERATION_CREATE_CELL,
        ActionSet.OPERATION_DESTROY_CELL,
    },
}
"""! Events a task labelled with the respective entity may use. GLOBAL tasks: GLOBAL_OPERATIONS"""

ALL_EVENT_OPERATIONS: List[str] = [
    value for name, value in vars(ActionSet).items() if name.startswith("OPERATION_")
]


def _is_not_empty(code) -> bool:
    """!

    True if the C++ snippet holds more than whitespace and comments. The
    default implementation of an action set's events is a to-do comment.

    """
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


def violations_of_task(task: Task) -> List[str]:
    """!

    Describe all events of the task's action sets which the task's entity does not permit

    @param task: peano4.api.taskgraph.Task.Task

    @return [str]
      One entry "<action set name>::<event>" per violation. Empty if the
      task is valid.

    """
    result = []
    for action_set in task.action_sets:
        for event in events_of_action_set(action_set):
            if task.entity.is_global:
                allowed = event in GLOBAL_OPERATIONS
            else:
                allowed = event in ALLOWED_OPERATIONS[task.entity]
            if not allowed:
                result.append(f"{action_set.get_action_set_name()}::{event}")
    return result


def validate(graph: Graph):
    """!

    Validate that every task's action sets only plug into events which match the task's entity

    All task graph compilers call this routine first. The rules are:

    - A task labelled CELL may only use touchCellFirstTime() and
      touchCellLastTime(). In addition, it may create and destroy cells and
      the hanging faces and vertices, i.e. interpolate and restrict
      (@ref exahype2.api.adrs).
    - A task labelled FACE may only use touchFaceFirstTime() and
      touchFaceLastTime(), a VERTEX task only touchVertexFirstTime() and
      touchVertexLastTime().
    - beginTraversal(), endTraversal(), prepareTraversal() and
      unprepareTraversal() require a GLOBAL, GLOBAL_START_OF_STEP or
      GLOBAL_END_OF_STEP task, and global tasks may use nothing else.

    A scheme which violates these rules has to split its task into
    several ones.

    @param graph: peano4.api.taskgraph.Graph

    @exception AssertionError
      If at least one task violates the rules. The message lists all
      violations.

    """
    messages = []
    for task in graph.tasks:
        violations = violations_of_task(task)
        if violations:
            messages.append(f"  Task '{task.name}' ({task.entity.name}): " + ", ".join(violations))
    assert not messages, (
        "Task graph is invalid: Action sets plug into events which their task's entity does not permit. "
        "Split the task into tasks per entity (GLOBAL tasks for the traversal events).\n" + "\n".join(messages)
    )
