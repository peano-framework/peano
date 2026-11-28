import peano4.api.solversteps
from peano4.api.taskgraph import Graph, Task
from peano4.api.taskgraph.compiler import FuseSteps

ActionSet = peano4.api.solversteps.ActionSet
Event = Task.TraversalEvent
CELL = Task.GridEntity.CELL


class _EventsActionSet:
    def __init__(self, name, *operations, prepare=False):
        self.name = name
        self.operations = operations
        self.prepare = prepare
        self.descend_invocation_order = 0
        self.parallel = False

    def get_action_set_name(self):
        return self.name

    def get_implementation_of_operation(self, operation_name):
        return "x;" if operation_name in self.operations else ""

    def get_implementation_of_prepareTraversal(self):
        return "x;" if self.prepare else ""

    def get_implementation_of_unprepareTraversal(self):
        return ""


def _task(name, *operations, prepare=False, dependencies=()):
    task = Task(name=name, entity=CELL)
    task.add_action_set(_EventsActionSet(name, *operations, prepare=prepare))
    for dependency in dependencies:
        task.add_dependency(dependency)
    return task


def _number_of_traversals(*tasks):
    graph = Graph()
    for task in tasks:
        graph.add_task(task)
    return len(FuseSteps.translate(graph, "Probe", None))


def test_label_is_first_and_last_non_empty_event():
    assert (
        _task(
            "t",
            ActionSet.OPERATION_TOUCH_VERTEX_FIRST_TIME,
            ActionSet.OPERATION_TOUCH_VERTEX_LAST_TIME,
        ).event_label
        == "VERTEX_FIRST_VERTEX_LAST"
    )
    assert (
        _task("t", ActionSet.OPERATION_TOUCH_VERTEX_FIRST_TIME).event_label
        == "VERTEX_FIRST"
    )
    assert (
        _task("t", ActionSet.OPERATION_TOUCH_VERTEX_LAST_TIME).event_label
        == "VERTEX_LAST"
    )


def test_events_in_between_do_not_show_in_the_label():
    task = _task(
        "t",
        ActionSet.OPERATION_TOUCH_VERTEX_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_CELL_LAST_TIME,
    )
    assert task.event_label == "VERTEX_FIRST_CELL_LAST"
    assert Event.CELL_FIRST in task.events


def test_creation_counts_as_first_and_destruction_as_last_touch():
    task = _task(
        "t",
        ActionSet.OPERATION_CREATE_HANGING_FACE,
        ActionSet.OPERATION_DESTROY_HANGING_FACE,
    )
    assert (task.first_event, task.last_event) == (Event.FACE_FIRST, Event.FACE_LAST)
    task = _task(
        "t", ActionSet.OPERATION_CREATE_CELL, ActionSet.OPERATION_DESTROY_HANGING_VERTEX
    )
    assert (task.first_event, task.last_event) == (Event.CELL_FIRST, Event.VERTEX_LAST)


def test_traversal_events_are_first_and_last_of_the_order():
    task = _task("t", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, prepare=True)
    assert task.event_label == "PREPARE_TRAVERSAL_CELL_FIRST"


def test_task_without_events_acts_in_the_first_event_of_its_declared_entity():
    task = Task(name="t", entity=CELL)
    task.add_action_set(_EventsActionSet("a"))
    assert task.events == {Event.CELL_FIRST}
    assert Task(name="t", entity=CELL).events == {Event.CELL_FIRST}
    assert Task(name="t", entity=Task.GridEntity.GLOBAL).events == frozenset()


def test_event_order_is_the_order_of_the_traversal():
    assert list(Event) == sorted(Event)
    assert Event.CELL_FIRST < Event.CELL_LAST < Event.FACE_LAST < Event.VERTEX_LAST


def test_tasks_fuse_if_last_event_of_first_is_not_after_first_event_of_second():
    a = _task("a", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)
    b = _task("b", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, dependencies=[a])
    assert _number_of_traversals(a, b) == 1
    c = _task("c", ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME)
    d = _task("d", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, dependencies=[c])
    assert _number_of_traversals(c, d) == 1


def test_task_with_late_event_does_not_fuse_with_task_with_earlier_event():
    a = _task(
        "a",
        ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_FACE_LAST_TIME,
    )
    b = _task("b", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, dependencies=[a])
    assert _number_of_traversals(a, b) == 2


def test_face_task_which_acts_in_last_touch_does_not_fuse_with_subsequent_cell_task():
    a = _task(
        "a",
        ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_FACE_LAST_TIME,
    )
    b = _task("b", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, dependencies=[a])
    assert _number_of_traversals(a, b) == 2


def test_face_last_touch_tasks_fuse_with_each_other():
    a = _task(
        "a",
        ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_FACE_LAST_TIME,
    )
    b = _task("b", ActionSet.OPERATION_TOUCH_FACE_LAST_TIME, dependencies=[a])
    assert _number_of_traversals(a, b) == 1


def test_independent_tasks_fuse_whatever_their_events_are():
    start = _task("start", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)
    late = _task(
        "late",
        ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_FACE_LAST_TIME,
        dependencies=[start],
    )
    middle = _task(
        "middle", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, dependencies=[start]
    )
    early = _task(
        "early", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, dependencies=[middle]
    )
    assert _number_of_traversals(start, late, middle, early) == 1


def test_dependent_task_still_waits_for_the_late_event_of_its_predecessor():
    start = _task("start", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)
    late = _task(
        "late",
        ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_FACE_LAST_TIME,
        dependencies=[start],
    )
    after_late = _task(
        "after late", ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, dependencies=[late]
    )
    assert _number_of_traversals(start, late, after_late) == 2
