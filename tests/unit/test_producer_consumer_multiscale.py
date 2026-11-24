import peano4.api.datamodel
import peano4.api.solversteps
from peano4.api.taskgraph import Graph, Task
from peano4.api.taskgraph.compiler import CellTaskProducerConsumer
from peano4.api.taskgraph.compiler import ValidateTaskGraph

ActionSet = peano4.api.solversteps.ActionSet
CELL = Task.GridEntity.CELL
FACE = Task.GridEntity.FACE
GLOBAL = Task.GridEntity.GLOBAL


class _ActionSet:
    def __init__(self, name, first="x = 1;", last="", guard_last="true"):
        self.name = name
        self.bodies = {
            ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME: first,
            ActionSet.OPERATION_TOUCH_CELL_LAST_TIME: last,
        }
        self.guard_last = guard_last
        self.descend_invocation_order = 0
        self.parallel = False

    def get_action_set_name(self):
        return self.name

    def get_body_of_operation(self, operation_name):
        return self.bodies.get(operation_name, "")

    def get_guard_of_operation(self, operation_name):
        return self.guard_last if operation_name == ActionSet.OPERATION_TOUCH_CELL_LAST_TIME else "true"

    def get_implementation_of_operation(self, operation_name):
        return self.get_body_of_operation(operation_name)

    def get_implementation_of_prepareTraversal(self):
        return ""

    def get_implementation_of_unprepareTraversal(self):
        return ""

    def get_body_of_prepareTraversal(self):
        return ""

    def get_body_of_unprepareTraversal(self):
        return ""

    def get_body_of_createGridControlEventsForNextTraversal(self):
        return ""

    def get_implementation_of_createGridControlEventsForNextTraversal(self):
        return ""

    def get_constructor_body(self):
        return ""

    def get_destructor_body(self):
        return ""

    def get_static_initialisations(self, full_qualified_classname):
        return ""

    def get_attributes(self):
        return ""

    def get_includes(self):
        return ""

    def may_run_in_background_task(self):
        return True

    def user_should_modify_template(self):
        return False


class _Submodel:
    def __init__(self, name):
        self.name = name
        self.generator = None


def _split(*action_sets):
    task = Task(name="Probe", entity=CELL)
    for action_set in action_sets:
        task.add_action_set(action_set)
    task.use_cell(_Submodel("Q"), Task.DataAccess.WRITE)
    graph = Graph()
    graph.add_task(task)
    split_graph, split_tasks = CellTaskProducerConsumer._split_cell_tasks(graph)
    labels = peano4.api.datamodel.DaStGen2("Labels")
    CellTaskProducerConsumer._spawn_tasks(split_tasks, labels)
    _, producer, consumer = split_tasks[0]
    return producer, consumer, labels


def test_first_and_last_time_are_wrapped_into_one_action_set():
    producer, consumer, _ = _split(_ActionSet("A", last="y = 2;"))
    [wrapper] = producer.action_sets
    first = wrapper.get_implementation_of_operation(ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)
    last = wrapper.get_implementation_of_operation(ActionSet.OPERATION_TOUCH_CELL_LAST_TIME)
    assert "x = 1;" in first and "y = 2;" not in first
    assert "y = 2;" in last and "x = 1;" not in last
    assert "spawnTask" in first and "spawnTask" in last


def test_last_task_depends_on_first_task_and_children():
    producer, _, _ = _split(_ActionSet("A", last="y = 2;"))
    last = producer.action_sets[0].get_implementation_of_operation(ActionSet.OPERATION_TOUCH_CELL_LAST_TIME)
    assert "ActionSet0FirstTaskNumber" in last
    assert "ActionSet0ChildrenTaskNumbers" in last
    assert "ThreePowerD" in last


def test_child_registers_at_parent_and_depends_on_parents_first_task():
    producer, _, _ = _split(_ActionSet("A", last="y = 2;"))
    first = producer.action_sets[0].get_implementation_of_operation(ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)
    assert "coarseGridCellLabels.setProbeActionSet0ChildrenTaskNumbers" in first
    assert "coarseGridCellLabels.getProbeActionSet0FirstTaskNumber()>=0" in first
    assert "fineGridCellLabels.setProbeActionSet0ChildrenTaskNumbers( i, tarch::multicore::TaskEnumerator::NoNumber )" in first


def test_false_guard_of_last_task_still_spawns_empty_task():
    producer, _, _ = _split(_ActionSet("A", last="y = 2;", guard_last="false"))
    last = producer.action_sets[0].get_implementation_of_operation(ActionSet.OPERATION_TOUCH_CELL_LAST_TIME)
    assert "else" in last and "[]() mutable {}" in last


def test_action_set_without_last_time_spawns_no_last_task():
    producer, _, _ = _split(_ActionSet("A"))
    wrapper = producer.action_sets[0]
    assert wrapper.get_implementation_of_operation(ActionSet.OPERATION_TOUCH_CELL_LAST_TIME) == ""
    first = wrapper.get_implementation_of_operation(ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)
    assert "setProbeActionSet0LastTaskNumber( tarch::multicore::TaskEnumerator::NoNumber )" in first


def test_last_time_only_action_set_is_wrapped():
    producer, _, _ = _split(_ActionSet("A", first="", last="y = 2;"))
    assert isinstance(producer.action_sets[0], CellTaskProducerConsumer.SpawnCellTaskActionSet)


def test_consumer_waits_for_first_and_last_task():
    producer, consumer, _ = _split(_ActionSet("A", last="y = 2;"))
    [waiter] = consumer.action_sets
    first = waiter.get_implementation_of_operation(ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)
    assert first.count("waitForTask") == 2
    assert "LastTaskNumber" in first and "FirstTaskNumber" in first


def test_labels_hold_two_numbers_and_children_array_per_action_set():
    _, _, labels = _split(_ActionSet("A", last="y = 2;"), _ActionSet("B"))
    names = [attribute.name for attribute in labels.data._attributes] if hasattr(labels.data, "_attributes") else []
    for stem in ("ProbeActionSet0", "ProbeActionSet1"):
        for suffix in ("FirstTaskNumber", "LastTaskNumber", "ChildrenTaskNumbers"):
            assert stem + suffix in names


def test_second_action_set_chains_on_first_and_last_of_first():
    producer, _, _ = _split(_ActionSet("A", last="y = 2;"), _ActionSet("B", last="z = 3;"))
    first = producer.action_sets[1].get_implementation_of_operation(ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)
    last = producer.action_sets[1].get_implementation_of_operation(ActionSet.OPERATION_TOUCH_CELL_LAST_TIME)
    assert "getProbeActionSet0FirstTaskNumber()>=0" in first
    assert "getProbeActionSet0LastTaskNumber()>=0" in last


def test_cell_task_with_late_cell_event_is_split_now():
    task = Task(name="Probe", entity=CELL)
    task.add_action_set(_ActionSet("A", last="y = 2;"))
    assert CellTaskProducerConsumer._has_events_after_touch_cell_first_time(task) == []


class _Validated(_ActionSet):
    def __init__(self, name, events):
        super().__init__(name)
        self.events = events

    def get_implementation_of_operation(self, operation_name):
        return "x;" if operation_name in self.events else ""


def _violations(entity, *events, prepare=False):
    action_set = _Validated("A", events)
    if prepare:
        action_set.get_implementation_of_prepareTraversal = lambda: "x;"
    task = Task(name="T", entity=entity)
    task.add_action_set(action_set)
    return ValidateTaskGraph.violations_of_task(task)


def test_cell_task_may_only_touch_cells():
    assert _violations(CELL, ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, ActionSet.OPERATION_TOUCH_CELL_LAST_TIME) == []
    assert _violations(CELL, ActionSet.OPERATION_TOUCH_FACE_LAST_TIME) == ["A::touchFaceLastTime"]
    assert _violations(CELL, ActionSet.OPERATION_TOUCH_VERTEX_FIRST_TIME) == ["A::touchVertexFirstTime"]


def test_cell_task_may_interpolate_and_restrict_but_face_task_may_not():
    assert _violations(CELL, ActionSet.OPERATION_CREATE_HANGING_FACE, ActionSet.OPERATION_DESTROY_HANGING_FACE) == []
    assert _violations(FACE, ActionSet.OPERATION_CREATE_HANGING_FACE) == ["A::createHangingFace"]


def test_face_task_may_only_touch_faces():
    assert _violations(FACE, ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME, ActionSet.OPERATION_TOUCH_FACE_LAST_TIME) == []
    assert _violations(FACE, ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME) == ["A::touchCellFirstTime"]


def test_traversal_events_require_a_global_task():
    assert _violations(CELL, prepare=True) == ["A::prepareTraversal"]
    assert _violations(CELL, ActionSet.OPERATION_BEGIN_TRAVERSAL) == ["A::beginTraversal"]
    assert _violations(GLOBAL, ActionSet.OPERATION_BEGIN_TRAVERSAL, ActionSet.OPERATION_END_TRAVERSAL, prepare=True) == []
    assert _violations(Task.GridEntity.GLOBAL_END_OF_STEP, ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME) == ["A::touchCellFirstTime"]


def test_comment_only_code_is_no_event():
    class Commented(_ActionSet):
        def get_implementation_of_operation(self, operation_name):
            return "// @todo Should be overwritten by mapping\n"

    task = Task(name="T", entity=CELL)
    task.add_action_set(Commented("A"))
    assert ValidateTaskGraph.violations_of_task(task) == []
