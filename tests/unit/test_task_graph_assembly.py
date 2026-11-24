import pytest

import peano4.api.solversteps
from peano4.api.taskgraph import Graph, Task
from peano4.api.taskgraph.compiler import TaskGraphAssembly
from peano4.api.taskgraph.compiler import Compilers
from peano4.api.taskgraph.compiler.SpawnTaskActionSet import SpawnTaskActionSet
from peano4.api.taskgraph.compiler.WaitForTasksActionSet import WaitForTasksActionSet

ActionSet = peano4.api.solversteps.ActionSet
Entity = Task.GridEntity
Operation = ActionSet


class _ActionSet:
    """! Duck-typed action set with a body per event"""

    def __init__(self, name, bodies=None, guards=None, may_run_in_task=True):
        self.name = name
        self.bodies = dict(bodies or {})
        self.guards = dict(guards or {})
        self.may_run_in_task = may_run_in_task
        self.descend_invocation_order = 0
        self.parallel = False

    def get_action_set_name(self):
        return self.name

    def get_body_of_operation(self, operation_name):
        return self.bodies.get(operation_name, "")

    def get_guard_of_operation(self, operation_name):
        return self.guards.get(operation_name, "true")

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
        return self.may_run_in_task

    def user_should_modify_template(self):
        return False


SIX_EVENTS = {
    Operation.OPERATION_TOUCH_VERTEX_FIRST_TIME: "vertexFirst();",
    Operation.OPERATION_TOUCH_VERTEX_LAST_TIME: "vertexLast();",
    Operation.OPERATION_TOUCH_FACE_FIRST_TIME: "faceFirst();",
    Operation.OPERATION_TOUCH_FACE_LAST_TIME: "faceLast();",
    Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "cellFirst();",
    Operation.OPERATION_TOUCH_CELL_LAST_TIME: "cellLast();",
}


def _task(name, bodies, entity=Entity.CELL, **kwargs):
    task = Task(name, entity)
    task.add_action_set(_ActionSet(f"{name}Actions", bodies, **kwargs))
    return task


def _compile(*tasks, fusion=False):
    graph = Graph()
    for task in tasks:
        graph.add_task(task)
    translate = (
        TaskGraphAssembly.translate_with_fusion
        if fusion
        else TaskGraphAssembly.translate_with_sequential
    )
    return translate(graph, "Probe", None)


def _wrappers(steps):
    return [
        action_set
        for step in steps
        for action_set in step.action_sets
        if isinstance(action_set, SpawnTaskActionSet)
    ]


def _wrapper_of(*tasks, **kwargs):
    return _wrappers(_compile(*tasks, **kwargs))[0]


def _code(wrapper, operation):
    return wrapper.get_implementation_of_operation(operation)


def test_every_mesh_task_is_split_into_producer_and_consumer():
    first = _task("First", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"})
    second = _task("Second", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "b();"})
    second.add_dependency(first)
    steps = _compile(first, second)
    names = [step.name for step in steps]
    assert len(steps) == 4
    suffixes = (TaskGraphAssembly.PRODUCER_SUFFIX, TaskGraphAssembly.CONSUMER_SUFFIX)
    for task_name in ("First", "Second"):
        for suffix in suffixes:
            assert any(task_name in name and suffix.strip() in name for name in names)
    # The second task follows the consumer of the first one
    assert names.index("Probe_First_Consumer") < names.index("Probe_Second_Producer")


def test_each_of_the_six_events_is_spawned_as_task():
    wrapper = _wrapper_of(_task("All", SIX_EVENTS, entity=Entity.CELL))
    for operation, body in SIX_EVENTS.items():
        code = _code(wrapper, operation)
        assert "spawnTask" in code, operation
        assert body in code, operation


def test_first_and_last_event_hold_independent_numbers():
    wrapper = _wrapper_of(_task("Cells", SIX_EVENTS))
    attributes = wrapper.attributes[TaskGraphAssembly.CELL]
    assert attributes.first != attributes.last
    first = _code(wrapper, Operation.OPERATION_TOUCH_CELL_FIRST_TIME)
    last = _code(wrapper, Operation.OPERATION_TOUCH_CELL_LAST_TIME)
    # The last number is reserved in the first touch, as the parent needs it
    assert "set" + attributes.last[0].upper() + attributes.last[1:] in first
    assert "firstTaskNumber" in first
    assert attributes.last[0].upper() + attributes.last[1:] in last


def test_task_without_last_event_has_no_last_task():
    wrapper = _wrapper_of(
        _task("Cells", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"})
    )
    assert "spawnTask" not in _code(wrapper, Operation.OPERATION_TOUCH_CELL_LAST_TIME)
    assert wrapper.attributes[TaskGraphAssembly.CELL].children is None


def test_last_task_is_empty_task_if_its_guard_fails():
    wrapper = _wrapper_of(
        _task(
            "Cells",
            {Operation.OPERATION_TOUCH_CELL_LAST_TIME: "a();"},
            guards={
                Operation.OPERATION_TOUCH_CELL_LAST_TIME: "marker.hasBeenRefined()"
            },
        )
    )
    code = _code(wrapper, Operation.OPERATION_TOUCH_CELL_LAST_TIME)
    assert "if (marker.hasBeenRefined())" in code
    assert "spawnEmptyTask" in code


def test_cells_depend_on_parent_and_children():
    wrapper = _wrapper_of(_task("Cells", SIX_EVENTS))
    attributes = wrapper.attributes[TaskGraphAssembly.CELL]
    first = _code(wrapper, Operation.OPERATION_TOUCH_CELL_FIRST_TIME)
    last = _code(wrapper, Operation.OPERATION_TOUCH_CELL_LAST_TIME)
    labels = wrapper.labels[TaskGraphAssembly.CELL]
    # Parent to child: the first task waits for the parent's first task
    assert f"coarseGridCell{labels}.get" in first
    # Child to parent: the child writes its last number into the parent
    assert f"coarseGridCell{labels}.set" in first
    assert "getIndexWithinParentCell" in first
    # The last task waits for all children
    assert attributes.children[0].upper() + attributes.children[1:] in last
    assert "ThreePowerD" in last


def test_faces_and_vertices_register_with_their_coarse_entities():
    wrapper = _wrapper_of(_task("Mesh", SIX_EVENTS))
    face = _code(wrapper, Operation.OPERATION_TOUCH_FACE_FIRST_TIME)
    vertex = _code(wrapper, Operation.OPERATION_TOUCH_VERTEX_FIRST_TIME)
    assert "coarseGridFaces" in face and "getIndexWithinParentFace" in face
    assert "coarseGridVertices" in vertex and "coincidesWithCoarseGridVertex" in vertex


def test_cell_depends_on_the_first_tasks_of_its_faces_and_vertices():
    wrapper = _wrapper_of(_task("Mesh", SIX_EVENTS))
    first = _code(wrapper, Operation.OPERATION_TOUCH_CELL_FIRST_TIME)
    assert "fineGridVertices" in first and "fineGridFaces" in first
    face = _code(wrapper, Operation.OPERATION_TOUCH_FACE_FIRST_TIME)
    assert "fineGridVertices" in face


def test_face_and_vertex_last_tasks_wait_for_the_cells_around():
    wrapper = _wrapper_of(_task("Mesh", SIX_EVENTS))
    for kind in (TaskGraphAssembly.VERTEX, TaskGraphAssembly.FACE):
        around = wrapper.attributes[kind].around
        assert around is not None
        assert around[0].upper() + around[1:] in _code(
            wrapper,
            Operation.OPERATION_TOUCH_VERTEX_LAST_TIME
            if kind == TaskGraphAssembly.VERTEX
            else Operation.OPERATION_TOUCH_FACE_LAST_TIME,
        )
    # The cell announces its tasks
    cell = _code(wrapper, Operation.OPERATION_TOUCH_CELL_FIRST_TIME)
    assert "getIndexOfCellAroundVertex" in cell
    assert "getIndexOfCellAroundFace" in cell


def test_no_bookkeeping_for_cells_if_nobody_waits_for_them():
    wrapper = _wrapper_of(
        _task("Cells", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"})
    )
    assert TaskGraphAssembly.FACE not in wrapper.attributes
    assert TaskGraphAssembly.VERTEX not in wrapper.attributes
    assert "getIndexOfCellAround" not in _code(
        wrapper, Operation.OPERATION_TOUCH_CELL_FIRST_TIME
    )


def test_destroying_persistent_entities_waits_for_their_tasks():
    wrapper = _wrapper_of(
        _task(
            "Mesh",
            {
                **SIX_EVENTS,
                Operation.OPERATION_DESTROY_PERSISTENT_FACE: "destroyFace();",
                Operation.OPERATION_DESTROY_CELL: "destroyCell();",
            },
        )
    )
    for operation, body in (
        (Operation.OPERATION_DESTROY_PERSISTENT_FACE, "destroyFace();"),
        (Operation.OPERATION_DESTROY_CELL, "destroyCell();"),
    ):
        code = _code(wrapper, operation)
        assert "waitForAndReleaseTask" in code
        assert code.index("waitForAndReleaseTask") < code.index(body)


def test_hanging_entities_wait_for_the_first_tasks_of_the_coarse_entity():
    wrapper = _wrapper_of(
        _task(
            "Mesh",
            {
                **SIX_EVENTS,
                Operation.OPERATION_CREATE_HANGING_FACE: "interpolate();",
                Operation.OPERATION_DESTROY_HANGING_VERTEX: "restrict();",
            },
        )
    )
    face = _code(wrapper, Operation.OPERATION_CREATE_HANGING_FACE)
    vertex = _code(wrapper, Operation.OPERATION_DESTROY_HANGING_VERTEX)
    assert "coarseGridFaces" in face and "waitForTask" in face
    assert face.index("waitForTask") < face.index("interpolate();")
    assert "coarseGridVertices" in vertex and "waitForTask" in vertex
    assert vertex.index("waitForTask") < vertex.index("restrict();")


def test_global_events_wait_for_all_tasks():
    wrapper = _wrapper_of(
        _task(
            "Cells",
            {
                Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();",
                Operation.OPERATION_END_TRAVERSAL: "reduce();",
            },
        )
    )
    code = _code(wrapper, Operation.OPERATION_END_TRAVERSAL)
    assert "waitForAllTasks" in code
    assert code.index("waitForAllTasks") < code.index("reduce();")


def test_producer_keeps_the_events_of_the_original_action_sets():
    task = _task("Cells", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"})
    steps = _compile(task)
    # The wrappers reset arrays in other events, but the compiler must not see these events
    producers = [
        step
        for step in steps
        if step.name.endswith(
            TaskGraphAssembly.PRODUCER_SUFFIX.strip().replace(" ", "_")
        )
    ]
    assert len(producers) == 1
    graph = Graph()
    graph.add_task(task)
    assembly = TaskGraphAssembly._Assembly(graph, "Probe")
    assembly.build()
    producer = [
        t
        for t in assembly.result.tasks
        if t.name.endswith(TaskGraphAssembly.PRODUCER_SUFFIX)
    ][0]
    assert producer.events == task.events
    assert producer.event_label == task.event_label


def test_consumer_waits_in_the_first_touch_of_every_entity():
    task = _task("Mesh", SIX_EVENTS)
    steps = _compile(task)
    consumer = [
        a
        for step in steps
        for a in step.action_sets
        if isinstance(a, WaitForTasksActionSet)
    ][0]
    for kind in (
        TaskGraphAssembly.VERTEX,
        TaskGraphAssembly.FACE,
        TaskGraphAssembly.CELL,
    ):
        assert kind in consumer.kinds()
    for operation in (
        Operation.OPERATION_TOUCH_VERTEX_FIRST_TIME,
        Operation.OPERATION_TOUCH_FACE_FIRST_TIME,
        Operation.OPERATION_TOUCH_CELL_FIRST_TIME,
    ):
        assert "waitForAndReleaseTask" in consumer.get_implementation_of_operation(
            operation
        )
    assert (
        consumer.get_implementation_of_operation(
            Operation.OPERATION_TOUCH_CELL_LAST_TIME
        )
        == ""
    )
    assert "return" in consumer.get_body_of_createGridControlEventsForNextTraversal()


def test_labels_are_written_by_producers_and_discarded_by_consumers():
    steps = _compile(_task("Mesh", SIX_EVENTS))
    producer, consumer = steps
    for step, access in (
        (producer, Task.DataAccess.WRITE),
        (consumer, Task.DataAccess.READ_AND_DISCARD),
    ):
        for accesses in (
            step.vertex_data_access,
            step.face_data_access,
            step.cell_data_access,
        ):
            assert list(accesses.values()) == [access]


def test_fused_step_with_consumer_and_producer_reads_and_writes_the_labels():
    first = _task("First", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"})
    second = _task("Second", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "b();"})
    second.add_dependency(first)
    steps = _compile(first, second, fusion=True)
    fused = [
        step
        for step in steps
        if any(isinstance(a, WaitForTasksActionSet) for a in step.action_sets)
        and any(isinstance(a, SpawnTaskActionSet) for a in step.action_sets)
    ]
    assert len(fused) == 1
    assert list(fused[0].cell_data_access.values()) == [Task.DataAccess.READ_AND_WRITE]


def test_labels_do_not_keep_independent_face_tasks_from_fusing():
    first = _task(
        "First", {Operation.OPERATION_TOUCH_FACE_LAST_TIME: "a();"}, entity=Entity.FACE
    )
    second = _task(
        "Second", {Operation.OPERATION_TOUCH_FACE_LAST_TIME: "b();"}, entity=Entity.FACE
    )
    steps = _compile(first, second, fusion=True)
    producers = [
        step
        for step in steps
        if any(isinstance(a, SpawnTaskActionSet) for a in step.action_sets)
    ]
    assert len(producers) == 1
    assert (
        len([a for a in producers[0].action_sets if isinstance(a, SpawnTaskActionSet)])
        == 2
    )


def test_tasks_that_have_to_run_on_the_main_thread_are_not_split():
    repeating = Task("Repeating", Entity.CELL, task_has_to_repeat="true")
    repeating.add_action_set(
        _ActionSet("A", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"})
    )
    main_thread = _task(
        "MainThread",
        {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"},
        may_run_in_task=False,
    )
    opt_out = _task("OptOut", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"})
    opt_out.may_spawn_background_tasks = False
    steps = _compile(repeating, main_thread, opt_out)
    assert len(steps) == 3
    assert not _wrappers(steps)


def test_later_task_depends_on_the_numbers_of_the_tasks_it_depends_on():
    first = _task(
        "First",
        {Operation.OPERATION_TOUCH_VERTEX_FIRST_TIME: "a();"},
        entity=Entity.VERTEX,
    )
    second = _task("Second", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "b();"})
    second.add_dependency(first)
    wrappers = _wrappers(_compile(first, second, fusion=True))
    assert len(wrappers) == 2
    code = _code(wrappers[1], Operation.OPERATION_TOUCH_CELL_FIRST_TIME)
    vertex_attribute = wrappers[0].attributes[TaskGraphAssembly.VERTEX].first
    assert vertex_attribute[0].upper() + vertex_attribute[1:] in code


def test_all_compilers_are_registered_under_their_names():
    assert (
        Compilers.COMPILERS["assembly"] is TaskGraphAssembly.translate_with_sequential
    )
    assert (
        Compilers.COMPILERS["assembly-fusion"]
        is TaskGraphAssembly.translate_with_fusion
    )
    assert "fusion" in Compilers.COMPILERS and "sequential" in Compilers.COMPILERS


def test_consumer_declares_the_face_and_vertex_data_of_its_producer():
    class _Submodel:
        def __init__(self, name):
            self.name = name
            self.generator = None

    face_data = _Submodel("Q")
    vertex_data = _Submodel("V")
    task = _task("Cells", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"})
    task.use_face(face_data, Task.DataAccess.READ_AND_DISCARD)
    task.use_vertex(vertex_data, Task.DataAccess.READ_AND_WRITE)
    graph = Graph()
    graph.add_task(task)
    assembly = TaskGraphAssembly._Assembly(graph, "Probe")
    assembly.build()
    consumer = [
        t
        for t in assembly.result.tasks
        if t.name.endswith(TaskGraphAssembly.CONSUMER_SUFFIX)
    ][0]
    # FuseSteps must see that the consumer stands for tasks which read and write these data
    assert consumer.face_data_access == {face_data: Task.DataAccess.READ_AND_WRITE}
    assert consumer.vertex_data_access == {vertex_data: Task.DataAccess.READ_AND_WRITE}


def test_the_end_of_the_traversal_waits_for_the_tasks_that_touch_exchanged_data():
    wrapper = _wrapper_of(_task("Mesh", SIX_EVENTS))
    end = _code(wrapper, Operation.OPERATION_END_TRAVERSAL)
    assert "waitForTasks( _tasksToFinishBeforeDataExchange )" in end
    assert "_tasksToFinishBeforeDataExchange" in wrapper.get_attributes()
    # All vertex tasks, the face tasks next to a parallel boundary
    assert "_tasksToFinishBeforeDataExchange.insert" in _code(
        wrapper, Operation.OPERATION_TOUCH_VERTEX_LAST_TIME
    )
    assert "isAdjacentToParallelBoundary" in _code(
        wrapper, Operation.OPERATION_TOUCH_FACE_LAST_TIME
    )


def test_cell_tasks_that_write_cell_data_only_may_outlive_the_traversal():
    cell_only = _wrapper_of(
        _task("Cells", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"})
    )
    assert "_tasksToFinishBeforeDataExchange.insert" not in _code(
        cell_only, Operation.OPERATION_TOUCH_CELL_FIRST_TIME
    )
