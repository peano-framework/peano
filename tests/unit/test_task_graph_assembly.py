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


def test_hanging_entities_wait_for_everything_spawned_so_far():
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
    for operation, body in (
        (Operation.OPERATION_CREATE_HANGING_FACE, "interpolate();"),
        (Operation.OPERATION_DESTROY_HANGING_VERTEX, "restrict();"),
    ):
        code = _code(wrapper, operation)
        assert "waitForTasks( _tasksSpawnedByThisTraversal )" in code
        assert code.index("waitForTasks") < code.index(body)


def test_new_cells_wait_for_the_first_tasks_of_the_parent_cell():
    wrapper = _wrapper_of(
        _task(
            "Mesh",
            {
                **SIX_EVENTS,
                Operation.OPERATION_CREATE_CELL: "interpolate();",
                Operation.OPERATION_DESTROY_CELL: "restrict();",
            },
        )
    )
    labels = wrapper.labels[TaskGraphAssembly.CELL]
    for operation, body in (
        (Operation.OPERATION_CREATE_CELL, "interpolate();"),
        (Operation.OPERATION_DESTROY_CELL, "restrict();"),
    ):
        code = _code(wrapper, operation)
        assert f"coarseGridCell{labels}" in code and "waitForTask(" in code
        assert code.index("waitForTask(") < code.index(body)


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


def test_consumer_declares_no_data_but_stays_away_from_conflicting_followers():
    class _Submodel:
        def __init__(self, name):
            self.name = name
            self.generator = None

    faces = _Submodel("Q")
    reading = _task("Reading", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "a();"})
    reading.use_face(faces, Task.DataAccess.READ_AND_DISCARD)
    writing = _task("Writing", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "b();"})
    writing.use_face(faces, Task.DataAccess.WRITE)
    writing.add_dependency(reading)
    unrelated = _task("Unrelated", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "c();"})
    unrelated.add_dependency(reading)
    graph = Graph()
    for task in (reading, writing, unrelated):
        graph.add_task(task)
    assembly = TaskGraphAssembly._Assembly(graph, "Probe")
    assembly.build()
    consumers = [
        t
        for t in assembly.result.tasks
        if t.name.endswith(TaskGraphAssembly.CONSUMER_SUFFIX)
    ]
    # A declaration would load or store data which the producer discards: Peano relays data strictly alternating
    for consumer in consumers:
        assert not consumer.face_data and not consumer.vertex_data
    # The consumer of the reader must not share a traversal with the writer, but it can with the other follower
    assert len(assembly.hazard_pairs) == 1
    consumer, follower = assembly.hazard_pairs[0]
    assert consumer.name.startswith("Reading") and follower.name.startswith("Writing")
    steps = _compile(reading, writing, unrelated, fusion=True)
    for step in steps:
        names = step.name
        assert not ("Reading_Consumer" in names and "Writing_Producer" in names)


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


def test_tasks_sharing_temporary_data_are_merged_into_one_traversal():
    class _Submodel:
        def __init__(self, name):
            self.name = name
            self.generator = None

    update = _Submodel("Update")
    clearing = _task(
        "Clearing",
        {Operation.OPERATION_TOUCH_FACE_FIRST_TIME: "clear();"},
        entity=Entity.FACE,
    )
    clearing.use_face(update, Task.DataAccess.TEMPORARY)
    projection = _task(
        "Projection", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "project();"}
    )
    projection.use_face(update, Task.DataAccess.TEMPORARY)
    projection.add_dependency(clearing)
    follower = _task(
        "Follower", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "follow();"}
    )
    follower.add_dependency(projection)
    graph = Graph()
    for task in (clearing, projection, follower):
        graph.add_task(task)
    merged = TaskGraphAssembly._merge_tasks_sharing_temporary_data(graph)
    assert len(merged.tasks) == 2
    assert [t.name for t in merged.tasks][0] == (
        "Clearing" + TaskGraphAssembly.MERGE_NAME_SEPARATOR + "Projection"
    )
    # Clearing runs before the projection within one task
    assert [a.get_action_set_name() for a in merged.tasks[0].action_sets] == [
        "ClearingActions",
        "ProjectionActions",
    ]
    assert merged.tasks[0].face_data_access[update] == Task.DataAccess.TEMPORARY
    assert merged.tasks[1].dependencies == [merged.tasks[0]]
    # The original graph is untouched
    assert len(graph.tasks) == 3
    steps = _compile(clearing, projection, follower, fusion=True)
    producers = [
        s
        for s in steps
        if any(isinstance(a, SpawnTaskActionSet) for a in s.action_sets)
    ]
    first_producer = [
        a for a in producers[0].action_sets if isinstance(a, SpawnTaskActionSet)
    ]
    assert len(first_producer) == 2


def test_tasks_are_not_merged_if_somebody_else_depends_on_the_first_one():
    class _Submodel:
        def __init__(self, name):
            self.name = name
            self.generator = None

    update = _Submodel("Update")
    first = _task(
        "First", {Operation.OPERATION_TOUCH_FACE_FIRST_TIME: "a();"}, entity=Entity.FACE
    )
    first.use_face(update, Task.DataAccess.TEMPORARY)
    second = _task("Second", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "b();"})
    second.use_face(update, Task.DataAccess.TEMPORARY)
    second.add_dependency(first)
    third = _task("Third", {Operation.OPERATION_TOUCH_CELL_FIRST_TIME: "c();"})
    third.add_dependency(first)
    graph = Graph()
    for task in (first, second, third):
        graph.add_task(task)
    assert len(TaskGraphAssembly._merge_tasks_sharing_temporary_data(graph).tasks) == 3


def test_all_entities_take_their_task_numbers_from_one_enumerator():
    # Different tarch::multicore::TaskEnumerator instances do not hand out disjoint
    # numbers: A face task and a cell task would get the same number and wait for each other
    wrapper = _wrapper_of(_task("Mesh", SIX_EVENTS))
    for operation in SIX_EVENTS:
        code = _code(wrapper, operation)
        assert ".getTaskEnumerator()" not in code.replace(
            "taskassembly::getTaskEnumerator()", ""
        ), operation
    assert "taskassembly::getTaskEnumerator().getNumber()" in _code(
        wrapper, Operation.OPERATION_TOUCH_FACE_FIRST_TIME
    )
    graph = Graph()
    graph.add_task(_task("Mesh", SIX_EVENTS))
    assembly = TaskGraphAssembly._Assembly(graph, "Probe")
    assembly.build()
    for labels in assembly.labels.values():
        assert "taskEnumerator" not in str(labels.data)


def _chain(*tasks):
    for earlier, later in zip(tasks, tasks[1:]):
        later.add_dependency(earlier)
    graph = Graph()
    for task in tasks:
        graph.add_task(task)
    return graph


def _cell_task(name, **kwargs):
    return _task(
        name,
        {
            Operation.OPERATION_TOUCH_CELL_FIRST_TIME: f"{name.lower()}();",
            Operation.OPERATION_TOUCH_CELL_LAST_TIME: f"{name.lower()}Last();",
        },
        **kwargs,
    )


def _concatenated(graph):
    return TaskGraphAssembly.translate_with_concatenation_and_sequential(
        graph, "Probe", None
    )


def _names(steps):
    return [step.name for step in steps]


def test_consumers_of_a_chain_vanish_except_the_last_one():
    first, second, third = (
        _cell_task("First"),
        _cell_task("Second"),
        _cell_task("Third"),
    )
    steps = _concatenated(_chain(first, second, third))
    assert _names(steps) == [
        "Probe_First_Producer",
        "Probe_Second_Producer",
        "Probe_Third_Producer",
        "Probe_Third_Consumer",
    ]
    consumers = [
        a for s in steps for a in s.action_sets if isinstance(a, WaitForTasksActionSet)
    ]
    assert len(consumers) == 1
    # It waits for the numbers of the whole chain: 3 tasks with 2 numbers each
    code = consumers[0].get_implementation_of_operation(
        Operation.OPERATION_TOUCH_CELL_FIRST_TIME
    )
    assert code.count("waitForAndReleaseTask") == 6
    for name in ("First", "Second", "Third"):
        assert name in code


def test_without_concatenation_every_task_keeps_its_consumer():
    first, second = _cell_task("First"), _cell_task("Second")
    assert len(_compile(first, second)) == 4
    second.dependencies = []
    steps = TaskGraphAssembly.translate_with_sequential(
        _chain(_cell_task("A"), _cell_task("B")), "Probe", None
    )
    assert len(steps) == 4


def test_the_follower_depends_on_the_numbers_of_the_removed_consumer():
    first, second = _cell_task("First"), _cell_task("Second")
    steps = _concatenated(_chain(first, second))
    wrappers = _wrappers(steps)
    code = _code(wrappers[1], Operation.OPERATION_TOUCH_CELL_FIRST_TIME)
    first_attributes = wrappers[0].attributes[TaskGraphAssembly.CELL]
    assert first_attributes.first[0].upper() + first_attributes.first[1:] in code
    # The last task of the first task runs, too: Its consumer has gone
    assert first_attributes.last[0].upper() + first_attributes.last[1:] in code


def test_the_labels_carry_the_numbers_from_producer_to_producer():
    steps = _concatenated(_chain(_cell_task("First"), _cell_task("Second")))
    producer_of_first, producer_of_second, consumer = steps
    assert list(producer_of_first.cell_data_access.values()) == [Task.DataAccess.WRITE]
    assert list(producer_of_second.cell_data_access.values()) == [
        Task.DataAccess.READ_AND_WRITE
    ]
    assert list(consumer.cell_data_access.values()) == [
        Task.DataAccess.READ_AND_DISCARD
    ]


def test_a_consumer_stays_if_two_tasks_follow():
    first = _cell_task("First")
    second, third = _cell_task("Second"), _cell_task("Third")
    second.add_dependency(first)
    third.add_dependency(first)
    graph = Graph()
    for task in (first, second, third):
        graph.add_task(task)
    assert "Probe_First_Consumer" in _names(_concatenated(graph))


def test_a_consumer_stays_in_front_of_a_task_that_is_not_split():
    first = _cell_task("First")
    global_task = Task("Reduction", Entity.GLOBAL_END_OF_STEP)
    steps = _concatenated(_chain(first, global_task))
    assert "Probe_First_Consumer" in _names(steps)


def test_a_consumer_stays_if_the_follower_has_main_thread_action_sets_on_the_mesh():
    first = _cell_task("First")
    second = _cell_task("Second")
    second.add_action_set(
        _ActionSet(
            "Plain",
            {Operation.OPERATION_CREATE_HANGING_FACE: "interpolate();"},
            may_run_in_task=False,
        )
    )
    assert "Probe_First_Consumer" in _names(_concatenated(_chain(first, second)))


def test_a_consumer_stays_if_a_follower_conflicts_on_face_data():
    class _Submodel:
        def __init__(self, name):
            self.name = name
            self.generator = None

    faces = _Submodel("Q")
    first = _cell_task("First")
    first.use_face(faces, Task.DataAccess.READ_AND_DISCARD)
    second = _cell_task("Second")
    third = _cell_task("Third")
    third.use_face(faces, Task.DataAccess.WRITE)
    steps = _concatenated(_chain(first, second, third))
    names = _names(steps)
    assert "Probe_First_Consumer" in names
    # The second one can still be dropped
    assert "Probe_Second_Consumer" not in names


def test_concatenating_compilers_are_registered():
    assert (
        Compilers.COMPILERS["assembly-concatenation"]
        is TaskGraphAssembly.translate_with_concatenation_and_sequential
    )
    assert (
        Compilers.COMPILERS["assembly-concatenation-fusion"]
        is TaskGraphAssembly.translate_with_concatenation_and_fusion
    )
