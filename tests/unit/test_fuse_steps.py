import pytest

from peano4.api.taskgraph import Graph, Task
from peano4.api.taskgraph.compiler import FuseSteps
from peano4.api.solversteps import SynchroniseActionSets, get_task_groups


VERTEX = Task.GridEntity.VERTEX
FACE = Task.GridEntity.FACE
CELL = Task.GridEntity.CELL

READ_AND_DISCARD = Task.DataAccess.READ_AND_DISCARD
WRITE = Task.DataAccess.WRITE
READ_AND_WRITE = Task.DataAccess.READ_AND_WRITE
TEMPORARY = Task.DataAccess.TEMPORARY


class _FakeActionSet:
    def __init__(self, name, descend_invocation_order=0):
        self.name = name
        self.descend_invocation_order = descend_invocation_order
        self.parallel = False

    def get_action_set_name(self):
        return self.name


class _FakeSubmodel:
    def __init__(self, name):
        self.name = name


def _chain(*entities):
    """A chain of tasks t0 -> t1 -> ... with the given grid entities"""
    graph = Graph()
    tasks = [Task(name=f"t{i}", entity=entity) for i, entity in enumerate(entities)]
    for previous_task, task in zip(tasks, tasks[1:]):
        task.add_dependency(previous_task)
    for task in tasks:
        task.add_action_set(_FakeActionSet(task.name))
        graph.add_task(task)
    return graph, tasks


def _groups(step):
    """Names of the action sets per task group of the generated observer"""
    return [
        [action_set.name for action_set in group]
        for group in get_task_groups(step.action_sets)
    ]


def _step_names(graph, non_fusable_task_pairs=None):
    return [
        step.name
        for step in FuseSteps.translate(graph, "S", None, non_fusable_task_pairs)
    ]


def test_entity_order_fuses_vertex_face_cell_but_not_cell_vertex():
    graph, _ = _chain(VERTEX, FACE, CELL, VERTEX, CELL)
    assert _step_names(graph) == ["S_T0__T1__T2", "S_T3__T4"]


def test_repeating_task_is_never_fused():
    graph = Graph()
    refine = Task("refine", CELL, task_has_to_repeat="condition")
    initial_condition = Task("init", CELL)
    initial_condition.add_dependency(refine)
    graph.add_task(refine)
    graph.add_task(initial_condition)
    steps = FuseSteps.translate(graph, "S", None)
    assert [step.name for step in steps] == ["S_Refine", "S_Init"]
    assert steps[0].has_to_repeat == "condition"


def test_non_fusable_pair_starts_new_traversal():
    graph, tasks = _chain(CELL, CELL)
    assert _step_names(graph) == ["S_T0__T1"]
    assert _step_names(graph, [(tasks[1], tasks[0])]) == ["S_T0", "S_T1"]


def test_cell_data_do_not_constrain_fusion():
    graph, tasks = _chain(CELL, CELL)
    patch = _FakeSubmodel("Q")
    tasks[0].use_cell(patch, WRITE)
    tasks[1].use_cell(patch, READ_AND_WRITE)
    assert _step_names(graph) == ["S_T0__T1"]


def test_face_data_written_then_read_prevents_fusion():
    graph, tasks = _chain(CELL, CELL)
    face_data = _FakeSubmodel("F")
    tasks[0].use_face(face_data, WRITE)
    tasks[1].use_face(face_data, READ_AND_DISCARD)
    assert _step_names(graph) == ["S_T0", "S_T1"]


def test_face_data_read_then_written_prevents_fusion():
    """The reconstruction/projection pattern of a Finite Volume time step"""
    graph, tasks = _chain(CELL, CELL, CELL)
    face_data = _FakeSubmodel("F")
    tasks[0].use_face(face_data, READ_AND_DISCARD)
    tasks[2].use_face(face_data, WRITE)
    assert _step_names(graph) == ["S_T0__T1", "S_T2"]


def test_vertex_data_written_twice_prevents_fusion():
    graph, tasks = _chain(CELL, CELL)
    vertex_data = _FakeSubmodel("V")
    tasks[0].use_vertex(vertex_data, WRITE)
    tasks[1].use_vertex(vertex_data, WRITE)
    assert _step_names(graph) == ["S_T0", "S_T1"]


def test_face_data_read_twice_or_temporary_does_not_prevent_fusion():
    graph, tasks = _chain(CELL, CELL, CELL)
    face_data = _FakeSubmodel("F")
    tasks[0].use_face(face_data, READ_AND_DISCARD)
    tasks[1].use_face(face_data, READ_AND_DISCARD)
    tasks[2].use_face(face_data, TEMPORARY)
    assert _step_names(graph) == ["S_T0__T1__T2"]


def test_independent_tasks_form_one_parallel_node():
    graph = Graph()
    first, second = Task("a", VERTEX), Task("b", CELL)
    for task in (first, second):
        task.add_action_set(_FakeActionSet(task.name))
        graph.add_task(task)
    steps = FuseSteps.translate(graph, "S", None)
    assert [step.name for step in steps] == ["S_A__B"]
    assert _groups(steps[0]) == [["a", "b"]]
    assert all(action_set.parallel for action_set in steps[0].action_sets)


def test_independent_tasks_with_conflicting_data_are_not_parallel():
    graph = Graph()
    first, second = Task("a", CELL), Task("b", CELL)
    face_data = _FakeSubmodel("F")
    first.use_face(face_data, WRITE)
    second.use_face(face_data, READ_AND_DISCARD)
    for task in (first, second):
        task.add_action_set(_FakeActionSet(task.name))
        graph.add_task(task)
    steps = FuseSteps.translate(graph, "S", None)
    assert [step.name for step in steps] == ["S_A", "S_B"]
    assert not any(
        action_set.parallel for step in steps for action_set in step.action_sets
    )


def test_independent_tasks_with_conflicting_cell_data_are_fused_but_not_parallel():
    graph = Graph()
    first, second, third = Task("a", CELL), Task("b", CELL), Task("c", CELL)
    cell_data = _FakeSubmodel("Q")
    first.use_cell(cell_data, READ_AND_WRITE)
    second.use_cell(cell_data, READ_AND_WRITE)
    third.use_cell(cell_data, READ_AND_DISCARD)
    for task in (first, second, third):
        task.add_action_set(_FakeActionSet(task.name))
        graph.add_task(task)
    steps = FuseSteps.translate(graph, "S", None)
    assert [step.name for step in steps] == ["S_A__B__C"]
    assert _groups(steps[0]) == [["a"], ["b"], ["c"]]
    assert not any(action_set.parallel for action_set in steps[0].action_sets)


def test_independent_tasks_reading_the_same_cell_data_are_parallel():
    graph = Graph()
    first, second = Task("a", CELL), Task("b", CELL)
    cell_data = _FakeSubmodel("Q")
    first.use_cell(cell_data, READ_AND_DISCARD)
    second.use_cell(cell_data, READ_AND_DISCARD)
    for task in (first, second):
        task.add_action_set(_FakeActionSet(task.name))
        graph.add_task(task)
    steps = FuseSteps.translate(graph, "S", None)
    assert _groups(steps[0]) == [["a", "b"]]


def test_adjacent_parallel_nodes_are_separated_by_a_synchronisation_marker():
    graph = Graph()
    v1, v2, c1, c2 = (
        Task("v1", VERTEX),
        Task("v2", VERTEX),
        Task("c1", CELL),
        Task("c2", CELL),
    )
    c1.add_dependency(v1)
    c2.add_dependency(v2)
    for task in (v1, v2, c1, c2):
        task.add_action_set(_FakeActionSet(task.name))
        graph.add_task(task)
    steps = FuseSteps.translate(graph, "S", None)
    assert [step.name for step in steps] == ["S_V1__V2__C1__C2"]
    assert [type(action_set) for action_set in steps[0].action_sets].count(
        SynchroniseActionSets
    ) == 1
    assert isinstance(steps[0].action_sets[2], SynchroniseActionSets)
    assert _groups(steps[0]) == [["v1", "v2"], ["c1", "c2"]]


def test_task_with_several_action_sets_is_never_part_of_a_parallel_node():
    graph = Graph()
    first, second = Task("a", CELL), Task("b", CELL)
    first.add_action_set(_FakeActionSet("a1"))
    first.add_action_set(_FakeActionSet("a2"))
    second.add_action_set(_FakeActionSet("b"))
    graph.add_task(first)
    graph.add_task(second)
    steps = FuseSteps.translate(graph, "S", None)
    assert [step.name for step in steps] == ["S_A__B"]
    assert not any(action_set.parallel for action_set in steps[0].action_sets)
    assert _groups(steps[0]) == [["a1"], ["a2"], ["b"]]


def test_face_task_writing_face_data_fuses_with_subsequent_cell_task():
    """Faces are touched for the first time before their adjacent cells, e.g. boundary conditions before a reconstruction"""
    graph, tasks = _chain(FACE, CELL)
    face_data = _FakeSubmodel("F")
    tasks[0].use_face(face_data, READ_AND_WRITE)
    tasks[1].use_face(face_data, READ_AND_DISCARD)
    assert _step_names(graph) == ["S_T0__T1"]


def test_face_task_reading_face_data_fuses_with_subsequent_cell_task_writing_it():
    graph, tasks = _chain(FACE, CELL)
    face_data = _FakeSubmodel("F")
    tasks[0].use_face(face_data, READ_AND_DISCARD)
    tasks[1].use_face(face_data, WRITE)
    assert _step_names(graph) == ["S_T0__T1"]


def test_face_tasks_with_conflicting_face_data_are_fused_but_not_parallel():
    graph = Graph()
    first, second = Task("a", FACE), Task("b", FACE)
    face_data = _FakeSubmodel("F")
    first.use_face(face_data, WRITE)
    second.use_face(face_data, READ_AND_WRITE)
    for task in (first, second):
        task.add_action_set(_FakeActionSet(task.name))
        graph.add_task(task)
    steps = FuseSteps.translate(graph, "S", None)
    assert [step.name for step in steps] == ["S_A__B"]
    assert _groups(steps[0]) == [["a"], ["b"]]
    assert not any(action_set.parallel for action_set in steps[0].action_sets)


def test_cell_task_writing_face_data_is_not_fused_with_subsequent_face_task():
    graph, tasks = _chain(CELL, FACE)
    face_data = _FakeSubmodel("F")
    tasks[0].use_face(face_data, WRITE)
    tasks[1].use_face(face_data, READ_AND_DISCARD)
    assert _step_names(graph) == ["S_T0", "S_T1"]


def test_vertex_task_writing_vertex_data_fuses_with_subsequent_cell_task():
    graph, tasks = _chain(VERTEX, CELL)
    vertex_data = _FakeSubmodel("V")
    tasks[0].use_vertex(vertex_data, WRITE)
    tasks[1].use_vertex(vertex_data, READ_AND_DISCARD)
    assert _step_names(graph) == ["S_T0__T1"]


def test_face_task_writing_vertex_data_prevents_fusion():
    """A vertex is adjacent to faces which are touched only after the cell"""
    graph, tasks = _chain(FACE, CELL)
    vertex_data = _FakeSubmodel("V")
    tasks[0].use_vertex(vertex_data, WRITE)
    tasks[1].use_vertex(vertex_data, READ_AND_DISCARD)
    assert _step_names(graph) == ["S_T0", "S_T1"]


START = Task.GridEntity.GLOBAL_START_OF_STEP
END = Task.GridEntity.GLOBAL_END_OF_STEP


def _graph_with_global_tasks(*tasks):
    graph = Graph()
    for task in tasks:
        if not task.action_sets:
            task.add_action_set(_FakeActionSet(task.name))
        graph.add_task(task)
    return graph


def _action_sets_per_step(graph):
    return [
        [action_set.name for action_set in step.action_sets]
        for step in FuseSteps.translate(graph, "S", None)
    ]


def test_start_and_end_of_step_tasks_open_and_close_the_first_and_last_traversal():
    cell_1, cell_2 = Task("c1", CELL), Task("c2", CELL)
    cell_2.add_dependency(cell_1)
    # They are the last and first in the graph, but ought to be first and last
    end = Task("end", END)
    end.add_dependency(cell_2)
    start = Task("start", START)
    graph = _graph_with_global_tasks(end, cell_1, cell_2, start)

    # No traversal of their own
    assert _action_sets_per_step(graph) == [["start", "c1", "c2", "end"]]


def test_end_of_step_tasks_come_after_other_global_tasks_and_follow_their_dependencies():
    cell = Task("c", CELL)
    global_task = Task("g", Task.GridEntity.GLOBAL)
    global_task.add_dependency(cell)
    first_end = Task("end1", END)
    second_end = Task("end2", END)
    second_end.add_dependency(first_end)
    graph = _graph_with_global_tasks(second_end, first_end, cell, global_task)

    assert _action_sets_per_step(graph) == [["c", "g", "end1", "end2"]]


def test_global_tasks_of_the_step_do_not_pull_other_tasks_into_the_traversal():
    """The start task is not a predecessor that delays: both cell tasks share a layer"""
    start = Task("start", START)
    cell_1, cell_2 = Task("c1", CELL), Task("c2", CELL)
    cell_1.add_dependency(start)
    cell_2.add_dependency(start)
    graph = _graph_with_global_tasks(start, cell_1, cell_2)

    assert _action_sets_per_step(graph) == [["start", "c1", "c2"]]
    assert [step.action_sets[1].parallel for step in FuseSteps.translate(graph, "S", None)] == [True]


def test_start_of_step_task_may_not_depend_on_anything():
    cell = Task("c", CELL)
    start = Task("start", START)
    start.add_dependency(cell)
    graph = _graph_with_global_tasks(cell, start)
    with pytest.raises(AssertionError, match="may not depend"):
        FuseSteps.translate(graph, "S", None)


def test_only_end_of_step_tasks_may_depend_on_end_of_step_tasks():
    end = Task("end", END)
    cell = Task("c", CELL)
    cell.add_dependency(end)
    graph = _graph_with_global_tasks(end, cell)
    with pytest.raises(AssertionError, match="only other GLOBAL_END_OF_STEP"):
        FuseSteps.translate(graph, "S", None)


def test_repeating_first_and_last_traversal_get_their_own_start_and_end_traversal():
    start = Task("start", START)
    refine = Task("refine", CELL, task_has_to_repeat="condition")
    plot = Task("plot", CELL, task_has_to_repeat="again")
    plot.add_dependency(refine)
    end = Task("end", END)
    end.add_dependency(plot)
    graph = _graph_with_global_tasks(start, refine, plot, end)

    # Otherwise, start and end would run in every repetition
    assert _action_sets_per_step(graph) == [["start"], ["refine"], ["plot"], ["end"]]


def test_start_and_end_of_step_tasks_without_other_tasks():
    graph = _graph_with_global_tasks(Task("start", START), Task("end", END))
    assert _action_sets_per_step(graph) == [["start", "end"]]


def test_ordered_tasks_put_start_first_and_end_last():
    cell = Task("c", CELL)
    end = Task("end", END)
    end.add_dependency(cell)
    start = Task("start", START)
    graph = Graph()
    for task in [end, cell, start]:
        graph.add_task(task)
    assert [task.name for task in graph.ordered_tasks()] == ["start", "c", "end"]


def test_global_tasks_may_not_hold_data_or_repeat():
    for entity in [START, END]:
        task = Task("t", entity)
        with pytest.raises(AssertionError):
            task.use_cell(_FakeSubmodel("Q"), WRITE)
        with pytest.raises(AssertionError):
            Task("t", entity, task_has_to_repeat="condition")
