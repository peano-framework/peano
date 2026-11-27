from peano4.api.taskgraph import Graph, Task
from peano4.api.taskgraph.compiler import CellTaskProducerConsumer

CELL = Task.GridEntity.CELL
FACE = Task.GridEntity.FACE
Access = Task.DataAccess
Separator = CellTaskProducerConsumer.CHAIN_NAME_SEPARATOR


class _ActionSet:
    def __init__(self, name, body="x = 1;", background=True):
        self.name = name
        self.body = body
        self.background = background
        self.descend_invocation_order = 0
        self.parallel = False

    def get_action_set_name(self):
        return self.name

    def get_body_of_operation(self, operation_name):
        return self.body if operation_name == "touchCellFirstTime" else ""

    def get_implementation_of_operation(self, operation_name):
        return ""

    def get_implementation_of_unprepareTraversal(self):
        return ""

    def may_run_in_background_task(self):
        return self.background

    def user_should_modify_template(self):
        return False


class _Submodel:
    def __init__(self, name):
        self.name = name


def _cell_task(name, patch, access=Access.READ_AND_WRITE, **kwargs):
    task = Task(name=name, entity=CELL)
    task.use_cell(patch, access)
    task.add_action_set(_ActionSet(name, **kwargs))
    return task


def _graph(*tasks):
    graph = Graph()
    for task in tasks:
        graph.add_task(task)
    return graph


def _chain_graph(length=2, **kwargs):
    patch = _Submodel("patch")
    tasks = [_cell_task(f"t{i}", patch, **kwargs) for i in range(length)]
    for previous, task in zip(tasks, tasks[1:]):
        task.add_dependency(previous)
    return _graph(*tasks), tasks


def test_two_tasks_of_a_chain_are_merged():
    graph, (a, b) = _chain_graph()

    merged = CellTaskProducerConsumer._merge_chains(graph)

    assert [task.name for task in merged.tasks] == [a.name + Separator + b.name]
    assert [action_set.name for action_set in merged.tasks[0].action_sets] == [
        "t0",
        "t1",
    ]
    assert len(graph.tasks) == 2 and b.dependencies == [a]


def test_longer_chains_are_merged_completely():
    graph, tasks = _chain_graph(length=4)

    merged = CellTaskProducerConsumer._merge_chains(graph)

    assert len(merged.tasks) == 1
    assert len(merged.tasks[0].action_sets) == 4


def test_merged_task_keeps_predecessors_and_successors():
    patch = _Submodel("patch")
    before = Task(name="before", entity=FACE)
    a, b = _cell_task("a", patch), _cell_task("b", patch)
    after = Task(name="after", entity=FACE)
    a.add_dependency(before)
    b.add_dependency(a)
    after.add_dependency(b)

    merged = CellTaskProducerConsumer._merge_chains(_graph(before, a, b, after))

    chain = [task for task in merged.tasks if Separator in task.name][0]
    after_copy = [task for task in merged.tasks if task.name == "after"][0]
    before_copy = [task for task in merged.tasks if task.name == "before"][0]
    assert chain.dependencies == [before_copy]
    assert after_copy.dependencies == [chain]


def test_task_with_two_successors_is_not_merged():
    patch = _Submodel("patch")
    a, b, c = (_cell_task(name, patch) for name in "abc")
    b.add_dependency(a)
    c.add_dependency(a)

    assert len(CellTaskProducerConsumer._merge_chains(_graph(a, b, c)).tasks) == 3


def test_task_with_two_predecessors_is_not_merged():
    patch = _Submodel("patch")
    a, b, c = (_cell_task(name, patch) for name in "abc")
    c.add_dependency(a)
    c.add_dependency(b)

    assert len(CellTaskProducerConsumer._merge_chains(_graph(a, b, c)).tasks) == 3


def test_redundant_edge_does_not_prevent_merge():
    graph, (a, _, c) = _chain_graph(length=3)
    c.add_dependency(a)  # redundant

    assert len(CellTaskProducerConsumer._merge_chains(graph).tasks) == 1


def test_plain_action_set_of_last_task_prevents_merge():
    patch = _Submodel("patch")
    a = _cell_task("a", patch)
    b = _cell_task("b", patch, body="")
    b.add_dependency(a)

    assert len(CellTaskProducerConsumer._merge_chains(_graph(a, b)).tasks) == 2


def test_plain_action_set_of_first_task_does_not_prevent_merge():
    patch = _Submodel("patch")
    a = _cell_task("a", patch, background=False)
    b = _cell_task("b", patch)
    b.add_dependency(a)

    assert len(CellTaskProducerConsumer._merge_chains(_graph(a, b)).tasks) == 1


def test_tasks_which_are_not_split_are_not_merged():
    patch = _Submodel("patch")
    a = _cell_task("a", patch)
    b = _cell_task("b", patch, access=Access.READ_AND_DISCARD)  # writes nothing
    b.add_dependency(a)
    c = _cell_task("c", patch)
    c.may_spawn_background_tasks = False
    c.add_dependency(b)

    assert len(CellTaskProducerConsumer._merge_chains(_graph(a, b, c)).tasks) == 3


def test_data_accesses_are_merged():
    first_patch, second_patch, third_patch = (_Submodel(name) for name in "xyz")
    a = _cell_task("a", first_patch, Access.WRITE)
    a.use_cell(second_patch, Access.READ_AND_DISCARD)
    b = _cell_task("b", first_patch, Access.READ_AND_DISCARD)
    b.use_cell(second_patch, Access.WRITE)
    b.use_cell(third_patch, Access.WRITE)
    b.add_dependency(a)

    merged = CellTaskProducerConsumer._merge_chains(_graph(a, b)).tasks[0]

    assert merged.cell_data_access[first_patch] == Access.TEMPORARY
    assert merged.cell_data_access[second_patch] == Access.READ_AND_WRITE
    assert merged.cell_data_access[third_patch] == Access.WRITE


def test_merged_tasks_need_fewer_mesh_traversals():
    def steps(translate):
        graph, _ = _chain_graph(length=2)
        return translate(graph, "S", None)

    plain = steps(CellTaskProducerConsumer.translate_with_fusion)
    chained = steps(CellTaskProducerConsumer.translate_with_chains_and_fusion)

    assert len(chained) < len(plain)
    assert len(chained) == 2
