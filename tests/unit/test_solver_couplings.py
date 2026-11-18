"""Unit tests for the registration of couplings of solvers and for the
flag add_global_tasks, with which a coupling leaves the load balancing switch and the
dynamic mesh refinement to one of its solvers."""

import types

import pytest

import peano4.api
import peano4.api.taskgraph
import exahype2.api
import toolbox.loadbalancing.api.actionsets

Task = peano4.api.taskgraph.Task

SCHEMES = {
    "fv": lambda solver: exahype2.api.fv.RusanovWithExplicitEuler(solver, 4),
    "fv-optimistic": lambda solver: exahype2.api.fv.RusanovWithOptimisticExplicitEuler(
        solver, 4
    ),
    "rkfd4": lambda solver: exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, patch_size=4, rk_order=2
    ),
}


def _solver(scheme, name="Probe"):
    solver = exahype2.api.Solver(name, 2)
    solver.add_pde_term(exahype2.api.Flux(implementation="F[0] = Q[0]; F[1] = Q[1];"))
    solver.add_pde_term(exahype2.api.MaxEigenvalue(implementation="*lambdaMax = 1.0;"))
    solver.add_pde_term(exahype2.api.RefineErasePointwisely(implementation=""))
    solver.add_pde_term(
        exahype2.api.InitPointWisely(implementation="Q[0] = 0.0; Q[1] = 0.0;")
    )
    if scheme is not None:
        solver.numerical_scheme = SCHEMES[scheme](solver)
    return solver


def _construct(solver, tmp_path, add_global_tasks):
    exahype2_project = types.SimpleNamespace(periodic_BC=[True, True, True], _dimensions=3)
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    time_step = peano4.api.taskgraph.Graph()
    solver.numerical_scheme.construct_output(
        exahype2_project, peano_project, time_step, add_global_tasks
    )
    return time_step


def _toggles_load_balancing(task):
    return any(
        isinstance(action_set, toolbox.loadbalancing.api.actionsets.ToggleLoadBalancing)
        or isinstance(action_set, toolbox.loadbalancing.api.actionsets.LoadBalancing)
        for action_set in task.action_sets
    )


def test_graph_append_connects_the_end_of_one_graph_to_the_start_of_the_other():
    first, second = (peano4.api.taskgraph.Graph() for _ in range(2))
    a, b, c, d, e = (Task(name=n, entity=Task.GridEntity.CELL) for n in "abcde")
    b.add_dependency(a)  # first: a -> b, and c alone (two sinks: b and c)
    d.add_dependency(e)  # second: e -> d, i.e. the source is e
    for task in (a, b, c):
        first.add_task(task)
    for task in (d, e):
        second.add_task(task)
    first.append(second)
    assert first.tasks == [a, b, c, d, e]
    # The source of the second graph waits for all sinks of the first graph
    assert set(e.dependencies) == {b, c}
    # Nothing else changes
    assert d.dependencies == [e] and b.dependencies == [a] and a.dependencies == []
    assert second.tasks == [d, e]


def test_graph_append_makes_the_sources_wait_for_global_sinks_too():
    first, second = (peano4.api.taskgraph.Graph() for _ in range(2))
    a = Task(name="a", entity=Task.GridEntity.CELL)
    end = Task(name="end", entity=Task.GridEntity.GLOBAL_END_OF_STEP)
    b = Task(name="b", entity=Task.GridEntity.CELL)
    end.add_dependency(a)
    for task in (a, end):
        first.add_task(task)
    second.add_task(b)
    first.append(second)
    # The task compiler makes sure that b does not run in the sweep of the end-of-step task
    assert b.dependencies == [end]
    assert first.tasks == [a, end, b]


def test_graph_append_rejects_shared_tasks():
    first, second = (peano4.api.taskgraph.Graph() for _ in range(2))
    a = Task(name="a", entity=Task.GridEntity.CELL)
    first.add_task(a)
    second.add_task(a)
    with pytest.raises(ValueError):
        first.append(second)


def test_graph_merge_unites_graphs_and_keeps_the_dependencies():
    first, second, merged = (peano4.api.taskgraph.Graph() for _ in range(3))
    a = Task(name="a", entity=Task.GridEntity.CELL)
    b = Task(name="b", entity=Task.GridEntity.CELL)
    c = Task(name="c", entity=Task.GridEntity.CELL)
    b.add_dependency(a)
    for task in (a, b):
        first.add_task(task)
    second.add_task(c)
    merged.merge(first)
    merged.merge(second)
    merged.merge(first)
    assert merged.tasks == [a, b, c]
    assert b.dependencies == [a]
    assert first.tasks == [a, b]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_schemes_add_the_global_tasks_by_default(scheme, tmp_path):
    name = exahype2.api.Solver.NumericalScheme.DYNAMIC_MESH_REFINEMENT_TASK_NAME
    graph = _construct(_solver(scheme), tmp_path, True)
    assert f"Probe_{name}" in [task.name for task in graph.tasks]
    assert any(_toggles_load_balancing(task) for task in graph.tasks)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_schemes_leave_the_global_tasks_to_the_host_on_request(scheme, tmp_path):
    graph = _construct(_solver(scheme), tmp_path, False)
    assert not any("Dynamic mesh refinement" in task.name for task in graph.tasks)
    assert not any("Disable load balancing" in task.name for task in graph.tasks)
    assert not any(_toggles_load_balancing(task) for task in graph.tasks)
    # The solver proper is intact
    assert any("Reconstruct" in task.name for task in graph.tasks)


def test_solver_without_global_tasks_does_not_contribute_to_the_mesh_adaptation(
    tmp_path,
):
    solver = _solver("fv")
    exahype2_project = exahype2.api.Project(
        ["probe"], "probe", "probe", directory=str(tmp_path)
    )
    exahype2_project.set_global_simulation_parameters(
        dimensions=2,
        offset=[0.0, 0.0],
        size=[1.0, 1.0],
        min_end_time=1.0,
        periodic_BC=[True, True],
    )
    solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
        solver, volume_size=0.1
    )
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))

    def realise(add_global_tasks):
        graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
        solver.generate_solver_realisation(
            exahype2_project, peano_project, *graphs, add_global_tasks=add_global_tasks
        )
        return graphs

    assert len(realise(True)[3].tasks) > 0
    assert len(realise(False)[3].tasks) == 0


class _Coupling:
    """! Minimal coupling: it brings its solvers along and realises them itself """

    def __init__(self, solvers):
        self.solvers = solvers
        self.realised = False

    def generate_solver_realisation(self, *args):
        self.realised = True


def test_project_registers_the_solvers_of_a_coupling_but_realises_the_coupling(
    tmp_path,
):
    first, second = _solver(None, "First"), _solver(None, "Second")
    third = _solver("fv", "Third")
    coupling = _Coupling([first, second])
    project = exahype2.api.Project(["probe"], "probe", "probe", directory=str(tmp_path))
    project.add_solver(third)
    project.add_solver(coupling)
    assert project._solvers == [third, first, second]
    assert project._realisations == [third, coupling]


def test_graph_source_tasks_are_the_counterpart_of_the_sink_tasks():
    graph = peano4.api.taskgraph.Graph()
    a, b, c = (Task(name=n, entity=Task.GridEntity.CELL) for n in "abc")
    b.add_dependency(a)
    for task in (a, b, c):
        graph.add_task(task)
    assert graph.source_tasks() == [a, c]
    assert graph.sink_tasks() == [b, c]
