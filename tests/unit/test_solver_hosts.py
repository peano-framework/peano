"""Unit tests for hosts of coupled solvers (@ref exahype2.api.SolverHost) and for the
flag add_global_tasks, with which a host leaves the load balancing switch and the
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


class _Hosted(exahype2.api.Solver):
    """! Solver which can be hosted: it returns one task per stage """

    def create_task_graph(self, exahype2_project, peano_project, stage, number_of_stages):
        graph = peano4.api.taskgraph.Graph()
        graph.add_task(
            Task(name=f"{self.name}_Reconstruct stage {stage}", entity=Task.GridEntity.CELL)
        )
        return graph


class _NoGridConstruction(exahype2.api.Solver.GridConstruction):
    def construct_output(self, exahype2_project, peano_project, setup_simulation):
        pass


def _solver(scheme, name="Probe", solver_class=exahype2.api.Solver):
    solver = solver_class(name, 2)
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
    exahype2_project = types.SimpleNamespace(periodic_BC=[True, True, True])
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


class _Chain(exahype2.api.SolverHost):
    """! Host which makes the first task of the second solver wait for the last of the first"""

    def couple(self, exahype2_project, peano_project, own_graphs, host_graphs):
        first, second = self.solvers
        last = own_graphs[first].time_step.sink_tasks()
        for graph in second.numerical_scheme.stage_graphs:
            for task in graph.source_tasks():
                for predecessor in last:
                    task.add_dependency(predecessor)


def test_host_realises_each_solver_into_graphs_of_its_own_and_concatenates(tmp_path):
    first, second = _solver(None, "First"), _solver(None, "Second", _Hosted)
    host = _Chain("Pair", [first, second])
    host.numerical_scheme = SCHEMES["fv"](first)
    assert host.global_tasks_solver is first
    assert first.numerical_scheme is host.numerical_scheme
    assert isinstance(second.numerical_scheme, exahype2.api.SolverHost.HostedScheme)
    assert second.numerical_scheme.patch_size == host.numerical_scheme.patch_size
    first.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
        first, volume_size=0.1
    )
    second.grid_construction = _NoGridConstruction(second)
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
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    host.generate_solver_realisation(exahype2_project, peano_project, *graphs)

    names = [task.name for task in graphs[1].tasks]
    assert [n for n in names if n.startswith("First_")] and [
        n for n in names if n.startswith("Second_")
    ]
    # Only the first solver adds the global tasks
    assert [n for n in names if "Dynamic mesh refinement" in n] == [
        "First_Dynamic mesh refinement and load balancing"
    ]
    # The coupling of the host connects the otherwise independent chains
    second_reconstruct = next(
        t for t in graphs[1].tasks if t.name == "Second_Reconstruct stage 0"
    )
    assert second_reconstruct.dependencies


def test_project_registers_the_solvers_of_a_host_but_realises_the_host(tmp_path):
    first, second = _solver(None, "First"), _solver(None, "Second", _Hosted)
    third = _solver("fv", "Third")
    host = _Chain("Pair", [first, second])
    project = exahype2.api.Project(["probe"], "probe", "probe", directory=str(tmp_path))
    project.add_solver(third)
    project.add_solver(host)
    assert project._solvers == [third, first, second]
    assert project._realisations == [third, host]


def test_a_host_needs_solvers():
    with pytest.raises(ValueError):
        _Chain("Empty", [])


def test_hosted_solvers_must_not_have_a_numerical_scheme_of_their_own(tmp_path):
    with pytest.raises(ValueError, match="numerical scheme of its own"):
        _Chain("Pair", [_solver("fv", "First"), _solver(None, "Second", _Hosted)])
    first, second = _solver(None, "First"), _solver(None, "Second", _Hosted)
    host = _Chain("Pair", [first, second])
    host.numerical_scheme = SCHEMES["fv"](first)
    # Assigned behind the back of the host
    second.numerical_scheme = SCHEMES["fv"](second)
    with pytest.raises(ValueError, match="numerical scheme of its own"):
        host.numerical_scheme = SCHEMES["fv"](first)
    exahype2_project = types.SimpleNamespace(periodic_BC=[True, True, True])
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    with pytest.raises(ValueError, match="numerical scheme of its own"):
        host.generate_solver_realisation(
            exahype2_project, peano_project, *(peano4.api.taskgraph.Graph() for _ in range(4))
        )


def test_a_host_without_a_numerical_scheme_cannot_be_realised(tmp_path):
    host = _Chain("Pair", [_solver(None, "First"), _solver(None, "Second", _Hosted)])
    with pytest.raises(ValueError, match="no numerical_scheme"):
        host.generate_solver_realisation(None, None, *(peano4.api.taskgraph.Graph() for _ in range(4)))


def test_graph_source_tasks_are_the_counterpart_of_the_sink_tasks():
    graph = peano4.api.taskgraph.Graph()
    a, b, c = (Task(name=n, entity=Task.GridEntity.CELL) for n in "abc")
    b.add_dependency(a)
    for task in (a, b, c):
        graph.add_task(task)
    assert graph.source_tasks() == [a, c]
    assert graph.sink_tasks() == [b, c]


def test_run_in_parallel_starts_all_branches_with_a_kick_off_task():
    branches = []
    for name in "xy":
        branch = peano4.api.taskgraph.Graph()
        first, second = (Task(name=name + n, entity=Task.GridEntity.CELL) for n in "12")
        second.add_dependency(first)
        branch.add_task(first)
        branch.add_task(second)
        branches.append(branch)
    graph = peano4.api.taskgraph.Graph()
    kick_off = exahype2.api.SolverHost.run_in_parallel(graph, branches)
    assert kick_off in graph.tasks and len(graph.tasks) == 5
    assert graph.source_tasks() == [kick_off]
    assert sorted(t.name for t in graph.sink_tasks()) == ["x2", "y2"]
    assert all(kick_off in b.source_tasks()[0].dependencies for b in branches)
