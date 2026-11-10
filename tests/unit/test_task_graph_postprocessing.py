import types

import pytest

import peano4.api
import peano4.api.taskgraph
import peano4.api.solversteps

import exahype2.api
import toolbox.blockstructured.api

Task = peano4.api.taskgraph.Task


def _cell_task(name):
    return Task(name=name, entity=Task.GridEntity.CELL)


def test_insert_after_redirects_successors():
    graph = peano4.api.taskgraph.Graph()
    a, b, c, d = (_cell_task(name) for name in "abcd")
    b.add_dependency(a)
    c.add_dependency(a)
    c.add_dependency(b)
    for task in [a, b, c]:
        graph.add_task(task)

    graph.insert_after(a, d)

    assert d in graph.tasks
    assert d.dependencies == [a]
    assert b.dependencies == [d]
    assert c.dependencies == [d, b]
    order = graph.ordered_tasks()
    assert order.index(a) < order.index(d) < order.index(b) < order.index(c)


def test_insert_after_last_task():
    graph = peano4.api.taskgraph.Graph()
    a, b = _cell_task("a"), _cell_task("b")
    graph.add_task(a)
    graph.insert_after(a, b)
    assert graph.ordered_tasks() == [a, b]


def test_insert_after_several_anchors():
    graph = peano4.api.taskgraph.Graph()
    a, b, c, d, e = (_cell_task(name) for name in "abcde")
    b.add_dependency(a)
    c.add_dependency(a)
    d.add_dependency(b)
    d.add_dependency(c)
    for task in [a, b, c, d]:
        graph.add_task(task)

    graph.insert_after([b, c], e)

    assert e.dependencies == [b, c]
    assert d.dependencies == [e]
    order = graph.ordered_tasks()
    assert order.index(b) < order.index(e) < order.index(d)
    assert order.index(c) < order.index(e)


def test_insert_after_rejects_dependent_anchors():
    graph = peano4.api.taskgraph.Graph()
    a, b, c, d = (_cell_task(name) for name in "abcd")
    b.add_dependency(a)
    c.add_dependency(b)
    for task in [a, b, c]:
        graph.add_task(task)
    with pytest.raises(AssertionError):
        graph.insert_after([a, c], d)


def test_find_tasks_by_tag():
    graph = peano4.api.taskgraph.Graph()
    a = Task(name="a", entity=Task.GridEntity.CELL, tags={"x"})
    b = _cell_task("b")
    c = _cell_task("c")
    c.tags.add("x")
    for task in [a, b, c]:
        graph.add_task(task)
    assert graph.find_tasks("x") == [a, c]
    assert graph.find_tasks("y") == []


def test_find_tasks_with_several_tags():
    graph = peano4.api.taskgraph.Graph()
    a = Task(name="a", entity=Task.GridEntity.CELL, tags=["x", "y"])
    b = Task(name="b", entity=Task.GridEntity.CELL, tags={"y"})
    c = _cell_task("c")
    for task in [a, b, c]:
        graph.add_task(task)
    assert graph.find_tasks("x") == [a]
    assert graph.find_tasks("y") == [a, b]
    assert graph.find_tasks("x", "y") == [a, b]


def _postprocessed_time_step(
    scheme_factory, grid_construction_factory, enforce, tmp_path
):
    """! Time step graph of a CCZ4 solver with the given constraint enforcement"""
    import exagrype.api

    solver = exagrype.api.FOCCZ4Solver("Probe")
    solver.numerical_scheme = scheme_factory(solver)
    solver.grid_construction = grid_construction_factory(solver)
    solver.enforce_ccz4_constraint(enforce)

    exahype2_project = exahype2.api.Project(
        ["probe"], "probe", "probe", directory=str(tmp_path)
    )
    exahype2_project.set_global_simulation_parameters(
        dimensions=3,
        offset=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 1.0],
        min_end_time=1.0,
        periodic_BC=[True, True, True],
    )
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    solver.generate_solver_realisation(exahype2_project, peano_project, *graphs)
    return solver, {task.name: task for task in graphs[1].tasks}


def _enforcements(tasks):
    return sorted(name for name in tasks if "_Enforce CCZ4 constraints after" in name)


def _runge_kutta(solver):
    return exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, patch_size=4, rk_order=3
    )


def _runge_kutta_grid(solver):
    return exahype2.api.rkfd4.SetupSimulationOverRegularGrid(solver, volume_size=0.1)


def _fv_grid(solver):
    return exahype2.api.fv.SetupSimulationOverRegularGrid(solver, volume_size=0.1)


def test_no_constraint_enforcement_by_default(tmp_path):
    import exagrype.api

    solver, tasks = _postprocessed_time_step(
        _runge_kutta,
        _runge_kutta_grid,
        exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint.No,
        tmp_path,
    )
    assert _enforcements(tasks) == []


def test_constraint_enforcement_after_each_runge_kutta_stage(tmp_path):
    import exagrype.api

    solver, tasks = _postprocessed_time_step(
        _runge_kutta,
        _runge_kutta_grid,
        exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint.IntermediateSteps,
        tmp_path,
    )
    assert len(_enforcements(tasks)) == 3
    for anchor, successor, patch in [
        ("Prepare stage 1", "Project stage 1 onto faces", solver._stage_input),
        ("Prepare stage 2", "Project stage 2 onto faces", solver._stage_input),
        ("Complete time step", "Project solution onto faces", solver._patch),
    ]:
        clip = tasks[f"Probe_Enforce CCZ4 constraints after {anchor}"]
        assert clip.dependencies == [tasks[f"Probe_{anchor}"]]
        assert tasks[f"Probe_{successor}"].dependencies == [clip]
        [action_set] = clip.action_sets
        assert isinstance(
            action_set, toolbox.blockstructured.api.ApplyPointWiselyToPatch
        )
        body = action_set.get_body_of_operation(
            peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
        )
        assert "::exagrype::ccz4::enforceCCZ4constraints(Q);" in body
        assert f"fineGridCell{patch.name}.value" in body
    # With adaptive time stepping, the maximum eigenvalue of the constrained
    # solution is determined after the projection
    assert (
        tasks["Probe_Project solution onto faces"]
        in tasks["Probe_Compute max eigenvalue"].dependencies
    )


def test_constraint_enforcement_after_final_runge_kutta_solution_only(tmp_path):
    import exagrype.api

    solver, tasks = _postprocessed_time_step(
        _runge_kutta,
        _runge_kutta_grid,
        exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint.OnlyFinalSolution,
        tmp_path,
    )
    assert _enforcements(tasks) == [
        "Probe_Enforce CCZ4 constraints after Complete time step"
    ]


def test_constraint_enforcement_after_both_writers_of_optimistic_finite_volumes(
    tmp_path,
):
    import exagrype.api

    solver, tasks = _postprocessed_time_step(
        lambda solver: exahype2.api.fv.RusanovWithOptimisticExplicitEuler(solver, 4),
        _fv_grid,
        exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint.OnlyFinalSolution,
        tmp_path,
    )
    [name] = _enforcements(tasks)
    clip = tasks[name]
    # One task after both branches, guarded by the plain cell guard
    assert set(clip.dependencies) == {
        tasks["Probe_Update cell"],
        tasks["Probe_Insert patch interior"],
    }
    [action_set] = clip.action_sets
    assert action_set.guard == solver.grid_selector.createCellGuard()
    for consumer in [
        "Compute max eigenvalue",
        "Spawn patch interior update",
        "Discard patch interior",
    ]:
        dependencies = tasks[f"Probe_{consumer}"].dependencies
        assert clip in dependencies
        assert tasks["Probe_Update cell"] not in dependencies
        assert tasks["Probe_Insert patch interior"] not in dependencies


def test_constraint_enforcement_after_plain_finite_volumes(tmp_path):
    import exagrype.api

    solver, tasks = _postprocessed_time_step(
        lambda solver: exahype2.api.fv.RusanovWithExplicitEuler(solver, 4),
        _fv_grid,
        exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint.IntermediateSteps,
        tmp_path,
    )
    clip = tasks["Probe_Enforce CCZ4 constraints after Update cell"]
    assert _enforcements(tasks) == [clip.name]
    assert tasks["Probe_Compute max eigenvalue"].dependencies == [clip]


def test_constraint_enforcement_rejects_unknown_scheme(tmp_path):
    import exagrype.api

    class UnknownScheme(exahype2.api.Solver.NumericalScheme):
        ## Required by the generation of the abstract solver
        patch_size = 4
        halo_size = 1

        def construct_output(
            self, exahype2_project, peano_project, time_step, add_global_tasks=True
        ):
            pass

    class NoGridConstruction(exahype2.api.Solver.GridConstruction):
        def construct_output(self, exahype2_project, peano_project, setup_simulation):
            pass

    with pytest.raises(TypeError, match="UnknownScheme"):
        _postprocessed_time_step(
            UnknownScheme,
            NoGridConstruction,
            exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint.OnlyFinalSolution,
            tmp_path,
        )


def test_sink_tasks_of_forking_graph():
    graph = peano4.api.taskgraph.Graph()
    a, b, c, d = (_cell_task(name) for name in "abcd")
    b.add_dependency(a)
    c.add_dependency(a)
    d.add_dependency(b)
    for task in [a, b, c, d]:
        graph.add_task(task)

    assert graph.sink_tasks() == [c, d]


def test_sink_tasks_of_chain_and_empty_graph():
    graph = peano4.api.taskgraph.Graph()
    assert graph.sink_tasks() == []
    a, b = _cell_task("a"), _cell_task("b")
    b.add_dependency(a)
    graph.add_task(a)
    graph.add_task(b)
    assert graph.sink_tasks() == [b]
