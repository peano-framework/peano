"""Unit tests for the load balancing within the setup over a regular grid
(@ref exahype2.api.fv.SetupSimulationOverRegularGrid,
@ref exahype2.api.rkfd4.SetupSimulationOverRegularGrid)."""

import types

import peano4.api
import peano4.api.taskgraph
import exahype2.api
import toolbox.loadbalancing.api.actionsets
import pytest


def _setup_graph(scheme, tmp_path):
    solver = exahype2.api.Solver("Probe", 4)
    solver.add_pde_term(exahype2.api.Flux(implementation=""))
    solver.add_pde_term(exahype2.api.MaxEigenvalue(implementation=""))
    solver.add_pde_term(exahype2.api.InitPointWisely(implementation=""))
    if scheme == "fv":
        solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(solver, 9)
        package = exahype2.api.fv
    else:
        solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
            solver, patch_size=9, rk_order=2
        )
        package = exahype2.api.rkfd4
    solver.grid_construction = package.SetupSimulationOverRegularGrid(
        solver, volume_size=0.02
    )
    peano_project = peano4.api.Project(["probe"], "probe", directory=str(tmp_path))
    exahype2_project = types.SimpleNamespace(periodic_BC=[True, True, True])
    graph = peano4.api.taskgraph.Graph()
    solver.grid_construction.construct_output(exahype2_project, peano_project, graph)
    return graph


@pytest.mark.parametrize("scheme", ["fv", "rkfd4"])
def test_task_waiting_for_the_load_balancing_informs_it(tmp_path, scheme):
    """!

    The load balancing only changes its state in finishStep(). A task which
    repeats until the load balancing has stagnated hence has to call it itself.
    If another task calls it, the repetition only terminates if the task graph
    compiler happens to fuse both tasks.

    """
    graph = _setup_graph(scheme, tmp_path)
    waiting = [
        task
        for task in graph.tasks
        if task.has_to_repeat and "hasStagnated()" in task.has_to_repeat
    ]
    assert waiting
    for task in waiting:
        assert any(
            isinstance(action_set, toolbox.loadbalancing.api.actionsets.LoadBalancing)
            for action_set in task.action_sets
        ), task.name


@pytest.mark.parametrize("scheme", ["fv", "rkfd4"])
def test_refinement_terminates_only_after_a_sweep_which_may_refine(scheme, tmp_path):
    """!

    The refinement pauses while the load balancing has split recently, and a
    paused refinement leaves the grid stationary. The repetition hence has to
    wait for hasMeshBeenStationary(), which holds one sweep after the pause.

    """
    graph = _setup_graph(scheme, tmp_path)
    [refine] = [
        task
        for task in graph.tasks
        if task.has_to_repeat and "hasStagnated()" in task.has_to_repeat
    ]
    assert "not ::toolbox" in refine.has_to_repeat
    assert "hasMeshBeenStationary()" in refine.has_to_repeat
    assert "hasSplitRecently()" not in refine.has_to_repeat
    guard = refine.action_sets[0].guard
    assert "hasSplitRecently()" in guard
