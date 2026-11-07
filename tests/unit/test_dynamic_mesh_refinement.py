import types

import pytest

import peano4.api
import peano4.api.taskgraph
import peano4.api.solversteps

import exahype2.api
import toolbox.loadbalancing.api.actionsets

Task = peano4.api.taskgraph.Task
ActionSet = peano4.api.solversteps.ActionSet

SCHEMES = {
    "fv": lambda solver, **bounds: exahype2.api.fv.RusanovWithExplicitEuler(
        solver, 4, **bounds
    ),
    "fv-optimistic": lambda solver, **bounds: exahype2.api.fv.RusanovWithOptimisticExplicitEuler(
        solver, 4, **bounds
    ),
    "rkfd4": lambda solver, **bounds: exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, patch_size=4, rk_order=2, **bounds
    ),
}


def _solver(scheme, criterion=True, **bounds):
    solver = exahype2.api.Solver("Probe", 2)
    solver.add_pde_term(exahype2.api.Flux(implementation="F[0] = Q[0]; F[1] = Q[1];"))
    solver.add_pde_term(exahype2.api.MaxEigenvalue(implementation="*lambdaMax = 1.0;"))
    if criterion:
        solver.add_pde_term(exahype2.api.RefineErasePointwisely(implementation=""))
    solver.numerical_scheme = SCHEMES[scheme](solver, **bounds)
    return solver


def _last_task(solver, tmp_path):
    """!
    The one task of the time step on which no other task depends, apart from
    the tasks that act at the end of the time step (such as the one that
    computes the time step size)
    """
    exahype2_project = types.SimpleNamespace(periodic_BC=[True, True, True])
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    time_step = peano4.api.taskgraph.Graph()
    solver.numerical_scheme.construct_output(exahype2_project, peano_project, time_step)
    dependencies = {
        id(dependency) for task in time_step.tasks for dependency in task.dependencies
    }
    [last] = [
        task
        for task in time_step.tasks
        if id(task) not in dependencies
        and task.entity != Task.GridEntity.GLOBAL_END_OF_STEP
    ]
    return last


def _criterion_body(action_set):
    return action_set.get_body_of_operation(ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_last_task_evaluates_criterion_and_switches_on_load_balancing(scheme, tmp_path):
    last = _last_task(_solver(scheme), tmp_path)
    assert last.name == "Probe_Dynamic mesh refinement"
    assert last.entity == Task.GridEntity.CELL
    dynamic_mesh_refinement, load_balancing = last.action_sets
    assert isinstance(
        dynamic_mesh_refinement, exahype2.api.blockstructured.DynamicMeshRefinement
    )
    assert isinstance(
        load_balancing, toolbox.loadbalancing.api.actionsets.LoadBalancing
    )
    body = load_balancing.get_body_of_unprepareTraversal()
    assert body.index("enable(true)") < body.index("finishStep()")


@pytest.mark.parametrize("scheme", SCHEMES)
def test_time_step_never_hands_events_to_peano(scheme, tmp_path):
    [dynamic_mesh_refinement, _] = _last_task(_solver(scheme), tmp_path).action_sets
    assert "RefinementControlService" not in (
        dynamic_mesh_refinement.get_body_of_createGridControlEventsForNextTraversal()
    )
    assert "refineErasePointwisely" in _criterion_body(dynamic_mesh_refinement)
    assert "RefinementControlService::getInstance().finishStep()" in (
        dynamic_mesh_refinement.get_body_of_unprepareTraversal()
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_no_criterion_means_no_evaluation(scheme, tmp_path):
    last = _last_task(_solver(scheme, criterion=False), tmp_path)
    assert last.name == "Probe_Dynamic mesh refinement"
    [load_balancing] = last.action_sets
    assert isinstance(
        load_balancing, toolbox.loadbalancing.api.actionsets.LoadBalancing
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_gradient_criterion_yields_gradient_action_set(scheme, tmp_path):
    solver = _solver(scheme, criterion=False)
    solver.add_pde_term(exahype2.api.GradientRefinement(implementation=""))
    [dynamic_mesh_refinement, _] = _last_task(solver, tmp_path).action_sets
    assert isinstance(
        dynamic_mesh_refinement,
        exahype2.api.blockstructured.DynamicMeshRefinementWithGradients,
    )
    body = _criterion_body(dynamic_mesh_refinement)
    assert "refineErasePointwisely" not in body
    assert "for (int normal=0; normal<Dimensions; normal++)" in body
    assert "for (int side=0; side<2; side++)" in body
    assert "::exahype2::fv::computeOneSidedGradient(" in body
    assert "refineEraseWithGradient(" in body
    assert "double gradient[2 + 0];" in body
    assert "anyVolumeWantsToRefine or command" in body
    assert "allVolumesWantToErase  and command" in body
    assert "finishStep()" in dynamic_mesh_refinement.get_body_of_unprepareTraversal()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_time_step_does_not_veto_erase(scheme, tmp_path):
    [dynamic_mesh_refinement, _] = _last_task(_solver(scheme), tmp_path).action_sets
    assert "RefinementCommand::Erase," in _criterion_body(dynamic_mesh_refinement)


def test_two_criteria_are_rejected(tmp_path):
    solver = _solver("fv")
    solver.add_pde_term(exahype2.api.GradientRefinement(implementation=""))
    with pytest.raises(ValueError):
        _last_task(solver, tmp_path)


def test_mesh_size_is_unconstrained_by_default(tmp_path):
    [dynamic_mesh_refinement, _] = _last_task(_solver("fv"), tmp_path).action_sets
    body = _criterion_body(dynamic_mesh_refinement)
    assert "tarch::la::max( marker.h() ) >=" not in body
    assert "3.0 * tarch::la::max( marker.h() ) <" not in body


@pytest.mark.parametrize("scheme", SCHEMES)
def test_numerical_scheme_bounds_mesh_size(scheme, tmp_path):
    solver = _solver(scheme, min_volume_size=0.25, max_volume_size=1.0)
    [dynamic_mesh_refinement, _] = _last_task(solver, tmp_path).action_sets
    body = _criterion_body(dynamic_mesh_refinement)
    # Volume sizes times patch size 4
    assert "tarch::la::max( marker.h() ) >= 1.0" in body
    assert "3.0 * tarch::la::max( marker.h() ) < 4.0" in body


@pytest.mark.parametrize("scheme", SCHEMES)
def test_numerical_scheme_rejects_inverted_bounds(scheme):
    with pytest.raises(AssertionError):
        _solver(scheme, min_volume_size=1.0, max_volume_size=0.5)
