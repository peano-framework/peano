"""Unit tests for the separate eigenvalue tasks of the Finite Volume schemes:
update cell -> compute max eigenvalue -> reduce max eigenvalue -> time step size
(@ref exahype2.api.fv.actionsets.ComputeMaxEigenvalue)."""

import types

import pytest

import peano4.api
import peano4.api.taskgraph
import exahype2.api
from peano4.api.taskgraph import Task
from peano4.api.taskgraph.compiler import FuseSteps

SCHEMES = {
    "explicit": exahype2.api.fv.RusanovWithExplicitEuler,
    "optimistic": exahype2.api.fv.RusanovWithOptimisticExplicitEuler,
}


def _time_step(scheme, tmp_path):
    solver = exahype2.api.Solver("Probe", 2)
    solver.add_pde_term(exahype2.api.Flux(implementation="F[0] = Q[0]; F[1] = Q[1];"))
    solver.add_pde_term(exahype2.api.MaxEigenvalue(implementation="*lambdaMax = 1.0;"))
    solver.numerical_scheme = SCHEMES[scheme](solver, 9)
    exahype2_project = types.SimpleNamespace(periodic_BC=[True, True, True])
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    graph = peano4.api.taskgraph.Graph()
    solver.numerical_scheme.construct_output(exahype2_project, peano_project, graph)
    return solver, {task.name: task for task in graph.tasks}, graph


@pytest.mark.parametrize("scheme", SCHEMES)
def test_eigenvalue_is_computed_stored_and_reduced_before_the_time_step_size(
    scheme, tmp_path
):
    _, tasks, _ = _time_step(scheme, tmp_path)
    compute = tasks["Probe_Compute max eigenvalue"]
    reduce = tasks["Probe_Reduce max eigenvalue"]
    cfl = tasks["Probe_Compute global CFL condition"]
    assert compute.entity == Task.GridEntity.CELL
    assert reduce.entity == Task.GridEntity.CELL
    assert cfl.entity == Task.GridEntity.GLOBAL_END_OF_STEP
    assert reduce.dependencies == [compute]
    assert reduce in cfl.dependencies
    assert tasks[f"Probe_{exahype2.api.Solver.NumericalScheme.DYNAMIC_MESH_REFINEMENT_TASK_NAME}"].dependencies.count(reduce) == 1


def test_time_step_size_is_computed_after_the_roll_over(tmp_path):
    # Sommerfeld roll overs read the time step size, which the CFL task replaces
    _, tasks, _ = _time_step("explicit", tmp_path)
    cfl = tasks["Probe_Compute global CFL condition"]
    assert "Probe_Roll over faces" not in tasks
    assert tasks["Probe_Project on faces"] in cfl.dependencies
    _, tasks, _ = _time_step("optimistic", tmp_path)
    cfl = tasks["Probe_Compute global CFL condition"]
    assert tasks["Probe_Project on faces"] in cfl.dependencies


def test_explicit_eigenvalue_follows_the_update_and_runs_beside_the_projection(
    tmp_path,
):
    _, tasks, _ = _time_step("explicit", tmp_path)
    assert tasks["Probe_Compute max eigenvalue"].dependencies == [
        tasks["Probe_Update cell"]
    ]
    assert tasks["Probe_Project on faces"].dependencies == [tasks["Probe_Update cell"]]


def test_optimistic_eigenvalue_follows_both_branches_and_runs_beside_the_spawn(
    tmp_path,
):
    _, tasks, _ = _time_step("optimistic", tmp_path)
    assert set(tasks["Probe_Compute max eigenvalue"].dependencies) == {
        tasks["Probe_Update cell"],
        tasks["Probe_Insert patch interior"],
    }
    spawn = tasks["Probe_Spawn patch interior update"]
    assert tasks["Probe_Compute max eigenvalue"] not in spawn.dependencies
    assert spawn in tasks["Probe_Compute global CFL condition"].dependencies


@pytest.mark.parametrize("scheme", SCHEMES)
def test_update_does_not_store_an_eigenvalue(scheme, tmp_path):
    _, tasks, _ = _time_step(scheme, tmp_path)
    for name, task in tasks.items():
        if "Update cell" in name or "Insert patch interior" in name:
            continue
        for action_set in task.action_sets:
            body = action_set.get_body_of_operation(
                peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
            )
            if type(action_set).__name__ == "ComputeMaxEigenvalue":
                assert "max_eigenvalue_kernel" not in body
                assert "setMaxEigenvalue( 0" in body
    for name in ["Probe_Update cell", "Probe_Update cell shell"]:
        if name not in tasks:
            continue
        for action_set in tasks[name].action_sets:
            body = action_set.get_body_of_operation(
                peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
            )
            assert "setMaxEigenvalue" not in body


@pytest.mark.parametrize("scheme", SCHEMES)
def test_update_eigenvalue_and_reduction_fold_into_one_traversal(scheme, tmp_path):
    _, tasks, graph = _time_step(scheme, tmp_path)
    steps = FuseSteps.translate(graph, "S", None, None)
    names = [
        [type(action_set).__name__ for action_set in step.action_sets] for step in steps
    ]
    holders = [
        step_names
        for step_names in names
        if "UpdateCell" in step_names and "ComputeMaxEigenvalue" in step_names
    ]
    assert len(holders) == 1, names
    # The projection writes the face data which the reconstruction reads at the
    # beginning of a traversal, i.e. it follows in the next one, together with
    # the reduction. The time step size acts in the last traversal
    assert "ProjectPatchOntoFaces" not in holders[0]
    assert "ReduceMaxEigenvalue" not in holders[0]
    # The reconstruction interpolates the hanging faces itself, i.e. no
    # traversal is spent on the interpolation
    assert len(names) == 2, names
    assert "ProjectPatchOntoFaces" in names[-1]
    assert "ReduceMaxEigenvalue" in names[-1]
    assert "ComputeGlobalCFLCondition" in names[-1]
