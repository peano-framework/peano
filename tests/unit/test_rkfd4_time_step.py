import types

import pytest

import peano4.api
import peano4.api.taskgraph
import peano4.api.solversteps

import exahype2.api


def _solver(time_step_size=None, rk_order=3, boundary="dirichlet"):
    """!

    @param boundary: str
      "dirichlet", "sommerfeld", or None for a solver without boundary
      condition term, i.e. for a domain that is periodic along all axes.

    """
    solver = exahype2.api.Solver("Probe", 2)
    solver.add_pde_term(exahype2.api.Flux(implementation="F[0] = Q[0]; F[1] = Q[1];"))
    solver.add_pde_term(exahype2.api.MaxEigenvalue(implementation="*lambdaMax = 1.0;"))
    if boundary == "dirichlet":
        solver.add_pde_term(
            exahype2.api.DirichletBoundaryConditions(implementation="Q[0] = 0.0;")
        )
    elif boundary == "sommerfeld":
        solver.add_pde_term(
            exahype2.api.SommerfeldBoundaryConditions(implementation="Q[0] = 0.0;")
        )
    solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, patch_size=4, rk_order=rk_order, time_step_size=time_step_size
    )
    return solver


def _construct_time_step(solver, directory, periodic_BC=(False, False, False)):
    exahype2_project = types.SimpleNamespace(periodic_BC=list(periodic_BC))
    peano_project = peano4.api.Project(["probe"], "probe", str(directory))
    time_step = peano4.api.taskgraph.Graph()
    solver.numerical_scheme.construct_output(exahype2_project, peano_project, time_step)
    return {task.name: task for task in time_step.tasks}


def test_one_reconstruction_per_stage_and_chain_of_stages(tmp_path):
    tasks = _construct_time_step(_solver(rk_order=3), tmp_path)
    assert set(tasks) == {
        "Probe_Apply Dirichlet BCs stage 0",
        "Probe_Apply Dirichlet BCs stage 1",
        "Probe_Apply Dirichlet BCs stage 2",
        "Probe_Reconstruct stage 0",
        "Probe_Reconstruct stage 1",
        "Probe_Reconstruct stage 2",
        "Probe_Disable load balancing",
        "Probe_Evaluate Runge-Kutta right hand side stage 0",
        "Probe_Evaluate Runge-Kutta right hand side stage 1",
        "Probe_Evaluate Runge-Kutta right hand side stage 2",
        "Probe_Prepare stage 1",
        "Probe_Prepare stage 2",
        "Probe_Complete time step",
        "Probe_Project stage 1 onto faces",
        "Probe_Project stage 2 onto faces",
        "Probe_Project solution onto faces",
        "Probe_Roll over stage 1 faces",
        "Probe_Roll over stage 2 faces",
        "Probe_Roll over solution faces",
        "Probe_Compute max eigenvalue",
        "Probe_Reduce max eigenvalue",
        "Probe_Compute time step size",
        f"Probe_{exahype2.api.Solver.NumericalScheme.DYNAMIC_MESH_REFINEMENT_TASK_NAME}",
    }
    # The maximum eigenvalue of the new solution is determined right after the
    # projection, i.e. in parallel to the roll over. If it ran in parallel to the
    # projection, its producer-consumer split would push the roll over into the
    # next mesh sweep, away from the temporary face data of the projection
    # (adrs/0002-eigenvalue-follows-projection.md)
    assert (
        tasks["Probe_Complete time step"]
        in tasks["Probe_Project solution onto faces"].dependencies
    )
    assert (
        tasks["Probe_Project solution onto faces"]
        in tasks["Probe_Compute max eigenvalue"].dependencies
    )
    assert (
        tasks["Probe_Compute max eigenvalue"]
        not in tasks["Probe_Roll over solution faces"].dependencies
    )
    # The eigenvalue is stored per cell, a task of its own reduces it, and the
    # time step size is computed from the outcome
    assert (
        tasks["Probe_Compute max eigenvalue"]
        in tasks["Probe_Reduce max eigenvalue"].dependencies
    )
    assert (
        tasks["Probe_Reduce max eigenvalue"]
        in tasks["Probe_Compute time step size"].dependencies
    )
    for stage in [1, 2]:
        assert (
            tasks[f"Probe_Prepare stage {stage}"]
            in tasks[f"Probe_Project stage {stage} onto faces"].dependencies
        )
        # The projection hosts the restriction, the roll over follows
        assert (
            tasks[f"Probe_Project stage {stage} onto faces"]
            in tasks[f"Probe_Roll over stage {stage} faces"].dependencies
        )
        # The reconstruction interpolates the faces itself, i.e. it waits for the roll over
        assert (
            tasks[f"Probe_Roll over stage {stage} faces"]
            in tasks[f"Probe_Reconstruct stage {stage}"].dependencies
        )
    # The boundary values have to be set after the previous stage's roll-over
    assert (
        tasks["Probe_Roll over stage 1 faces"]
        in tasks["Probe_Apply Dirichlet BCs stage 1"].dependencies
    )
    assert (
        tasks["Probe_Apply Dirichlet BCs stage 1"]
        in tasks["Probe_Reconstruct stage 1"].dependencies
    )
    assert (
        tasks["Probe_Roll over solution faces"]
        in tasks["Probe_Compute time step size"].dependencies
    )
    # The load balancing is switched off before the first stage only
    assert (
        tasks["Probe_Disable load balancing"]
        in tasks["Probe_Apply Dirichlet BCs stage 0"].dependencies
    )
    # The reconstruction holds the interpolation and the reconstruction
    assert [
        type(a).__name__ for a in tasks["Probe_Reconstruct stage 1"].action_sets
    ] == [
        "InterpolateHangingFaces",
        "ReconstructPatchWithHalo",
    ]
    assert [
        type(a).__name__ for a in tasks["Probe_Project stage 1 onto faces"].action_sets
    ] == ["ProjectPatchOntoFaces", "RestrictHangingFaces"]
    # All other tasks of the scheme hold exactly one action set, so that the task
    # graph compiler can run the tasks of several solvers in parallel
    for name in [
        "Probe_Evaluate Runge-Kutta right hand side stage 1",
        "Probe_Roll over stage 1 faces",
    ]:
        assert len(tasks[name].action_sets) == 1, name


def _tagged(tasks, tag):
    return {name for name, task in tasks.items() if tag in task.tags}


def test_stage_combinations_are_tagged_as_solution_updates(tmp_path):
    solver = _solver(rk_order=3)
    tasks = _construct_time_step(solver, tmp_path)
    final = _tagged(tasks, exahype2.api.Solver.SolutionUpdateTaskTag)
    intermediate = _tagged(tasks, exahype2.api.Solver.IntermediateSolutionUpdateTag)
    assert set(final) == {"Probe_Complete time step"}
    assert set(intermediate) == {"Probe_Prepare stage 1", "Probe_Prepare stage 2"}
    tagged = final | intermediate
    # The combination must not project onto the faces itself: the projection
    # has to see the postprocessed solution
    for name in tagged:
        assert not any(
            isinstance(action_set, exahype2.api.rkfd4.actionsets.ProjectPatchOntoFaces)
            for action_set in tasks[name].action_sets
        )


def test_boundary_values_use_stage_time_stamp(tmp_path):
    tasks = _construct_time_step(_solver(rk_order=3), tmp_path)
    [action_set] = tasks["Probe_Apply Dirichlet BCs stage 2"].action_sets
    body = action_set.get_body_of_operation(
        peano4.api.solversteps.ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME
    )
    # c_2 = 1 for Kutta's third-order scheme
    assert (
        "repositories::instanceOfProbe.getMinTimeStamp() + 1.0 * "
        "repositories::instanceOfProbe.getMinTimeStepSize()," in body
    )


def test_periodic_domain_needs_no_boundary_tasks(tmp_path):
    tasks = _construct_time_step(
        _solver(rk_order=2, boundary=None), tmp_path, periodic_BC=(True, True, True)
    )
    assert not any("Dirichlet" in name for name in tasks)
    assert not any(
        isinstance(
            action_set,
            exahype2.api.rkfd4.actionsets.RollOverUpdatedFaceWithSommerfeldBoundaryConditions,
        )
        for task in tasks.values()
        for action_set in task.action_sets
    )


def test_periodic_domain_rejects_boundary_condition_term(tmp_path):
    with pytest.raises(ValueError, match="all axes are periodic"):
        _construct_time_step(_solver(), tmp_path, periodic_BC=(True, True, True))


def _roll_overs(tasks):
    return {
        name: action_set
        for name, task in tasks.items()
        for action_set in task.action_sets
        if isinstance(action_set, exahype2.api.rkfd4.actionsets.RollOverUpdatedFace)
    }


def test_sommerfeld_replaces_roll_over_and_omits_boundary_tasks(tmp_path):
    tasks = _construct_time_step(_solver(rk_order=3, boundary="sommerfeld"), tmp_path)
    assert not any("Dirichlet" in name for name in tasks)
    roll_overs = _roll_overs(tasks)
    assert set(roll_overs) == {
        "Probe_Roll over stage 1 faces",
        "Probe_Roll over stage 2 faces",
        "Probe_Roll over solution faces",
    }
    # Kutta's third-order scheme has c = 0, 1/2, 1, i.e. the roll overs
    # advance the ghost grid cells by 1/2, 1/2 and 0 times the time step size
    for name, increment in [
        ("Probe_Roll over stage 1 faces", "0.5"),
        ("Probe_Roll over stage 2 faces", "0.5"),
        ("Probe_Roll over solution faces", "0.0"),
    ]:
        action_set = roll_overs[name]
        assert isinstance(
            action_set,
            exahype2.api.rkfd4.actionsets.RollOverUpdatedFaceWithSommerfeldBoundaryConditions,
        )
        body = action_set.get_body_of_operation(
            peano4.api.solversteps.ActionSet.OPERATION_TOUCH_FACE_LAST_TIME
        )
        assert "::exahype2::blockstructured::applySommerfeldBoundaryConditions(" in body
        assert (
            f"{increment} * repositories::instanceOfProbe.getMinTimeStepSize()," in body
        )
        assert "tarch::la::Vector<Dimensions, double>({0.0, 0.0, 0.0})" in body
        # The Sommerfeld update has to precede the copy into the persistent data
        assert body.index("applySommerfeldBoundaryConditions") < body.index(
            "copyHalfOfHalo"
        )


def test_sommerfeld_rejects_decreasing_stage_time_stamps(tmp_path, monkeypatch):
    solver = _solver(rk_order=3, boundary="sommerfeld")
    # The third-order scheme of Shu and Osher has c = 0, 1, 1/2
    monkeypatch.setattr(
        solver.numerical_scheme.butcher_tableau,
        "relative_time_step_sizes",
        lambda: [0.0, 1.0, 0.5],
    )
    with pytest.raises(ValueError, match="decrease"):
        _construct_time_step(solver, tmp_path)


def test_dirichlet_accepts_decreasing_stage_time_stamps(tmp_path, monkeypatch):
    solver = _solver(rk_order=3)
    monkeypatch.setattr(
        solver.numerical_scheme.butcher_tableau,
        "relative_time_step_sizes",
        lambda: [0.0, 1.0, 0.5],
    )
    tasks = _construct_time_step(solver, tmp_path)
    assert "Probe_Apply Dirichlet BCs stage 2" in tasks


def test_stage_combinations_use_rows_of_butcher_tableau(tmp_path):
    solver = _solver(rk_order=4, boundary=None)
    tasks = _construct_time_step(solver, tmp_path, periodic_BC=(True, True, True))
    tableau = solver.numerical_scheme.butcher_tableau

    def combination(task_name):
        [action_set] = [
            action_set
            for action_set in tasks[task_name].action_sets
            if isinstance(
                action_set, exahype2.api.rkfd4.actionsets.ComputeLinearCombination
            )
        ]
        return action_set

    for stage in range(1, 4):
        action_set = combination(f"Probe_Prepare stage {stage}")
        assert action_set.weights == tableau.weight_matrix()[stage]
        assert action_set.output is solver._stage_input
    final = combination("Probe_Complete time step")
    assert final.weights == tableau.final_estimate_weights()
    assert final.output is solver._patch


def test_kernels_are_generated_for_both_dimensions(tmp_path):
    _construct_time_step(_solver(), tmp_path)
    kernels = tmp_path / "kernels"
    for name in [
        "Probe_fd4_2d",
        "Probe_fd4_3d",
        "Probe_fd4_eigenvalue_2d",
        "Probe_fd4_eigenvalue_3d",
        "rungekutta_linear_combination",
    ]:
        assert (kernels / (name + ".h")).exists(), name
        assert (kernels / (name + ".cpph")).exists(), name
    wrapper = (kernels / "Probe_fd4.h").read_text()
    assert '#include "Probe_fd4_2d.h"' in wrapper
    assert '#include "Probe_fd4_eigenvalue_3d.h"' in wrapper
    assert "Probe_rightHandSideWithFD4" in (kernels / "Probe_fd4_2d.cpph").read_text()


def test_fixed_time_step_size_skips_eigenvalue(tmp_path):
    solver = _solver(time_step_size=0.01, boundary=None)
    tasks = _construct_time_step(solver, tmp_path, periodic_BC=(True, True, True))
    assert not (tmp_path / "kernels" / "Probe_fd4_eigenvalue_2d.h").exists()
    assert not any(
        isinstance(action_set, exahype2.api.rkfd4.actionsets.ComputeMaxEigenvalue)
        for task in tasks.values()
        for action_set in task.action_sets
    )
    [time_step_size] = tasks["Probe_Compute time step size"].action_sets
    assert (
        "const double admissibleTimeStepSize = 0.01;"
        in time_step_size.get_body_of_unprepareTraversal()
    )


def test_right_hand_side_uses_stage_time_stamp_and_estimate_slot(tmp_path):
    solver = _solver(rk_order=4, boundary=None)
    tasks = _construct_time_step(solver, tmp_path, periodic_BC=(True, True, True))
    [action_set] = tasks[
        "Probe_Evaluate Runge-Kutta right hand side stage 2"
    ].action_sets
    body = action_set.get_body_of_operation(
        peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
    )
    # c_2 = 2/3 for Kutta's 3/8 rule, and each stage's estimate holds 2*4^d values
    assert "getMinTimeStamp() + 0.6666666666666666 * timeStepSize" in body
    assert "fineGridCellProbeQRhsEstimates.value + 64;" in body
    assert "fineGridCellProbeQRhsEstimates.value + 256;" in body
    assert (
        "Probe_rightHandSideWithFD4<\n    4,\n    2,\n    2,\n    0\n>(\n    patchData,\n    [&]("
        in body
    )


def test_adaptive_time_stepping_requires_max_eigenvalue(tmp_path):
    solver = exahype2.api.Solver("Probe", 1)
    solver.add_pde_term(exahype2.api.Flux(implementation="F[0] = Q[0];"))
    solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, patch_size=4, rk_order=2
    )
    with pytest.raises(AssertionError):
        _construct_time_step(solver, tmp_path, periodic_BC=(True, True, True))


def test_stage_input_is_loaded_once_after_each_store(tmp_path):
    """!
    Peano's stacks require that each stored datum is loaded by the next
    sweep which uses it. The reconstruction of a later stage is the last
    reader of the stage input, as the next combination writes it from
    scratch, so it must not store it.
    """
    solver = _solver(rk_order=4, boundary=None)
    tasks = _construct_time_step(solver, tmp_path, periodic_BC=(True, True, True))
    Access = peano4.api.taskgraph.Task.DataAccess
    assert (
        tasks["Probe_Reconstruct stage 0"].cell_data_access[solver._patch]
        == Access.READ_AND_WRITE
    )
    for stage in range(1, 4):
        reconstruction = tasks[f"Probe_Reconstruct stage {stage}"]
        assert (
            reconstruction.cell_data_access[solver._stage_input]
            == Access.READ_AND_DISCARD
        )
        assert (
            tasks[f"Probe_Prepare stage {stage}"].cell_data_access[solver._stage_input]
            == Access.WRITE
        )


@pytest.mark.parametrize("time_step_size", [None, 0.01])
def test_time_step_size_acts_at_the_end_of_the_time_step_and_needs_no_traversal(
    tmp_path, time_step_size
):
    solver = _solver(rk_order=2, boundary=None, time_step_size=time_step_size)
    tasks = _construct_time_step(solver, tmp_path, periodic_BC=(True, True, True))
    task = tasks["Probe_Compute time step size"]
    assert task.entity == peano4.api.taskgraph.Task.GridEntity.GLOBAL_END_OF_STEP
    assert not task.cell_data
    # Nothing, not even the refinement, waits for a task that runs at the very end
    assert not any(task in other.dependencies for other in tasks.values())
    # The mesh refinement is independent of the time step size
    assert (
        task
        not in tasks[
            f"Probe_{exahype2.api.Solver.NumericalScheme.DYNAMIC_MESH_REFINEMENT_TASK_NAME}"
        ].dependencies
    )


def test_estimates_are_discarded_last_but_not_written_first(tmp_path):
    tasks = _construct_time_step(_solver(rk_order=3), tmp_path)
    Access = peano4.api.taskgraph.Task.DataAccess

    def access_to_estimates(name):
        (qualifier,) = [
            access
            for data, access in tasks[name].cell_data_access.items()
            if data.name.endswith("RhsEstimates")
        ]
        return qualifier

    # Write-only first would leak one patch per cell and time step
    assert (
        access_to_estimates("Probe_Evaluate Runge-Kutta right hand side stage 0")
        == Access.READ_AND_WRITE
    )
    assert (
        access_to_estimates("Probe_Evaluate Runge-Kutta right hand side stage 1")
        == Access.READ_AND_WRITE
    )
    assert access_to_estimates("Probe_Prepare stage 1") == Access.READ_AND_WRITE
    assert access_to_estimates("Probe_Prepare stage 2") == Access.READ_AND_WRITE
    assert access_to_estimates("Probe_Complete time step") == Access.READ_AND_DISCARD
