import types

import pytest

import peano4.api
import peano4.api.solversteps
import peano4.api.taskgraph

import exahype2.api
import exagrype.api


def _solver(quantities=None, sigma=8.0, rk_order=2):
    solver = exagrype.api.FOCCZ4Solver("CCZ4")
    solver.numerical_scheme = exagrype.api.FD4WithKODissipation(
        solver,
        patch_size=5,
        rk_order=rk_order,
        kreiss_oliger_sigma=sigma,
        quantities=quantities,
    )
    return solver


def _construct_time_step(solver, directory):
    exahype2_project = types.SimpleNamespace(periodic_BC=[True, True, True])
    peano_project = peano4.api.Project(["probe"], "probe", str(directory))
    time_step = peano4.api.taskgraph.Graph()
    solver.numerical_scheme.construct_output(exahype2_project, peano_project, time_step)
    return {task.name: task for task in time_step.tasks}


def _body(action_set):
    return action_set.get_body_of_operation(
        peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
    )


def test_scheme_widens_halo_to_three():
    solver = _solver()
    assert solver.numerical_scheme.overlap == 3
    assert solver._patch_with_halo.dim == (11, 11, 11)
    assert solver._patch_overlap_current_time_step.dim == (6, 5, 5)


def test_plain_fd4_keeps_halo_of_two():
    solver = exagrype.api.FOCCZ4Solver("CCZ4")
    scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, patch_size=5, rk_order=2
    )
    assert scheme.overlap == 2
    assert not hasattr(scheme, "kreiss_oliger_sigma")


def test_dissipation_applies_to_all_unknowns_by_default():
    solver = _solver()
    assert solver.numerical_scheme.dissipative_quantities == list(range(59))


def test_quantities_accept_names_and_indices():
    solver = _solver(quantities=["lapse", "G", 58, 0])
    assert solver.numerical_scheme.dissipative_quantities == [0, 1, 2, 3, 4, 5, 16, 58]


@pytest.mark.parametrize("quantities", [["lapsee"], [59], [-1], [1.5], []])
def test_invalid_quantities_are_rejected(quantities):
    with pytest.raises((ValueError, TypeError)):
        _solver(quantities=quantities)


def test_negative_sigma_is_rejected():
    with pytest.raises(ValueError):
        _solver(sigma=-1.0)


def test_dissipation_runs_after_right_hand_side_in_each_stage(tmp_path):
    solver = _solver(quantities=["lapse", "phi"], rk_order=2)
    tasks = _construct_time_step(solver, tmp_path)
    for stage in range(2):
        evaluation = tasks[f"CCZ4_Evaluate Runge-Kutta right hand side stage {stage}"]
        dissipation = tasks[f"CCZ4_Add Kreiss-Oliger dissipation stage {stage}"]
        assert [type(a) for a in evaluation.action_sets] == [
            exahype2.api.rkfd4.actionsets.EvaluateRungeKuttaRightHandSide
        ]
        assert [type(a) for a in dissipation.action_sets] == [
            exagrype.api.actionsets.AddKreissOligerDissipation
        ]
        assert dissipation.dependencies == [evaluation]
        body = _body(dissipation.action_sets[0])
        assert "::exagrype::ccz4::addKreissOligerDissipation(" in body
        assert "constexpr int dissipativeQuantities[] = { 16, 24 };" in body
        # Estimate slot of the stage: stage * 59 unknowns * 5^3 grid cells
        assert f"fineGridCellCCZ4QRhsEstimates.value + {stage * 59 * 125}," in body
        assert "    3,\n" in body
        assert "    8.0,\n" in body


def test_kernel_call_takes_no_dissipation_argument(tmp_path):
    solver = _solver()
    _construct_time_step(solver, tmp_path)
    # The kernel gets the halo of three, and the functors follow patchData directly
    assert "<\n    5,\n    3,\n    59,\n    0\n>" in solver.compute_kernel_call
    assert "(\n    patchData,\n    [&](" in solver.compute_kernel_call
