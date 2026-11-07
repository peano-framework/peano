import types

import pytest

import peano4.api
import peano4.api.solversteps
import peano4.api.taskgraph

import exahype2.api
import exagrype.api

Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel


def _solver():
    solver = exahype2.api.Solver("Probe", {"u": 2, "v": 3})
    solver.add_pde_term(exahype2.api.InitPointWisely(implementation="Q[0] = 0.0;"))
    return solver


def _scheme(solver, **arguments):
    scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, patch_size=5, rk_order=2, time_step_size=0.1, **arguments
    )
    solver.numerical_scheme = scheme
    return scheme


def _lopsided(solver, **arguments):
    return _scheme(
        solver,
        right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
        advection_velocity="v",
        advected_quantities=["u"],
        **arguments,
    )


def _tasks(solver, directory):
    exahype2_project = types.SimpleNamespace(periodic_BC=[True, True, True])
    peano_project = peano4.api.Project(["probe"], "probe", str(directory))
    time_step = peano4.api.taskgraph.Graph()
    solver.numerical_scheme.construct_output(exahype2_project, peano_project, time_step)
    return {task.name: task for task in time_step.tasks}


def test_central_differences_are_the_default(tmp_path):
    solver = _solver()
    scheme = _scheme(solver)
    assert scheme.right_hand_side_kernel == Kernel.CentralDifferences
    assert scheme.overlap == 2
    tasks = _tasks(solver, tmp_path)
    assert [
        type(action_set)
        for action_set in tasks[
            "Probe_Evaluate Runge-Kutta right hand side stage 1"
        ].action_sets
    ] == [exahype2.api.rkfd4.actionsets.EvaluateRungeKuttaRightHandSide]
    assert "Probe_Add lopsided advection stage 1" not in tasks


def test_lopsided_advection_runs_after_central_differences(tmp_path):
    solver = _solver()
    scheme = _lopsided(solver)
    assert scheme.overlap == 3
    assert solver._patch_with_halo.dim == (11, 11, 11)
    # The velocity is an unknown, i.e. it follows u
    assert scheme.advection_velocity == [2, 3, 4]
    assert scheme.advected_quantities == [0, 1]
    tasks = _tasks(solver, tmp_path)
    for stage in range(2):
        # One task per action set, so the compiler can run them in parallel to
        # the tasks of other solvers. They form a chain.
        evaluation = tasks[f"Probe_Evaluate Runge-Kutta right hand side stage {stage}"]
        advection = tasks[f"Probe_Add lopsided advection stage {stage}"]
        assert [type(a) for a in evaluation.action_sets] == [
            exahype2.api.rkfd4.actionsets.EvaluateRungeKuttaRightHandSide
        ]
        assert [type(a) for a in advection.action_sets] == [
            exahype2.api.rkfd4.actionsets.AddLopsidedAdvection
        ]
        assert advection.dependencies == [evaluation]
        body = advection.action_sets[0].get_body_of_operation(
            peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
        )
        assert "::exahype2::fd::addLopsidedAdvection(" in body
        assert "constexpr int advectionVelocity[] = { 2, 3, 4 };" in body
        assert "constexpr int advectedQuantities[] = { 0, 1 };" in body
        # Estimate slot of the stage: stage * 5 unknowns * 5^3 grid cells
        assert f"fineGridCellProbeQRhsEstimates.value + {stage * 5 * 125};" in body


def test_lopsided_advection_needs_velocity_and_quantities():
    with pytest.raises(ValueError):
        _scheme(
            _solver(),
            right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
            advected_quantities=["u"],
        )
    with pytest.raises(ValueError):
        _scheme(
            _solver(),
            right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
            advection_velocity="v",
        )


def test_central_differences_reject_advection_arguments():
    with pytest.raises(ValueError):
        _scheme(_solver(), advection_velocity="v", advected_quantities=["u"])


def test_kernel_has_to_be_an_enum():
    with pytest.raises(TypeError):
        _scheme(_solver(), right_hand_side_kernel="lopsided")


def test_advected_quantities_have_to_exist():
    with pytest.raises(ValueError):
        _scheme(
            _solver(),
            right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
            advection_velocity="v",
            advected_quantities=["w"],
        )


def test_velocity_needs_one_entry_per_dimension(tmp_path):
    solver = _solver()
    _scheme(
        solver,
        right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
        advection_velocity=[2, 3],
        advected_quantities=["u"],
    )
    with pytest.raises(ValueError):
        _tasks(solver, tmp_path)


def test_resolve_quantities_selects_names_and_indices():
    solver = _solver()
    resolve = exahype2.api.rkfd4.resolve_quantities
    assert resolve(solver, ["u", 1]) == [0, 1]
    assert resolve(solver, ["v"]) == [2, 3, 4]
    for quantities in (["w"], [5], [], [True], [1.5]):
        with pytest.raises((ValueError, TypeError)):
            resolve(solver, quantities)


def test_ccz4_lopsided_advection_quantities():
    solver = exagrype.api.FOCCZ4Solver("CCZ4", parameters={"CCZ4bs": 1.0, "CCZ4sk": 1.0})
    scheme = exagrype.api.FD4WithKODissipation(
        solver,
        10,
        4,
        kreiss_oliger_sigma=1.0,
        right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
        advection_velocity=exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY,
        advected_quantities=solver.lopsided_advection_quantities(),
    )
    assert scheme.advection_velocity == [17, 18, 19]
    # Everything but G (0-5), lapse (16), shift (17-19), phi (24) and K0 (58)
    assert scheme.advected_quantities == (
        list(range(6, 16)) + list(range(20, 24)) + list(range(25, 58))
    )
    # Without the advection of the shift, b and the derivatives of the shift
    # are not advected
    assert "b" not in exagrype.api.FOCCZ4Solver("CCZ4").lopsided_advection_quantities()


def test_dissipation_runs_after_lopsided_advection(tmp_path):
    solver = exagrype.api.FOCCZ4Solver("CCZ4", parameters={"CCZ4bs": 1.0, "CCZ4sk": 1.0})
    solver.numerical_scheme = exagrype.api.FD4WithKODissipation(
        solver,
        5,
        2,
        kreiss_oliger_sigma=1.0,
        right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
        advection_velocity="shift",
        advected_quantities=solver.lopsided_advection_quantities(),
    )
    assert solver.numerical_scheme.overlap == 3
    tasks = _tasks(solver, tmp_path)
    chain = [
        "CCZ4_Evaluate Runge-Kutta right hand side stage 0",
        "CCZ4_Add lopsided advection stage 0",
        "CCZ4_Add Kreiss-Oliger dissipation stage 0",
    ]
    assert [[type(a) for a in tasks[name].action_sets] for name in chain] == [
        [exahype2.api.rkfd4.actionsets.EvaluateRungeKuttaRightHandSide],
        [exahype2.api.rkfd4.actionsets.AddLopsidedAdvection],
        [exagrype.api.actionsets.AddKreissOligerDissipation],
    ]
    assert tasks[chain[1]].dependencies == [tasks[chain[0]]]
    assert tasks[chain[2]].dependencies == [tasks[chain[1]]]
