import pytest

import peano4.api
import peano4.api.datamodel

import exahype2.api


def _scheme(patch_size=5, rk_order=3, time_step_size=None):
    solver = exahype2.api.Solver("Probe", 4)
    scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, patch_size=patch_size, rk_order=rk_order, time_step_size=time_step_size
    )
    solver.numerical_scheme = scheme
    return solver, scheme


def test_patches_have_halo_of_two():
    solver, scheme = _scheme()
    assert scheme.overlap == 2
    assert solver._patch.dim == (5, 5, 5)
    assert solver._patch.no_of_unknowns == 4
    assert solver._stage_input.dim == (5, 5, 5)
    assert solver._stage_input.no_of_unknowns == 4
    assert solver._patch_with_halo.dim == (9, 9, 9)
    assert solver._patch_with_halo.no_of_unknowns == 4
    for overlap in (
        solver._patch_overlap_current_time_step,
        solver._patch_overlap_update,
    ):
        assert overlap.dim == (4, 5, 5)
        assert overlap.no_of_unknowns == 4


def test_rhs_estimates_hold_one_patch_of_unknowns_per_stage():
    solver, scheme = _scheme(rk_order=3)
    assert scheme.rk_order == 3
    assert solver._rhs_estimates.dim == (15, 5, 5)
    # Auxiliary variables do not evolve in time, so there is no estimate for them
    assert solver._rhs_estimates.no_of_unknowns == 4


def test_data_names_carry_solver_name():
    solver, _ = _scheme()
    assert [
        data.name
        for data in [
            solver._patch,
            solver._stage_input,
            solver._patch_with_halo,
            solver._rhs_estimates,
            solver._patch_overlap_current_time_step,
            solver._patch_overlap_update,
            solver._max_eigenvalue,
        ]
    ] == [
        "ProbeQ",
        "ProbeQStageInput",
        "ProbeQWithHalo",
        "ProbeQRhsEstimates",
        "ProbeQCurrentTimeStep",
        "ProbeQUpdate",
        "ProbeMaxEigenvalue",
    ]


def _reduction_methods(solver):
    return [declaration for declaration, _ in solver._abstract_solver_methods]


def test_adaptive_time_step_size_stores_the_eigenvalue_per_cell_and_reduces_it_in_the_solver():
    """! Per cell, then the reduction is a global datum which the solver hosts """
    solver, scheme = _scheme()
    assert scheme.has_adaptive_time_step_size
    assert solver._max_eigenvalue is not None
    assert _reduction_methods(solver) == [
        "void reduceMaxEigenvalue(double eigenvalue);",
        "double getAndResetMaxEigenvalue();",
    ]


def test_fixed_time_step_size_needs_no_eigenvalue():
    solver, scheme = _scheme(time_step_size=0.01)
    assert not scheme.has_adaptive_time_step_size
    assert solver._max_eigenvalue is None
    assert _reduction_methods(solver) == []


def test_add_data_to_repository_exchanges_current_face_data_only():
    solver, scheme = _scheme()
    project = peano4.api.Project(["probe"], "probe", ".")
    scheme._CentralDifferencesWithRungeKutta__add_data_to_repository(project)

    for patch in scheme._patches():
        assert isinstance(
            patch.generator, peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer
        )
    face_guard = solver.grid_selector.createFaceGuard()
    assert (
        solver._patch_overlap_current_time_step.generator.send_condition == face_guard
    )
    assert solver._patch_overlap_update.generator.send_condition == "false"
    assert solver._patch_overlap_update.generator.receive_and_merge_condition == "false"


@pytest.mark.parametrize("patch_size", [1])
def test_patch_smaller_than_halo_is_rejected(patch_size):
    with pytest.raises(ValueError):
        _scheme(patch_size=patch_size)


@pytest.mark.parametrize("rk_order", [0, 5])
def test_unsupported_runge_kutta_order_is_rejected(rk_order):
    with pytest.raises(ValueError):
        _scheme(rk_order=rk_order)


@pytest.mark.parametrize(
    "arguments",
    [
        {"time_step_size": 0.0},
        {"cfl_constant": 0.0},
    ],
)
def test_invalid_arguments_are_rejected(arguments):
    solver = exahype2.api.Solver("Probe", 1)
    with pytest.raises(ValueError):
        exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
            solver, patch_size=5, rk_order=2, **arguments
        )
