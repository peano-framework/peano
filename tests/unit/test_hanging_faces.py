"""Hanging faces on adaptive meshes: the toolbox action sets
(toolbox.blockstructured.api.InterpolateHangingFaces,
toolbox.blockstructured.api.RestrictHangingFaces) and how the Finite
Volume scheme uses them."""

import peano4.api
import peano4.api.datamodel
import peano4.api.solversteps
import peano4.api.taskgraph
import exahype2.api
import toolbox.blockstructured.api

ActionSet = peano4.api.solversteps.ActionSet
Interpolate = toolbox.blockstructured.api.InterpolateHangingFaces
Restrict = toolbox.blockstructured.api.RestrictHangingFaces


def _face_patch():
    return peano4.api.datamodel.Patch((2, 9, 9), 4, "QFace")


def test_interpolation_defaults_to_linear_in_create_hanging_face():
    action_set = Interpolate(_face_patch())
    body = action_set.get_body_of_operation(ActionSet.OPERATION_CREATE_HANGING_FACE)
    assert "interpolateHaloLayer_AoS_linear_with_constant_extrapolation(" in body
    assert "coarseGridFacesQFace(marker.getSelectedFaceNumber()).value" in body
    assert "fineGridFaceQFace.value" in body
    for operation in [
        ActionSet.OPERATION_DESTROY_HANGING_FACE,
        ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME,
        ActionSet.OPERATION_CREATE_CELL,
    ]:
        assert action_set.get_body_of_operation(operation) == ""


def test_piecewise_constant_interpolation():
    action_set = Interpolate(_face_patch(), Interpolate.Scheme.PiecewiseConstant)
    body = action_set.get_body_of_operation(ActionSet.OPERATION_CREATE_HANGING_FACE)
    assert "interpolateHaloLayer_AoS_piecewise_constant(" in body


def test_averaging_restricts_in_destroy_hanging_face_and_clears_faces():
    action_set = Restrict(_face_patch(), clear_guard="myClearGuard")
    body = action_set.get_body_of_operation(ActionSet.OPERATION_DESTROY_HANGING_FACE)
    assert "restrictInnerHalfOfHaloLayer_AoS_averaging(" in body
    assert "coarseGridFacesQFace(marker.getSelectedFaceNumber()).value" in body
    clear = action_set.get_body_of_operation(ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME)
    assert "clearHaloLayerAoS(" in clear
    assert (
        action_set.get_guard_of_operation(ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME)
        == "myClearGuard"
    )


def test_injection_does_not_clear_faces():
    action_set = Restrict(_face_patch(), Restrict.Scheme.Inject)
    body = action_set.get_body_of_operation(ActionSet.OPERATION_DESTROY_HANGING_FACE)
    assert "restrictInnerHalfOfHaloLayer_AoS_inject(" in body
    assert (
        action_set.get_body_of_operation(ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME)
        == ""
    )


def _graphs(tmp_path, **scheme_arguments):
    solver = exahype2.api.Solver("Probe", 4)
    solver.add_pde_term(exahype2.api.Flux(implementation=""))
    solver.add_pde_term(exahype2.api.MaxEigenvalue(implementation=""))
    solver.add_pde_term(exahype2.api.InitPointWisely(implementation=""))
    solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(
        solver, 9, **scheme_arguments
    )
    solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
        solver, volume_size=0.1
    )
    exahype2_project = exahype2.api.Project(
        ["probe"], "probe", "probe", directory=str(tmp_path)
    )
    exahype2_project.set_global_simulation_parameters(
        dimensions=2,
        offset=[0.0, 0.0],
        size=[1.0, 1.0],
        min_end_time=1.0,
        periodic_BC=[True, True, True],
    )
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    solver.generate_solver_realisation(exahype2_project, peano_project, *graphs)
    return solver, [{task.name: task for task in graph.tasks} for graph in graphs[:2]]


def _action_sets_of_type(task, action_set_type):
    return [a for a in task.action_sets if isinstance(a, action_set_type)]


def test_reconstruction_interpolates_before_it_reconstructs_and_restricts_projection(
    tmp_path,
):
    solver, (setup, time_step) = _graphs(tmp_path)
    (interpolation,) = _action_sets_of_type(
        time_step["Probe_Reconstruct"], Interpolate
    )
    # Interpolation is no task of its own: it acts before the reconstruction
    assert "Probe_Interpolate hanging faces" not in time_step
    actions = time_step["Probe_Reconstruct"].action_sets
    assert actions.index(interpolation) < min(
        i for i, a in enumerate(actions) if type(a).__name__ == "ReconstructPatchWithHalo"
    )
    assert interpolation.scheme == Interpolate.Scheme.Linear
    assert interpolation.d["FACE_NAME"] == solver._patch_overlap_current_time_step.name
    assert "Probe_Restrict on hanging faces" not in time_step
    (restriction,) = _action_sets_of_type(
        time_step["Probe_Project on faces"], Restrict
    )
    assert restriction.scheme == Restrict.Scheme.Averaging
    assert restriction.d["FACE_NAME"] == solver._patch_overlap_update.name
    (setup_restriction,) = _action_sets_of_type(
        setup["Probe Project on faces"], Restrict
    )
    assert setup_restriction.d["FACE_NAME"] == solver._patch_overlap_update.name


def test_schemes_are_configurable(tmp_path):
    _, (setup, time_step) = _graphs(
        tmp_path,
        interpolation=Interpolate.Scheme.PiecewiseConstant,
        restriction=Restrict.Scheme.Inject,
    )
    (interpolation,) = _action_sets_of_type(
        time_step["Probe_Reconstruct"], Interpolate
    )
    assert interpolation.scheme == Interpolate.Scheme.PiecewiseConstant
    for task in [
        time_step["Probe_Project on faces"],
        setup["Probe Project on faces"],
    ]:
        (restriction,) = _action_sets_of_type(task, Restrict)
        assert restriction.scheme == Restrict.Scheme.Inject


def test_projection_hosts_the_restriction_and_the_roll_over(tmp_path):
    # The roll over copies the update, which lives within one traversal only
    # (src/exahype2/api/fv/adrs/0003-roll-over-is-part-of-the-projection.md).
    # The restriction precedes the roll over, as the roll over has to copy the
    # restricted data
    _, (_, time_step) = _graphs(tmp_path)
    projection = time_step["Probe_Project on faces"]
    assert "Probe_Restrict on hanging faces" not in time_step
    assert "Probe_Roll over faces" not in time_step
    assert [type(a).__name__ for a in projection.action_sets] == [
        "ProjectPatchOntoFaces",
        "RestrictHangingFaces",
        "RollOverUpdatedFace",
    ]
