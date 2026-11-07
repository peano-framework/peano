import pytest

import peano4.api
import peano4.api.taskgraph
import peano4.api.solversteps

import exahype2.api
import toolbox.blockstructured.api
import toolbox.loadbalancing.api.actionsets

Task = peano4.api.taskgraph.Task
ActionSet = peano4.api.solversteps.ActionSet
Cells = toolbox.blockstructured.api.InterpolateAndRestrictCells


def _solver(scheme):
    solver = exahype2.api.Solver("Probe", 2)
    solver.add_pde_term(exahype2.api.Flux(implementation="F[0] = Q[0]; F[1] = Q[1];"))
    solver.add_pde_term(exahype2.api.MaxEigenvalue(implementation="*lambdaMax = 1.0;"))
    solver.add_pde_term(exahype2.api.InitPointWisely(implementation="Q[0] = 1.0;"))
    if scheme == "fv":
        solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(solver, 4)
        solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
            solver, 0.1
        )
    elif scheme == "fv-optimistic":
        solver.numerical_scheme = exahype2.api.fv.RusanovWithOptimisticExplicitEuler(
            solver, 4
        )
        solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
            solver, 0.1
        )
    else:
        solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
            solver, patch_size=4, rk_order=2
        )
        solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
            solver, 0.1
        )
    return solver


def _mesh_adaptation_graph(solver, tmp_path):
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
    return graphs[3]


@pytest.mark.parametrize("scheme", ["fv", "fv-optimistic"])
def test_finite_volumes_adapt_mesh_and_balance_load_by_default(scheme, tmp_path):
    solver = _solver(scheme)
    graph = _mesh_adaptation_graph(solver, tmp_path)
    assert isinstance(solver.mesh_adaptation, exahype2.api.fv.AdaptMeshAndBalanceLoad)
    adapt, project = graph.ordered_tasks()
    assert adapt.name == "Probe_Adapt mesh and balance load"
    assert adapt.has_to_repeat is not None
    assert project.name == "Probe_Project on faces"
    assert project.dependencies == [adapt]
    kinds = [type(action_set) for action_set in adapt.action_sets]
    assert exahype2.api.blockstructured.ConsumeRefinementRequests in kinds
    assert Cells in kinds
    assert toolbox.loadbalancing.api.actionsets.LoadBalancing in kinds
    discards = exahype2.api.fv.actionsets.DiscardPatchInterior in kinds
    assert discards == (scheme == "fv-optimistic")


def test_finite_differences_adapt_mesh_and_balance_load_by_default(tmp_path):
    solver = _solver("rkfd4")
    graph = _mesh_adaptation_graph(solver, tmp_path)
    assert isinstance(
        solver.mesh_adaptation, exahype2.api.rkfd4.AdaptMeshAndBalanceLoad
    )
    adapt, project = graph.ordered_tasks()
    kinds = [type(action_set) for action_set in adapt.action_sets]
    assert Cells in kinds
    assert exahype2.api.fv.actionsets.DiscardPatchInterior not in kinds
    projection_kinds = [type(action_set) for action_set in project.action_sets]
    assert exahype2.api.rkfd4.actionsets.ProjectPatchOntoFaces in projection_kinds
    assert toolbox.blockstructured.api.RestrictHangingFaces in projection_kinds


def _time_step_tasks(solver, tmp_path):
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
    return {task.name: task for task in graphs[1].tasks}


def test_finite_differences_handle_hanging_faces_in_every_stage(tmp_path):
    solver = _solver("rkfd4")
    tasks = _time_step_tasks(solver, tmp_path)
    for stage in range(2):
        interpolation = tasks[f"Probe_Interpolate hanging faces stage {stage}"]
        assert [type(a) for a in interpolation.action_sets] == [
            toolbox.blockstructured.api.InterpolateHangingFaces
        ]
        # The interpolation follows the reconstruction and precedes the right-hand side
        assert tasks[f"Probe_Reconstruct stage {stage}"] in interpolation.dependencies
        assert (
            interpolation
            in tasks[f"Probe_Evaluate Runge-Kutta right hand side stage {stage}"].dependencies
        )
    # The projection, the restriction and the roll over form a chain of tasks,
    # as the roll over has to copy the restricted data
    for suffix in ["stage 1", "solution"]:
        projection = tasks[f"Probe_Project {suffix} onto faces"]
        restriction = tasks[f"Probe_Restrict {suffix} on hanging faces"]
        roll_over = tasks[f"Probe_Roll over {suffix} faces"]
        assert [type(a) for a in projection.action_sets] == [
            exahype2.api.rkfd4.actionsets.ProjectPatchOntoFaces
        ]
        assert [type(a) for a in restriction.action_sets] == [
            toolbox.blockstructured.api.RestrictHangingFaces
        ]
        assert restriction.dependencies == [projection]
        assert roll_over.dependencies == [restriction]


def test_user_defined_mesh_adaptation_is_kept(tmp_path):
    solver = _solver("fv")
    solver.mesh_adaptation = exahype2.api.fv.AdaptMeshAndBalanceLoad(
        solver, interpolation=Cells.Interpolation.PiecewiseConstant
    )
    mine = solver.mesh_adaptation
    graph = _mesh_adaptation_graph(solver, tmp_path)
    assert solver.mesh_adaptation is mine
    adapt = graph.ordered_tasks()[0]
    [cells] = [a for a in adapt.action_sets if isinstance(a, Cells)]
    assert "interpolateCell_AoS_piecewise_constant" in cells.get_body_of_operation(
        ActionSet.OPERATION_CREATE_CELL
    )


def test_cells_are_interpolated_cleared_and_restricted():
    patch = peano4.api.datamodel.Patch((3, 3, 3), 5, "Q")
    cells = Cells(patch)
    created = cells.get_body_of_operation(ActionSet.OPERATION_CREATE_CELL)
    assert "interpolateCell_AoS_linear_with_constant_extrapolation" in created
    assert "coarseGridCellQ.value,\n    fineGridCellQ.value" in created
    touched = cells.get_body_of_operation(ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)
    assert "marker.hasBeenRefined() and not marker.willBeRefined()" in touched
    assert "clearCell" in touched
    destroyed = cells.get_body_of_operation(ActionSet.OPERATION_DESTROY_CELL)
    assert "restrictCell_AoS_averaging" in destroyed
    assert "fineGridCellQ.value,\n    coarseGridCellQ.value" in destroyed


def test_consumption_hands_over_requests_and_resets_time_step_size():
    solver = _solver("fv")
    consume = exahype2.api.blockstructured.ConsumeRefinementRequests(solver)
    assert (
        "RefinementControlService"
        in consume.get_body_of_createGridControlEventsForNextTraversal()
    )
    unprepare = consume.get_body_of_unprepareTraversal()
    assert "RefinementControlService::getInstance().finishStep()" in unprepare
    assert "resetTimeStepSize()" in unprepare
    for operation in [
        ActionSet.OPERATION_CREATE_CELL,
        ActionSet.OPERATION_DESTROY_CELL,
    ]:
        assert "_meshHasChanged = true" in consume.get_body_of_operation(operation)
