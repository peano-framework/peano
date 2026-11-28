"""Unit tests for the setup from a checkpoint
(@ref exahype2.api.blockstructured.LoadCheckpoint.LoadCheckpoint)."""

import types

import peano4.api
import peano4.api.taskgraph
import exahype2.api
import toolbox.blockstructured.api
import pytest


def _solver(scheme):
    solver = exahype2.api.Solver("Probe", 4)
    solver.add_pde_term(exahype2.api.Flux(implementation=""))
    solver.add_pde_term(exahype2.api.MaxEigenvalue(implementation=""))
    if scheme == "fv":
        solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(solver, 9)
        package = exahype2.api.fv
    else:
        solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
            solver, patch_size=9, rk_order=2
        )
        package = exahype2.api.rkfd4
    return solver, package


def _setup_graph(solver, tmp_path):
    peano_project = peano4.api.Project(["probe"], "probe", directory=str(tmp_path))
    exahype2_project = types.SimpleNamespace(periodic_BC=[True, True, True])
    graph = peano4.api.taskgraph.Graph()
    solver.grid_construction.construct_output(exahype2_project, peano_project, graph)
    return graph


def test_both_schemes_offer_the_same_setup():
    assert exahype2.api.fv.LoadCheckpoint is exahype2.api.rkfd4.LoadCheckpoint


@pytest.mark.parametrize("scheme,halo", [("fv", 1), ("fd4", 2)])
def test_setup_refines_consolidates_and_loads(tmp_path, scheme, halo):
    solver, package = _solver(scheme)
    solver.grid_construction = package.LoadCheckpoint(
        solver, output_path="restart", time_stamp=0.5
    )
    graph = _setup_graph(solver, tmp_path)
    assert [task.name for task in graph.tasks] == [
        "Probe_Refine mesh from checkpoint",
        "Probe_Consolidate mesh",
        "Probe_Consolidate load balancing",
        "Probe_Load checkpoint",
    ]
    refine, _, _, load = graph.tasks
    assert refine.has_to_repeat is not None
    events = refine.action_sets[0].get_body_of_createGridControlEventsForNextTraversal()
    assert (
        f'::exahype2::CheckpointReader::getInstance("restart/checkpoint-Probe.peano-patch-file", 0.5, {halo}, DomainOffset, DomainSize)'
        in events
    )

    load_patches, project = load.action_sets
    assert isinstance(
        project, toolbox.blockstructured.api.ProjectPatchWithHaloOntoFaces
    )
    assert "fineGridCellProbeQWithHalo" in load_patches.get_body_of_operation(
        "touchCellFirstTime"
    )
    assert "fromCheckpointMetaData" in load_patches.get_body_of_unprepareTraversal()
    assert load.cell_data_access[solver._patch_with_halo] == (
        peano4.api.taskgraph.Task.DataAccess.TEMPORARY
    )


def test_latest_checkpoint_by_default(tmp_path):
    solver, package = _solver("fv")
    solver.grid_construction = package.LoadCheckpoint(solver)
    graph = _setup_graph(solver, tmp_path)
    events = (
        graph.tasks[0]
        .action_sets[0]
        .get_body_of_createGridControlEventsForNextTraversal()
    )
    assert '"./checkpoint-Probe.peano-patch-file", -1.0,' in events


def test_restore_moves_plotters_on(tmp_path):
    solver, package = _solver("fv")
    solver.add_plotter(package.PlotAsVTKFile(solver))
    solver.add_plotter(package.WriteCheckpoint(solver))
    solver.grid_construction = package.LoadCheckpoint(solver)
    _setup_graph(solver, tmp_path)
    peano_project = peano4.api.Project(["probe"], "probe", directory=str(tmp_path))
    exahype2_project = types.SimpleNamespace(periodic_BC=[True, True, True])
    for plotter in solver.plotters:
        plotter.construct_output(
            exahype2_project, peano_project, peano4.api.taskgraph.Graph()
        )
    (definition,) = [
        generator("probe::AbstractProbe")
        for declaration, generator in solver._abstract_solver_methods
        if declaration.startswith("void fromCheckpointMetaData")
    ]
    assert definition.index("_nextPlotAsVTKFileTimeStamp =") < definition.index(
        "finishPlotAsVTKFile();"
    )
    assert "finishWriteCheckpoint();" in definition
