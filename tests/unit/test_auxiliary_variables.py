"""Unit tests for sets of auxiliary variables (@ref exahype2.api.AuxiliaryVariables)."""

import types

import peano4.api
import peano4.api.taskgraph
import pytest

import exahype2.api


def _solver(name="Probe"):
    solver = exahype2.api.Solver(name, {"h": 1, "hu": 2})
    solver.add_pde_term(exahype2.api.Flux(implementation="F[0] = 1.0;", stateless=True))
    solver.add_pde_term(
        exahype2.api.NonconservativeProduct(implementation="", stateless=True)
    )
    solver.add_pde_term(exahype2.api.MaxEigenvalue(implementation="", stateless=False))
    solver.add_pde_term(exahype2.api.InitPointWisely(implementation=""))
    return solver


def _two_sets():
    return (
        exahype2.api.AuxiliaryVariables("Bathymetry", {"b": 1}),
        exahype2.api.AuxiliaryVariables("Distortion", {"jacobian": 4, "shift": 1}),
    )


def test_set_knows_its_variables():
    distortion = exahype2.api.AuxiliaryVariables("Distortion", {"jacobian": 4, "shift": 1})
    assert distortion.size == 5
    assert distortion.variable_positions == {"jacobian": 0, "shift": 4}
    assert distortion.argument_name == "distortion"
    assert distortion.delta_argument_name == "deltaDistortion"
    assert "static constexpr int shift = 4;" in distortion.variables_declaration()


@pytest.mark.parametrize("name", ["bathymetry", "1B", "B a"])
def test_invalid_names_are_rejected(name):
    with pytest.raises(ValueError):
        exahype2.api.AuxiliaryVariables(name, {"b": 1})


def test_solver_has_arbitrary_number_of_sets_with_offsets_in_kernel_buffer():
    solver = _solver()
    bathymetry, distortion = _two_sets()
    solver.add_auxiliary_variables(bathymetry)
    solver.add_auxiliary_variables(distortion)
    assert solver.auxiliary_variables == [bathymetry, distortion]
    assert solver.number_of_auxiliary_variables == 6
    # Behind the three unknowns
    assert solver.auxiliary_variable_offset(bathymetry) == 3
    assert solver.auxiliary_variable_offset(distortion) == 4


def test_same_set_cannot_be_attached_twice():
    solver = _solver()
    bathymetry, _ = _two_sets()
    solver.add_auxiliary_variables(bathymetry)
    with pytest.raises(ValueError):
        solver.add_auxiliary_variables(bathymetry)
    with pytest.raises(ValueError):
        solver.add_auxiliary_variables(exahype2.api.AuxiliaryVariables("Bathymetry", {"c": 1}))


def test_first_solver_owns_set():
    first, second = _solver("First"), _solver("Second")
    bathymetry, _ = _two_sets()
    first.add_auxiliary_variables(bathymetry)
    second.add_auxiliary_variables(bathymetry)
    assert bathymetry.owner is first


def test_signatures_are_unchanged_without_auxiliary_variables():
    solver = _solver()
    flux = solver.get_pde_term("flux", True)
    assert solver.signature_of(flux) == flux.SIGNATURE


def test_pointers_follow_each_q_and_delta_q():
    solver = _solver()
    for auxiliary in _two_sets():
        solver.add_auxiliary_variables(auxiliary)
    flux = solver.signature_of(solver.get_pde_term("flux", True))
    assert flux.startswith(
        "const double* __restrict__ Q, const double* __restrict__ bathymetry, "
        "const double* __restrict__ distortion, const tarch::la::Vector<Dimensions, double>& x"
    )
    ncp = solver.signature_of(solver.get_pde_term("nonconservativeProduct", True))
    assert (
        "const double* __restrict__ Q, const double* __restrict__ bathymetry, "
        "const double* __restrict__ distortion, "
        "const double* __restrict__ deltaQ, const double* __restrict__ deltaBathymetry, "
        "const double* __restrict__ deltaDistortion, " in ncp
    )


def test_only_the_owner_writes_the_set_in_the_initial_condition():
    owner, other = _solver("Owner"), _solver("Other")
    bathymetry, _ = _two_sets()
    owner.add_auxiliary_variables(bathymetry)
    other.add_auxiliary_variables(bathymetry)
    assert "double* __restrict__ Q, double* __restrict__ bathymetry" in owner.signature_of(
        owner.get_pde_term("initPointWisely")
    )
    assert "double* __restrict__ Q, const double* __restrict__ bathymetry" in other.signature_of(
        other.get_pde_term("initPointWisely")
    )


def test_abstract_solver_declares_structs_and_kernel_adapters(tmp_path):
    solver = _solver()
    for auxiliary in _two_sets():
        solver.add_auxiliary_variables(auxiliary)
    declaration = solver.get_pde_term("flux", True).generator_for(solver).generate_declaration()
    # The extended term and the adapter which the kernels call
    assert declaration.count("static GPUCallableMethod void flux(") == 2
    definition = (
        solver.get_pde_term("flux", True)
        .generator_for(solver)
        .generate_out_of_class_definitions("probe::AbstractProbe")
    )
    assert "flux(Q, Q + 3, Q + 4, x, normal, F, Offloadable::Yes);" in definition
    ncp = (
        solver.get_pde_term("nonconservativeProduct", True)
        .generator_for(solver)
        .generate_out_of_class_definitions("probe::AbstractProbe")
    )
    assert "deltaQ, deltaQ + 3, deltaQ + 4" in ncp


def test_stateful_kernel_call_slices_the_kernel_buffer():
    solver = _solver()
    for auxiliary in _two_sets():
        solver.add_auxiliary_variables(auxiliary)
    from exahype2.api.AuxiliaryVariables import forwarding_arguments

    assert forwarding_arguments(solver.get_pde_term("maxEigenvalue").SIGNATURE, solver) == [
        "Q",
        "Q + 3",
        "Q + 4",
        "x",
        "normal",
        "lambdaMax",
    ]


def _graphs(solvers, tmp_path):
    exahype2_project = types.SimpleNamespace(periodic_BC=[True] * 3)
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    setup = peano4.api.taskgraph.Graph()
    time_step = peano4.api.taskgraph.Graph()
    for solver in solvers:
        solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(solver, 0.1)
        solver.grid_construction.construct_output(exahype2_project, peano_project, setup)
        solver.numerical_scheme.construct_output(exahype2_project, peano_project, time_step)
    return peano_project, setup, time_step


def _fv_solver(name):
    solver = _solver(name)
    solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(solver, 6)
    return solver


def test_shared_set_is_held_by_one_cell_data_object(tmp_path):
    first, second = _fv_solver("First"), _fv_solver("Second")
    bathymetry, _ = _two_sets()
    first.add_auxiliary_variables(bathymetry)
    second.add_auxiliary_variables(bathymetry)
    peano_project, setup, time_step = _graphs([first, second], tmp_path)
    cells = [cell.name for cell in peano_project.datamodel.cell_data]
    assert cells.count("BathymetryAuxiliaryVariables") == 1
    # The unknowns do not hold auxiliary variables
    assert first._patch.no_of_unknowns == 3
    # The halos are temporary, i.e. each solver has its own
    assert "BathymetryAuxiliaryVariablesWithHaloOfFirst" in cells
    assert "BathymetryAuxiliaryVariablesWithHaloOfSecond" in cells
    # The second solver initialises after the owner
    tasks = {task.name: task for task in setup.tasks}
    assert (
        tasks["First Initial condition"] in tasks["Second Initial condition"].dependencies
    )


def test_sharing_solver_has_to_be_added_after_owner(tmp_path):
    first, second = _fv_solver("First"), _fv_solver("Second")
    bathymetry, _ = _two_sets()
    first.add_auxiliary_variables(bathymetry)
    second.add_auxiliary_variables(bathymetry)
    with pytest.raises(ValueError, match="before"):
        _graphs([second, first], tmp_path)


def test_distinct_sets_get_distinct_cell_data(tmp_path):
    solver = _fv_solver("Probe")
    for auxiliary in _two_sets():
        solver.add_auxiliary_variables(auxiliary)
    peano_project, setup, time_step = _graphs([solver], tmp_path)
    cells = [cell.name for cell in peano_project.datamodel.cell_data]
    assert "BathymetryAuxiliaryVariables" in cells
    assert "DistortionAuxiliaryVariables" in cells


def test_optimistic_scheme_rejects_auxiliary_variables(tmp_path):
    solver = _solver()
    solver.add_auxiliary_variables(_two_sets()[0])
    solver.numerical_scheme = exahype2.api.fv.RusanovWithOptimisticExplicitEuler(solver, 6)
    with pytest.raises(NotImplementedError):
        solver.numerical_scheme.construct_output(
            types.SimpleNamespace(periodic_BC=[True] * 3),
            peano4.api.Project(["probe"], "probe", str(tmp_path)),
            peano4.api.taskgraph.Graph(),
        )


def test_gpu_update_assembles_buffers_before_copying_to_the_device():
    solver = _solver()
    for auxiliary in _two_sets():
        solver.add_auxiliary_variables(auxiliary)
    solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(solver, 6)
    solver._auxiliary_variable_halos = {
        "Bathymetry": types.SimpleNamespace(name="BathymetryHalo"),
        "Distortion": types.SimpleNamespace(name="DistortionHalo"),
    }
    solver.compute_kernel_call = "kernel();"
    body = exahype2.api.fv.actionsets.UpdateCellGPU(solver, "true").get_body_of_operation(
        peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
    )
    # Stride of the kernels: three unknowns plus six auxiliary variables
    assert "int inputStride = 9 *" in body
    assert body.index("std::vector<double> kernelIn") < body.index("CopyCellDataGPU")
    assert "fineGridCellBathymetryHalo.value[volume*1 + k]" in body
    assert "fineGridCellDistortionHalo.value[volume*5 + k]" in body
    assert body.index("copyBackAndDestroy") < body.rindex("kernelOut[volume*9 + k]")


def test_fd4_reconstructs_the_halos_of_auxiliary_variables(tmp_path):
    solver = _solver()
    solver.add_auxiliary_variables(_two_sets()[0])
    solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, patch_size=6, rk_order=2, time_step_size=0.1
    )
    exahype2_project = types.SimpleNamespace(periodic_BC=[True] * 3)
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    time_step = peano4.api.taskgraph.Graph()
    solver.numerical_scheme.construct_output(exahype2_project, peano_project, time_step)
    tasks = {task.name: task for task in time_step.tasks}
    # The unknowns and each set of auxiliary variables are reconstructed by tasks
    # of their own: the halos of the auxiliary variables follow the unknowns
    reconstruction = tasks["Probe_Reconstruct stage 0"]
    auxiliary = tasks["Probe_Reconstruct auxiliary variables stage 0"]
    assert [type(a).__name__ for a in reconstruction.action_sets] == [
        "ReconstructPatchWithHalo"
    ]
    assert [type(a).__name__ for a in auxiliary.action_sets] == [
        "ReconstructPatchWithHalo"
    ]
    assert auxiliary.dependencies == [reconstruction]
    update = tasks["Probe_Evaluate Runge-Kutta right hand side stage 0"]
    body = update.action_sets[0].get_body_of_operation(
        peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
    )
    assert "std::vector<double> kernelIn" in body
    # The kernels see the unknowns plus the auxiliary variables
    assert "Probe_rightHandSideWithFD4<\n    6,\n    2,\n    3,\n    1\n>" in solver.compute_kernel_call
