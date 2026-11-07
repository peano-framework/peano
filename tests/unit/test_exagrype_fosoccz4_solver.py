import pytest

import peano4.api
import peano4.api.taskgraph
import exahype2.api
import exagrype.api


def _realise(solver, tmp_path, enforce=None):
    """! Realise solver (and thus its reconstructor) and return the time step graph"""
    Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
    # Lopsided advection widens the halo to the three that the reconstructor needs
    solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver,
        9,
        rk_order=4,
        right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
        advection_velocity=exagrype.api.FOSOCCZ4Solver.ADVECTION_VELOCITY,
        advected_quantities=solver.lopsided_advection_quantities(),
    )
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        solver, volume_size=1.0 / 27.0
    )
    if enforce is not None:
        solver.enforce_ccz4_constraint(enforce)
    exahype2_project = exahype2.api.Project(
        ["probe"], "probe", "probe", directory=str(tmp_path)
    )
    exahype2_project.set_global_simulation_parameters(
        dimensions=3,
        offset=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 1.0],
        min_end_time=1.0,
        periodic_BC=[True, True, True],
    )
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    solver.generate_solver_realisation(exahype2_project, peano_project, *graphs)
    return graphs[1]


def test_so_solver_splits_the_59_entries_into_25_unknowns_and_34_derivatives():
    first_order = exagrype.api.FOCCZ4Solver("FO")
    second_order = exagrype.api.FOSOCCZ4Solver("SO")
    assert first_order.unknowns == 59
    assert second_order.unknowns == 25
    assert second_order.number_of_auxiliary_variables == 34
    assert second_order.derivatives.size == 34
    assert second_order.auxiliary_variables == [second_order.derivatives]
    assert second_order.derivatives.owner is second_order
    # Unknowns plus derivatives are the unknowns of the first-order solver, in order
    assert (
        second_order.variable_names_per_entry()
        + second_order.derivatives.variable_names_per_entry()
        != []
    )
    assert list(exagrype.api.FOSOCCZ4Solver.Unknowns) + list(
        exagrype.api.FOSOCCZ4Solver.DerivativeUnknowns
    ) == list(exagrype.api.FOCCZ4Solver.Unknowns)
    assert (
        exagrype.api.FOSOCCZ4Solver.Unknowns
        | exagrype.api.FOSOCCZ4Solver.DerivativeUnknowns
        == exagrype.api.FOCCZ4Solver.Unknowns
    )


def test_so_solver_has_no_formulation_switch_parameter():
    assert "CCZ4SO" not in exagrype.api.FOSOCCZ4Solver("SO").parameters
    assert "CCZ4SO" not in exagrype.api.FOCCZ4Solver("FO").parameters
    with pytest.raises(KeyError):
        exagrype.api.FOSOCCZ4Solver("SO", parameters={"CCZ4SO": 0})


def test_so_solver_calls_the_second_order_kernels():
    first_order = exagrype.api.FOCCZ4Solver("FO")
    second_order = exagrype.api.FOSOCCZ4Solver("SO")
    for term, kernel in [
        ("nonconservativeProduct", "::exagrype::ccz4::ncp%s("),
        ("volumetricSource", "::exagrype::ccz4::source%s("),
    ]:
        assert kernel % "FirstOrder" in first_order.get_pde_term(term, stateless=True).implementation
        assert kernel % "SecondOrder" in second_order.get_pde_term(term, stateless=True).implementation
    for solver in (first_order, second_order):
        assert (
            "::exagrype::ccz4::maxEigenvalue("
            in solver.get_pde_term("maxEigenvalue", stateless=True).implementation
        )
    # The kernels get the primary entries and, as a second pointer, the
    # derivatives, i.e. the auxiliary variables, without any copy
    ncp = second_order.get_pde_term("nonconservativeProduct", stateless=True).implementation
    assert "Q, derivatives," in ncp
    assert "deltaQ, deltaDerivatives," in ncp
    source = second_order.get_pde_term("volumetricSource", stateless=True).implementation
    assert "Q, derivatives," in source
    # The first-order solver passes the secondary entries behind the primary ones
    assert "Q, Q + 25," in first_order.get_pde_term("volumetricSource", stateless=True).implementation
    assert "deltaQ, deltaQ + 25," in first_order.get_pde_term("nonconservativeProduct", stateless=True).implementation
    # The second-order solver has no storage for the secondary result
    assert "S,\n    Q, derivatives," in source
    assert "[59]" not in ncp + source


def test_so_solver_splits_the_initial_condition():
    implementation = (
        exagrype.api.FOSOCCZ4Solver("SO").get_pde_term("initPointWisely").implementation
    )
    assert "gaugeWave" in implementation
    assert "initialValues[59]" in implementation
    assert "Q[i] = initialValues[i]" in implementation
    assert "derivatives[i] = initialValues[25 + i]" in implementation


def test_so_solver_advects_all_evolved_unknowns_with_lopsided_stencils():
    # The second-order kernels remove the advection of all evolved unknowns from the kernels
    moving_puncture = exagrype.api.FOSOCCZ4Solver(
        "SO", parameters={"CCZ4bs": 1.0, "CCZ4sk": 1.0}
    )
    assert moving_puncture.lopsided_advection_quantities() == list(
        exagrype.api.FOSOCCZ4Solver.Unknowns
    )
    # The default CCZ4bs=CCZ4sk=0 does not advect the shift and b
    default = exagrype.api.FOSOCCZ4Solver("SO")
    assert "shift" not in default.lopsided_advection_quantities()
    assert "b" not in default.lopsided_advection_quantities()
    assert "G" in default.lopsided_advection_quantities()
    assert "lapse" in default.lopsided_advection_quantities()


def test_so_solver_enforces_only_the_constraints_of_the_evolved_unknowns():
    assert (
        "enforceCCZ4constraintsOfEvolvedVariables"
        in exagrype.api.FOSOCCZ4Solver("SO")._enforcement_implementation()
    )
    assert (
        "enforceCCZ4constraintsOfEvolvedVariables"
        not in exagrype.api.FOCCZ4Solver("FO")._enforcement_implementation()
    )


def test_solvers_share_the_base_class():
    assert issubclass(exagrype.api.FOCCZ4Solver, exagrype.api.CCZ4SolverBase)
    assert issubclass(exagrype.api.FOSOCCZ4Solver, exagrype.api.CCZ4SolverBase)
    assert (
        exagrype.api.FOSOCCZ4Solver.EnforceCCZ4Constraint
        is exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint
    )


def test_reconstructor_belongs_to_the_solver():
    solver = exagrype.api.FOSOCCZ4Solver("SO")
    reconstructor = solver.reconstructor
    assert isinstance(reconstructor, exagrype.api.FOSOCCZ4Reconstructor)
    assert reconstructor.solver is solver
    # Same data, swapped roles
    assert reconstructor.unknowns == 34
    assert reconstructor.number_of_auxiliary_variables == 25


def test_reconstructor_takes_turns_with_each_stage_of_the_solver(tmp_path):
    solver = exagrype.api.FOSOCCZ4Solver("SO")
    time_step = _realise(solver, tmp_path)
    order = time_step.ordered_tasks()
    for stage in range(4):
        reconstruct = next(
            task for task in order if task.name == f"SO_Reconstruct stage {stage}"
        )
        derivatives = next(
            task
            for task in order
            if task.name == f"SOReconstructor_Reconstruct derivatives for stage {stage}"
        )
        assert derivatives in reconstruct.dependencies
        assert order.index(derivatives) < order.index(reconstruct)
        if stage > 0:
            # The reconstructor follows what the solver's reconstruction followed
            assert any(
                dependency.name.startswith("SO_Roll over stage")
                or dependency.name.startswith("SO_Roll over solution")
                for dependency in derivatives.dependencies
            )
            assert f"SO_Prepare stage {stage}" in {
                task.name for task in order[: order.index(derivatives)]
            }


def test_reconstructor_keeps_the_data_it_shares_with_the_solver_alive(tmp_path):
    solver = exagrype.api.FOSOCCZ4Solver("SO")
    time_step = _realise(solver, tmp_path)
    Access = peano4.api.taskgraph.Task.DataAccess
    task = next(
        task
        for task in time_step.tasks
        if task.name == "SOReconstructor_Reconstruct derivatives for stage 1"
    )
    assert task.cell_data_access[solver._stage_input] == Access.READ_AND_WRITE
    assert (
        task.face_data_access[solver._patch_overlap_current_time_step]
        == Access.READ_AND_WRITE
    )
    assert task.cell_data_access[solver.derivatives.patch] == Access.READ_AND_WRITE
    assert (
        task.face_data_access[solver.derivatives.patch_overlap_current_time_step]
        == Access.READ_AND_WRITE
    )


def test_reconstructor_works_together_with_the_constraint_enforcement(tmp_path):
    solver = exagrype.api.FOSOCCZ4Solver("SO")
    time_step = _realise(
        solver,
        tmp_path,
        enforce=exagrype.api.FOSOCCZ4Solver.EnforceCCZ4Constraint.IntermediateSteps,
    )
    names = [task.name for task in time_step.ordered_tasks()]
    assert any("Enforce CCZ4 constraints" in name for name in names)
    assert sum("Reconstruct derivatives" in name for name in names) == 4


def test_reconstructor_runs_in_front_of_the_finite_volume_update(tmp_path):
    solver = exagrype.api.FOSOCCZ4Solver("SO")
    solver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(solver, 9)
    solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
        solver, volume_size=1.0 / 27.0
    )
    exahype2_project = exahype2.api.Project(
        ["probe"], "probe", "probe", directory=str(tmp_path)
    )
    exahype2_project.set_global_simulation_parameters(
        dimensions=3,
        offset=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 1.0],
        min_end_time=1.0,
        periodic_BC=[True, True, True],
    )
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    solver.generate_solver_realisation(exahype2_project, peano_project, *graphs)
    order = graphs[1].ordered_tasks()
    reconstruct = next(task for task in order if task.name == "SO_Reconstruct")
    derivatives = next(
        task
        for task in order
        if task.name == "SOReconstructor_Reconstruct derivatives"
    )
    assert derivatives in reconstruct.dependencies
    assert order.index(derivatives) < order.index(reconstruct)
    assert sum("Reconstruct derivatives" in task.name for task in order) == 1


def test_reconstructor_rejects_unsupported_schemes(tmp_path):
    solver = exagrype.api.FOSOCCZ4Solver("SO")
    solver.numerical_scheme = exahype2.api.fv.RusanovWithOptimisticExplicitEuler(solver, 9)
    with pytest.raises(Exception):
        solver.reconstructor.generate_solver_realisation(
            None, None, *[peano4.api.taskgraph.Graph() for _ in range(4)]
        )


def test_reconstructor_rejects_a_halo_of_two(tmp_path):
    solver = exagrype.api.FOSOCCZ4Solver("SO")
    solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, 9, rk_order=4
    )
    assert solver.numerical_scheme.HALO_SIZE == 2
    with pytest.raises(ValueError, match="at least three"):
        solver.reconstructor.generate_solver_realisation(
            None, None, *[peano4.api.taskgraph.Graph() for _ in range(4)]
        )


def test_reconstructor_adapts_to_the_larger_halo_of_the_dissipation_scheme(tmp_path):
    solver = exagrype.api.FOSOCCZ4Solver("SO")
    solver.numerical_scheme = exagrype.api.FD4WithKODissipation(
        solver, 9, rk_order=4, kreiss_oliger_sigma=1.0
    )
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        solver, volume_size=1.0 / 27.0
    )
    exahype2_project = exahype2.api.Project(
        ["probe"], "probe", "probe", directory=str(tmp_path)
    )
    exahype2_project.set_global_simulation_parameters(
        dimensions=3,
        offset=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 1.0],
        min_end_time=1.0,
        periodic_BC=[True, True, True],
    )
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    solver.generate_solver_realisation(exahype2_project, peano_project, *graphs)
    assert sum("Reconstruct derivatives" in task.name for task in graphs[1].tasks) == 4


@pytest.mark.parametrize(
    "solver_class", [exagrype.api.FOCCZ4Solver, exagrype.api.FOSOCCZ4Solver]
)
def test_pde_terms_are_stateless(solver_class):
    """! Finite Volumes and Finite Differences both call the static terms """
    solver = solver_class("S")
    for name in ["nonconservativeProduct", "volumetricSource", "maxEigenvalue"]:
        assert solver.get_pde_term(name, stateless=True).stateless
        with pytest.raises(KeyError):
            solver.get_pde_term(name)
