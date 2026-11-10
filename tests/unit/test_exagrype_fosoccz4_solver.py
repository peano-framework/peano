import pytest

import peano4.api
import peano4.api.taskgraph
import exahype2.api
import exagrype.api


def _project(tmp_path):
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
    return exahype2_project, peano4.api.Project(["probe"], "probe", str(tmp_path))


def _realise_pair(pair, tmp_path):
    """! Let the host realise the pair and return the time step graph of the project """
    exahype2_project, peano_project = _project(tmp_path)
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    pair.generate_solver_realisation(exahype2_project, peano_project, *graphs)
    return graphs[1]


def _configure(pair, enforce=None):
    """! Configure the CCZ4 solver of the pair with FD4 and lopsided advection """
    solver = pair.ccz4
    Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
    # Lopsided advection widens the halo to the three that the reconstruction needs
    pair.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
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


def _realise(pair, tmp_path, enforce=None):
    """! Configure the pair with FD4, realise both solvers and return the time step graph """
    _configure(pair, enforce)
    return _realise_pair(pair, tmp_path)


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


def test_fososolver_hosts_the_ccz4_solver_and_the_reconstruction_solver():
    pair = exagrype.api.FOSOSolver("SO")
    assert isinstance(pair.ccz4, exagrype.api.FOSOCCZ4Solver)
    assert isinstance(pair.reconstruction, exagrype.api.FOSOReconstructionSolver)
    assert pair.ccz4.name == "SO"
    assert pair.reconstruction.name == "SOReconstruction"
    assert pair.reconstruction.ccz4 is pair.ccz4
    # Same data, swapped roles
    assert pair.ccz4.unknowns == 25
    assert pair.ccz4.number_of_auxiliary_variables == 34
    assert pair.reconstruction.unknowns == 34
    # The unknowns of the reconstruction are the auxiliary variables of the CCZ4 solver
    assert pair.reconstruction.variable_names_per_entry() == [
        name.removeprefix("Derivatives.")
        for name in pair.ccz4.derivatives.variable_names_per_entry()
    ]
    assert pair.reconstruction.pde_terms == []
    assert pair.reconstruction.plotters == []


def test_project_registers_the_solvers_of_the_host_but_lets_the_host_realise_them(tmp_path):
    pair = exagrype.api.FOSOSolver("SO", reconstruction_name="Recon")
    project, _ = _project(tmp_path)
    project.add_solver(pair)
    assert project._solvers == [pair.ccz4, pair.reconstruction]
    assert project._realisations == [pair]
    assert pair.reconstruction.name == "Recon"


def test_solvers_build_separate_graphs_which_the_host_concatenates(tmp_path):
    """! Neither solver sees the tasks of the other one, only the host's graph holds both """
    pair = exagrype.api.FOSOSolver("SO")
    _configure(pair)
    exahype2_project, peano_project = _project(tmp_path)
    recorded = {}
    for solver in (pair.ccz4, pair.reconstruction):
        original = solver.generate_solver_realisation

        def record(*args, _solver=solver, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            # A snapshot, as the host adds dependencies to the shared tasks afterwards
            recorded[_solver.name] = (
                args[3],
                [(t.name, len(t.dependencies)) for t in args[3].tasks],
                kwargs,
            )
            return result

        solver.generate_solver_realisation = record
    project_graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    pair.generate_solver_realisation(exahype2_project, peano_project, *project_graphs)

    ccz4_graph, ccz4_tasks, ccz4_kwargs = recorded["SO"]
    recon_graph, recon_tasks, recon_kwargs = recorded["SOReconstruction"]
    assert ccz4_graph is not recon_graph
    assert ccz4_graph is not project_graphs[1]
    assert not any("Reconstruct derivatives" in name for name, _ in ccz4_tasks)
    chain = [
        ("Reconstruct primary variables", 1),
        ("Interpolate primary variables on hanging faces", 0),
        ("Interpolate derivative variables on hanging faces", 0),
        ("Reconstruct derivative variables", 1),
        ("Reconstruct derivatives", 2),
        ("Project derivative variables onto faces", 1),
        ("Restrict derivative variables on hanging faces", 1),
        ("Roll over derivative variable faces", 1),
    ]
    assert recon_tasks == [
        (f"SOReconstruction_{task_name} stage {stage}", dependencies)
        for stage in range(4)
        for task_name, dependencies in chain
    ]
    # Concatenation: the project holds the tasks of both, wrapped into the global tasks
    assert len(project_graphs[1].tasks) == len(ccz4_tasks) + len(recon_tasks) + 2
    # Both are intermediate steps: The host adds the global tasks afterwards
    assert ccz4_kwargs == {"add_global_tasks": False}
    assert recon_kwargs == {"add_global_tasks": False}


def test_host_reports_the_concatenated_graph(tmp_path, monkeypatch):
    titles = {}

    def report(self, title="Task graph"):
        titles[title] = [t.name for t in self.tasks]

    monkeypatch.setattr(peano4.api.taskgraph.Graph, "log_as_text", report)
    monkeypatch.setattr(peano4.api.taskgraph.Graph, "visualise", lambda self, title="": None)
    previous = peano4.api.Logger().verbosity
    peano4.api.Logger().verbosity = peano4.api.LoggerVerbosity.TEXT
    try:
        _realise(exagrype.api.FOSOSolver("SO"), tmp_path)
    finally:
        peano4.api.Logger().verbosity = previous
    own_ccz4 = titles["Time stepping task graph for SO (CCZ4 part)"]
    own_reconstruction = titles[
        "Time stepping task graph for SOReconstruction (reconstruction part of SO)"
    ]
    host = titles[
        "Time stepping task graph for host SO (concatenation of the graphs of SO, SOReconstruction)"
    ]
    assert not any("Reconstruct derivatives" in name for name in own_ccz4)
    assert len(own_reconstruction) == 4 * 8
    # The host adds the global tasks after it has concatenated the graphs
    assert sorted(host) == sorted(
        own_ccz4
        + own_reconstruction
        + [
            "SO_Disable load balancing",
            "SO_Dynamic mesh refinement and load balancing",
        ]
    )
    assert not any("Disable load balancing" in name for name in own_ccz4)


def test_reconstruction_solver_forwards_the_time_step_data_of_the_ccz4_solver(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    _realise(pair, tmp_path)
    methods = dict(pair.reconstruction._abstract_solver_methods)
    for getter in ["getMinTimeStamp", "getMaxTimeStamp", "getMinTimeStepSize", "getMaxTimeStepSize"]:
        definition = methods[f"double {getter}() const override;"]("A")
        assert f"repositories::instanceOfSO.{getter}()" in definition
    assert "SolverRepository.h" in pair.reconstruction.user_solver_includes


def test_reconstruction_solver_takes_turns_with_each_stage_of_the_solver(tmp_path):
    time_step = _realise(exagrype.api.FOSOSolver("SO"), tmp_path)
    order = time_step.ordered_tasks()
    for stage in range(4):
        reconstruct = next(
            task
            for task in order
            if task.name == f"SO_Interpolate Derivatives on hanging faces stage {stage}"
        )
        chain = {
            task.name: task
            for task in order
            if task.name.startswith("SOReconstruction_")
            and task.name.endswith(f" stage {stage}")
        }
        first_tasks = [
            chain[f"SOReconstruction_Interpolate primary variables on hanging faces stage {stage}"],
            chain[
                f"SOReconstruction_Interpolate derivative variables on hanging faces stage {stage}"
            ],
        ]
        last = chain[f"SOReconstruction_Roll over derivative variable faces stage {stage}"]
        assert last in reconstruct.dependencies
        assert order.index(last) < order.index(reconstruct)
        if stage > 0:
            # The reconstruction follows what the solver's reconstruction followed
            for first in first_tasks:
                assert any(
                    dependency.name.startswith("SO_Roll over stage")
                    or dependency.name.startswith("SO_Roll over solution")
                    for dependency in first.dependencies
                )
                assert f"SO_Prepare stage {stage}" in {
                    task.name for task in order[: order.index(first)]
                }


def test_reconstruction_keeps_the_data_it_shares_with_the_solver_alive(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    solver = pair.ccz4
    time_step = _realise(pair, tmp_path)
    Access = peano4.api.taskgraph.Task.DataAccess
    tasks = {task.name: task for task in time_step.tasks}
    primary = tasks["SOReconstruction_Reconstruct primary variables stage 1"]
    assert primary.cell_data_access[solver._stage_input] == Access.READ_AND_WRITE
    assert (
        primary.face_data_access[solver._patch_overlap_current_time_step]
        == Access.READ_AND_WRITE
    )
    derivatives = tasks["SOReconstruction_Reconstruct derivative variables stage 1"]
    assert derivatives.cell_data_access[solver.derivatives.patch] == Access.READ_AND_WRITE
    assert (
        derivatives.face_data_access[solver.derivatives.patch_overlap_current_time_step]
        == Access.READ_AND_WRITE
    )


def test_reconstruction_works_together_with_the_constraint_enforcement(tmp_path):
    time_step = _realise(
        exagrype.api.FOSOSolver("SO"),
        tmp_path,
        enforce=exagrype.api.FOSOCCZ4Solver.EnforceCCZ4Constraint.IntermediateSteps,
    )
    names = [task.name for task in time_step.ordered_tasks()]
    assert any("Enforce CCZ4 constraints" in name for name in names)
    assert sum("Reconstruct derivatives" in name for name in names) == 4


def test_reconstruction_runs_in_front_of_the_finite_volume_update(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    solver = pair.ccz4
    pair.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(solver, 9)
    solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
        solver, volume_size=1.0 / 27.0
    )
    order = _realise_pair(pair, tmp_path).ordered_tasks()
    reconstruct = next(
        task
        for task in order
        if task.name == "SO_Interpolate Derivatives on hanging faces"
    )
    last = next(
        task
        for task in order
        if task.name == "SOReconstruction_Roll over derivative variable faces"
    )
    assert last in reconstruct.dependencies
    assert order.index(last) < order.index(reconstruct)
    assert sum("Reconstruct derivatives" in task.name for task in order) == 1


def test_reconstruction_rejects_unsupported_schemes(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    pair.numerical_scheme = exahype2.api.fv.RusanovWithOptimisticExplicitEuler(pair.ccz4, 9)
    with pytest.raises(Exception):
        pair.reconstruction.create_task_graph(None, None, 0, 1)


def test_reconstruction_rejects_a_halo_of_two(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    pair.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        pair.ccz4, 9, rk_order=4
    )
    assert pair.ccz4.numerical_scheme.HALO_SIZE == 2
    with pytest.raises(ValueError, match="at least three"):
        pair.reconstruction.create_task_graph(None, None, 0, 1)


def test_reconstruction_adapts_to_the_larger_halo_of_the_dissipation_scheme(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    solver = pair.ccz4
    pair.numerical_scheme = exagrype.api.FD4WithKODissipation(
        solver, 9, rk_order=4, kreiss_oliger_sigma=1.0
    )
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        solver, volume_size=1.0 / 27.0
    )
    time_step = _realise_pair(pair, tmp_path)
    assert sum("Reconstruct derivatives" in task.name for task in time_step.tasks) == 4
    assert pair.reconstruction.numerical_scheme.halo_size == 3


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


def test_interpolations_and_reconstructions_of_the_ccz4_solver_run_in_parallel(tmp_path):
    time_step = _realise(exagrype.api.FOSOSolver("SO"), tmp_path)
    tasks = {task.name: task for task in time_step.tasks}
    for stage in range(4):
        unknowns = tasks[f"SO_Interpolate hanging faces stage {stage}"]
        derivatives = tasks[f"SO_Interpolate Derivatives on hanging faces stage {stage}"]
        reconstruct = tasks[f"SO_Reconstruct stage {stage}"]
        reconstruct_derivatives = tasks[
            f"SO_Reconstruct auxiliary variables Derivatives stage {stage}"
        ]
        chain_end = tasks[
            f"SOReconstruction_Roll over derivative variable faces stage {stage}"
        ]
        # The unknowns do not wait for the derivatives and vice versa
        assert unknowns not in derivatives.dependencies
        assert derivatives not in unknowns.dependencies
        assert unknowns in reconstruct.dependencies
        assert derivatives in reconstruct_derivatives.dependencies
        assert reconstruct_derivatives not in reconstruct.dependencies
        assert reconstruct not in reconstruct_derivatives.dependencies
        # The reconstruction chain is a block in front of both
        assert chain_end in reconstruct.dependencies
        # The derivatives are interpolated after they have been recomputed
        assert chain_end in derivatives.dependencies
        # The right-hand side waits for both reconstructions
        rhs = tasks[f"SO_Evaluate Runge-Kutta right hand side stage {stage}"]
        assert reconstruct in rhs.dependencies
        assert reconstruct_derivatives in rhs.dependencies
