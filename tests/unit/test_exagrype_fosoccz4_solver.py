import pytest

import peano4.api
import peano4.api.solversteps
import peano4.api.taskgraph
import exahype2.api
import exagrype.api
import toolbox.blockstructured.api


def _project(tmp_path, periodic_BC=(True, True, True)):
    exahype2_project = exahype2.api.Project(
        ["probe"], "probe", "probe", directory=str(tmp_path)
    )
    exahype2_project.set_global_simulation_parameters(
        dimensions=3,
        offset=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 1.0],
        min_end_time=1.0,
        periodic_BC=list(periodic_BC),
    )
    return exahype2_project, peano4.api.Project(["probe"], "probe", str(tmp_path))


def _realise_pair(pair, tmp_path):
    """! Let the host realise the pair and return the time step graph of the project"""
    exahype2_project, peano_project = _project(tmp_path)
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    pair.generate_solver_realisation(exahype2_project, peano_project, *graphs)
    return graphs[1]


def _configure(pair, enforce=None):
    """! Configure the CCZ4 solver of the pair with FD4 and lopsided advection"""
    solver = pair.ccz4
    Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
    # Lopsided advection widens the halo to three
    pair.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver,
        9,
        rk_order=4,
        right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
        advection_velocity=exagrype.api.FOSOCCZ4Solver.ADVECTION_VELOCITY,
        advected_quantities=solver.advected_quantities(),
        auxiliary_derivatives=solver.auxiliary_derivatives(),
    )
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        solver, volume_size=1.0 / 27.0
    )
    if enforce is not None:
        solver.enforce_ccz4_constraint(enforce)


def _realise(pair, tmp_path, enforce=None):
    """! Configure the pair with FD4, realise both solvers and return the time step graph"""
    _configure(pair, enforce)
    return _realise_pair(pair, tmp_path)


def test_so_solver_splits_the_58_entries_into_25_unknowns_and_33_derivatives():
    first_order = exagrype.api.FOCCZ4Solver("FO")
    second_order = exagrype.api.FOSOCCZ4Solver("SO")
    assert first_order.unknowns == 58
    assert second_order.unknowns == 25
    assert second_order.number_of_auxiliary_variables == 33
    assert second_order.derivatives.size == 33
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


def _ncp_call(solver):
    """!
    C++ which calls the kernel of the non-conservative product. The solver
    registers its non-conservative product only when it is realised, so it
    is taken from the hook which writes it.
    """
    return solver._ncp_implementation()


def test_so_solver_calls_the_second_order_kernels():
    first_order = exagrype.api.FOCCZ4Solver("FO")
    second_order = exagrype.api.FOSOCCZ4Solver("SO")
    assert "::exagrype::ccz4::ncpFirstOrder(" in _ncp_call(first_order)
    assert "::exagrype::ccz4::ncpSecondOrder(" in _ncp_call(second_order)
    for solver, order in [(first_order, "FirstOrder"), (second_order, "SecondOrder")]:
        assert (
            f"::exagrype::ccz4::source{order}("
            in solver.get_pde_term("volumetricSource", stateless=True).implementation
        )
    for solver in (first_order, second_order):
        assert (
            "::exagrype::ccz4::maxEigenvalue("
            in solver.get_pde_term("maxEigenvalue", stateless=True).implementation
        )
    # The kernels get the primary entries and, as a second pointer, the
    # derivatives, i.e. the auxiliary variables, without any copy
    ncp = _ncp_call(second_order)
    assert "Q, derivatives," in ncp
    assert "deltaQ, deltaDerivatives," in ncp
    source = second_order.get_pde_term(
        "volumetricSource", stateless=True
    ).implementation
    assert "Q, derivatives," in source
    # The first-order solver passes the secondary entries behind the primary ones
    assert (
        "Q, Q + 25,"
        in first_order.get_pde_term("volumetricSource", stateless=True).implementation
    )
    assert "deltaQ, deltaQ + 25," in _ncp_call(first_order)
    # The second-order solver has no storage for the secondary result
    assert "S,\n    Q, derivatives," in source
    assert "[58]" not in ncp + source


def test_so_solver_initialises_the_evolved_unknowns_only():
    solver = exagrype.api.FOSOCCZ4Solver("SO")
    term = solver.get_pde_term("initPointWisely")
    assert "gaugeWaveEvolvedVariables(Q, x, 0.0);" in term.implementation
    assert "derivatives" not in term.implementation
    assert "derivatives" not in solver.signature_of(term)


@pytest.mark.parametrize(
    "initial_condition",
    [
        exagrype.api.initialvalues.gauge_wave,
        exagrype.api.initialvalues.diagonal_gauge_wave,
        exagrype.api.initialvalues.linear_wave,
        exagrype.api.initialvalues.flat,
        exagrype.api.initialvalues.single_black_hole,
    ],
)
def test_so_solver_rejects_initial_conditions_which_fill_all_58_entries(
    initial_condition,
):
    with pytest.raises(ValueError, match="_evolved_variables"):
        exagrype.api.FOSOCCZ4Solver("SO", initial_condition=initial_condition())


@pytest.mark.parametrize(
    "name", ["gauge_wave", "diagonal_gauge_wave", "linear_wave", "flat"]
)
def test_so_solver_accepts_the_initial_conditions_of_the_evolved_unknowns(name):
    initial_condition = getattr(
        exagrype.api.initialvalues, name + "_evolved_variables"
    )()
    solver = exagrype.api.FOSOCCZ4Solver("SO", initial_condition=initial_condition)
    assert solver.get_pde_term("initPointWisely") is initial_condition


@pytest.mark.parametrize("finite_volumes", [True, False])
def test_setup_reconstructs_the_derivatives_behind_the_projection_of_the_unknowns(
    tmp_path, finite_volumes
):
    pair = exagrype.api.FOSOSolver("SO")
    solver = pair.ccz4
    if finite_volumes:
        pair.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(solver, 9)
        solver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
            solver, volume_size=1.0 / 27.0
        )
    else:
        _configure(pair)
    exahype2_project, peano_project = _project(tmp_path)
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    pair.generate_solver_realisation(exahype2_project, peano_project, *graphs)
    setup = {task.name: task for task in graphs[0].tasks}

    project_name = (
        f"SO {exahype2.api.Solver.GridConstruction.PROJECT_ON_FACES_TASK_NAME}"
    )
    project_unknowns = setup[project_name]
    # Three tasks: the halo of the unknowns, the stencil and the projection
    chain = [
        task for task in graphs[0].tasks if task.name.startswith("SOReconstruction_")
    ]
    assert len(chain) == 3
    halo, stencil, project_derivatives = chain
    assert project_unknowns in halo.dependencies
    assert stencil.dependencies == [halo]
    assert project_derivatives.dependencies == [stencil]
    # Only the last task writes the face data of the derivatives
    faces = solver.derivatives.patch_overlap_current_time_step.name

    def uses_derivative_faces(task):
        return any(data.name == faces for data in task.face_data)

    assert uses_derivative_faces(project_derivatives)
    assert not any(
        uses_derivative_faces(task)
        for task in graphs[0].tasks
        if task is not project_derivatives
    )
    assert not any("Initialise auxiliary variables" in name for name in setup)


def test_so_solver_advects_all_evolved_unknowns_with_lopsided_stencils():
    # The second-order kernels remove the advection of all evolved unknowns from the kernels
    moving_puncture = exagrype.api.FOSOCCZ4Solver(
        "SO", parameters={"CCZ4b": 1, "CCZ4s": 1}
    )
    assert moving_puncture.advected_quantities() == list(
        exagrype.api.FOSOCCZ4Solver.Unknowns
    )
    # The default CCZ4s=0 does not advect the shift and b
    default = exagrype.api.FOSOCCZ4Solver("SO")
    assert "shift" not in default.advected_quantities()
    assert "b" not in default.advected_quantities()
    assert "G" in default.advected_quantities()
    assert "lapse" in default.advected_quantities()
    # Neither does the non-shifting shift
    non_shifting = exagrype.api.FOSOCCZ4Solver(
        "SO", parameters={"CCZ4b": 0, "CCZ4s": 1}
    ).advected_quantities()
    assert set(moving_puncture.advected_quantities()) - set(non_shifting) == {
        "shift",
        "b",
    }


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
    assert pair.ccz4.number_of_auxiliary_variables == 33
    assert pair.reconstruction.unknowns == 33
    # The unknowns of the reconstruction are the auxiliary variables of the CCZ4 solver
    assert pair.reconstruction.variable_names_per_entry() == [
        name.removeprefix("Derivatives.")
        for name in pair.ccz4.derivatives.variable_names_per_entry()
    ]
    assert pair.reconstruction.pde_terms == []
    assert pair.reconstruction.plotters == []


def test_project_registers_the_solvers_of_the_host_but_lets_the_host_realise_them(
    tmp_path,
):
    pair = exagrype.api.FOSOSolver("SO", reconstruction_name="Recon")
    project, _ = _project(tmp_path)
    project.add_solver(pair)
    assert project._solvers == [pair.ccz4, pair.reconstruction]
    assert project._realisations == [pair]
    assert pair.reconstruction.name == "Recon"


def test_solvers_build_separate_graphs_which_the_host_concatenates(tmp_path):
    """! Neither solver sees the tasks of the other one, only the host's graph holds both"""
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
    # Per stage: two reconstructions of the halos, the stencil and the
    # projection, i.e. one chain without dependencies on other stages
    assert [dependencies for _, dependencies in recon_tasks] == [0, 0, 2, 1] * 4
    assert all(
        name.endswith(f" stage {index // 4}")
        for index, (name, _) in enumerate(recon_tasks)
    )
    # Concatenation: the project holds the tasks of both, wrapped into the global tasks
    assert len(project_graphs[1].tasks) == len(ccz4_tasks) + len(recon_tasks) + 3
    # Both are intermediate steps: The host adds the global tasks afterwards
    assert ccz4_kwargs == {"add_global_tasks": False}
    assert recon_kwargs == {"add_global_tasks": False}


def test_host_reports_the_concatenated_graph(tmp_path, monkeypatch):
    titles = {}

    def report(self, title="Task graph"):
        titles[title] = [t.name for t in self.tasks]

    monkeypatch.setattr(peano4.api.taskgraph.Graph, "log_as_text", report)
    monkeypatch.setattr(
        peano4.api.taskgraph.Graph, "visualise", lambda self, title="": None
    )
    previous = peano4.api.Logger().verbosity
    peano4.api.Logger().verbosity = peano4.api.LoggerVerbosity.TEXT
    try:
        _realise(exagrype.api.FOSOSolver("SO"), tmp_path)
    finally:
        peano4.api.Logger().verbosity = previous
    own_ccz4 = titles["Time stepping task graph for SO"]
    own_reconstruction = titles["Time stepping task graph for SOReconstruction"]
    host = titles["Time stepping task graph for host SO"]
    assert not any("Reconstruct derivatives" in name for name in own_ccz4)
    assert len(own_reconstruction) == 4 * 4
    # The host adds the global tasks after it has concatenated the graphs
    scheme = exahype2.api.Solver.NumericalScheme
    assert sorted(host) == sorted(
        own_ccz4
        + own_reconstruction
        + [
            f"SO_{scheme.DISABLE_LOAD_BALANCING_TASK_NAME}",
            f"SO_{scheme.DYNAMIC_MESH_REFINEMENT_TASK_NAME}",
            f"SO_{scheme.ENABLE_LOAD_BALANCING_TASK_NAME}",
        ]
    )
    assert not any("Disable load balancing" in name for name in own_ccz4)


def test_reconstruction_solver_forwards_the_time_step_data_of_the_ccz4_solver(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    _realise(pair, tmp_path)
    methods = dict(pair.reconstruction._abstract_solver_methods)
    for getter in [
        "getMinTimeStamp",
        "getMaxTimeStamp",
        "getMinTimeStepSize",
        "getMaxTimeStepSize",
    ]:
        definition = methods[f"double {getter}() const override;"]("A")
        assert f"repositories::instanceOfSO.{getter}()" in definition
    assert "SolverRepository.h" in pair.reconstruction.user_solver_includes


def _chain(pair, stage):
    """! First tasks and last task of the reconstruction chain of a stage"""
    graph = pair.reconstruction.numerical_scheme.stage_graphs[stage]
    (last,) = graph.sink_tasks()
    return graph.source_tasks(), last


def _partners(pair, time_step, stage):
    """! The tasks of the CCZ4 solver which the chain of a stage runs in front of"""
    tasks = {task.name: task for task in time_step.tasks}
    return [tasks[name] for name in pair.reconstruction.partner_task_names(stage)]


def test_reconstruction_solver_takes_turns_with_each_stage_of_the_solver(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    time_step = _realise(pair, tmp_path)
    order = time_step.ordered_tasks()
    for stage in range(4):
        first_tasks, last = _chain(pair, stage)
        assert len(first_tasks) == 2
        partners = _partners(pair, time_step, stage)
        assert len(partners) == 2
        for partner in partners:
            assert last in partner.dependencies
            assert order.index(last) < order.index(partner)
            # The chain follows what the CCZ4 solver's reconstructions followed
            for first in first_tasks:
                for dependency in partner.dependencies:
                    if dependency is not last:
                        assert dependency in first.dependencies
        if stage > 0:
            # ... which is the previous stage
            for first in first_tasks:
                assert first.dependencies
                assert all(
                    order.index(dependency) < order.index(first)
                    for dependency in first.dependencies
                )


def test_reconstruction_waits_for_the_dirichlet_boundary_conditions(tmp_path):
    """!

    The chain reads the faces of the unknowns, whose ghost values the
    boundary task of the CCZ4 solver sets. Only the reconstruction of the
    unknowns waits for the boundary task, the one of the derivatives does not.

    """
    pair = exagrype.api.FOSOSolver("SO")
    _configure(pair)
    pair.ccz4.add_pde_term(
        exahype2.api.DirichletBoundaryConditions(implementation="Q[0] = 1.0;")
    )
    exahype2_project, peano_project = _project(
        tmp_path, periodic_BC=(False, True, True)
    )
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    pair.generate_solver_realisation(exahype2_project, peano_project, *graphs)
    time_step = graphs[1]
    order = time_step.ordered_tasks()
    Boundary = exahype2.api.rkfd4.actionsets.ApplyDirichletBoundaryConditions
    for stage in range(4):
        reconstruct, reconstruct_derivatives = _partners(pair, time_step, stage)
        (boundary,) = [
            task
            for task in reconstruct.dependencies
            if any(isinstance(action_set, Boundary) for action_set in task.action_sets)
        ]
        assert boundary not in reconstruct_derivatives.dependencies
        first_tasks, _ = _chain(pair, stage)
        for first in first_tasks:
            assert boundary in first.dependencies
            assert order.index(boundary) < order.index(first)


def test_reconstruction_interpolates_and_restricts_within_the_tasks_using_the_faces(
    tmp_path,
):
    """!

    src/exahype2/api/adrs/0001-interpolation-and-restriction-ride-along-with-face-users.md:
    no task of the chain holds nothing but an interpolation or a restriction

    """
    pair = exagrype.api.FOSOSolver("SO")
    _realise(pair, tmp_path)
    Interpolate = toolbox.blockstructured.api.InterpolateHangingFaces
    Restrict = toolbox.blockstructured.api.RestrictHangingFaces
    Reconstruct = toolbox.blockstructured.api.ReconstructPatchWithHalo
    Project = exahype2.api.rkfd4.actionsets.ProjectPatchOntoFaces
    for stage in range(4):
        graph = pair.reconstruction.numerical_scheme.stage_graphs[stage]
        for task in graph.tasks:
            assert not all(
                isinstance(action_set, (Interpolate, Restrict))
                for action_set in task.action_sets
            )
        first_tasks, _ = _chain(pair, stage)
        for first in first_tasks:
            types = [type(action_set) for action_set in first.action_sets]
            assert types == [Interpolate, Reconstruct]
        (project,) = [
            task
            for task in graph.tasks
            if any(isinstance(action_set, Project) for action_set in task.action_sets)
        ]
        # The roll over shares the temporary update with the projection
        assert [type(action_set) for action_set in project.action_sets] == [
            Project,
            Restrict,
            exahype2.api.rkfd4.actionsets.RollOverUpdatedFace,
        ]


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
    assert (
        derivatives.cell_data_access[solver.derivatives.patch] == Access.READ_AND_WRITE
    )
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
    time_step = _realise_pair(pair, tmp_path)
    order = time_step.ordered_tasks()
    _, last = _chain(pair, 0)
    for partner in _partners(pair, time_step, 0):
        assert last in partner.dependencies
        assert order.index(last) < order.index(partner)
    assert sum("Reconstruct derivatives" in task.name for task in order) == 1


def test_reconstruction_rejects_unsupported_schemes(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    pair.numerical_scheme = exahype2.api.fv.RusanovWithOptimisticExplicitEuler(
        pair.ccz4, 9
    )
    with pytest.raises(Exception):
        pair.reconstruction.create_task_graph(None, None, 0, 1)


def test_reconstruction_takes_the_halo_of_two_of_the_plain_scheme(tmp_path):
    """!

    src/exagrype/ccz4/adrs/0004-the-foso-reconstruction-takes-the-halo-of-the-scheme.md:
    the reconstruction computes the derivatives of the cell only, so the halo of
    the plain Finite Differences suffices

    """
    pair = exagrype.api.FOSOSolver("SO")
    solver = pair.ccz4
    pair.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, 9, rk_order=4, auxiliary_derivatives=solver.auxiliary_derivatives()
    )
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        solver, volume_size=1.0 / 27.0
    )
    assert solver.numerical_scheme.HALO_SIZE == 2
    _realise_pair(pair, tmp_path)
    assert pair.reconstruction.numerical_scheme.halo_size == 2
    Functor = exagrype.api.actionsets.ApplyReconstructionFunctor
    for stage in range(4):
        graph = pair.reconstruction.numerical_scheme.stage_graphs[stage]
        (functor,) = [
            action_set
            for task in graph.tasks
            for action_set in task.action_sets
            if isinstance(action_set, Functor)
        ]
        body = functor.get_body_of_operation(
            peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
        )
        # Patch size 9, halo 2, 25 unknowns, 33 derivatives
        assert "oldQWithHalo, marker.h(), 9, 2, 25, 33," in body


def test_derivative_table_covers_each_derivative_once():
    """!

    src/exagrype/ccz4/adrs/0005-one-table-defines-the-auxiliary-derivatives.md

    """
    Solver = exagrype.api.FOSOCCZ4Solver
    unknown, axis, factor = Solver("SO").derivative_table()
    assert len(unknown) == len(axis) == len(factor) == 33
    unknowns = sum(Solver.Unknowns.values())
    assert all(0 <= index < unknowns for index in unknown)
    assert set(axis) == {0, 1, 2}
    # Each derivative of a variable along an axis appears once
    assert len(set(zip(unknown, axis))) == 33


def test_reconstruction_hands_the_derivative_table_to_the_kernel(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    _realise(pair, tmp_path)
    unknown, axis, factor = pair.ccz4.derivative_table()
    Functor = exagrype.api.actionsets.ApplyReconstructionFunctor
    graph = pair.reconstruction.numerical_scheme.stage_graphs[0]
    (functor,) = [
        action_set
        for task in graph.tasks
        for action_set in task.action_sets
        if isinstance(action_set, Functor)
    ]
    body = functor.get_body_of_operation(
        peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME
    )
    assert "{" + ", ".join(str(i) for i in unknown) + "}" in body
    assert "{" + ", ".join(str(i) for i in axis) + "}" in body
    assert "{" + ", ".join(repr(f) for f in factor) + "}" in body


@pytest.mark.parametrize(
    "table, message",
    [
        ([("dLapse", "G", 0, 1.0)], "has 3 entries, but 'G' has 6"),
        ([("dLapse[3]", "lapse", 0, 1.0)], "is no auxiliary variable"),
        ([("dLapse[0]", "alpha", 0, 1.0)], "is no unknown"),
        ([("dLapse[0]", "lapse", 3, 1.0)], "has to be 0, 1 or 2"),
        (
            [("dLapse[0]", "lapse", 0, 1.0), ("dLapse[0]", "lapse", 1, 1.0)],
            "defined twice",
        ),
        ([("dLapse[0]", "lapse", 0, 1.0)], "does not define"),
    ],
)
def test_derivative_table_rejects_inconsistent_definitions(monkeypatch, table, message):
    Solver = exagrype.api.FOSOCCZ4Solver
    monkeypatch.setattr(Solver, "auxiliary_derivatives", classmethod(lambda cls: table))
    with pytest.raises(ValueError, match=message):
        Solver("SO").derivative_table()


@pytest.mark.parametrize(
    "auxiliary_derivatives",
    [None, [(f"dLapse[{axis}]", "lapse", axis, 1.0) for axis in range(3)]],
)
def test_finite_differences_have_to_get_the_derivative_table(
    tmp_path, auxiliary_derivatives
):
    """!

    src/exagrype/ccz4/adrs/0006-the-second-order-solver-uses-compact-second-derivatives.md:
    strict, i.e. without the table of the solver or with another one, the
    realisation fails

    """
    pair = exagrype.api.FOSOSolver("SO")
    solver = pair.ccz4
    pair.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, 9, rk_order=4, auxiliary_derivatives=auxiliary_derivatives
    )
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        solver, volume_size=1.0 / 27.0
    )
    with pytest.raises(
        ValueError, match="auxiliary_derivatives=solver.auxiliary_derivatives()"
    ):
        _realise_pair(pair, tmp_path)


def test_reconstruction_adapts_to_the_larger_halo_of_the_dissipation_scheme(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    solver = pair.ccz4
    pair.numerical_scheme = exagrype.api.FD4WithKODissipation(
        solver,
        9,
        rk_order=4,
        kreiss_oliger_sigma=1.0,
        auxiliary_derivatives=solver.auxiliary_derivatives(),
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
    """!
    Finite Volumes and Finite Differences both call the static terms. The
    non-conservative product is registered when the solver is realised
    (test_realisation_registers_the_ncp_which_fits_the_numerical_scheme in
    test_exagrype_ccz4_solver.py checks that it is stateless).
    """
    solver = solver_class("S")
    for name in ["volumetricSource", "maxEigenvalue"]:
        assert solver.get_pde_term(name, stateless=True).stateless
        with pytest.raises(KeyError):
            solver.get_pde_term(name)


def test_reconstructions_of_the_ccz4_solver_run_in_parallel(tmp_path):
    pair = exagrype.api.FOSOSolver("SO")
    time_step = _realise(pair, tmp_path)
    for stage in range(4):
        reconstruct, reconstruct_derivatives = _partners(pair, time_step, stage)
        _, chain_end = _chain(pair, stage)
        # The unknowns do not wait for the derivatives and vice versa
        assert reconstruct_derivatives not in reconstruct.dependencies
        assert reconstruct not in reconstruct_derivatives.dependencies
        # The reconstruction chain is a block in front of both
        assert chain_end in reconstruct.dependencies
        assert chain_end in reconstruct_derivatives.dependencies
        # The right-hand side waits for both reconstructions
        right_hand_side = [
            task for task in time_step.tasks if reconstruct in task.dependencies
        ]
        assert right_hand_side
        for task in right_hand_side:
            assert reconstruct_derivatives in task.dependencies
