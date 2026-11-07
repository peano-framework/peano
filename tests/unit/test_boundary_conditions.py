"""Boundary condition terms of exahype2.api and how the numerical schemes
realise them (@ref exahype2.api.blockstructured.BoundaryConditions)."""
import types

import pytest

import peano4.api
import peano4.api.solversteps
import peano4.api.taskgraph

import exahype2.api


TOUCH_FACE_LAST_TIME = peano4.api.solversteps.ActionSet.OPERATION_TOUCH_FACE_LAST_TIME


def _project(periodic_BC):
    return types.SimpleNamespace(periodic_BC=list(periodic_BC))


def _solver(*boundary_terms, max_eigenvalue=True):
    solver = exahype2.api.Solver("Probe", 2)
    solver.add_pde_term(exahype2.api.Flux(implementation="F[0] = Q[0]; F[1] = Q[1];"))
    solver.add_pde_term(
        exahype2.api.InitPointWisely(
            implementation="Q[0] = 1.0; Q[1] = 0.0; Q[2] = 0.0;"
        )
    )
    if max_eigenvalue:
        solver.add_pde_term(
            exahype2.api.MaxEigenvalue(implementation="*lambdaMax = 1.0;")
        )
    for term in boundary_terms:
        solver.add_pde_term(term)
    return solver


def _sommerfeld(**kwargs):
    return exahype2.api.SommerfeldBoundaryConditions(
        implementation="Q[0] = 1.0; Q[1] = 0.0;", **kwargs
    )


def _dirichlet():
    return exahype2.api.DirichletBoundaryConditions(implementation="Q[0] = 1.0;")


def test_has_domain_boundary():
    has_domain_boundary = exahype2.api.blockstructured.BoundaryConditions.has_domain_boundary
    assert not has_domain_boundary(_project([True, True, True]))
    assert has_domain_boundary(_project([True, False, True]))
    assert has_domain_boundary(_project([False, False, False]))


def test_get_boundary_conditions_returns_the_one_term():
    get_boundary_conditions = exahype2.api.blockstructured.BoundaryConditions.get_boundary_conditions
    dirichlet = _dirichlet()
    assert (
        get_boundary_conditions(_solver(dirichlet), _project([False] * 3)) is dirichlet
    )
    sommerfeld = _sommerfeld()
    assert (
        get_boundary_conditions(_solver(sommerfeld), _project([True, False]))
        is sommerfeld
    )
    assert get_boundary_conditions(_solver(), _project([True] * 3)) is None


@pytest.mark.parametrize(
    "solver, periodic_BC, message",
    [
        (lambda: _solver(_dirichlet()), [True] * 3, "all axes are periodic"),
        (lambda: _solver(_sommerfeld()), [True] * 3, "all axes are periodic"),
        (lambda: _solver(), [False] * 3, "no boundary condition PDE term"),
        (
            lambda: _solver(_dirichlet(), _sommerfeld()),
            [False] * 3,
            "more than one boundary condition term",
        ),
        (
            lambda: _solver(_sommerfeld(), max_eigenvalue=False),
            [False] * 3,
            "has no MaxEigenvalue",
        ),
        (
            lambda: _solver(_sommerfeld(origin=[0.0, 0.0])),
            [False] * 3,
            "has 2 entries, but the domain has 3 dimensions",
        ),
    ],
)
def test_get_boundary_conditions_rejects_inconsistent_terms(
    solver, periodic_BC, message
):
    with pytest.raises(ValueError, match=message):
        exahype2.api.blockstructured.BoundaryConditions.get_boundary_conditions(
            solver(), _project(periodic_BC)
        )


def test_sommerfeld_term_generates_far_field_solution():
    term = _sommerfeld(origin=[1.0, 2.0])
    assert term.name == "sommerfeldFarFieldSolution"
    assert term.get_origin(2) == [1.0, 2.0]
    assert _sommerfeld().get_origin(3) == [0.0, 0.0, 0.0]
    declaration = term.generator.generate_declaration()
    assert "virtual void sommerfeldFarFieldSolution(" in declaration
    assert "const tarch::la::Vector<Dimensions, double>& x" in declaration


def _realisation(
    scheme_factory, grid_construction_factory, solver, periodic_BC, tmp_path
):
    """! Setup and time step task graphs of solver over the unit cube"""
    solver.numerical_scheme = scheme_factory(solver)
    solver.grid_construction = grid_construction_factory(solver)
    exahype2_project = exahype2.api.Project(
        ["probe"], "probe", "probe", directory=str(tmp_path)
    )
    exahype2_project.set_global_simulation_parameters(
        dimensions=3,
        offset=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 1.0],
        min_end_time=1.0,
        periodic_BC=periodic_BC,
    )
    peano_project = peano4.api.Project(["probe"], "probe", str(tmp_path))
    graphs = [peano4.api.taskgraph.Graph() for _ in range(4)]
    solver.generate_solver_realisation(exahype2_project, peano_project, *graphs)
    return [{task.name: task for task in graph.tasks} for graph in graphs[:2]]


SCHEMES = {
    "fv": (
        lambda solver: exahype2.api.fv.RusanovWithExplicitEuler(solver, 4),
        lambda solver: exahype2.api.fv.SetupSimulationOverRegularGrid(
            solver, volume_size=0.1
        ),
        exahype2.api.fv.actionsets,
    ),
    "optimistic fv": (
        lambda solver: exahype2.api.fv.RusanovWithOptimisticExplicitEuler(solver, 4),
        lambda solver: exahype2.api.fv.SetupSimulationOverRegularGrid(
            solver, volume_size=0.1
        ),
        exahype2.api.fv.actionsets,
    ),
    "rkfd4": (
        lambda solver: exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
            solver, patch_size=4, rk_order=2
        ),
        lambda solver: exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
            solver, volume_size=0.1
        ),
        exahype2.api.rkfd4.actionsets,
    ),
}


def _roll_over_bodies(tasks, actionsets):
    return [
        action_set.get_body_of_operation(TOUCH_FACE_LAST_TIME)
        for task in tasks.values()
        for action_set in task.action_sets
        if isinstance(action_set, actionsets.RollOverUpdatedFace)
    ]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_sommerfeld_replaces_roll_over_of_setup_and_time_step(scheme, tmp_path):
    scheme_factory, grid_construction_factory, actionsets = SCHEMES[scheme]
    setup, time_step = _realisation(
        scheme_factory,
        grid_construction_factory,
        _solver(_sommerfeld(origin=[0.5, 0.5, 0.5])),
        [False, False, False],
        tmp_path,
    )
    assert not any("Dirichlet" in name for name in time_step)

    # The setup initialises the ghost cells with the initial condition
    [setup_roll_over] = _roll_over_bodies(setup, actionsets)
    assert "applyDirichletBoundaryConditions(" in setup_roll_over
    assert "initPointWisely(x, Q)" in setup_roll_over
    assert "applySommerfeldBoundaryConditions(" not in setup_roll_over

    # Every roll over of the time step advances the ghost cells first
    roll_overs = _roll_over_bodies(time_step, actionsets)
    assert roll_overs
    for body in roll_overs:
        assert "applySommerfeldBoundaryConditions(" in body
        assert "sommerfeldFarFieldSolution(x, Q)" in body
        assert "maxEigenvalue(Q, x, normal, &lambdaMax)" in body
        assert "tarch::la::Vector<Dimensions, double>({0.5, 0.5, 0.5})" in body
        assert body.index("applySommerfeldBoundaryConditions") < body.index(
            "copyHalfOfHalo"
        )


def _accesses_to_persistent_face_data(solver, tasks):
    return {
        task.face_data_access[solver._patch_overlap_current_time_step]
        for task in tasks.values()
        if solver._patch_overlap_current_time_step in task.face_data
    }


@pytest.mark.parametrize("scheme", SCHEMES)
def test_sommerfeld_keeps_old_face_data_alive(scheme, tmp_path):
    """! The roll over reads the old face data, so no task may discard it,
    and the projection has to load it"""
    scheme_factory, grid_construction_factory, actionsets = SCHEMES[scheme]
    solver = _solver(_sommerfeld())
    setup, time_step = _realisation(
        scheme_factory, grid_construction_factory, solver, [False] * 3, tmp_path
    )
    # The roll over of the rkfd4 scheme declares the data as temporary, as the
    # head of its chain of tasks declares it for the whole traversal
    accesses = _accesses_to_persistent_face_data(solver, time_step)
    accesses.discard(peano4.api.taskgraph.Task.DataAccess.TEMPORARY)
    assert accesses == {peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE}


@pytest.mark.parametrize("scheme", SCHEMES)
def test_dirichlet_discards_face_data_after_reconstruction(scheme, tmp_path):
    scheme_factory, grid_construction_factory, actionsets = SCHEMES[scheme]
    solver = _solver(_dirichlet())
    setup, time_step = _realisation(
        scheme_factory, grid_construction_factory, solver, [False] * 3, tmp_path
    )
    assert (
        peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD
        in _accesses_to_persistent_face_data(solver, time_step)
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_dirichlet_keeps_boundary_task_and_plain_roll_over(scheme, tmp_path):
    scheme_factory, grid_construction_factory, actionsets = SCHEMES[scheme]
    setup, time_step = _realisation(
        scheme_factory,
        grid_construction_factory,
        _solver(_dirichlet()),
        [False, False, False],
        tmp_path,
    )
    assert any("Apply Dirichlet BCs" in name for name in time_step)
    for body in _roll_over_bodies(setup, actionsets) + _roll_over_bodies(
        time_step, actionsets
    ):
        assert "BoundaryConditions(" not in body


@pytest.mark.parametrize("scheme", SCHEMES)
def test_periodic_domain_has_no_boundary_treatment(scheme, tmp_path):
    scheme_factory, grid_construction_factory, actionsets = SCHEMES[scheme]
    setup, time_step = _realisation(
        scheme_factory,
        grid_construction_factory,
        _solver(),
        [True, True, True],
        tmp_path,
    )
    assert not any("Dirichlet" in name for name in time_step)
    for body in _roll_over_bodies(setup, actionsets) + _roll_over_bodies(
        time_step, actionsets
    ):
        assert "BoundaryConditions(" not in body


@pytest.mark.parametrize("scheme", SCHEMES)
def test_periodic_domain_rejects_boundary_condition_term(scheme, tmp_path):
    scheme_factory, grid_construction_factory, actionsets = SCHEMES[scheme]
    with pytest.raises(ValueError, match="all axes are periodic"):
        _realisation(
            scheme_factory,
            grid_construction_factory,
            _solver(_sommerfeld()),
            [True, True, True],
            tmp_path,
        )


def test_sommerfeld_prefers_stateless_max_eigenvalue():
    get_term = exahype2.api.blockstructured.BoundaryConditions.get_max_eigenvalue_term_for_sommerfeld
    solver = _solver(_sommerfeld())
    assert not get_term(solver).stateless
    stateless = exahype2.api.MaxEigenvalue(
        implementation="*lambdaMax = 1.0;", stateless=True
    )
    solver.add_pde_term(stateless)
    assert get_term(solver) is stateless
