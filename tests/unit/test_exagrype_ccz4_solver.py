import os
import re

import pytest

import peano4.api
import peano4.api.taskgraph
import exahype2.api
import exagrype.api


with open(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "exagrype", "ccz4", "Kernels.h"
    )
) as kernels_header:
    KERNELS_HEADER = kernels_header.read()


def test_solver_has_58_named_unknowns():
    solver = exagrype.api.FOCCZ4Solver("CCZ4")
    assert solver.unknowns == 58
    assert solver.unknowns == exagrype.api.FOCCZ4Solver.NUMBER_OF_CCZ4_ENTRIES
    assert solver.number_of_auxiliary_variables == 0
    assert solver.variable_names_per_entry()[16] == "lapse"
    assert solver.variable_names_per_entry()[23] == "Kex"
    assert solver.variable_names_per_entry()[24] == "phi"
    # The last unknowns are the derivatives of phi: K0 is no unknown
    assert solver.variable_names_per_entry()[55:] == ["P[0]", "P[1]", "P[2]"]


def _call(solver, term):
    """!
    C++ which calls the kernel of term. The solver registers its
    non-conservative product only when it is realised, so the call of
    the kernel is taken from the hook which writes it.
    """
    if term == exahype2.api.NonconservativeProduct.NAME:
        return solver._ncp_implementation()
    return solver.get_pde_term(term, stateless=True).implementation


def test_solver_registers_ccz4_pde_terms():
    """!
    The non-conservative product is missing until the solver is realised
    (test_realisation_registers_the_ncp_which_fits_the_numerical_scheme)
    """
    solver = exagrype.api.FOCCZ4Solver("CCZ4")
    names = {term.name for term in solver.pde_terms}
    assert names == {
        exahype2.api.VolumetricSource.NAME,
        exahype2.api.MaxEigenvalue.NAME,
    }
    assert "::exagrype::ccz4::ncpFirstOrder(" in _call(
        solver, exahype2.api.NonconservativeProduct.NAME
    )
    assert (
        "::exagrype::ccz4::sourceFirstOrder("
        in solver.get_pde_term("volumetricSource", stateless=True).implementation
    )
    assert (
        "::exagrype::ccz4::maxEigenvalue("
        in solver.get_pde_term("maxEigenvalue", stateless=True).implementation
    )
    # All terms are generated, so no user class is required
    assert not solver.requires_user_class


def test_solver_has_no_initial_condition_of_its_own():
    solver = exagrype.api.FOCCZ4Solver("CCZ4")
    assert "initPointWisely" not in {term.name for term in solver.pde_terms}


def test_initial_condition_is_added_from_outside():
    solver = exagrype.api.FOCCZ4Solver("CCZ4")
    solver.add_pde_term(exagrype.api.initialvalues.gauge_wave())
    assert "gaugeWave" in solver.get_pde_term("initPointWisely").implementation
    solver.add_pde_term(exagrype.api.initialvalues.flat())
    assert (
        "::exagrype::initialvalues::flat("
        in solver.get_pde_term("initPointWisely").implementation
    )


def test_parameters_override_defaults():
    solver = exagrype.api.FOCCZ4Solver(
        "CCZ4", parameters={"CCZ4k1": 0.0, "CCZ4LapseType": 1}
    )
    assert solver.parameters["CCZ4k1"] == 0.0
    assert solver.parameters["CCZ4LapseType"] == 1
    assert (
        solver.parameters["CCZ4GLMd"]
        == exagrype.api.FOCCZ4Solver.DefaultParameters["CCZ4GLMd"]
    )


def test_unknown_parameter_is_rejected():
    with pytest.raises(KeyError):
        exagrype.api.FOCCZ4Solver("CCZ4", parameters={"CCZ4kk1": 0.0})


@pytest.mark.parametrize("key", ["CCZ4ds", "CCZ4xi", "CCZ4sk", "CCZ4bs", "CCZ4SO"])
def test_removed_parameters_are_rejected(key):
    assert key not in exagrype.api.FOCCZ4Solver.DefaultParameters
    with pytest.raises(KeyError):
        exagrype.api.FOCCZ4Solver("CCZ4", parameters={key: 1})


def test_switches_are_integers():
    defaults = exagrype.api.FOCCZ4Solver.DefaultParameters
    assert defaults["CCZ4b"] == 1
    assert defaults["CCZ4s"] == 0
    for key in exagrype.api.FOCCZ4Solver.Switches:
        assert isinstance(defaults[key], int)

    solver = exagrype.api.FOCCZ4Solver("CCZ4", parameters={"CCZ4b": 1, "CCZ4s": 1})
    attributes = "\n".join(solver._abstract_solver_attributes)
    assert "static constexpr int CCZ4b = 1;" in attributes
    assert "static constexpr int CCZ4s = 1;" in attributes
    assert "static constexpr double CCZ4f = 0.75;" in attributes


@pytest.mark.parametrize("value", [1.0, 2, -1, True, "1"])
def test_switches_reject_anything_but_zero_and_one(value):
    with pytest.raises(TypeError):
        exagrype.api.FOCCZ4Solver("CCZ4", parameters={"CCZ4s": value})
    with pytest.raises(TypeError):
        exagrype.api.FOSOCCZ4Solver("SO", parameters={"CCZ4b": value})


@pytest.mark.parametrize(
    "solver_class", [exagrype.api.FOCCZ4Solver, exagrype.api.FOSOCCZ4Solver]
)
@pytest.mark.parametrize(
    "term", ["nonconservativeProduct", "volumetricSource", "maxEigenvalue"]
)
def test_kernel_calls_pass_the_parameters_of_the_kernel_signature(solver_class, term):
    """!

    The CCZ4 parameters of a kernel call match the kernel's declaration

    The parameters are doubles and integers, i.e. the compiler does not
    notice if a call passes them in the wrong order. The test reads the
    name of the kernel from the generated call and the parameters from its
    declaration in Kernels.h, i.e. it holds no names of its own
    (tests/unit/adrs/0001-unit-tests-query-names-from-the-code.md).

    """
    solver = solver_class("CCZ4")
    call = _call(solver, term)
    kernel = re.search(r"::exagrype::ccz4::(\w+)\(", call).group(1)
    declaration = re.search(rf"void {kernel}\((.*?)\);", KERNELS_HEADER, re.DOTALL)
    declared_parameters = re.findall(r"\bCCZ4\w+", declaration.group(1))

    assert declared_parameters
    assert re.findall(r"\bCCZ4\w+", call) == declared_parameters
    assert set(declared_parameters) <= set(solver_class.DefaultParameters)


def test_constraint_enforcement_is_opt_in():
    solver = exagrype.api.FOCCZ4Solver("CCZ4")
    Enforce = exagrype.api.FOCCZ4Solver.EnforceCCZ4Constraint
    assert solver._enforce_ccz4_constraint == Enforce.No
    solver.enforce_ccz4_constraint(Enforce.IntermediateSteps)
    assert solver._enforce_ccz4_constraint == Enforce.IntermediateSteps
    solver.enforce_ccz4_constraint(Enforce.No)
    assert solver._enforce_ccz4_constraint == Enforce.No
    with pytest.raises(TypeError):
        solver.enforce_ccz4_constraint(True)


Kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel


def _central_differences(solver):
    return exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(solver, 9, rk_order=4)


def _dissipation(solver):
    return exagrype.api.FD4WithKODissipation(solver, 9, 4, kreiss_oliger_sigma=1.0)


def _lopsided(solver, advected_quantities=None, advection_velocity=None):
    """! FD4 with lopsided advection, by default of the quantities the solver asks for"""
    return exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver,
        9,
        rk_order=4,
        right_hand_side_kernel=Kernel.CentralDifferencesWithLopsidedAdvection,
        advection_velocity=advection_velocity or type(solver).ADVECTION_VELOCITY,
        advected_quantities=(
            solver.advected_quantities()
            if advected_quantities is None
            else advected_quantities
        ),
    )


def _finite_volumes(solver):
    return exahype2.api.fv.RusanovWithExplicitEuler(solver, 9)


@pytest.mark.parametrize(
    "solver_class", [exagrype.api.FOCCZ4Solver, exagrype.api.FOSOCCZ4Solver]
)
@pytest.mark.parametrize(
    "parameters", [{}, {"CCZ4s": 1, "CCZ4b": 1}, {"CCZ4s": 1, "CCZ4b": 0}]
)
@pytest.mark.parametrize(
    "scheme_factory, advection",
    [
        (_central_differences, True),
        (_dissipation, True),
        (_lopsided, False),
        (_finite_volumes, False),
    ],
)
def test_ncp_adds_advection_iff_the_scheme_does_not(
    solver_class, parameters, scheme_factory, advection
):
    """!

    The non-conservative product adds the advection terms if and only if the
    Finite Differences do not (src/exagrype/ccz4/adrs/0003-the-solver-picks-who-adds-the-advection.md).
    The Finite Volumes add none, and neither does the non-conservative product.

    """
    solver = solver_class("CCZ4", parameters=parameters)
    solver.numerical_scheme = scheme_factory(solver)
    assert solver.ncp_adds_advection() == advection


@pytest.mark.parametrize(
    "solver_class", [exagrype.api.FOCCZ4Solver, exagrype.api.FOSOCCZ4Solver]
)
def test_ncp_adds_advection_rejects_lopsided_advection_of_other_quantities(
    solver_class,
):
    """!

    A scheme which advects other quantities than those the kernels leave
    out would add some advection terms twice or miss some

    """
    frozen_shift = solver_class("CCZ4")
    moving_shift = solver_class("CCZ4", parameters={"CCZ4s": 1, "CCZ4b": 1})
    wrong_choices = [
        dict(advected_quantities=frozen_shift.advected_quantities()[1:]),
        dict(advected_quantities=moving_shift.advected_quantities()),
        dict(advection_velocity="b"),
    ]
    for choice in wrong_choices:
        solver = solver_class("CCZ4")
        solver.numerical_scheme = _lopsided(solver, **choice)
        with pytest.raises(ValueError):
            solver.ncp_adds_advection()


def test_ncp_adds_advection_rejects_unknown_schemes():
    solver = exagrype.api.FOCCZ4Solver("CCZ4")
    with pytest.raises(TypeError):
        solver.ncp_adds_advection()


def _constexpr_list(code, name):
    return [
        int(entry)
        for entry in re.search(rf"constexpr int {name}\[\] = {{(.*?)}};", code)
        .group(1)
        .split(",")
    ]


@pytest.mark.parametrize(
    "solver_class", [exagrype.api.FOCCZ4Solver, exagrype.api.FOSOCCZ4Solver]
)
@pytest.mark.parametrize(
    "parameters",
    [{}, {"CCZ4s": 1, "CCZ4b": 1}, {"CCZ4s": 1, "CCZ4b": 0}, {"CCZ4s": 0, "CCZ4b": 1}],
)
def test_central_advection_adds_what_the_lopsided_advection_would_add(
    solver_class, parameters
):
    """!

    The advection which the non-conservative product appends covers exactly
    the quantities and the velocity which the lopsided advection of the
    scheme gets, i.e. either way the same terms are added. Both read them
    from advected_quantities() and ADVECTION_VELOCITY, the one
    definition of the advection terms. The call passes the arguments of
    exahype2::fd::addCentralAdvection() in the order of its declaration.

    """
    solver = solver_class("CCZ4", parameters=parameters)
    scheme = _lopsided(solver)
    code = solver._advection_implementation()

    quantities = _constexpr_list(code, "quantities")
    assert quantities == scheme.advected_quantities
    assert _constexpr_list(code, "velocity") == scheme.advection_velocity
    assert all(index < solver.unknowns for index in quantities)

    call = re.search(r"::exahype2::fd::addCentralAdvection\((.*?)\);", code)
    assert call.group(1).split(", ") == [
        "Q",
        "deltaQ",
        "BTimesDeltaQ",
        "normal",
        "velocity",
        "quantities",
        str(len(quantities)),
    ]
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "src",
            "exahype2",
            "fd",
            "CentralAdvection.h",
        )
    ) as header:
        declaration = re.search(
            r"void addCentralAdvection\((.*?)\);", header.read(), re.DOTALL
        )
    declared = [parameter.split()[-1] for parameter in declaration.group(1).split(",")]
    assert declared == [
        "Q",
        "deltaQ",
        "BTimesDeltaQ",
        "normal",
        "velocity",
        "quantities",
        "numberOfQuantities",
    ]


def _realise(solver, tmp_path):
    """! Realise a first-order solver with gauge wave initial data on the unit cube"""
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


def _solver_with(scheme_factory):
    solver = exagrype.api.FOCCZ4Solver("CCZ4", parameters={"CCZ4s": 1, "CCZ4b": 1})
    solver.add_pde_term(exagrype.api.initialvalues.gauge_wave())
    solver.numerical_scheme = scheme_factory(solver)
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(
        solver, volume_size=1.0 / 27.0
    )
    return solver


def _non_conservative_products(solver):
    return [
        term
        for term in solver.pde_terms
        if term.name == exahype2.api.NonconservativeProduct.NAME
    ]


@pytest.mark.parametrize(
    "scheme_factory, advection",
    [(_central_differences, True), (_lopsided, False)],
)
def test_realisation_registers_the_ncp_which_fits_the_numerical_scheme(
    scheme_factory, advection, tmp_path
):
    """!

    The solver is created before its numerical scheme, so it can pick the
    non-conservative product only when it is realised. It registers a
    single, stateless one

    """
    solver = _solver_with(scheme_factory)
    assert not _non_conservative_products(solver)
    _realise(solver, tmp_path)

    (term,) = _non_conservative_products(solver)
    assert term.stateless
    assert "::exagrype::ccz4::ncpFirstOrder(" in term.implementation
    assert ("::exahype2::fd::addCentralAdvection(" in term.implementation) == advection


def test_realising_twice_replaces_the_ncp_of_the_first_realisation(tmp_path):
    """! A project can be generated several times"""
    solver = _solver_with(_central_differences)
    _realise(solver, tmp_path / "first")
    _realise(solver, tmp_path / "second")
    (term,) = _non_conservative_products(solver)
    assert term.implementation.count("::exahype2::fd::addCentralAdvection(") == 1


@pytest.mark.parametrize("stateless", [True, False])
def test_realisation_rejects_an_ncp_added_by_somebody_else(stateless, tmp_path):
    """!

    The solver would replace it silently, and it might add advection terms
    the scheme adds as well, or miss them
    (src/exagrype/ccz4/adrs/0003-the-solver-picks-who-adds-the-advection.md)

    """
    solver = _solver_with(_central_differences)
    solver.add_pde_term(
        exahype2.api.NonconservativeProduct(implementation="", stateless=stateless)
    )
    with pytest.raises(ValueError, match="add_pde_term"):
        _realise(solver, tmp_path)
