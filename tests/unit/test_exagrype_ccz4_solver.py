import os
import re

import pytest

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


def test_solver_registers_ccz4_pde_terms():
    solver = exagrype.api.FOCCZ4Solver("CCZ4")
    names = {term.name for term in solver.pde_terms}
    assert names == {
        "nonconservativeProduct",
        "volumetricSource",
        "maxEigenvalue",
        "initPointWisely",
    }
    assert (
        "::exagrype::ccz4::ncpFirstOrder("
        in solver.get_pde_term("nonconservativeProduct", stateless=True).implementation
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


def test_default_initial_condition_is_gauge_wave():
    solver = exagrype.api.FOCCZ4Solver("CCZ4")
    assert "gaugeWave" in solver.get_pde_term("initPointWisely").implementation


def test_initial_condition_can_be_replaced():
    solver = exagrype.api.FOCCZ4Solver(
        "CCZ4", initial_condition=exagrype.api.initialvalues.flat()
    )
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
    call = solver.get_pde_term(term, stateless=True).implementation
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
