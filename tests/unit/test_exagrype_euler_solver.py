import pytest

import exahype2.api
import exagrype.api


def test_solver_has_five_named_unknowns():
    solver = exagrype.api.euler.Euler("Euler")
    assert solver.unknowns == 5
    assert solver.number_of_auxiliary_variables == 0


def test_solver_registers_pde_terms():
    solver = exagrype.api.euler.Euler("Euler")
    names = {term.name for term in solver.pde_terms}
    assert names == {"flux", "volumetricSource", "maxEigenvalue", "initPointWisely"}
    assert not solver.requires_user_class


def test_gamma_has_to_exceed_one():
    with pytest.raises(ValueError):
        exagrype.api.euler.Euler("Euler", gamma=1.0)


def test_hooks_extend_the_solver():
    class WithSource(exagrype.api.euler.Euler):
        def _source_implementation(self):
            return "tarch::memset(S, 0, 5 * sizeof(double));"

    names = {term.name for term in WithSource("Euler").pde_terms}
    assert "volumetricSource" in names


def test_flux_source_and_eigenvalue_see_the_ccz4_fields():
    solver = exagrype.api.euler.Euler("Euler")
    solver.add_auxiliary_variables(
        exahype2.api.AuxiliaryVariables("Spacetime", {"lapse": 1})
    )
    ccz4_fields = exahype2.api.AuxiliaryVariables(
        "CCZ4Fields", dict(exagrype.api.FOCCZ4Solver.Unknowns)
    )
    solver.add_auxiliary_variables(ccz4_fields)
    terms = {term.name: term.implementation for term in solver.pde_terms}
    for name in ["flux", "volumetricSource", "maxEigenvalue"]:
        assert f"{ccz4_fields.argument_name}," in terms[name]
        assert "spacetime" not in terms[name]


def test_two_sets_with_the_ccz4_size_are_ambiguous():
    solver = exagrype.api.euler.Euler("Euler")
    unknowns = dict(exagrype.api.FOCCZ4Solver.Unknowns)
    solver.add_auxiliary_variables(exahype2.api.AuxiliaryVariables("One", unknowns))
    with pytest.raises(ValueError):
        solver.add_auxiliary_variables(exahype2.api.AuxiliaryVariables("Two", unknowns))


def test_solver_without_the_ccz4_fields_cannot_be_generated():
    solver = exagrype.api.euler.Euler("Euler")
    solver.add_auxiliary_variables(
        exahype2.api.AuxiliaryVariables("Spacetime", {"lapse": 1})
    )
    with pytest.raises(ValueError):
        solver.generate_solver_realisation(None, None, None, None, None, None)


def test_solver_with_the_ccz4_fields_passes_the_check(monkeypatch):
    monkeypatch.setattr(
        exahype2.api.Solver, "generate_solver_realisation", lambda self, *a, **k: "ok"
    )
    solver = exagrype.api.euler.Euler("Euler")
    solver.add_auxiliary_variables(
        exahype2.api.AuxiliaryVariables(
            "CCZ4Fields", dict(exagrype.api.FOCCZ4Solver.Unknowns)
        )
    )
    assert solver.generate_solver_realisation(None, None, None, None, None, None) == "ok"
