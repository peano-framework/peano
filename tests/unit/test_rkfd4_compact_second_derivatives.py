"""Compact second derivatives of auxiliary variables which are derivatives
(@ref exahype2.api.rkfd4.CompactSecondDerivatives,
src/exahype2/api/rkfd4/adrs/0003-compact-second-derivatives-of-auxiliary-derivatives.md)."""
import os
import types

import numpy as np
import pytest

import peano4.api
import peano4.api.taskgraph
import exahype2.api
from exahype2.api.Quantities import resolve_auxiliary_derivatives
from exahype2.api.rkfd4 import CompactSecondDerivatives
from exahype2.api.rkfd4 import FD4Kernels


def _solver(auxiliary_sets=None):
    """! Two unknowns u (2 entries) and v, and a set of derivatives of u"""
    solver = exahype2.api.Solver("Probe", {"u": 2, "v": 1})
    solver.add_pde_term(
        exahype2.api.NonconservativeProduct(implementation="BTimesDeltaQ[0] = 0.0;")
    )
    solver.add_pde_term(exahype2.api.MaxEigenvalue(implementation="*lambdaMax = 1.0;"))
    for auxiliary in auxiliary_sets or [
        exahype2.api.AuxiliaryVariables("Derivatives", {"du": 2, "dv": 1})
    ]:
        solver.add_auxiliary_variables(auxiliary)
    return solver


TABLE = [("du", "u", 0, 0.5), ("dv", "v", 1, 1.0)]


def test_compact_stencil_is_exact_for_quintics_and_sees_the_highest_frequency():
    h = 0.1
    x = 0.37
    weights = CompactSecondDerivatives.COMPACT_SECOND_DERIVATIVE
    for degree in range(6):
        value = sum(w * (x + s * h) ** degree for s, w in weights) / (12.0 * h * h)
        exact = degree * (degree - 1) * x ** max(degree - 2, 0) if degree >= 2 else 0.0
        assert value == pytest.approx(exact, abs=1e-9)
    # The highest frequency (-1)^i: the first derivative of the first derivative
    # returns zero, the compact stencil does not
    first = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    first_of_first = sum(
        a * b * (-1) ** (s + t) for s, a in first.items() for t, b in first.items()
    )
    assert first_of_first == 0.0
    assert sum(w * (-1) ** s for s, w in weights) == -64.0


def test_names_are_resolved_into_indices_of_the_kernel_input():
    solver = _solver()
    # Unknowns u[0], u[1], v; then du[0], du[1], dv
    assert resolve_auxiliary_derivatives(solver, TABLE) == [
        (3, 0, 0, 0.5),
        (4, 1, 0, 0.5),
        (5, 2, 1, 1.0),
    ]
    assert resolve_auxiliary_derivatives(
        solver, [("Derivatives.du[1]", "u[1]", 2, 2)]
    ) == [(4, 1, 2, 2.0)]


@pytest.mark.parametrize(
    "table, message",
    [
        ([("dw", "u", 0, 1.0)], "is no auxiliary variable"),
        ([("du", "w", 0, 1.0)], "is no unknown"),
        ([("du", "v", 0, 1.0)], "has 2 entries, but 'v' has 1"),
        ([("du", "u", 3, 1.0)], "has to be 0, 1 or 2"),
        ([("du", "u", 0, 1.0), ("du[0]", "u[0]", 1, 1.0)], "defined twice"),
    ],
)
def test_inconsistent_tables_are_rejected(table, message):
    with pytest.raises(ValueError, match=message):
        resolve_auxiliary_derivatives(_solver(), table)


def test_names_in_several_sets_have_to_be_qualified():
    solver = _solver(
        [
            exahype2.api.AuxiliaryVariables("First", {"d": 1}),
            exahype2.api.AuxiliaryVariables("Second", {"d": 1}),
        ]
    )
    with pytest.raises(ValueError, match="Qualify it"):
        resolve_auxiliary_derivatives(solver, [("d", "v", 0, 1.0)])
    assert resolve_auxiliary_derivatives(solver, [("Second.d", "v", 0, 1.0)]) == [
        (4, 2, 0, 1.0)
    ]


def test_the_source_overwrites_delta_behind_its_assignment():
    rows = resolve_auxiliary_derivatives(_solver(), TABLE)
    source = CompactSecondDerivatives.right_hand_side_source(
        FD4Kernels.rightHandSideWithFD43D, rows, 3
    )
    lines = [line.strip() for line in source.splitlines()]
    delta_x = next(i for i, line in enumerate(lines) if line.startswith("delta_x = "))
    # Consecutive entries with the same factor form one block. Grid cell c is
    # delta_x[..., c-2], i.e. the left-hand side uses -2:NC-2 along x
    assert lines[delta_x + 1] == (
        "delta_x[3:5, -2:NC-2, 0:NC, 0:NC] = 0.5 * (1.0 / 12.0) * NC / h(0) * "
        "(- 1.0 * input[0:2, -2:NC-2, 0:NC, 0:NC] + 16.0 * input[0:2, -1:NC-1, 0:NC, 0:NC] "
        "- 30.0 * input[0:2, 0:NC, 0:NC, 0:NC] + 16.0 * input[0:2, 1:NC+1, 0:NC, 0:NC] "
        "- 1.0 * input[0:2, 2:NC+2, 0:NC, 0:NC])"
    )
    assert lines[delta_x + 2].startswith("ncp_x = ")
    delta_y = next(i for i, line in enumerate(lines) if line.startswith("delta_y = "))
    assert lines[delta_y + 1].startswith("delta_y[5:6, 0:NC, -2:NC-2, 0:NC] = 1.0 * ")
    delta_z = next(i for i, line in enumerate(lines) if line.startswith("delta_z = "))
    assert lines[delta_z + 1].startswith("ncp_z = ")


def test_the_two_dimensional_kernel_ignores_the_third_axis():
    rows = resolve_auxiliary_derivatives(_solver(), [("dv", "v", 2, 1.0)])
    source = CompactSecondDerivatives.right_hand_side_source(
        FD4Kernels.rightHandSideWithFD42D, rows, 2
    )
    assert "delta_x[" not in source and "delta_y[" not in source


def _generated_kernel(solver, directory):
    peano_project = peano4.api.Project(["probe"], "probe", str(directory))
    solver.numerical_scheme.construct_output(
        types.SimpleNamespace(periodic_BC=[True, True, True]),
        peano_project,
        peano4.api.taskgraph.Graph(),
    )
    with open(
        os.path.join(peano_project.temporary_directory, "kernels", "Probe_fd4_3d.cpph")
    ) as file:
        return file.read()


def test_without_table_the_kernel_is_unchanged(tmp_path):
    plain = _solver()
    plain.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        plain, 4, rk_order=2
    )
    assert plain.numerical_scheme.auxiliary_derivatives is None
    reference = _generated_kernel(plain, tmp_path / "plain")
    other = _solver()
    other.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        other, 4, rk_order=2
    )
    assert _generated_kernel(other, tmp_path / "other") == reference


def test_generated_kernel_writes_the_grid_cells_of_the_patch(tmp_path):
    """! The guard against delta[..., 0:NC], which writes two grid cells off"""
    solver = _solver()
    solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
        solver, 4, rk_order=2, auxiliary_derivatives=TABLE
    )
    kernel = _generated_kernel(solver, tmp_path)
    overwrites = [
        line
        for line in kernel.splitlines()
        if line.strip().startswith("delta_x[i][") and "* (1.0 / 12.0)) * NC)" in line
    ]
    assert len(overwrites) == 1
    # Storage index l of delta_x is grid cell l: (l + ((- 2) + 2))
    assert "(l + ((- 2) + 2))" in overwrites[0]
