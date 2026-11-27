# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Compact second derivatives for auxiliary variables which are derivatives

The right-hand side kernels of @ref exahype2.api.rkfd4.FD4Kernels hand the
non-conservative product the fourth-order first derivative of every entry,
times the mesh width (deltaQ). If an auxiliary variable a is itself the
derivative of an unknown u along axis k, a = c d_k u, its derivative along
k is a second derivative of u. The fourth-order first derivative of a then
is the first derivative of the first derivative, a nine-point stencil with
the error h^4/15 f^(6) which returns zero for the highest frequency on the
grid. The compact five-point stencil COMPACT_SECOND_DERIVATIVE of u has the
error h^4/90 f^(6) and resolves this frequency.

right_hand_side_source() takes the source of a right-hand side kernel and
overwrites, after each deltaQ along k, the rows of the auxiliary variables
which are derivatives along k by c times the compact second derivative of u,
times the mesh width. The rows of derivatives along other axes (mixed second
derivatives) keep the first derivative of the first derivative, so no
stencil reaches into the diagonal blocks of the halo. See
src/exahype2/api/rkfd4/adrs/0003-compact-second-derivatives-of-auxiliary-derivatives.md.

"""
import inspect
from typing import List, Tuple

COMPACT_SECOND_DERIVATIVE = ((-2, -1.0), (-1, 16.0), (0, -30.0), (1, 16.0), (2, -1.0))
"""!
(shift, weight) of @f$ \\partial _{xx} f \\approx \\sum w_s f_{i+s} / (12 \\Delta x^2) @f$
"""

AXIS_NAMES = "xyz"


def _window(axis: int, dimensions: int, shift: int) -> str:
    """! Slices of the grid cells of a patch, shifted along axis"""
    slices = []
    for direction in range(dimensions):
        if direction == axis and shift != 0:
            slices.append(f"{shift}:NC{shift:+d}")
        else:
            slices.append("0:NC")
    return ", ".join(slices)


def _blocks(rows, axis: int) -> List[Tuple[int, int, int, float]]:
    """!

    Group the rows along axis into blocks of consecutive entries

    @return List of (first derivative, first unknown, number of entries, factor)

    """
    blocks = []
    for derivative, unknown, row_axis, factor in sorted(rows):
        if row_axis != axis:
            continue
        if blocks:
            first_derivative, first_unknown, count, block_factor = blocks[-1]
            if (
                derivative == first_derivative + count
                and unknown == first_unknown + count
                and factor == block_factor
            ):
                blocks[-1] = (first_derivative, first_unknown, count + 1, factor)
                continue
        blocks.append((derivative, unknown, 1, factor))
    return blocks


def overwrite_statements(rows, axis: int, dimensions: int, indent: str) -> List[str]:
    """!

    DSLHyPE statements which overwrite the rows of delta along axis

    DSLHyPE gives the temporary delta the index range of its first operand,
    i.e. -2:NC-2 along axis (@ref exahype2.api.rkfd4.FD4Kernels). Grid cell c
    of the patch hence is delta[..., c-2] along axis, and the left-hand side
    uses -2:NC-2 along axis, not 0:NC: The latter would write two grid cells
    off, partly beyond the end of delta.

    @param rows: List[Tuple[int, int, int, float]]
      (derivative, unknown, axis, factor) per entry, indices into the input
      of the kernel (@ref exahype2.api.Quantities.resolve_auxiliary_derivatives()).

    """
    name = AXIS_NAMES[axis]
    statements = []
    for derivative, unknown, count, factor in _blocks(rows, axis):
        target = f"{derivative}:{derivative + count}"
        source = f"{unknown}:{unknown + count}"
        stencil = " ".join(
            f"{'+' if weight > 0 else '-'} {abs(weight)!r} * input[{source}, {_window(axis, dimensions, shift)}]"
            for shift, weight in COMPACT_SECOND_DERIVATIVE
        )
        statements.append(
            f"{indent}delta_{name}[{target}, {_window(axis, dimensions, -2)}] = "
            f"{factor!r} * (1.0 / 12.0) * NC / h({axis}) * ({stencil})"
        )
    return statements


def right_hand_side_source(function, rows, dimensions: int) -> str:
    """!

    Source of a right-hand side kernel with compact second derivatives

    @param function: Python function
      exahype2.api.rkfd4.FD4Kernels.rightHandSideWithFD42D or ...3D.

    @param rows: List[Tuple[int, int, int, float]]
      (derivative, unknown, axis, factor) per entry. Rows along an axis
      beyond the dimensions of the kernel are ignored, as the kernel of the
      other dimension is generated as well.

    @param dimensions: int
      2 or 3.

    @return str
      The source of the function with the statements of
      overwrite_statements() behind each assignment of delta_x, delta_y
      (and delta_z).

    """
    lines = []
    found = set()
    for line in inspect.getsource(function).splitlines():
        lines.append(line)
        stripped = line.lstrip()
        for axis in range(dimensions):
            if stripped.startswith(f"delta_{AXIS_NAMES[axis]} = "):
                found.add(axis)
                lines += overwrite_statements(
                    rows, axis, dimensions, line[: len(line) - len(stripped)]
                )
    if found != set(range(dimensions)):
        raise RuntimeError(
            f"{function.__name__} does not assign delta along all {dimensions} axes"
        )
    return "\n".join(lines) + "\n"
