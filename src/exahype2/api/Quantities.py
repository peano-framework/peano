# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from typing import List, Tuple, Union


def resolve_quantities(
    solver,
    quantities: List[Union[str, int]],
) -> List[int]:
    """!

    Translate names and indices of the unknowns Q into sorted indices

    @param solver: exahype2.api.Solver
      The solver whose entries of Q we refer to.

    @param quantities: List[str or int]
      A string is the name of a variable as passed to the solver (e.g.
      "lapse" or "G" for exagrype.api.FOCCZ4Solver) and selects all of its
      entries. An int is the index of one entry of Q. The list must not be
      empty.

    @return List[int]
      Sorted indices without duplicates.

    """
    entries = solver.unknowns
    if len(quantities) == 0:
        raise ValueError(f"Solver {solver.name}: the list of quantities is empty")
    names = solver.variable_names_per_entry()[:entries]
    indices = set()
    for quantity in quantities:
        if isinstance(quantity, bool) or not isinstance(quantity, (int, str)):
            raise TypeError(
                f"Solver {solver.name}: quantities have to be names (str) or "
                f"indices (int) of entries of Q, but got {quantity!r}"
            )
        if isinstance(quantity, int):
            if not 0 <= quantity < entries:
                raise ValueError(
                    f"Solver {solver.name}: quantity index {quantity} is out of "
                    f"range, as only {entries} entries can be selected"
                )
            indices.add(quantity)
        else:
            matches = [
                index
                for index, name in enumerate(names)
                if name == quantity or name.startswith(quantity + "[")
            ]
            if not matches:
                raise ValueError(
                    f"Solver {solver.name}: {quantity!r} is not the name of a "
                    f"variable which can be selected. Available names are "
                    f"{sorted(set(name.split('[')[0] for name in names))}."
                )
            indices.update(matches)
    return sorted(indices)


def _entries_by_name(names: List[str], name: str) -> List[int]:
    """! Positions of the entries called name or name[k], in ascending order"""
    return [
        position
        for position, entry in enumerate(names)
        if entry == name or entry.startswith(name + "[")
    ]


def resolve_auxiliary_derivatives(
    solver,
    auxiliary_derivatives: List[Tuple[str, str, int, float]],
) -> List[Tuple[int, int, int, float]]:
    """!

    Translate a table of auxiliary variables which are derivatives into indices

    Each row (derivative, unknown, axis, factor) states that the auxiliary
    variable derivative equals factor times the derivative of the unknown
    along axis, entry by entry. A name selects all entries of a variable
    (e.g. "G"), a name with an index one entry (e.g. "dLapse[0]"). The names
    of auxiliary variables may be qualified by the name of their set
    ("Derivatives.dLapse[0]") and have to be qualified if several sets
    hold a variable of this name.

    @param solver: exahype2.api.Solver
      The solver whose unknowns and auxiliary variables the names refer to.

    @param auxiliary_derivatives: List[Tuple[str, str, int, float]]
      The table, @see exagrype.api.FOSOCCZ4Solver.FOSOCCZ4Solver.auxiliary_derivatives()
      for an example.

    @return List[Tuple[int, int, int, float]]
      One row per entry: the index of the derivative and the one of the
      unknown in the input of the kernels (unknowns first, then the
      auxiliary variables, @see
      exahype2.api.Solver.Solver.auxiliary_variable_offset()), the axis and
      the factor, sorted by the index of the derivative.

    @exception ValueError if a name selects nothing or is ambiguous, if a row
      pairs variables with different numbers of entries, if an axis is not
      0, 1 or 2, or if an entry of the auxiliary variables appears twice.

    """
    unknown_names = solver.variable_names_per_entry()[: solver.unknowns]
    auxiliary_names = []
    kernel_input_index = []
    for auxiliary in solver.auxiliary_variables:
        offset = solver.auxiliary_variable_offset(auxiliary)
        auxiliary_names += auxiliary.variable_names_per_entry()
        kernel_input_index += [offset + k for k in range(auxiliary.size)]
    unqualified_names = [name.split(".", 1)[1] for name in auxiliary_names]

    def derivative_entries(name: str) -> List[int]:
        entries = _entries_by_name(auxiliary_names, name) or _entries_by_name(
            unqualified_names, name
        )
        if not entries:
            raise ValueError(
                f"Solver {solver.name}: {name!r} is no auxiliary variable. "
                f"Available are {sorted(set(n.split('[')[0] for n in unqualified_names))}."
            )
        sets = {auxiliary_names[entry].split(".", 1)[0] for entry in entries}
        if len(sets) > 1:
            raise ValueError(
                f"Solver {solver.name}: {name!r} names variables of the sets "
                f"{sorted(sets)}. Qualify it by the name of the set."
            )
        return [kernel_input_index[entry] for entry in entries]

    def unknown_entries(name: str) -> List[int]:
        entries = _entries_by_name(unknown_names, name)
        if not entries:
            raise ValueError(
                f"Solver {solver.name}: {name!r} is no unknown. Available are "
                f"{sorted(set(n.split('[')[0] for n in unknown_names))}."
            )
        return entries

    rows = {}
    for derivative, unknown, axis, factor in auxiliary_derivatives:
        if axis not in (0, 1, 2):
            raise ValueError(
                f"Solver {solver.name}: the axis of {derivative!r} has to be 0, 1 "
                f"or 2, but is {axis!r}"
            )
        targets = derivative_entries(derivative)
        sources = unknown_entries(unknown)
        if len(targets) != len(sources):
            raise ValueError(
                f"Solver {solver.name}: {derivative!r} has {len(targets)} entries, "
                f"but {unknown!r} has {len(sources)}"
            )
        for target, source in zip(targets, sources):
            if target in rows:
                raise ValueError(
                    f"Solver {solver.name}: the auxiliary variable entry {target} "
                    f"is defined twice"
                )
            rows[target] = (target, source, axis, float(factor))
    return [rows[target] for target in sorted(rows)]
