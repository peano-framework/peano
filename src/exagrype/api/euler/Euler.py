""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api

from ..FOCCZ4Solver import FOCCZ4Solver
from . import initialvalues


class Euler(exahype2.api.Solver):
    """!

    Compressible Euler equations of an ideal gas

    The solver evolves the five conserved quantities density, momentum and
    total energy,

    @f$
      \\partial_t \\begin{pmatrix} \\rho \\\\ j \\\\ E \\end{pmatrix}
      + \\nabla \\cdot \\begin{pmatrix} j \\\\ j \\otimes j / \\rho + p I \\\\ (E+p) j / \\rho \\end{pmatrix} = 0,
      \\qquad p = (\\gamma-1) \\left(E - \\frac{|j|^2}{2\\rho}\\right),
    @f$

    in three dimensions. It has a flux, a maximum eigenvalue, a source term and an initial
    condition, all of them stateless. The source term is zero.
    Flux, source and eigenvalue see the CCZ4 fields, even though they ignore
    them at the moment. The solver always holds a
    set of auxiliary variables with the 59 entries of first-order CCZ4,
    which the user attaches (add_auxiliary_variables()); generating the
    solver without it fails. By default, the
    initial condition is a density wave which the fluid advects with
    constant velocity (initialvalues.smooth_wave()).

    The solver is deliberately plain. Its purpose within ExaGRyPE is to be
    the matter ingredient which one couples to exagrype.api.FOCCZ4Solver. The
    assembly of such a coupled system is not part of this class, but done by
    the user (see benchmarks/exagrype/gauge-wave-with-Euler), who typically
    attaches auxiliary variables (@ref exahype2.api.AuxiliaryVariables) with
    the spacetime to the solver, and replaces the flux and the source term
    through the hooks _flux_implementation() and _source_implementation().

    ## Shortcomings

    - The equations are those of flat-space hydrodynamics, i.e. the solver
      knows nothing about general relativity on its own.
    - Only ideal gases are supported, and there is no positivity limiter.

    """

    Unknowns = {
        "rho": 1,
        "j": 3,
        "E": 1,
    }
    """! The five conserved quantities in the order the flux expects them """

    def __init__(
        self,
        name: str,
        gamma: float = 1.4,
        initial_condition: exahype2.api.InitPointWisely = None,
    ):
        """!

        Create an Euler solver

        @param name: str
          Name of the solver, @ref exahype2.api.Solver.

        @param gamma: float
          Adiabatic index of the ideal gas. Has to be greater than one.

        @param initial_condition: exahype2.api.InitPointWisely
          Initial condition. Defaults to exagrype.api.euler.initialvalues.smooth_wave().

        """
        if gamma <= 1.0:
            raise ValueError(f"Solver {name}: gamma has to be greater than one")
        super().__init__(name, unknowns=self.Unknowns)
        self._gamma = gamma

        self.add_user_solver_includes(
            '#include "exagrype/euler/EulerKernels.h"\n'
            '#include "exagrype/euler/initialvalues/InitialValues.h"\n'
        )
        self.add_abstract_solver_attribute(
            'static_assert(Dimensions == 3, "The Euler solver of ExaGRyPE is three-dimensional only");'
        )
        self.add_abstract_solver_attribute(
            f"static constexpr double Gamma = {gamma!r};"
        )

        self._flux = exahype2.api.Flux(implementation="", stateless=True)
        self._source = exahype2.api.VolumetricSource(implementation="", stateless=True)
        self._max_eigenvalue = exahype2.api.MaxEigenvalue(
            implementation="", stateless=True
        )
        self.add_pde_term(self._flux)
        self.add_pde_term(self._source)
        self.add_pde_term(self._max_eigenvalue)
        self.add_pde_term(
            initial_condition if initial_condition is not None else initialvalues.smooth_wave()
        )

    def add_auxiliary_variables(self, auxiliary: exahype2.api.AuxiliaryVariables):
        """!

        Attach a set of auxiliary variables and hand it to flux, source and eigenvalue

        The sets are not known when the constructor creates the PDE terms, so
        the terms are (re)built from the hooks whenever a set arrives. They
        stay empty until the first set has been attached.

        @param auxiliary: exahype2.api.AuxiliaryVariables
          The set, @ref exahype2.api.Solver.add_auxiliary_variables().

        """
        super().add_auxiliary_variables(auxiliary)
        self._refresh_pde_terms()

    def generate_solver_realisation(self, *args, **kwargs):
        """!

        Reject solvers that do not see the CCZ4 fields

        The Euler solver is the matter part of a coupled system with
        first-order CCZ4. Hence, it always holds a set of auxiliary variables
        with as many entries as FOCCZ4Solver has unknowns. Flux, source and
        eigenvalue rely on this and do not handle solvers without
        auxiliary variables. The check runs at generation time, as the sets
        are attached after the constructor.

        """
        size = sum(FOCCZ4Solver.Unknowns.values())
        if self._ccz4_fields_set() is None:
            raise ValueError(
                f"Solver {self.name}: the Euler solver needs a set of auxiliary variables "
                f"with the {size} entries of first-order CCZ4, but has "
                f"{[(a.name, a.size) for a in self.auxiliary_variables]}"
            )
        return super().generate_solver_realisation(*args, **kwargs)

    def _refresh_pde_terms(self):
        """!

        Rebuild the implementations of flux, source and eigenvalue from the hooks

        Nothing happens before the set with the CCZ4 fields has been attached,
        as the kernels need it.

        """
        if self._ccz4_fields_set() is None:
            return
        self._flux.implementation = self._flux_implementation()
        self._source.implementation = self._source_implementation()
        self._max_eigenvalue.implementation = self._max_eigenvalue_implementation()

    def _ccz4_fields_set(self):
        """!

        The set of auxiliary variables which holds the CCZ4 fields

        It is the set with as many entries as FOCCZ4Solver has unknowns. Other
        sets (such as the lapse of a toy coupling) are not handed to the kernels.

        @return exahype2.api.AuxiliaryVariables
          The set, or None if it has not been attached yet.

        """
        size = sum(FOCCZ4Solver.Unknowns.values())
        candidates = [a for a in self.auxiliary_variables if a.size == size]
        if len(candidates) > 1:
            raise ValueError(
                f"Solver {self.name}: more than one set of auxiliary variables has the {size} entries of CCZ4: "
                f"{[a.name for a in candidates]}"
            )
        return candidates[0] if candidates else None

    def _ccz4_fields(self) -> str:
        """!

        C++ expression of the pointer to the CCZ4 fields in a PDE term

        The kernels of exagrype::euler take it as ccz4Fields.

        """
        return self._ccz4_fields_set().argument_name

    @property
    def gamma(self) -> float:
        """! Adiabatic index of the gas"""
        return self._gamma

    def _flux_implementation(self) -> str:
        """!

        C++ of the flux along normal

        Hook for subclasses, which can scale or modify the flux, e.g. by the
        lapse of a spacetime they hold as auxiliary variables. The code has
        the variables Q, x, normal and F at hand, plus one pointer per set of
        auxiliary variables (the CCZ4 fields: _ccz4_fields()). The default
        calls exagrype::euler::flux().

        """
        return f"""
  ::exagrype::euler::flux( Q, {self._ccz4_fields()}, normal, Gamma, F );
"""

    def _source_implementation(self) -> str:
        """!

        C++ of the volumetric source

        Hook for subclasses. The code has Q, x, t and S at hand, plus the
        arguments of _flux_implementation(). The default calls
        exagrype::euler::source(), which yields zero.

        """
        return f"""
  ::exagrype::euler::source( Q, {self._ccz4_fields()}, Gamma, S );
"""

    def _max_eigenvalue_implementation(self) -> str:
        return f"""
  *lambdaMax = ::exagrype::euler::maxEigenvalue( Q, {self._ccz4_fields()}, normal, Gamma );
"""
