""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

from typing import Dict, Union

import exahype2.api


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

    in three dimensions. It has a flux, a maximum eigenvalue and an initial
    condition, all of them stateless, and no source term. By default, the
    initial condition is a density wave which the fluid advects with
    constant velocity (smooth_wave()).

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
          Initial condition. Defaults to smooth_wave().

        """
        if gamma <= 1.0:
            raise ValueError(f"Solver {name}: gamma has to be greater than one")
        super().__init__(name, unknowns=self.Unknowns)
        self._gamma = gamma

        self.add_abstract_solver_attribute(
            'static_assert(Dimensions == 3, "The Euler solver of ExaGRyPE is three-dimensional only");'
        )
        self.add_abstract_solver_attribute(
            f"static constexpr double Gamma = {gamma!r};"
        )

        self.add_pde_term(
            exahype2.api.Flux(
                implementation=self._flux_implementation(), stateless=True
            )
        )
        source = self._source_implementation()
        if source is not None:
            self.add_pde_term(
                exahype2.api.VolumetricSource(implementation=source, stateless=True)
            )
        self.add_pde_term(
            exahype2.api.MaxEigenvalue(
                implementation=self._max_eigenvalue_implementation(), stateless=True
            )
        )
        self.add_pde_term(
            initial_condition if initial_condition is not None else self.smooth_wave()
        )

    @property
    def gamma(self) -> float:
        """! Adiabatic index of the gas"""
        return self._gamma

    @staticmethod
    def smooth_wave(
        amplitude: float = 0.2, velocity: float = 1.0, pressure: float = 1.0
    ) -> exahype2.api.InitPointWisely:
        """!

        Density wave on constant pressure

        The density is 1 + amplitude * sin(2 pi x) and the fluid moves along
        the x-axis with constant velocity. Pressure and velocity are
        constant, i.e. this is an exact solution of the Euler equations
        which merely advects the wave. Any additional entries behind the
        five unknowns, i.e. auxiliary variables which the solver owns, are
        not touched.

        @param amplitude: float
          Amplitude of the density wave. Has to be smaller than one to keep
          the density positive.

        @param velocity: float
          Speed along the x-axis.

        @param pressure: float
          Constant pressure.

        """
        if abs(amplitude) >= 1.0:
            raise ValueError("amplitude has to be smaller than one in modulus")
# @todo These snippets should go into a C++ file called Kernels.h/Kernels.cpp but in a subdirectory euler parallel to ccz4        
        return exahype2.api.InitPointWisely(
            implementation=f"""
  const double rho = 1.0 + {amplitude!r} * std::sin( 2.0 * M_PI * x(0) );
  Q[0] = rho;
  Q[1] = rho * {velocity!r};
  Q[2] = 0.0;
  Q[3] = 0.0;
  Q[4] = {pressure!r} / (Gamma - 1.0) + 0.5 * rho * {velocity!r} * {velocity!r};
""",
        )

    def _flux_implementation(self) -> str:
        """!

        C++ of the flux along normal

        Hook for subclasses, which can scale or modify the flux, e.g. by the
        lapse of a spacetime they hold as auxiliary variables. The code has
        the variables Q, x, normal and F at hand, as well as the pressure.

        """
# @todo These snippets should go into a C++ file called Kernels.h/Kernels.cpp but in a subdirectory euler parallel to ccz4. This is a general idea that should be documented like this in the guidelines
        return """
  const double invRho = 1.0 / Q[0];
  const double u      = Q[1 + normal] * invRho;
  const double p      = (Gamma - 1.0) * (Q[4] - 0.5 * invRho * (Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3]));

  F[0] = Q[1 + normal];
  F[1] = Q[1] * u;
  F[2] = Q[2] * u;
  F[3] = Q[3] * u;
  F[1 + normal] += p;
  F[4] = (Q[4] + p) * u;
"""

    def _source_implementation(self) -> Union[str, None]:
        """!

        C++ of the volumetric source or None if the system has none

        Hook for subclasses.

        """
        return None

    def _max_eigenvalue_implementation(self) -> str:
# @todo These snippets should go into a C++ file called Kernels.h/Kernels.cpp but in a subdirectory euler parallel to ccz4. This is a general idea that should be documented like this in the guidelines
        return """
  const double invRho = 1.0 / Q[0];
  const double p      = (Gamma - 1.0) * (Q[4] - 0.5 * invRho * (Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3]));
  const double c      = std::sqrt( Gamma * p * invRho );
  *lambdaMax          = std::abs( Q[1 + normal] * invRho ) + c;
"""
