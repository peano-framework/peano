""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api

from ..FOCCZ4Solver import FOCCZ4Solver


class ScalarField(exahype2.api.Solver):
    """!

    Four scalar fields which a spacetime advects

    The solver has four unknowns phi_0, ..., phi_3. They obey the advection equation

    @f$ \\partial_t \\phi + v \\partial_x \\phi = 0, \\qquad v = \\alpha^2, @f$

    along the x-axis, where @f$ \\alpha @f$ is the lapse of first-order CCZ4. The
    lapse enters through a set of auxiliary variables, which holds the 58
    entries of CCZ4 (@ref exagrype.api.FOCCZ4Solver). A coupling creates the
    set, fills it, and hands it over to the constructor. The coupling is a
    fake coupling: the spacetime advects the field, but the field does not
    feed back into the spacetime. The velocity is the metric component g_xx
    of the gauge wave, i.e. the equation has a closed-form solution on the
    gauge wave (@ref benchmarks/exagrype/gauge-wave-with-scalar-field).

    The PDE terms are a non-conservative product, a maximum eigenvalue and an
    initial condition (four sine waves along the x-axis with phase shifts). All of them
    are stateless and call @ref exagrype::scalarfield::nonconservativeProduct()
    and friends. The solver has no flux: the advective form is what the exact
    solution refers to.

    The solver is deliberately plain. Its numerical scheme is assigned like for any
    other solver (outer coupling), or by the coupling (inner coupling).

    """

    Unknowns = {"phi": 4}
    """! The four advected fields """

    def __init__(self, name: str, ccz4_fields: exahype2.api.AuxiliaryVariables):
        """!

        @param name: str
          Name of the solver, @ref exahype2.api.Solver.

        @param ccz4_fields: exahype2.api.AuxiliaryVariables
          The set which holds the 58 entries of first-order CCZ4. The solver
          attaches it. The coupling fills it.

        """
        if ccz4_fields.size != sum(FOCCZ4Solver.Unknowns.values()):
            raise ValueError(
                f"Solver {name}: the auxiliary set has to hold the "
                f"{sum(FOCCZ4Solver.Unknowns.values())} entries of first-order CCZ4, "
                f"but holds {ccz4_fields.size}"
            )
        super().__init__(name, unknowns=self.Unknowns)
        self.add_user_solver_includes('#include "exagrype/scalarfield/ScalarFieldKernels.h"\n')
        self.add_auxiliary_variables(ccz4_fields)
        fields = ccz4_fields.argument_name
        self.add_pde_term(
            exahype2.api.NonconservativeProduct(
                implementation=f"::exagrype::scalarfield::nonconservativeProduct({fields}, deltaQ, normal, BTimesDeltaQ);",
                stateless=True,
            )
        )
        self.add_pde_term(
            exahype2.api.MaxEigenvalue(
                implementation=f"*lambdaMax = ::exagrype::scalarfield::maxEigenvalue({fields}, normal);",
                stateless=True,
            )
        )
        self.add_pde_term(
            exahype2.api.InitPointWisely(
                implementation=f"::exagrype::scalarfield::initialValues(Q, {fields}, x);"
            )
        )
