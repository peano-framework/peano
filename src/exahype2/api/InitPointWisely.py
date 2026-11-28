# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from __future__ import annotations

from .PDETerm import PDETerm


class InitPointWisely(PDETerm):
    """!

    Point-wise initial condition: given the physical position of a single
    degree of freedom, write its initial Q there.

    Unlike the legacy, cell-wise initialCondition() (handed a whole mesh
    cell's centre and extent plus gridIsConstructed, and expected to fill an
    entire patch itself - see python/exahype2/solvers/rkfd/*.template.h),
    this term is evaluated once per degree of freedom, exactly like Flux and
    MaxEigenvalue are:

    @code
    virtual void initPointWisely(
      const tarch::la::Vector<Dimensions, double>& x,
      double* __restrict__                         Q // Q[{{NUMBER_OF_UNKNOWNS}}]
    );
    @endcode

    Q holds the unknowns. The term receives no auxiliary variables
    (@ref exahype2.api.AuxiliaryVariables), even if the solver has some. Set
    them via exahype2.api.InitAuxiliaryVariablesPointWisely, or in setup
    tasks of your own.

    """

    SIGNATURE = (
        "const tarch::la::Vector<Dimensions, double>& x, "
        "double* __restrict__ Q"
    )

    NAME = "initPointWisely"

    RECEIVES_AUXILIARY_VARIABLES = False

    def __init__(
        self,
        implementation: str = None,
        stateless: bool = False,
    ):
        """!

        implementation: str
          A C++ snippet that becomes initPointWisely()'s body (the empty
          string is a valid, generated no-op body, though a no-op initial
          condition leaves Q uninitialised), or None (the default) to leave
          initPointWisely() pure virtual for you to implement by hand.

          A grid construction such as exahype2.api.fv.SetupSimulationOverRegularGrid
          validates that the solver it is attached to has this term
          registered before it does anything else, since it relies on it to
          fill in the initial condition.

        """
        super().__init__(
            name=self.NAME,
            implementation=implementation,
            stateless=stateless,
        )

    class Generator(PDETerm.Generator):
        """!

        InitPointWisely-specific code generation

        Reuses PDETerm.Generator's declaration/definition/GPU-offload logic
        as-is - InitPointWisely.SIGNATURE is picked up automatically.
        Declared explicitly (rather than left to be inherited implicitly)
        for the same reason as Flux.Generator: the obvious place to add
        initPointWisely-specific generation behaviour later, and so
        init_point_wisely_term.generator reports its type as
        InitPointWisely.Generator rather than the more generic
        PDETerm.Generator.

        """
