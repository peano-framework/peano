# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from __future__ import annotations

from .PDETerm import PDETerm


class InitAuxiliaryVariablesPointWisely(PDETerm):
    """!

    Point-wise initial values of the auxiliary variables which a solver owns

    exahype2.api.InitPointWisely sets the unknowns only. If the values of
    auxiliary variables (@ref exahype2.api.AuxiliaryVariables) come from the
    user, such as a sea floor, add this term. The setup calls it in a task of
    its own, which follows the initial condition, once per volume or grid
    cell. For a solver with a set Bathymetry, the signature reads

    @code
    virtual void initAuxiliaryVariablesPointWisely(
      const tarch::la::Vector<Dimensions, double>& x,
      const double* __restrict__                   Q,
      double* __restrict__                         bathymetry
    );
    @endcode

    Q holds the unknowns which initPointWisely() has set. A pointer to each
    set of the solver follows behind Q, in the order in which the sets have
    been attached. The solver receives writable pointers to the sets it owns,
    and const pointers to the sets it shares. The term sees one volume only.
    Values which depend on neighbouring volumes, such as derivatives, need
    setup tasks of their own (@ref exagrype.api.FOSOSolver).

    The term runs on the host.

    """

    SIGNATURE = (
        "const tarch::la::Vector<Dimensions, double>& x, "
        "const double* __restrict__ Q"
    )

    NAME = "initAuxiliaryVariablesPointWisely"

    WRITES_OWNED_AUXILIARY_VARIABLES = True

    def __init__(self, implementation: str = None):
        """!

        @param implementation: str
          A C++ snippet that becomes the body of
          initAuxiliaryVariablesPointWisely(), or None (the default) to leave
          the method pure virtual for you to implement by hand.

        """
        super().__init__(name=self.NAME, implementation=implementation)

    class Generator(PDETerm.Generator):
        """! Code generation is the one of any PDE term"""
