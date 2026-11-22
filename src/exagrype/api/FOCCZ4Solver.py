""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

from typing import Dict, Union

import exahype2.api

from .CCZ4SolverBase import CCZ4SolverBase


class FOCCZ4Solver(CCZ4SolverBase):
    """!

    Solver for the Einstein equations in the first-order CCZ4 formulation

    The solver is a plain exahype2.api.Solver with the 58 CCZ4 unknowns and
    the PDE terms of CCZ4 pre-registered. All terms delegate to the
    ExaGRyPE library (configure Peano with --enable-exagrype):

    - the non-conservative product to exagrype::ccz4::ncpFirstOrder(),
      plus the advection terms if the numerical scheme does not add them
      (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.ncp_adds_advection());
      it is registered only when the solver is realised,
    - the volumetric source to exagrype::ccz4::sourceFirstOrder(),
    - the maximum eigenvalue to exagrype::ccz4::maxEigenvalue().

    The solver has no initial condition. Add one as for any other solver,
    e.g. a routine of exagrype.api.initialvalues via add_pde_term().

    CCZ4 has no conservative flux. You still have to assign a numerical
    scheme, a grid construction and (optionally) a plotter, exactly as for
    any other exahype2.api.Solver, and to add the solver to an
    exahype2.api.Project. The project has to be three-dimensional: the
    generated solver class refuses to compile otherwise.

    ## Shortcomings

    - The fourth-order Finite Differences of the production runs are
      available through exahype2.api.rkfd4, i.e. pass a
      CentralDifferencesWithRungeKutta as numerical scheme. They have no
      artificial dissipation. exagrype.api.FD4WithKODissipation is the same
      scheme with a Kreiss-Oliger dissipation on top. Unlike the legacy
      ExaGRyPE, neither supports adaptive mesh refinement yet.
    - The solution is pushed back onto the CCZ4 constraints
      (exagrype::ccz4::enforceCCZ4constraints()) only if you switch it on
      via enforce_ccz4_constraint(). Without it, long runs drift away from
      the constraint surface.
    - The advection terms @f$ \\beta^k \\partial _k Q @f$ of all
      unknowns besides the metric, the lapse, the shift and phi (whose
      advection is part of the algebraic source) are added either by the
      fourth-order Finite Differences, if you select their kernel
      CentralDifferencesWithLopsidedAdvection with ADVECTION_VELOCITY and
      advected_quantities(), or by the non-conservative product
      with central differences otherwise. The Finite Volumes of
      exahype2.api.fv add none of them, i.e. they evolve CCZ4 without
      these advection terms
      (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.ncp_adds_advection()).
    - Black holes are limited to a single one
      (exagrype.api.initialvalues.single_black_hole()). Tracers and the
      ADM/Psi4 postprocessing are not available yet. The second-order
      formulation is exagrype.api.FOSOCCZ4Solver.

    """

    Unknowns = {
        "G": 6,
        "K": 6,
        "theta": 1,
        "Z": 3,
        "lapse": 1,
        "shift": 3,
        "b": 3,
        "Kex": 1,
        "phi": 1,
        "dLapse": 3,
        "dxShift": 3,
        "dyShift": 3,
        "dzShift": 3,
        "dxG": 6,
        "dyG": 6,
        "dzG": 6,
        "P": 3,
    }
    """!

    The 58 unknowns of the first-order CCZ4 formulation in the order the
    kernels in exagrype::ccz4 expect them: 25 primary entries (up to phi)
    followed by the 33 first derivatives. Kex is the trace of the extrinsic
    curvature, whereas K holds the six entries of the trace-free conformal
    extrinsic curvature.

    """

    def __init__(
        self,
        name: str,
        parameters: Dict[str, Union[int, float]] = None,
    ):
        """!

        Create a first-order CCZ4 solver

        @param name: str
          Name of the solver (@ref exahype2.api.Solver).

        @param parameters: Dict[str, Union[int, float]]
          CCZ4 parameters that differ from DefaultParameters, e.g.
          {"CCZ4k1": 0.0}. Keys which are not in DefaultParameters are
          rejected, as they most likely are typos.

        """
        super().__init__(name, self.Unknowns, parameters)

    def _ncp_implementation(self) -> str:
        """! The secondary entries follow the 25 primary ones"""
        return """
  ::exagrype::ccz4::ncpFirstOrder(
    BTimesDeltaQ, BTimesDeltaQ + 25,
    Q, Q + 25,
    deltaQ, deltaQ + 25,
    normal, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, CCZ4b, CCZ4s, CCZ4mu
  );
"""

    def _source_implementation(self) -> str:
        return f"""
  tarch::memset(S, 0, {sum(self.Unknowns.values())} * sizeof(double));
  ::exagrype::ccz4::sourceFirstOrder(
    S, S + 25,
    Q, Q + 25,
    CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, CCZ4b, CCZ4s, CCZ4itau, CCZ4eta, CCZ4k1, CCZ4k2, CCZ4k3
  );
"""
