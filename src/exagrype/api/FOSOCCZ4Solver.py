""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

from typing import Dict, List, Union

import exahype2.api

from .CCZ4SolverBase import CCZ4SolverBase


class FOSOCCZ4Solver(CCZ4SolverBase):
    """!

    Solver for the Einstein equations in the first-/second-order reconstructed
    CCZ4 formulation

    The solver evolves the first 25 of the 59 CCZ4 unknowns (see
    Unknowns), i.e. the conformal metric, the extrinsic curvature, the
    gauge and the auxiliary evolution variables. The remaining 34 entries
    (DerivativeUnknowns) are the first derivatives of the lapse, the shift,
    the conformal metric and phi, plus K0. They are not evolved but held as
    auxiliary variables (@ref exahype2.api.AuxiliaryVariables) named
    "Derivatives". The PDE terms hand the 25 unknowns and, as a second
    pointer, the 34 auxiliary variables to the very same kernels as the
    first-order solver exagrype.api.FOCCZ4Solver. The second-order kernels set the time derivative of the 34
    derivatives to zero, and the solver does not even ask for it.

    ## Ping-pong with the reconstruction solver

    Nobody updates the derivatives on their own. A second solver,
    exagrype.api.FOSOReconstructionSolver, does this. Do not create the pair
    yourself, but use exagrype.api.FOSOSolver, which hosts this solver (as
    ccz4) and its partner. The reconstruction solver
    runs in front of each stage of the
    Runge-Kutta scheme of this solver, recomputes the 34 derivatives from the
    25 unknowns of the stage and projects them onto the faces, so this solver
    finds them with a halo when it evaluates the stage. Without the
    reconstruction, the solver sees the derivatives of the initial
    condition forever.

    The solver works with exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
    (and its subclass exagrype.api.FD4WithKODissipation) and with
    exahype2.api.fv.RusanovWithExplicitEuler. The latter reconstructs the
    derivatives with second-order central differences.

    For Finite Differences, the numerical scheme has to have a halo of at least three grid cells:
    The solver needs the derivatives one layer outside of its cell (the
    fluxes and non-conservative products on the cell's faces use the values
    left and right of the face), and the stencil of the reconstruction
    reaches two layers further out. The plain scheme has a halo of two;
    lopsided advection or exagrype.api.FD4WithKODissipation widen it to
    three. The reconstruction solver rejects a smaller halo.

    @ref page_exagrype_fo_so_project_plan for the design and its status.

    ## Shortcomings

    - The second-order kernels do not advect the evolved unknowns. The
      numerical scheme has to do it, @ref lopsided_advection_quantities().
    - Only the two schemes above are supported.
    - Black holes (TwoPunctures), tracers and the ADM/Psi4 postprocessing are
      not available yet.
    - Checkpoints do not cover auxiliary variables, i.e. the solver cannot
      write checkpoints.

    """

    Unknowns = {
        "G": 6,
        "K": 6,
        "theta": 1,
        "Z": 3,
        "lapse": 1,
        "shift": 3,
        "b": 3,
        "traceK": 1,
        "phi": 1,
    }
    """!

    The 25 evolved unknowns in the order the kernels in exagrype::ccz4
    expect them, i.e. the first 25 entries of
    exagrype.api.FOCCZ4Solver.Unknowns.

    """

    DerivativeUnknowns = {
        "dLapse": 3,
        "dxShift": 3,
        "dyShift": 3,
        "dzShift": 3,
        "dxG": 6,
        "dyG": 6,
        "dzG": 6,
        "P": 3,
        "K0": 1,
    }
    """!

    The 34 entries of CCZ4 which this solver does not evolve, in the order the
    kernels expect them behind the unknowns. They are the auxiliary variables
    "Derivatives" and the 34 unknowns of exagrype.api.FOSOReconstructionSolver.

    """

    def __init__(
        self,
        name: str,
        parameters: Dict[str, Union[int, float]] = None,
        initial_condition: exahype2.api.InitPointWisely = None,
    ):
        """!

        Create a second-order reconstructed CCZ4 solver

        @param name: str
          Name of the solver, @see exahype2.api.Solver.

        @param parameters: Dict[str, Union[int, float]]
          CCZ4 parameters that differ from DefaultParameters, e.g.
          {"CCZ4k1": 0.0}. Keys which are not in DefaultParameters are
          rejected, as they most likely are typos.

        @param initial_condition: exahype2.api.InitPointWisely
          Initial condition. It fills the 59 entries of the first-order
          formulation, i.e. it can be any routine from
          exagrype.api.initialvalues. The solver takes the first 25 as its
          unknowns, and uses the other 34 as the initial value of the
          derivatives. Defaults to exagrype.api.initialvalues.gauge_wave().

        """
        super().__init__(name, self.Unknowns, parameters, initial_condition)
        self.derivatives = exahype2.api.AuxiliaryVariables(
            "Derivatives",
            self.DerivativeUnknowns,
            mode=exahype2.api.AuxiliaryVariables.Mode.CHANGING_ON_FACES,
            data_name="DerivativeVariables",
        )
        self.add_auxiliary_variables(self.derivatives)

    @property
    def task_graph_title_suffix(self) -> str:
        return " (CCZ4 part)"

    def lopsided_advection_quantities(self) -> List[str]:
        """!

        The unknowns whose advection terms the kernels omit

        exagrype::ccz4::sourceSecondOrder() and exagrype::ccz4::ncpSecondOrder() omit
        the advection @f$ \\beta^k \\partial _k Q @f$ of all 25 evolved
        unknowns, so the numerical scheme has to add it for all of them, e.g.
        via

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
          solver, patch_size, rk_order=4,
          right_hand_side_kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel.CentralDifferencesWithLopsidedAdvection,
          advection_velocity     = exagrype.api.FOSOCCZ4Solver.ADVECTION_VELOCITY,
          advected_quantities    = solver.lopsided_advection_quantities(),
        )
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        The advection of the shift and of b is scaled with CCZ4bs and
        CCZ4sk in the kernels (@see exagrype.api.FOCCZ4Solver.lopsided_advection_quantities()
        for the coupling). A kernel which advects each quantity on its own
        reproduces this only for CCZ4bs=CCZ4sk=1. If either of them is
        zero, the routine leaves the shift and b out.

        @return List[str]
          Names of unknowns, @see Unknowns.

        """
        excluded = set()
        if self._parameters["CCZ4bs"] == 0 or self._parameters["CCZ4sk"] == 0:
            excluded |= {"shift", "b"}
        return [name for name in self.Unknowns if name not in excluded]

    def _enforcement_implementation(self) -> str:
        return "::exagrype::ccz4::enforceCCZ4constraintsOfEvolvedVariables(Q);"

    @classmethod
    def _number_of_evolved_entries(cls) -> int:
        return sum(cls.Unknowns.values())

    @classmethod
    def _number_of_derivatives(cls) -> int:
        return sum(cls.DerivativeUnknowns.values())

    def _ncp_implementation(self) -> str:
        """! The secondary entries are the auxiliary variables"""
        return """
  ::exagrype::ccz4::ncpSecondOrder(
    BTimesDeltaQ,
    Q, derivatives,
    deltaQ, deltaDerivatives,
    normal, CCZ4LapseType, CCZ4ds, CCZ4c, CCZ4e, CCZ4f, CCZ4bs, CCZ4sk, CCZ4xi, CCZ4mu
  );
"""

    def _source_implementation(self) -> str:
        return f"""
  tarch::memset(S, 0, {sum(self.Unknowns.values())} * sizeof(double));
  ::exagrype::ccz4::sourceSecondOrder(
    S,
    Q, derivatives,
    CCZ4LapseType, CCZ4ds, CCZ4c, CCZ4e, CCZ4f, CCZ4bs, CCZ4sk, CCZ4xi, CCZ4itau, CCZ4eta, CCZ4k1, CCZ4k2, CCZ4k3
  );
"""

    def _adapt_initial_condition(
        self, initial_condition: exahype2.api.InitPointWisely
    ) -> exahype2.api.InitPointWisely:
        """!

        The initial condition fills 59 entries; we split them up

        The routine wraps the user's implementation. It runs the
        implementation on a local array of 59 entries (the implementation
        sees it as Q) and then copies the first 25 entries onto Q and the
        other 34 onto the derivatives.

        """
        if initial_condition.implementation is None:
            raise ValueError(
                f"Solver {self.name}: the initial condition needs an implementation "
                "which fills the 59 entries of the first-order formulation"
            )
        evolved = self._number_of_evolved_entries()
        return exahype2.api.InitPointWisely(
            implementation=f"""
  double initialValues[{self.NUMBER_OF_CCZ4_ENTRIES}];
  {{
    double* __restrict__ Q = initialValues;
{initial_condition.implementation}
  }}
  for (int i = 0; i < {evolved}; i++) Q[i] = initialValues[i];
  for (int i = 0; i < {self._number_of_derivatives()}; i++) derivatives[i] = initialValues[{evolved} + i];
"""
        )
