""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

from typing import Dict, List, Tuple, Union

import exahype2.api
from exahype2.api.Quantities import resolve_auxiliary_derivatives

from . import initialvalues
from .CCZ4SolverBase import CCZ4SolverBase


class FOSOCCZ4Solver(CCZ4SolverBase):
    """!

    Solver for the Einstein equations in the first-/second-order reconstructed
    CCZ4 formulation

    The solver evolves the first 25 of the 58 CCZ4 unknowns (see
    Unknowns), i.e. the conformal metric, the extrinsic curvature, the
    gauge and the auxiliary evolution variables. The remaining 33 entries
    (DerivativeUnknowns) are the first derivatives of the lapse, the shift,
    the conformal metric and phi. They are not evolved but held as
    auxiliary variables (@ref exahype2.api.AuxiliaryVariables) named
    "Derivatives". The PDE terms hand the 25 unknowns and, as a second
    pointer, the 33 auxiliary variables to the very same kernels as the
    first-order solver exagrype.api.FOCCZ4Solver. The second-order kernels set the time derivative of the 33
    derivatives to zero, and the solver does not even ask for it.

    ## Ping-pong with the reconstruction solver

    Nobody updates the derivatives on their own. A second solver,
    exagrype.api.FOSOReconstructionSolver, does this. Do not create the pair
    yourself, but use exagrype.api.FOSOSolver, which hosts this solver (as
    ccz4) and its partner. The reconstruction solver
    runs in front of each stage of the
    Runge-Kutta scheme of this solver, recomputes the 33 derivatives from the
    25 unknowns of the stage and projects them onto the faces, so this solver
    finds them with a halo when it evaluates the stage. Without the
    reconstruction, the solver sees the derivatives of the initial
    condition forever.

    The solver works with exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
    (and its subclass exagrype.api.FD4WithKODissipation) and with
    exahype2.api.fv.RusanovWithExplicitEuler. The latter reconstructs the
    derivatives with second-order central differences.

    For Finite Differences, the solver takes the halo of the numerical
    scheme, as the first-order solver does: two grid cells for the plain
    scheme, three for lopsided advection or exagrype.api.FD4WithKODissipation.
    The reconstruction computes the derivatives of the cell's own grid cells
    only, and its stencil reaches two layers out. The derivatives of the halo
    are the ones which the neighbouring cells have computed, exchanged via the
    faces
    (src/exagrype/ccz4/adrs/0004-the-foso-reconstruction-takes-the-halo-of-the-scheme.md).

    The second derivatives are first derivatives of the derivatives, except
    for the pure ones: With Finite Differences, the scheme computes the
    derivative of, e.g., dLapse[0] along x as the compact second derivative
    of the lapse. Hand auxiliary_derivatives() to the scheme for this, i.e.

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
      solver, patch_size, rk_order=4,
      auxiliary_derivatives = solver.auxiliary_derivatives(),
      ...
    )
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    The solver refuses a Finite Difference scheme without this table
    (src/exagrype/ccz4/adrs/0006-the-second-order-solver-uses-compact-second-derivatives.md).

    @ref page_exagrype_fo_so_project_plan for the design and its status.

    ## Shortcomings

    - The source holds no advection terms. They are added either by the
      Finite Differences with lopsided advection
      (@ref exagrype.api.FOSOCCZ4Solver.FOSOCCZ4Solver.advected_quantities())
      or by the non-conservative product
      (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.ncp_adds_advection()).
      The Finite Volumes add none of them.
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
        "Kex": 1,
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
    }
    """!

    The 33 entries of CCZ4 which this solver does not evolve, in the order the
    kernels expect them behind the unknowns. They are the auxiliary variables
    "Derivatives" and the 33 unknowns of exagrype.api.FOSOReconstructionSolver.

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
          Name of the solver (@ref exahype2.api.Solver).

        @param parameters: Dict[str, Union[int, float]]
          CCZ4 parameters that differ from DefaultParameters, e.g.
          {"CCZ4k1": 0.0}. Keys which are not in DefaultParameters are
          rejected, as they most likely are typos.

        @param initial_condition: exahype2.api.InitPointWisely
          Initial condition. It fills the 58 entries of the first-order
          formulation, i.e. it can be any routine from
          exagrype.api.initialvalues. The solver takes the first 25 as its
          unknowns, and uses the other 33 as the initial value of the
          derivatives. Defaults to exagrype.api.initialvalues.gauge_wave().

        """
        super().__init__(name, self.Unknowns, parameters)
        self.add_pde_term(
            self._adapt_initial_condition(
                initial_condition
                if initial_condition is not None
                else initialvalues.gauge_wave()
            )
        )
        self.derivatives = exahype2.api.AuxiliaryVariables(
            "Derivatives",
            self.DerivativeUnknowns,
            mode=exahype2.api.AuxiliaryVariables.Mode.CHANGING_ON_FACES,
            data_name="DerivativeVariables",
        )
        self.add_auxiliary_variables(self.derivatives)

    def advected_quantities(self) -> List[str]:
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
          advected_quantities    = solver.advected_quantities(),
        )
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        The advection of the shift and of b belongs to the shifting shift
        only, i.e. the routine leaves them out unless CCZ4s=1 and CCZ4b=1
        (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.advected_quantities()
        explains what the non-shifting shift does to the equation of b).
        Either the numerical scheme or the non-conservative product adds the
        advection of the unknowns returned
        (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.ncp_adds_advection()).

        @return List[str]
          Names of unknowns
          (@ref exagrype.api.FOSOCCZ4Solver.FOSOCCZ4Solver.Unknowns).

        """
        excluded = set()
        if self._parameters["CCZ4b"] == 0 or self._parameters["CCZ4s"] == 0:
            excluded |= {"shift", "b"}
        return [name for name in self.Unknowns if name not in excluded]

    @classmethod
    def auxiliary_derivatives(cls) -> List[Tuple[str, str, int, float]]:
        """!

        Which unknown each auxiliary variable differentiates

        Each entry (derivative, unknown, axis, factor) states that the
        auxiliary variable derivative equals factor times the derivative of
        the unknown along axis, entry by entry: dxG holds
        @f$ D_{xij} = \\frac{1}{2} \\partial _x \\tilde \\gamma _{ij} @f$, i.e. the
        six entries of dxG belong to the six entries of G with the factor
        one half. A name with an index (dLapse[0]) selects one entry of a
        variable, a name without an index all of them.

        The table is the only definition of the derivatives. The
        reconstruction (exagrype.api.FOSOReconstructionSolver) hands it to
        exagrype::ccz4::reconstructFirstDerivativesFD4(), and it must not be
        repeated anywhere else
        (src/exagrype/ccz4/adrs/0005-one-table-defines-the-auxiliary-derivatives.md).

        @return List[Tuple[str, str, int, float]]
          Names of a derivative (@ref DerivativeUnknowns) and of an unknown
          (@ref Unknowns), the axis and the factor. Together, the entries
          cover each of the 33 derivatives exactly once.

        """
        result = []
        for axis, name in enumerate("xyz"):
            result += [
                (f"dLapse[{axis}]", "lapse", axis, 1.0),
                (f"d{name}Shift", "shift", axis, 1.0),
                (f"d{name}G", "G", axis, 0.5),
                (f"P[{axis}]", "phi", axis, 1.0),
            ]
        return result

    def derivative_table(self) -> Tuple[List[int], List[int], List[float]]:
        """!

        auxiliary_derivatives() per entry of the auxiliary variables

        The names are resolved as for the compact second derivatives of the
        Finite Differences (@ref exahype2.api.Quantities.resolve_auxiliary_derivatives()),
        i.e. the reconstruction and the kernels read the same rows.

        @return differentiated_unknown, axis, factor
          Three lists with one element per entry of DerivativeUnknowns: the
          index of the differentiated unknown, the axis and the factor.

        @exception ValueError if auxiliary_derivatives() names an unknown
          variable, pairs variables with different numbers of entries or does
          not cover each derivative exactly once.

        """
        rows = resolve_auxiliary_derivatives(self, self.auxiliary_derivatives())
        offset = self.auxiliary_variable_offset(self.derivatives)
        size = self.derivatives.size
        missing = sorted(
            set(range(size)) - {derivative - offset for derivative, _, _, _ in rows}
        )
        if missing:
            raise ValueError(
                f"Solver {self.name}: auxiliary_derivatives() does not define the "
                f"derivative entries {missing}"
            )
        rows = sorted(rows)
        return (
            [unknown for _, unknown, _, _ in rows],
            [axis for _, _, axis, _ in rows],
            [factor for _, _, _, factor in rows],
        )

    def generate_solver_realisation(
        self,
        exahype2_project,
        peano_project,
        setup_simulation,
        time_step,
        plot,
        mesh_adaptation,
        add_global_tasks=True,
    ):
        """!

        Check the scheme, then realise the solver as any CCZ4 solver

        The Finite Differences have to compute the pure second derivatives
        with the compact stencil, i.e. the user has to hand
        auxiliary_derivatives() to the scheme (strict:
        src/exagrype/ccz4/adrs/0006-the-second-order-solver-uses-compact-second-derivatives.md).
        The solver checks this here rather than filling the table into the
        scheme itself, as it must not change the scheme which the user has
        created, and as the script states what the kernels do.

        @exception ValueError if the scheme is an
          exahype2.api.rkfd4.CentralDifferencesWithRungeKutta without the table
          of this solver.

        """
        scheme = self.numerical_scheme
        if isinstance(scheme, exahype2.api.rkfd4.CentralDifferencesWithRungeKutta):
            expected = resolve_auxiliary_derivatives(self, self.auxiliary_derivatives())
            if scheme.auxiliary_derivatives != expected:
                raise ValueError(
                    f"Solver {self.name}: the scheme has to compute the pure second "
                    "derivatives of the unknowns with the compact stencil. Pass "
                    "auxiliary_derivatives=solver.auxiliary_derivatives() to "
                    f"{type(scheme).__name__}."
                )
        super().generate_solver_realisation(
            exahype2_project,
            peano_project,
            setup_simulation,
            time_step,
            plot,
            mesh_adaptation,
            add_global_tasks,
        )

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
    normal, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, CCZ4b, CCZ4s, CCZ4mu
  );
"""

    def _source_implementation(self) -> str:
        return f"""
  tarch::memset(S, 0, {sum(self.Unknowns.values())} * sizeof(double));
  ::exagrype::ccz4::sourceSecondOrder(
    S,
    Q, derivatives,
    CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, CCZ4b, CCZ4s, CCZ4itau, CCZ4eta, CCZ4k1, CCZ4k2, CCZ4k3
  );
"""

    def _adapt_initial_condition(
        self, initial_condition: exahype2.api.InitPointWisely
    ) -> exahype2.api.InitPointWisely:
        """!

        The initial condition fills 58 entries; we split them up

        The routine wraps the user's implementation. It runs the
        implementation on a local array of 58 entries (the implementation
        sees it as Q) and then copies the first 25 entries onto Q and the
        other 33 onto the derivatives.

        """
        if initial_condition.implementation is None:
            raise ValueError(
                f"Solver {self.name}: the initial condition needs an implementation "
                "which fills the 58 entries of the first-order formulation"
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
