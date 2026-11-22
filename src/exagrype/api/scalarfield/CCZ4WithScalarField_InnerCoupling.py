""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api
import peano4.api

from ..FOCCZ4Solver import FOCCZ4Solver
from .ScalarField import ScalarField


class CCZ4WithScalarField_InnerCoupling:
    """!

    First-order CCZ4 and an advected scalar field with one numerical scheme for both

    In an inner coupling, the numerical scheme sits in the coupled solver, i.e.
    in this class. The two aggregated solvers hold no numerical scheme of
    their own, and the user must not assign one to them. Instead, the user
    attaches a "generator" (a numerical scheme of the coupled system) to the
    coupled solver:

        coupled = exagrype.api.scalarfield.CCZ4WithScalarField_InnerCoupling()
        coupled.numerical_scheme = exagrype.api.scalarfield.RungeKuttaFD4InnerAdditive(patch_size=9, rk_order=4)

    There are four generators, one for each combination of Finite Volumes
    (explicit Euler) or Runge-Kutta with fourth-order Finite Differences, and
    of multiplicative or additive coupling. They exchange the data in each
    stage of the time integrator, i.e. both solvers advance the same stage,
    and the solvers see each other's data of the same time level. This gives
    the time integration order of the scheme: the lapse that the scalar field
    sees in stage k is the lapse of the stage input of CCZ4.
    The generators build the whole task graph of the combination, see
    @ref exagrype.api.scalarfield.InnerCouplingGenerator.

    Grid construction and plotters are not part of a generator, they are
    assigned to the solvers (host.ccz4, host.scalar_field) as always. The
    coupling is one-directional (fake coupling): CCZ4 does not see the field.
    The auxiliary variables of the scalar field hold the CCZ4 fields and change
    in each stage (exahype2.api.AuxiliaryVariables.Mode.CHANGING_ON_FACES).

    """

    def __init__(self, ccz4_name: str = "CCZ4", scalar_field_name: str = "ScalarField"):
        """!

        @param ccz4_name: str
          Name of the CCZ4 solver.

        @param scalar_field_name: str
          Name of the scalar field solver.

        """
        self.ccz4 = FOCCZ4Solver(ccz4_name)
        self.ccz4_auxiliary_copies = exahype2.api.AuxiliaryVariables(
            "CCZ4Fields",
            dict(FOCCZ4Solver.Unknowns),
            mode=exahype2.api.AuxiliaryVariables.Mode.CHANGING_ON_FACES,
            data_name="CCZ4FieldsForScalarField",
        )
        self.scalar_field = ScalarField(scalar_field_name, self.ccz4_auxiliary_copies)
        self._name = f"{ccz4_name}With{scalar_field_name}"
        self._numerical_scheme = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def solvers(self):
        """ The solvers the project has to register; exahype2.api.Project.add_solver() asks for them """
        return [self.ccz4, self.scalar_field]

    @property
    def numerical_scheme(self):
        """! The generator, i.e. the numerical scheme of the coupled system """
        return self._numerical_scheme

    @numerical_scheme.setter
    def numerical_scheme(self, generator):
        """!

        Attach the generator of the coupled system

        The aggregated solvers must not hold a numerical scheme of their own yet: the
        generator installs the building blocks on the solvers itself.

        """
        for solver in self.solvers:
            if solver.numerical_scheme is not None:
                raise ValueError(
                    f"Solver {solver.name} holds a numerical scheme of its own. "
                    f"In an inner coupling, assign the numerical scheme to the coupled solver {self.name} only"
                )
        generator.attach_to(self)
        self._numerical_scheme = generator

    def generate_solver_realisation(
        self,
        exahype2_project,
        peano_project,
        setup_simulation,
        time_step,
        plot,
        mesh_adaptation,
    ):
        """!

        Let the generator build the task graph of the coupled system and add it to the project's graphs

        The arguments are those of exahype2.api.Solver.generate_solver_realisation().

        """
        peano4.api.Logger().new_section("Host " + self.name)
        assert self.numerical_scheme is not None, (
            f"Coupled solver {self.name} has no numerical_scheme set. Assign a generator, "
            "e.g. exagrype.api.scalarfield.RungeKuttaFD4InnerAdditive, to numerical_scheme."
        )
        self.numerical_scheme.construct_output(
            exahype2_project,
            peano_project,
            setup_simulation,
            time_step,
            plot,
            mesh_adaptation,
        )
