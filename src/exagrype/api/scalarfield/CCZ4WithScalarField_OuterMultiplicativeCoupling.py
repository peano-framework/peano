""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api
import peano4.api
import peano4.api.taskgraph

from ..FOCCZ4Solver import FOCCZ4Solver
from ..actionsets import SynchroniseTimeStepSizes
from .ScalarField import ScalarField
from .create_copy_task import create_copy_task


class CCZ4WithScalarField_OuterMultiplicativeCoupling:
    """!

    First-order CCZ4 and an advected scalar field, run one after the other within each time step

    The class is the sibling of exagrype.api.euler.CCZ4WithEuler_OuterMultiplicativeCoupling,
    but the coupling is one-directional (fake coupling): The scalar field sees
    the spacetime, CCZ4 does not see the field. We get one copy task only:

        graph of CCZ4 -> copy (CCZ4 into scalar field) -> graph of scalar field -> synchronise

    i.e. the scalar field sees the CCZ4 fields after CCZ4 has completed its step.
    Each solver brings its own numerical scheme, assigned to the solvers
    (@ref exahype2.api.rkfd4.CentralDifferencesWithRungeKutta, ...), and
    the solvers do not exchange data within their Runge-Kutta stages. The
    spacetime is hence frozen at the end of the time step while the scalar field
    performs its whole step, and the coupling is first order in the time step
    size, however high the order of the time integrators is. An inner coupling
    (@ref exagrype.api.scalarfield.CCZ4WithScalarField_InnerCoupling) yields the order of the integrators.

    The graphs are chained with peano4.api.taskgraph.Graph.append(). Each solver graph
    ends in a GLOBAL_END_OF_STEP task that computes the next time step size.
    A final GLOBAL_END_OF_STEP task enforces the minimum of both time step sizes.

    """

    COPY_TASK_NAME = "Copy CCZ4 fields into {}"
    """! Name of the task that fills the auxiliary variables; the argument is the name of the scalar field solver """
    SYNCHRONISE_TASK_NAME = "Synchronise time step sizes"
    """! Name of the final task """

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
            mode=exahype2.api.AuxiliaryVariables.Mode.STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES,
            data_name="CCZ4FieldsForScalarField",
        )
        self.scalar_field = ScalarField(scalar_field_name, self.ccz4_auxiliary_copies)
        self._name = f"{ccz4_name}With{scalar_field_name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def solvers(self):
        """ The solvers the project has to register; exahype2.api.Project.add_solver() asks for them """
        return [self.ccz4, self.scalar_field]

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

        Realise both solvers, chain their time stepping graphs, and add the outcome to the project's graphs

        The arguments are those of exahype2.api.Solver.generate_solver_realisation().
        Both solvers are realised without global tasks. The load balancing and the dynamic
        mesh refinement wrap the chained time step graph afterwards, through the scheme of CCZ4.

        """
        peano4.api.Logger().new_section("Host " + self.name)
        Entity = peano4.api.taskgraph.Task.GridEntity

        ccz4_time_step = peano4.api.taskgraph.Graph()
        scalar_field_time_step = peano4.api.taskgraph.Graph()
        for solver, graph in [
            (self.ccz4, ccz4_time_step),
            (self.scalar_field, scalar_field_time_step),
        ]:
            solver.generate_solver_realisation(
                exahype2_project,
                peano_project,
                setup_simulation,
                graph,
                plot,
                mesh_adaptation,
                add_global_tasks=False,
            )

        copy = peano4.api.taskgraph.Graph()
        copy.add_task(
            create_copy_task(
                self.COPY_TASK_NAME.format(self.scalar_field.name),
                self.ccz4._patch,
                self.scalar_field,
                self.ccz4_auxiliary_copies,
            )
        )

        ## graph of CCZ4 -> copy -> graph of the scalar field
        coupled_time_step = peano4.api.taskgraph.Graph()
        coupled_time_step.append(ccz4_time_step)
        coupled_time_step.append(copy)
        coupled_time_step.append(scalar_field_time_step)

        ## Both solvers have computed their next time step size by now
        synchronise = peano4.api.taskgraph.Task(
            name=self.SYNCHRONISE_TASK_NAME, entity=Entity.GLOBAL_END_OF_STEP
        )
        for task in coupled_time_step.sink_tasks():
            synchronise.add_dependency(task)
        synchronise.add_action_set(
            SynchroniseTimeStepSizes([self.ccz4, self.scalar_field])
        )
        coupled_time_step.add_task(synchronise)

        self.ccz4.numerical_scheme.wrap_task_graph_in_load_balancing_and_amr_tasks(
            coupled_time_step
        )
        if self.ccz4.mesh_adaptation is not None:
            self.ccz4.mesh_adaptation.construct_output(
                exahype2_project, peano_project, mesh_adaptation
            )

        title = f"Time stepping task graph for host {self.name}"
        if peano4.api.Logger().is_visualisation:
            coupled_time_step.visualise(title=title)
        if peano4.api.Logger().is_text:
            coupled_time_step.log_as_text(title=title)
        time_step.merge(coupled_time_step)
