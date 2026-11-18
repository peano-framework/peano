""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api
import peano4.api
import peano4.api.taskgraph
import toolbox.blockstructured.api

from .euler import Euler
from .FOCCZ4Solver import FOCCZ4Solver
from .actionsets import CopyPatchWithLinearInterpolation, SynchroniseTimeStepSizes


class CCZ4WithEuler_OuterAdditiveCoupling:
    """!

    First-order CCZ4 and an Euler fluid, coupled only at the begin and the end of each time step

    The class hosts one exagrype.api.FOCCZ4Solver (ccz4) and one
    exagrype.api.euler.Euler (euler). Add the host, not its solvers, to the project.
    The coupling is additive and affects the begin and the end of a time step
    only. In between, the solvers are independent, i.e. each solver brings its
    own numerical scheme and the schemes can differ in their time stepping, patch
    size and halo. Assign them to the solvers as for any other solver:

        coupled = exagrype.api.CCZ4WithEuler_OuterAdditiveCoupling()
        coupled.ccz4.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(coupled.ccz4, 8, rk_order=4)
        coupled.euler.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(coupled.euler, 17)

    The host sets no initial conditions. Add them to the solvers as to any other solver:

        coupled.ccz4.add_pde_term(exagrype.api.initialvalues.gauge_wave())

    (Euler brings a smooth wave by default.)

    If you prefer one numerical scheme for the whole compound, use the
    coupling per stage instead (@ref page_exahype_programmers_guide_coupling).

    ## Data

    Each solver reads all variables of the other one as auxiliary variables:
    ccz4 holds the set EulerFields (5 entries, resolved on the patch of CCZ4),
    euler holds the set CCZ4Fields (58 entries, resolved on the patch of Euler).
    The sets are read-only for the solvers, but they sit on the faces, and they
    change from time step to time step. The solvers hold no persistent copy of
    the fields of the other solver, i.e. the sets are temporary in the sense
    that they are overwritten at the begin of each time step. They do not
    change within a time step
    (exahype2.api.AuxiliaryVariables.Mode.STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES),
    i.e. the Runge-Kutta scheme of CCZ4 reconstructs the halo of the Euler
    fields in its first stage only and holds it through the later stages.

    ## Task graph

    The time step graph of the host is the concatenation of the graphs of the
    two solvers, preceded and followed by tasks of the host:

    1. Two prep tasks run in parallel. Each fills the auxiliary variables of one
       solver with the current solution of the other solver (if the patches have a
       different size, piecewise linear interpolation, @ref
       exagrype.api.actionsets.CopyPatchWithLinearInterpolation), projects them onto the faces,
       restricts the hanging faces onto the coarse ones, and rolls the faces over. The solvers reconstruct the halos of the auxiliary
       variables from these faces. Interpolation onto hanging faces
       rides along with the reconstruction of the solvers.
    2. All first tasks of both solvers wait for both prep tasks, so neither solver
       writes data that a copy is still reading.
    3. The solvers run their graphs independently.
    4. A GLOBAL_END_OF_STEP task waits for the last tasks of both solvers and enforces the
       minimum of the time step sizes on both of them
       (@ref exagrype.api.actionsets.SynchroniseTimeStepSizes).

    @dot
    digraph G {
      rankdir=LR;
      node [shape=box, style=rounded, fontname=Helvetica];
      prepCCZ4 [label="Prepare auxiliary variables\nfrom CCZ4\n(copy, project, restrict, roll over)"];
      prepEuler [label="Prepare auxiliary variables\nfrom Euler\n(copy, project, restrict, roll over)"];
      ccz4 [label="Task graph of CCZ4\n(RK4 with FD4)", shape=box3d];
      euler [label="Task graph of Euler\n(Finite Volumes)", shape=box3d];
      sync [label="Synchronise time step sizes\n(GLOBAL_END_OF_STEP)"];
      prepCCZ4 -> ccz4; prepCCZ4 -> euler;
      prepEuler -> ccz4; prepEuler -> euler;
      ccz4 -> sync; euler -> sync;
    }
    @enddot

    The two prep tasks run in parallel. Both solver graphs wait for both prep tasks, i.e.
    the arrows into the solver graphs end at the first tasks of each graph, and the arrows
    out of them start at their last tasks. Load balancing and dynamic mesh refinement wrap the whole picture.

    The synchronisation takes the minimum of the two time step sizes, as this is the
    conservative choice: the common step then respects the CFL condition of both solvers.

    The restriction sits in the prep task although the mesh of the benchmark is regular. The
    prep task is the one place that writes new face data, and restriction belongs to
    the writer (@ref exahype2.api.adrs). It facilitates overlapping domain decomposition and
    flux-preserving interpolation, which this class does not use.

    The coupling is bi-directional in the data, but a solver only sees the state
    of the other one at the begin of the time step. Neither the Einstein equations nor
    the Euler equations make use of the data yet: the PDE terms ignore the
    auxiliary variables unless you overwrite them.

    ## Global tasks

    CCZ4 owns the load balancing and the dynamic mesh refinement.

    """

    EULER_AUXILIARY_COPIES_NAME = "EulerFields"
    """! Name of the auxiliary variables of CCZ4 which hold the Euler fields """
    CCZ4_AUXILIARY_COPIES_NAME = "CCZ4Fields"
    """! Name of the auxiliary variables of Euler which hold the CCZ4 fields """
    PREP_TASK_NAME = "Prepare auxiliary variables from {}"
    """! Name of the tasks that fill the auxiliary variables; the argument is the name of the solver that provides the data """
    SYNCHRONISE_TASK_NAME = "Synchronise time step sizes"
    """! Name of the final task """

    def __init__(self, ccz4_name: str = "CCZ4", euler_name: str = "Euler"):
        """!

        Create the two solvers and the auxiliary variables through which they see each other

        The coupling happens at the begin and at the end of the time step
        only, so the numerical schemes of the solvers remain flexible.

        @param ccz4_name: str
          Name of the CCZ4 solver.

        @param euler_name: str
          Name of the Euler solver.

        """
        self.ccz4 = FOCCZ4Solver(ccz4_name)
        self.euler = Euler(euler_name)
        Mode = exahype2.api.AuxiliaryVariables.Mode
        self.euler_auxiliary_copies = exahype2.api.AuxiliaryVariables(
            self.EULER_AUXILIARY_COPIES_NAME,
            dict(Euler.Unknowns),
            mode=Mode.STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES,
            data_name="EulerFieldsForCCZ4",
        )
        self.ccz4_auxiliary_copies = exahype2.api.AuxiliaryVariables(
            self.CCZ4_AUXILIARY_COPIES_NAME,
            dict(FOCCZ4Solver.Unknowns),
            mode=Mode.STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES,
            data_name="CCZ4FieldsForEuler",
        )
        self.ccz4.add_auxiliary_variables(self.euler_auxiliary_copies)
        self.euler.add_auxiliary_variables(self.ccz4_auxiliary_copies)
        self._name = f"{ccz4_name}With{euler_name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def solvers(self):
        """ The solvers the project has to register; exahype2.api.Project.add_solver() asks for them """
        return [self.ccz4, self.euler]

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

        Realise both solvers, couple their time stepping graphs, and add the outcome to the project's graphs

        The arguments are those of exahype2.api.Solver.generate_solver_realisation().
        Both solvers are realised without global tasks. The load balancing and the dynamic
        mesh refinement wrap the coupled time step graph afterwards, through the scheme of CCZ4.

        """
        peano4.api.Logger().new_section("Host " + self.name)
        coupled_time_step = peano4.api.taskgraph.Graph()
        for solver in self.solvers:
            own_time_step = peano4.api.taskgraph.Graph()
            solver.generate_solver_realisation(
                exahype2_project,
                peano_project,
                setup_simulation,
                own_time_step,
                plot,
                mesh_adaptation,
                add_global_tasks=False,
            )
            coupled_time_step.merge(own_time_step)
        self._couple(exahype2_project, peano_project, coupled_time_step)

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

    def _couple(self, exahype2_project, peano_project, time_step):
        """!

        Add the preamble and the final task around the concatenated graphs

        @param exahype2_project: exahype2.api.Project
          The project that hosts the system.

        @param peano_project: peano4.api.Project
          The Peano project that the exahype2 project assembles.

        @param time_step: peano4.api.taskgraph.Graph
          The concatenation of the time step graphs of both solvers. It
          receives the new tasks. All its source tasks are first tasks of a
          solver, and all its sink tasks (but the global ones) are last tasks
          of a solver.

        """
        peano4.api.Logger().new_section(
            "Coupling " + self.ccz4.name + " and " + self.euler.name
        )
        Entity = peano4.api.taskgraph.Task.GridEntity

        ## Determine them before the prep tasks are added: those are sources, too
        first_tasks = time_step.source_tasks()
        last_tasks = [
            task for task in time_step.sink_tasks() if task.entity != Entity.GLOBAL
        ]

        ## Two prep tasks run in parallel. Each reads the persistent patch of
        ## one solver, writes the auxiliary variables of the other one, and
        ## projects them onto the faces, where the reconstruction of the halo
        ## finds them.
        preps = [
            self._create_prep_task(time_step, self.ccz4, self.euler, self.ccz4_auxiliary_copies),
            self._create_prep_task(time_step, self.euler, self.ccz4, self.euler_auxiliary_copies),
        ]
        ## The solvers start only once both prep tasks have finished
        for first_task in first_tasks:
            for prep in preps:
                first_task.add_dependency(prep)

        ## Synchronisation at the end
        synchronise = peano4.api.taskgraph.Task(
            name=self.SYNCHRONISE_TASK_NAME, entity=Entity.GLOBAL_END_OF_STEP
        )
        for task in last_tasks:
            synchronise.add_dependency(task)
        synchronise.add_action_set(SynchroniseTimeStepSizes([self.ccz4, self.euler]))
        time_step.add_task(synchronise)


    def _create_prep_task(
        self, time_step, source_solver, target_solver, auxiliary_copies
    ):
        """!

        Create the task that fills the auxiliary variables of one solver with the solution of the other one

        auxiliary copies (auxiliary_copies) of the target solver (copy, with
        auxiliary variables auxiliary_copies of the target solver (copy, with
        interpolation if the patches differ in size), projects them onto the
        faces, restricts them on hanging faces, and rolls the faces over. It is
        added to time_step, but no dependencies are set: the caller wires it.
        It is the one place that writes new face data.

        @param time_step: peano4.api.taskgraph.Graph
          Receives the task.

        @param source_solver: exahype2.api.Solver
          Provides the data.

        @param target_solver: exahype2.api.Solver
          Holds the auxiliary variables.

        @param auxiliary_copies: exahype2.api.AuxiliaryVariables
          The auxiliary variables of the target solver.

        @return peano4.api.taskgraph.Task
          The new task.

        """
        Access = peano4.api.taskgraph.Task.DataAccess
        Entity = peano4.api.taskgraph.Task.GridEntity
        actionsets = exahype2.api.rkfd4.actionsets
        includes = (
            target_solver._get_default_includes() + target_solver.user_solver_includes
        )
        prep = peano4.api.taskgraph.Task(
            name=self.PREP_TASK_NAME.format(source_solver.name),
            entity=Entity.CELL,
        )
        prep.use_cell(source_solver._patch, Access.READ_AND_WRITE)
        prep.use_cell(auxiliary_copies.patch, Access.READ_AND_WRITE)
        prep.use_face(auxiliary_copies.patch_overlap_update, Access.TEMPORARY)
        prep.use_face(auxiliary_copies.patch_overlap_current_time_step, Access.READ_AND_WRITE)
        prep.add_action_set(
            CopyPatchWithLinearInterpolation(
                source_patch=source_solver._patch,
                target_patch=auxiliary_copies.patch,
                guard=target_solver.grid_selector.createCellGuard(),
                additional_includes=includes,
            )
        )
        prep.add_action_set(
            actionsets.ProjectPatchOntoFaces(
                auxiliary_copies.patch,
                auxiliary_copies.patch_overlap_update,
                guard=target_solver.grid_selector.createCellGuard(),
                includes=includes,
            )
        )
        prep.add_action_set(
            toolbox.blockstructured.api.RestrictHangingFaces(
                auxiliary_copies.patch_overlap_update,
                scheme=target_solver.numerical_scheme.restriction,
                clear_guard=target_solver.grid_selector.createFaceGuard(),
                additional_includes=includes,
            )
        )
        prep.add_action_set(
            actionsets.RollOverUpdatedFace(
                auxiliary_copies.patch_overlap_current_time_step,
                auxiliary_copies.patch_overlap_update,
                guard=target_solver.grid_selector.createFaceGuard(),
                includes=includes,
            )
        )
        time_step.add_task(prep)
        return prep
