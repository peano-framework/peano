"""!

Assembly of the CCZ4 solver and the Euler solver into one coupled system

All ingredients live in exagrype.api (exagrype.api.FOCCZ4Solver and
exagrype.api.euler.Euler). This file assembles them in a host
(@ref exahype2.api.SolverHost): Each solver builds a task graph of its own and
reports it, and the host concatenates the graphs, inserts the KickOff and WrapUp
tasks, and reports the outcome. The coupling is additive and
volumetric, with a global synchronisation at the end of each Runge-Kutta
stage (@ref page_exahype_programmers_guide_coupling):

- Both solvers evolve in parallel, within the same mesh sweeps. The
  host wraps the concatenated graph into the load balancing switch and the
  dynamic mesh refinement through the CCZ4 solver
  (global_tasks_solver, see @ref exahype2.api.SolverHost), i.e. these exist once.
- The Euler solver holds a snapshot of the lapse of CCZ4 as auxiliary
  variable "Spacetime". A KickOff task in front of each Runge-Kutta stage
  fills it from the stage's input of CCZ4. Euler hence never reads data which
  CCZ4 writes at the same time, i.e. there is no data race.
- A WrapUp task after each stage waits for the end of the stage of both
  solvers and calls the functor synchronise() of the generated abstract Euler
  solver class. It only acts in unprepareTraversal(), i.e. it needs no mesh
  traversal: The WrapUp tasks of the stages are GLOBAL tasks that the task
  graph compiler appends to the traversal of the stage, and the very last one
  is a GLOBAL_END_OF_STEP task, which runs after everything else, notably after
  the solvers have computed their next time step sizes
  (@ref peano4.api.taskgraph.Task.Task.GridEntity).

The coupling is uni-directional: CCZ4 does not see the fluid.

"""

import exahype2.api
import exagrype.api
import peano4.api
import peano4.api.taskgraph
import peano4.api.datamodel
import peano4.api.solversteps
import toolbox.blockstructured.api


class EulerOnSpacetime(exagrype.api.euler.Euler):
    """!

    Euler solver on a spacetime which is given by a snapshot of the lapse

    The fluxes and the maximum eigenvalue are multiplied by the lapse, i.e.
    the system is the flat-space Euler system with a clock that runs with the
    lapse. This is a toy coupling, not the equations of general relativistic
    hydrodynamics. The lapse is the auxiliary variable "Spacetime" and starts
    with the value 1 until the first KickOff task has overwritten it.

    """

    def __init__(self, name: str, gamma: float = 1.4):
        """!

        @param name: str
          Name of the solver, @ref exahype2.api.Solver.

        @param gamma: float
          Adiabatic index of the gas.

        """
        # Before the superclass, as its hooks refer to the set
        self.spacetime = exahype2.api.AuxiliaryVariables("Spacetime", {"lapse": 1})
        wave = exagrype.api.euler.initialvalues.smooth_wave()
        super().__init__(
            name,
            gamma=gamma,
            initial_condition=exahype2.api.InitPointWisely(
                implementation=wave.implementation + f"\n  {self._lapse()} = 1.0;\n"
            ),
        )
        # The Euler solver demands the CCZ4 fields as auxiliary variables.
        # This host fills the lapse only, i.e. the set is stationary and unused.
        self.ccz4_fields = exahype2.api.AuxiliaryVariables(
            "CCZ4Fields",
            dict(exagrype.api.FOCCZ4Solver.Unknowns),
            mode=exahype2.api.AuxiliaryVariables.Mode.STATIONARY_ON_FACES,
        )
        self.add_auxiliary_variables(self.spacetime)
        self.add_auxiliary_variables(self.ccz4_fields)

    def _lapse(self) -> str:
        """! C++ expression of the lapse in a PDE term"""
        return f"{self.spacetime.argument_name}[0]"

    def _flux_implementation(self) -> str:
        return (
            super()._flux_implementation()
            + f"""
  for (int i=0; i<5; i++) F[i] *= {self._lapse()};
"""
        )

    def _max_eigenvalue_implementation(self) -> str:
        return (
            super()._max_eigenvalue_implementation()
            + f"""
  *lambdaMax *= {self._lapse()};
"""
        )


class WrapUp(peano4.api.solversteps.ActionSet):
    """!

    Global synchronisation of the coupled solvers

    The action set does nothing but call the functor synchronise() of the
    solver after the traversal. It is the very place where the solvers can
    exchange global information, e.g. agree on a common time step size.

    """

    def __init__(self, solver):
        """!

        @param solver: exahype2.api.Solver
          The solver whose synchronise() functor the action set calls.

        """
        super().__init__(descend_invocation_order=1, parallel=False)
        self._solver = solver

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_") + "_WrapUp"

    def user_should_modify_template(self):
        return False

    def get_includes(self):
        return '#include "repositories/SolverRepository.h"\n'

    def get_body_of_unprepareTraversal(self):
        return f"""
  repositories::{self._solver.get_name_of_global_instance()}.synchronise();
"""


class CCZ4WithEuler(exahype2.api.SolverHost):
    """!

    CCZ4 and Euler as one coupled system

    The object hosts one exagrype.api.FOCCZ4Solver (ccz4) and one
    EulerOnSpacetime (euler). The host holds both numerical schemes
    (numerical_scheme for ccz4, set_numerical_scheme() for euler); configure
    grid construction and plotter of the solvers, and then add the host to the project with
    exahype2.api.Project.add_solver(). Both solvers have
    to use exahype2.api.rkfd4.CentralDifferencesWithRungeKutta with the same
    patch size and Runge-Kutta order. The halos may differ.

    The generated abstract Euler solver class gets a functor synchronise()
    which the WrapUp tasks call. It does nothing as long as you do not
    supply a functor through setSynchronise(), e.g. in the main routine, to
    exchange global information between the two solvers.

    """

    def __init__(self, ccz4_name: str = "CCZ4", euler_name: str = "Euler"):
        """!

        @param ccz4_name: str
          Name of the CCZ4 solver.

        @param euler_name: str
          Name of the Euler solver.

        """
        self.ccz4 = exagrype.api.FOCCZ4Solver(
            ccz4_name, initial_condition=exagrype.api.initialvalues.gauge_wave()
        )
        self.euler = EulerOnSpacetime(euler_name)
        # CCZ4 first, as it owns the global tasks and Euler reads its data
        super().__init__(f"{ccz4_name}With{euler_name}", [self.ccz4, self.euler])
        self.euler.add_abstract_solver_attribute(
            "std::function<void()> _synchroniseFunctor = nullptr;"
        )
        self.euler.add_abstract_solver_method(
            "void setSynchronise(std::function<void()> functor);",
            lambda qualified_class_name: "void "
            + qualified_class_name
            + "::setSynchronise(std::function<void()> functor) {\n"
            "  _synchroniseFunctor = functor;\n"
            "}\n",
        )
        self.euler.add_abstract_solver_method(
            "void synchronise();",
            lambda qualified_class_name: "void "
            + qualified_class_name
            + "::synchronise() {\n"
            "  if (_synchroniseFunctor) {\n"
            "    _synchroniseFunctor();\n"
            "  }\n"
            "}\n",
        )

    def couple(self, exahype2_project, peano_project, own_graphs, host_graphs):
        """!

        Insert KickOff and WrapUp tasks around each Runge-Kutta stage

        Called once the graphs of the two solvers have been concatenated into
        the time step graph of the host. Per stage k, we build

            end of stage k-1 -> KickOff(k) -> reconstructions of CCZ4 and Euler of stage k

        with the end of stage k-1 being whatever the two reconstructions
        waited for, i.e. the end of the previous stage of both solvers. A
        WrapUp(k-1) task waits for the same tasks. No task depends on it, as
        it acts in unprepareTraversal() only. The last WrapUp is a
        GLOBAL_END_OF_STEP task which waits for all tasks that nothing else
        depends on, i.e. it also runs after the computation of the time
        step sizes.

        @param exahype2_project: exahype2.api.Project
          The project that hosts the system.

        @param peano_project: peano4.api.Project
          The Peano project that the exahype2 project assembles.

        @param own_graphs: Dict[exahype2.api.Solver, exahype2.api.PhaseGraphs]
          The graphs of the two solvers (unused).

        @param host_graphs: exahype2.api.PhaseGraphs
          The concatenated graphs. The time step graph receives the new tasks.

        """
        time_step = host_graphs.time_step
        peano4.api.Logger().new_section(
            "Coupling " + self.ccz4.name + " and " + self.euler.name
        )
        ccz4, euler = self.ccz4, self.euler
        scheme = ccz4.numerical_scheme
        euler_scheme = euler.numerical_scheme
        for candidate in (scheme, euler_scheme):
            if not isinstance(
                candidate, exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
            ):
                raise TypeError(
                    "Both solvers have to use exahype2.api.rkfd4.CentralDifferencesWithRungeKutta"
                )
        if (
            scheme.patch_size != euler_scheme.patch_size
            or scheme.rk_order != euler_scheme.rk_order
        ):
            raise ValueError(
                "Both solvers need the same patch size and Runge-Kutta order"
            )

        Access = peano4.api.taskgraph.Task.DataAccess
        Entity = peano4.api.taskgraph.Task.GridEntity
        n = scheme.patch_size
        h = scheme.HALO_SIZE
        spacetime = euler.spacetime
        lapse = ccz4.variable_names_per_entry().index("lapse")
        cell_guard = ccz4.grid_selector.createCellGuard()
        face_guard = ccz4.grid_selector.createFaceGuard()
        actionsets = exahype2.api.rkfd4.actionsets
        includes = (
            ccz4._get_default_includes()
            + ccz4.user_solver_includes
            + '\n#include "exahype2/enumerator/AoSLexicographicEnumerator.h"\n'
        )

        # Scratch array: CCZ4's stage input with halo. Nobody besides KickOff reads it
        halo = peano4.api.datamodel.Patch(
            (n + 2 * h, n + 2 * h, n + 2 * h), ccz4.unknowns, "KickOffCCZ4WithHalo"
        )
        halo.grid_selector = ccz4.grid_selector
        halo.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
            halo, "double"
        )
        peano_project.datamodel.add_cell(halo)

        functor = f"""
  const ::exahype2::enumerator::AoSLexicographicEnumerator enumerator(1,{n},{h},{ccz4.unknowns},0);
  for (int x=0; x<{n}; x++)
  for (int y=0; y<{n}; y++)
  for (int z=0; z<{n}; z++) {{
    tarch::la::Vector<Dimensions,int> volume;
    volume(0) = x;
    volume(1) = y;
    volume(2) = z;
    fineGridCell{spacetime.patch.name}.value[ x + {n}*y + {n*n}*z ] = oldQWithHalo[ enumerator(0, volume, {lapse}) ];
  }}
"""

        def find(solver, stage):
            tasks = [
                task
                for task in time_step.tasks
                if task.name == f"{solver.name}_Reconstruct stage {stage}"
            ]
            if len(tasks) != 1:
                raise ValueError(
                    f"Cannot find the reconstruction of stage {stage} of {solver.name}"
                )
            return tasks[0]

        for stage in range(scheme.rk_order):
            stage_input = ccz4._patch if stage == 0 else ccz4._stage_input
            ccz4_reconstruction = find(ccz4, stage)
            euler_reconstruction = find(euler, stage)

            kick_off = peano4.api.taskgraph.Task(
                name=f"KickOff stage {stage}", entity=Entity.CELL
            )
            kick_off.use_cell(stage_input, Access.READ_AND_WRITE)
            kick_off.use_face(
                ccz4._patch_overlap_current_time_step, Access.READ_AND_WRITE
            )
            kick_off.use_cell(halo, Access.TEMPORARY)
            kick_off.use_cell(spacetime.patch, Access.READ_AND_WRITE)
            kick_off.use_face(spacetime.patch_overlap_update, Access.TEMPORARY)
            kick_off.use_face(
                spacetime.patch_overlap_current_time_step, Access.READ_AND_WRITE
            )
            kick_off.add_action_set(
                toolbox.blockstructured.api.ReconstructPatchAndApplyFunctor(
                    patch=stage_input,
                    patch_overlap=ccz4._patch_overlap_current_time_step,
                    reconstructed_patch=halo,
                    functor_implementation=functor,
                    guard=cell_guard,
                    add_assertions_to_halo_exchange=False,
                    additional_includes=includes,
                )
            )
            kick_off.add_action_set(
                actionsets.ProjectPatchOntoFaces(
                    spacetime.patch,
                    spacetime.patch_overlap_update,
                    guard=cell_guard,
                    includes=includes,
                )
            )
            kick_off.add_action_set(
                actionsets.RollOverUpdatedFace(
                    spacetime.patch_overlap_current_time_step,
                    spacetime.patch_overlap_update,
                    guard=face_guard,
                    includes=includes,
                )
            )

            # The end of the previous stage of both solvers
            predecessors = []
            for task in (
                ccz4_reconstruction.dependencies + euler_reconstruction.dependencies
            ):
                if task not in predecessors:
                    predecessors.append(task)
            for task in predecessors:
                kick_off.add_dependency(task)
            if stage > 0:
                time_step.add_task(
                    self.__create_wrap_up(
                        f"WrapUp stage {stage - 1}",
                        predecessors,
                        peano4.api.taskgraph.Task.GridEntity.GLOBAL,
                    )
                )
            time_step.add_task(kick_off)
            ccz4_reconstruction.dependencies = [kick_off]
            euler_reconstruction.dependencies = [kick_off]

        Entity = peano4.api.taskgraph.Task.GridEntity
        # The WrapUp tasks of the stages are global, and nothing may depend on them
        sinks = [
            task for task in time_step.sink_tasks() if task.entity != Entity.GLOBAL
        ]
        time_step.add_task(
            self.__create_wrap_up(
                f"WrapUp stage {scheme.rk_order - 1}",
                sinks,
                Entity.GLOBAL_END_OF_STEP,
            )
        )

    def __create_wrap_up(self, name, predecessors, entity):
        """!

        @param name: str
          Name of the task.

        @param predecessors: List[peano4.api.taskgraph.Task]
          The tasks that have to finish before the task.

        @param entity: peano4.api.taskgraph.Task.GridEntity
          A global kind: The task uses no data and only acts in
          unprepareTraversal().

        """
        task = peano4.api.taskgraph.Task(name=name, entity=entity)
        for predecessor in predecessors:
            task.add_dependency(predecessor)
        task.add_action_set(WrapUp(self.euler))
        return task
