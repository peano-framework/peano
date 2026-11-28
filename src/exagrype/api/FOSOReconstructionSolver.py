""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api
import peano4.api
import peano4.api.datamodel
import peano4.api.taskgraph
import toolbox.blockstructured.api

from .actionsets import ApplyReconstructionFunctor


class FOSOReconstructionSolver(exahype2.api.Solver):
    """!

    Reconstructs the first derivatives which exagrype.api.FOSOCCZ4Solver needs

    The solver is the second partner in the ping-pong of the first-/second-order
    reconstructed CCZ4 formulation (@ref page_exagrype_fo_so_ccz4). Where the
    CCZ4 solver sees 25 unknowns and 33 auxiliary variables (the derivatives),
    the reconstruction solver sees the same 58 entries the other way round: its
    33 unknowns are the derivatives, and its 25 auxiliary variables are the
    unknowns of the CCZ4 solver. Nothing is copied: the derivatives are the
    very patch that the CCZ4 solver holds as auxiliary variables "Derivatives",
    and the reconstruction reads the CCZ4 solver's patch.

    It is a real exahype2.api.Solver (it has a solver class and a global
    instance in the solver repository), yet it is much simpler than a normal
    solver:

    - It has no PDE terms and no compute kernel. Its only compute step
      evaluates the stencil on the unknowns of the partner.
    - It computes no time step size. Its C++ class forwards the time stamps
      and time step sizes of the CCZ4 solver, so the global minimum over all
      solvers is the one of the CCZ4 solver.
    - It does neither add lopsided advection nor enforce constraints.
    - It has no grid construction of its own, no plotter and no mesh
      adaptation. The CCZ4 solver builds the mesh and initialises the
      derivatives.

    Users do not create the solver themselves: exagrype.api.FOSOSolver
    creates it and realises it right after the CCZ4 solver. The solver has no
    numerical scheme of its own: the scheme of the host builds the CCZ4 solver's
    graph and asks this solver for its graph per stage (create_task_graph()). The reconstruction
    has to be realised after the CCZ4 solver, as it uses its data.

    ## Task graph

    For each stage of the Runge-Kutta scheme of the CCZ4 solver (once for FV),
    the solver creates a chain of tasks in a task graph of its own:

    1. Reconstruct the primary variables and the derivative variables
       (halo from the cell and the faces). The two tasks run in parallel.
       Each first interpolates the coarse face data onto the hanging faces.
       From the point of view of this solver, the primary variables are
       auxiliary variables, and the derivative variables are the unknowns.
    2. Apply the stencil, i.e. recompute the derivatives from the primary variables.
    3. Project the derivatives onto the faces, restrict them from the
       hanging faces onto the coarse faces and roll the face data over. The
       restriction exists only if the derivatives' mode
       (@ref exahype2.api.AuxiliaryVariables.Mode) is CHANGING_ON_FACES,
       which it is.

    Interpolation and restriction are no tasks of their own, but ride along
    with the tasks which read and write the face data
    (src/exahype2/api/adrs/0001-interpolation-and-restriction-ride-along-with-face-users.md):
    Peano creates a hanging face before it enters the adjacent fine cell, and
    destroys it after it has left the cell. Within one task, the interpolation
    hence always precedes the reconstruction, and the restriction always
    follows the projection, whatever traversals the task graph compiler
    builds. The CCZ4 solver interpolates the faces of the unknowns once more
    in its own reconstruction. This second interpolation is required: Its
    reconstruction runs several mesh traversals later, when the hanging faces
    of the first one do not exist anymore.

    The roll over is part of the projection, too, as in
    src/exahype2/api/fv/adrs/0003-roll-over-is-part-of-the-projection.md:
    The update of the face data is temporary, i.e. it lives for one mesh
    traversal. Peano touches a face for the last time after both adjacent
    cells have projected onto it, and the roll over acts then. As a task of
    its own, it would copy the update only if the task graph compiler happens
    to fuse it into the traversal of the projection. The roll over of the
    derivatives reads neither the time stamp nor the time step size, i.e.
    src/exahype2/api/rkfd4/adrs/0001-time-step-size-after-roll-over.md does
    not constrain it. The CCZ4 solver's own roll over (of exahype2.api.rkfd4)
    is part of its projection, too
    (src/exahype2/api/rkfd4/adrs/0004-roll-over-is-part-of-the-projection.md).

    The graph has no dependencies between stages. The host concatenates the graph with the
    one of the CCZ4 solver (exagrype.api.FOSOSolver._couple()): The first tasks of
    a stage wait for what the CCZ4 solver's reconstructions of the unknowns and
    of the derivatives in this stage waited for (partner_task_names()). Both
    read the face data which the chain reads or writes, and each interpolates
    its face data onto the hanging faces itself.
    The chain is a block in front of the rest of the stage: Both
    reconstructions of the CCZ4 solver wait for the last task. In terms of
    @ref page_exahype_programmers_guide_coupling, this is a multiplicative
    coupling, with the coupling repeated per stage.

    ## Shortcomings

    - Only periodic domains are supported. The task graph is ready for
      Dirichlet boundary conditions (the reconstruction waits for the
      boundary task of the CCZ4 solver), but the generated code does not
      compile: The boundary routines of the schemes call
      dirichletBoundaryConditions() and the Sommerfeld routines without
      the auxiliary variables, which the solver's signatures expect.
      Furthermore, nobody sets the derivatives in the ghost grid cells
      outside of the domain.
    - Supported are exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
      (fourth-order stencil, the scheme's halo of two or more grid cells, one
      task per Runge-Kutta stage) and exahype2.api.fv.RusanovWithExplicitEuler
      (second-order central differences, overlap of one is enough, a single
      task as there is a single stage). The optimistic Finite Volume scheme is
      not supported.

    """

    class NoGridConstruction(exahype2.api.Solver.GridConstruction):
        """!

        The reconstruction solver builds no mesh

        The CCZ4 solver constructs the mesh, and initialises the derivatives
        which are the unknowns of the reconstruction solver.

        """

        def construct_output(self, exahype2_project, peano_project, setup_simulation):
            pass

    def __init__(self, name: str, ccz4: "FOSOCCZ4Solver"):
        """!

        @param name: str
          Name of the solver, @see exahype2.api.Solver.

        @param ccz4: exagrype.api.FOSOCCZ4Solver
          The CCZ4 solver whose derivatives this solver reconstructs.

        """
        if not hasattr(ccz4, "derivatives"):
            raise TypeError(
                f"The reconstruction serves a FOSOCCZ4Solver, but got {type(ccz4).__name__}"
            )
        super().__init__(name, dict(ccz4.DerivativeUnknowns))
        self.ccz4 = ccz4
        """! The partner whose derivatives we reconstruct """
        self.grid_selector = ccz4.grid_selector
        self._scratch_halos = None
        self.grid_construction = FOSOReconstructionSolver.NoGridConstruction(self)
        self.add_user_solver_includes('\n#include "repositories/SolverRepository.h"\n')
        self._forward_time_stamps_and_time_step_sizes()

    def _forward_time_stamps_and_time_step_sizes(self):
        """!

        The solver does not time step. It therefore hands out what the CCZ4
        solver hands out. Otherwise, the global minimum and maximum which the
        solver repository evaluates over all solvers would be spoilt.

        """
        for getter in [
            "getMinTimeStamp",
            "getMaxTimeStamp",
            "getMinTimeStepSize",
            "getMaxTimeStepSize",
            "getAdmissibleTimeStepSize",
        ]:
            self.add_abstract_solver_method(
                f"double {getter}() const override;",
                lambda qualified_class_name, getter=getter: (
                    f"double {qualified_class_name}::{getter}() const {{\n"
                    f"  return repositories::{self.ccz4.get_name_of_global_instance()}.{getter}();\n"
                    "}\n"
                ),
            )

    def _layout(self):
        """!

        Validate the scheme of the CCZ4 solver and return how to reconstruct

        @return scheme, actionsets, reconstruction_function, stages, per_stage

        """
        reconstruction = self
        ccz4 = reconstruction.ccz4
        name = reconstruction.name
        scheme = ccz4.numerical_scheme
        match scheme:
            case exahype2.api.rkfd4.CentralDifferencesWithRungeKutta():
                # The stencil reaches two layers out, i.e. any halo of the scheme
                # suffices. The derivatives of the halo come from the
                # neighbour via the faces.
                actionsets = exahype2.api.rkfd4.actionsets
                reconstruction_function = "reconstructFirstDerivativesFD4"
                stages = scheme.rk_order
                per_stage = True
            case exahype2.api.fv.RusanovWithExplicitEuler():
                # Second-order central differences reach one layer out, and the
                # derivatives of the halo come from the neighbour via the faces
                actionsets = exahype2.api.fv.actionsets
                reconstruction_function = (
                    "reconstructFirstDerivativesCentralDifferences"
                )
                stages = 1
                per_stage = False
            case _:
                raise TypeError(
                    f"Reconstruction {name}: solver {ccz4.name} has to use "
                    "exahype2.api.rkfd4.CentralDifferencesWithRungeKutta (or a subclass) or "
                    "exahype2.api.fv.RusanovWithExplicitEuler as numerical scheme, "
                    f"but uses {type(scheme).__name__}"
                )
        derivatives = ccz4.derivatives
        if derivatives.patch is None:
            raise ValueError(
                f"Reconstruction {name}: solver {ccz4.name} has not been realised yet. "
                "The CCZ4 solver has to be realised before its reconstruction."
            )
        return scheme, actionsets, reconstruction_function, stages, per_stage

    def partner_task_names(self, stage: int):
        """!

        Names of the tasks of the CCZ4 solver which the reconstruction of a stage interacts with

        These are the two tasks of the stage which read the face data that
        the reconstruction reads or writes. Each of them also interpolates its
        face data onto the hanging faces, as interpolation is no task of its
        own (src/exahype2/api/adrs/0001-interpolation-and-restriction-ride-along-with-face-users.md).

        @param stage: int
          Stage of the Runge-Kutta scheme, 0 for Finite Volumes.

        @return unknowns_reconstruction, derivatives_reconstruction
          Names of the tasks that reconstruct the halo of the unknowns and
          the one of the derivatives. The first reconstruction tasks of the
          stage wait for what these two wait for, and both wait for the last
          reconstruction task. @ref exagrype.api.FOSOSolver._couple()

        @todo The suffix " stage k" is a convention which this routine and
          exahype2.api.rkfd4.CentralDifferencesWithRungeKutta spell out
          independently. If the scheme changes it, the lookup in _couple()
          fails. The schemes should rather offer a routine which yields the
          name of the reconstruction task of stage k (the Finite Volumes
          without suffix), and this routine should call it.

        """
        ccz4 = self.ccz4
        suffix = f" stage {stage}" if self._layout()[4] else ""
        derivatives = exahype2.api.AuxiliaryVariables.RECONSTRUCTION_TASK_NAME.format(
            ccz4.derivatives.name
        )
        return (
            f"{ccz4.name}_{ccz4.numerical_scheme.RECONSTRUCTION_TASK_NAME}{suffix}",
            f"{ccz4.name}_{derivatives}{suffix}",
        )

    def _reconstruction_call(self, reconstruction_function: str, n: int, h: int) -> str:
        """!

        C++ which recomputes the derivatives of a cell from its unknowns

        The call hands the table of the CCZ4 solver
        (@ref exagrype.api.FOSOCCZ4Solver.FOSOCCZ4Solver.derivative_table()) to
        the C++ routine, which knows nothing about CCZ4 itself
        (src/exagrype/ccz4/adrs/0005-one-table-defines-the-auxiliary-derivatives.md).

        @param reconstruction_function: str
          exagrype::ccz4::reconstructFirstDerivativesFD4 or
          exagrype::ccz4::reconstructFirstDerivativesCentralDifferences.

        @param n: int
          Grid cells per axis of a patch.

        @param h: int
          Halo of the reconstructed unknowns.

        """
        ccz4 = self.ccz4
        unknown, axis, factor = ccz4.derivative_table()
        return f"""{{
    constexpr int    differentiatedUnknown[] = {{{", ".join(str(i) for i in unknown)}}};
    constexpr int    axis[]                  = {{{", ".join(str(i) for i in axis)}}};
    constexpr double factor[]                = {{{", ".join(repr(f) for f in factor)}}};
    ::exagrype::ccz4::{reconstruction_function}(
      oldQWithHalo, marker.h(), {n}, {h}, {ccz4.unknowns}, {len(unknown)},
      differentiatedUnknown, axis, factor, newQ
    );
  }}"""

    def create_task_graph(
        self, exahype2_project, peano_project, stage: int, number_of_stages: int
    ):
        """!

        Create the chain of reconstruction tasks of one stage

        The CCZ4 solver has to be realised before, as the routine needs its
        data. The graph holds one chain of tasks, without any dependency on the
        CCZ4 solver's tasks. Its sources are the two reconstruction tasks of
        the halos (each interpolates the hanging faces first), its only sink
        the projection (which restricts and rolls over). The host (exagrype.api.FOSOSolver) wires the chain
        into the stage of the CCZ4 solver.

        @param exahype2_project: exahype2.api.Project
          The project that hosts the solvers (unused).

        @param peano_project: peano4.api.Project
          The Peano project that the exahype2 project assembles.

        @param stage: int
          Stage of the Runge-Kutta scheme of the CCZ4 solver, 0 for Finite Volumes.

        @param number_of_stages: int
          Number of stages (unused).

        @return peano4.api.taskgraph.Graph

        """
        reconstruction = self
        ccz4 = self.ccz4
        name = self.name
        scheme, actionsets, reconstruction_function, stages, per_stage = self._layout()
        derivatives = ccz4.derivatives
        graph = peano4.api.taskgraph.Graph()
        Access = peano4.api.taskgraph.Task.DataAccess
        Entity = peano4.api.taskgraph.Task.GridEntity
        n = scheme.patch_size
        h = scheme.HALO_SIZE if per_stage else scheme.overlap
        cell_guard = ccz4.grid_selector.createCellGuard()
        face_guard = ccz4.grid_selector.createFaceGuard()
        includes = (
            ccz4._get_default_includes()
            + ccz4.user_solver_includes
            + '\n#include "exagrype/ccz4/SecondOrderAuxiliaryVariablesReconstruction.h"\n'
        )

        # The reconstructed primary variables and derivative variables.
        # Nobody besides the tasks of the stage reads them, i.e. they are
        # scratch arrays.
        def create_scratch_halo(unknowns, halo_name):
            halo = peano4.api.datamodel.Patch(
                (n + 2 * h, n + 2 * h, n + 2 * h), unknowns, halo_name
            )
            halo.grid_selector = ccz4.grid_selector
            halo.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
                halo, "double"
            )
            peano_project.datamodel.add_cell(halo)
            return halo

        # One pair for all stages
        if self._scratch_halos is None:
            self._scratch_halos = (
                create_scratch_halo(ccz4.unknowns, name + "PrimaryVariablesWithHalo"),
                create_scratch_halo(
                    derivatives.size, name + derivatives.data_name + "WithHalo"
                ),
            )
        primary_halo, derivative_halo = self._scratch_halos

        functor = self._reconstruction_call(reconstruction_function, n, h)

        stage_input = ccz4._patch if stage == 0 else ccz4._stage_input
        suffix = f" stage {stage}" if per_stage else ""

        def create_task(task_name):
            task = peano4.api.taskgraph.Task(
                name=f"{name}_{task_name}{suffix}", entity=Entity.CELL
            )
            graph.add_task(task)
            return task

        ## Reconstruct, both the primary variables (the auxiliary
        ## variables of this solver) and the derivative variables (its
        ## unknowns). The two are independent. The CCZ4 solver's
        ## reconstruction reads the same data, so it has to survive.
        reconstruct_primary = create_task("Reconstruct primary variables")
        reconstruct_primary.use_cell(stage_input, Access.READ_AND_WRITE)
        reconstruct_primary.use_cell(primary_halo, Access.WRITE)
        reconstruct_primary.use_face(
            ccz4._patch_overlap_current_time_step, Access.READ_AND_WRITE
        )
        reconstruct_primary.add_action_set(
            toolbox.blockstructured.api.InterpolateHangingFaces(
                ccz4._patch_overlap_current_time_step,
                scheme=scheme.interpolation,
                additional_includes=includes,
            )
        )
        reconstruct_primary.add_action_set(
            toolbox.blockstructured.api.ReconstructPatchWithHalo(
                patch=stage_input,
                patch_overlap=ccz4._patch_overlap_current_time_step,
                reconstructed_patch=primary_halo,
                guard=cell_guard,
                add_assertions_to_halo_exchange=False,
                additional_includes=includes,
            )
        )
        reconstruct_derivatives = create_task("Reconstruct derivative variables")
        reconstruct_derivatives.use_cell(derivatives.patch, Access.READ_AND_WRITE)
        reconstruct_derivatives.use_cell(derivative_halo, Access.WRITE)
        reconstruct_derivatives.use_face(
            derivatives.patch_overlap_current_time_step, Access.READ_AND_WRITE
        )
        reconstruct_derivatives.add_action_set(
            toolbox.blockstructured.api.InterpolateHangingFaces(
                derivatives.patch_overlap_current_time_step,
                scheme=scheme.interpolation,
                additional_includes=includes,
            )
        )
        reconstruct_derivatives.add_action_set(
            toolbox.blockstructured.api.ReconstructPatchWithHalo(
                patch=derivatives.patch,
                patch_overlap=derivatives.patch_overlap_current_time_step,
                reconstructed_patch=derivative_halo,
                guard=cell_guard,
                add_assertions_to_halo_exchange=False,
                additional_includes=includes,
            )
        )

        ## The stencil
        apply_stencil = create_task("Reconstruct derivatives")
        apply_stencil.use_cell(primary_halo, Access.READ_AND_DISCARD)
        apply_stencil.use_cell(derivative_halo, Access.READ_AND_DISCARD)
        apply_stencil.use_cell(derivatives.patch, Access.READ_AND_WRITE)
        apply_stencil.add_action_set(
            ApplyReconstructionFunctor(
                patch=derivatives.patch,
                primary_variables_with_halo=primary_halo,
                derivative_variables_with_halo=derivative_halo,
                functor_implementation=functor,
                guard=cell_guard,
                additional_includes=includes,
            )
        )
        apply_stencil.add_dependency(reconstruct_primary)
        apply_stencil.add_dependency(reconstruct_derivatives)

        ## Project the derivatives onto the faces
        project = create_task("Project derivative variables onto faces")
        project.use_cell(derivatives.patch, Access.READ_AND_WRITE)
        project.use_face(derivatives.patch_overlap_update, Access.TEMPORARY)
        ## The roll over writes the face data. The declaration also makes the
        ## compiler start a new mesh traversal here (it does not fuse tasks
        ## with conflicting face data).
        project.use_face(derivatives.patch_overlap_current_time_step, Access.WRITE)
        project.add_action_set(
            actionsets.ProjectPatchOntoFaces(
                derivatives.patch,
                derivatives.patch_overlap_update,
                guard=cell_guard,
                includes=includes,
            )
        )
        ## Restrict from the hanging faces onto the coarse faces, but only
        ## if the derivatives are allowed to change, i.e. in the sense of the
        ## auxiliary variables' mode CHANGING_ON_FACES. Peano destroys a
        ## hanging face after the fine cell has projected onto it.
        if derivatives.mode == exahype2.api.AuxiliaryVariables.Mode.CHANGING_ON_FACES:
            project.add_action_set(
                toolbox.blockstructured.api.RestrictHangingFaces(
                    derivatives.patch_overlap_update,
                    scheme=scheme.restriction,
                    clear_guard=face_guard,
                    additional_includes=includes,
                )
            )
        ## Roll over within the same task, as the update is temporary
        project.add_action_set(
            actionsets.RollOverUpdatedFace(
                derivatives.patch_overlap_current_time_step,
                derivatives.patch_overlap_update,
                guard=face_guard,
                includes=includes,
            )
        )
        project.add_dependency(apply_stencil)
        return graph

    def create_readme_descriptor(self, domain_offset, domain_size) -> str:
        return (
            f"Reconstruction {self.name}: reconstructs the {self.unknowns} first "
            f"derivatives of solver {self.ccz4.name} from its {self.ccz4.unknowns} unknowns"
        )
