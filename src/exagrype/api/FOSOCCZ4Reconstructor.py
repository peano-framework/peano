""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api
import peano4.api
import peano4.api.datamodel
import peano4.api.taskgraph
import toolbox.blockstructured.api


class FOSOCCZ4Reconstructor:
    """!

    Reconstructs the first derivatives which exagrype.api.FOSOCCZ4Solver needs

    The reconstructor is the second partner in the ping-pong of the
    first-/second-order reconstructed CCZ4 formulation. Where the solver
    sees 25 unknowns and 34 auxiliary variables (the derivatives), the
    reconstructor sees the same 59 entries the other way round: its 34
    unknowns are the derivatives, and its 25 auxiliary variables are the
    unknowns of the solver. The reconstructor evolves nothing. In each
    task graph it runs in front of each stage of the solver's Runge-Kutta
    scheme, and does the following:

    1. It builds the solver's input of the stage with its halo, exactly as the
       solver does later for its own right-hand side.
    2. It calls exagrype::ccz4::reconstructFirstDerivativesFD4(), which wraps
       the reconstruction of the legacy ExaGRyPE
       (exagrype::ccz4::recomputeAuxiliaryVariablesFD4_4thOrder()), per cell.
       The outcome are the 34 derivatives (K0 excepted) in the cell.
    3. It projects the derivatives onto the faces (the same mechanism as for
       the unknowns of any other solver), so the solver can build the
       derivatives with a halo.

    The solver then evaluates the stage on the unknowns of the stage and the
    freshly reconstructed derivatives.

    The reconstructor is not an exahype2.api.Solver: it has no unknowns,
    no numerical scheme and no PDE terms of its own, as it does not evolve
    anything. Its 34 "unknowns" are the auxiliary variables
    (exahype2.api.AuxiliaryVariables) which the solver owns. Users do not
    create it: exagrype.api.FOSOCCZ4Solver creates its reconstructor
    (exagrype.api.FOSOCCZ4Solver.reconstructor) and realises it right after
    itself. The project hence only sees the solver, and the solver's task
    graph and the reconstructor's task graph are one:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    solver = exagrype.api.FOSOCCZ4Solver("SO")
    solver.numerical_scheme = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(solver, patch_size, rk_order=4)
    ...
    project.add_solver(solver)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @ref page_exagrype_fo_so_project_plan

    ## Shortcomings

    - The reconstruction runs in front of each stage, i.e. the time
      stepping graph of the solver and the one of the reconstructor take turns.
    - Domain boundaries other than periodic ones and Sommerfeld boundary
      conditions are not supported, as the reconstruction reads the halo of the
      solver's unknowns before the Dirichlet boundary conditions
      are applied. The reconstructor takes the same dependencies as the
      solver's reconstruction, which makes it work in principle, but this has
      not been tested.
    - Supported are exahype2.api.rkfd4.CentralDifferencesWithRungeKutta
      (fourth-order stencil, halo of at least three grid cells, one task per
      Runge-Kutta stage) and exahype2.api.fv.RusanovWithExplicitEuler
      (second-order central differences, overlap of one is enough, a single
      task as there is a single stage). The optimistic Finite Volume scheme is
      not supported.

    """

    def __init__(self, solver: "FOSOCCZ4Solver", name: str = None):
        """!

        @param solver: exagrype.api.FOSOCCZ4Solver
          The solver whose derivatives this object reconstructs.

        @param name: str
          Name of the reconstructor, which prefixes the names of its tasks
          and of its data. Defaults to the solver's name plus "Reconstructor".

        """
        if not hasattr(solver, "derivatives"):
            raise TypeError(
                f"The reconstructor serves a FOSOCCZ4Solver, but got {type(solver).__name__}"
            )
        self._solver = solver
        self._name = name if name is not None else solver.name + "Reconstructor"

    @property
    def name(self) -> str:
        return self._name

    @property
    def solver(self) -> "FOSOCCZ4Solver":
        return self._solver

    @property
    def unknowns(self) -> int:
        """! The reconstructor's unknowns, i.e. the derivatives of the solver."""
        return self._solver.derivatives.size

    @property
    def number_of_auxiliary_variables(self) -> int:
        """! The reconstructor's auxiliary variables, i.e. the unknowns of the solver."""
        return self._solver.unknowns

    def create_readme_descriptor(self, domain_offset, domain_size) -> str:
        return (
            f"Reconstructor {self._name}: reconstructs the {self.unknowns} first "
            f"derivatives of solver {self._solver.name} from its {self._solver.unknowns} unknowns"
        )

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

        Insert the reconstruction in front of each stage of the solver

        The routine is called by exagrype.api.FOSOCCZ4Solver.generate_solver_realisation()
        after the solver has been realised.

        For each stage of the Runge-Kutta scheme, the routine creates one task.
        The task depends on whatever the solver's reconstruction of this stage
        depends on, and the solver's reconstruction depends on the new task.
        Within the task, we visit each cell once, build the stage's
        input with its halo, reconstruct the derivatives of the cell and
        project them onto the faces.

        """
        peano4.api.Logger().new_section("Reconstructor " + self._name)
        solver = self._solver
        scheme = solver.numerical_scheme
        match scheme:
            case exahype2.api.rkfd4.CentralDifferencesWithRungeKutta():
                if scheme.HALO_SIZE < 3:
                    raise ValueError(
                        f"Reconstructor {self._name}: the halo of the scheme of {solver.name} has to be "
                        f"at least three, but is {scheme.HALO_SIZE}. The solver evaluates fluxes and "
                        "non-conservative products on the faces of the cell, i.e. it needs the "
                        "derivatives one layer outside of the cell, and the fourth-order stencil reaches "
                        "two further layers. Use a scheme with lopsided advection or "
                        "exagrype.api.FD4WithKODissipation."
                    )
                actionsets = exahype2.api.rkfd4.actionsets
                reconstruction_function = "reconstructFirstDerivativesFD4"
                stages = scheme.rk_order
                per_stage = True
            case exahype2.api.fv.RusanovWithExplicitEuler():
                # Second-order central differences reach one layer out, and the
                # derivatives of the halo come from the neighbour via the faces
                actionsets = exahype2.api.fv.actionsets
                reconstruction_function = "reconstructFirstDerivativesCentralDifferences"
                stages = 1
                per_stage = False
            case _:
                raise TypeError(
                    f"Reconstructor {self._name}: solver {solver.name} has to use "
                    "exahype2.api.rkfd4.CentralDifferencesWithRungeKutta (or a subclass) or "
                    "exahype2.api.fv.RusanovWithExplicitEuler as numerical scheme, "
                    f"but uses {type(scheme).__name__}"
                )
        derivatives = solver.derivatives
        if derivatives.patch is None:
            raise ValueError(
                f"Reconstructor {self._name}: solver {solver.name} has not been realised yet. "
                "The solver has to be realised before its reconstructor."
            )

        Access = peano4.api.taskgraph.Task.DataAccess
        Entity = peano4.api.taskgraph.Task.GridEntity
        n = scheme.patch_size
        h = scheme.HALO_SIZE if per_stage else scheme.overlap
        cell_guard = solver.grid_selector.createCellGuard()
        face_guard = solver.grid_selector.createFaceGuard()
        includes = (
            solver._get_default_includes()
            + solver.user_solver_includes
            + '\n#include "exagrype/ccz4/SecondOrderAuxiliaryVariablesReconstruction.h"\n'
        )

        # The input of a stage with halo. Nobody besides the reconstruction
        # task reads it, i.e. it is a scratch array.
        halo = peano4.api.datamodel.Patch(
            (n + 2 * h, n + 2 * h, n + 2 * h),
            solver.unknowns,
            self._name + "PrimaryVariablesWithHalo",
        )
        halo.grid_selector = solver.grid_selector
        halo.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
            halo, "double"
        )
        peano_project.datamodel.add_cell(halo)

        functor = (
            f"::exagrype::ccz4::{reconstruction_function}("
            f"oldQWithHalo, marker.h(), {n}, {h}, "
            f"fineGridCell{derivatives.patch.name}.value);"
        )

        ## Stages is the Runge-Kutta stages or 1 for FV
        for stage in range(stages):
            stage_input = solver._patch if stage == 0 else solver._stage_input
            suffix = f" stage {stage}" if per_stage else ""
            solver_reconstruction = [
                task
                for task in time_step.tasks
                if task.name == f"{solver.name}_Reconstruct{suffix}"
            ]
            if len(solver_reconstruction) != 1:
                raise ValueError(
                    f"Reconstructor {self._name}: cannot find the reconstruction{suffix} of solver {solver.name} in the time stepping task graph"
                )
            solver_reconstruction = solver_reconstruction[0]

            task = peano4.api.taskgraph.Task(
                name=f"{self._name}_Reconstruct derivatives" + (f" for stage {stage}" if per_stage else ""),
                entity=Entity.CELL,
            )
            # The solver's reconstruction reads the same data, so it has to survive.
            task.use_cell(stage_input, Access.READ_AND_WRITE)
            task.use_face(
                solver._patch_overlap_current_time_step, Access.READ_AND_WRITE
            )
            task.use_cell(halo, Access.TEMPORARY)
            task.use_cell(derivatives.patch, Access.READ_AND_WRITE)
            task.use_face(derivatives.patch_overlap_update, Access.TEMPORARY)
            # Loaded and stored, so Peano's stacks hold exactly one entry per face
            # when the next task (the solver's reconstruction) loads it again
            task.use_face(
                derivatives.patch_overlap_current_time_step, Access.READ_AND_WRITE
            )
            task.add_action_set(
                toolbox.blockstructured.api.ReconstructPatchAndApplyFunctor(
                    patch=stage_input,
                    patch_overlap=solver._patch_overlap_current_time_step,
                    reconstructed_patch=halo,
                    functor_implementation=functor,
                    guard=cell_guard,
                    add_assertions_to_halo_exchange=False,
                    additional_includes=includes,
                )
            )
            task.add_action_set(
                actionsets.ProjectPatchOntoFaces(
                    derivatives.patch,
                    derivatives.patch_overlap_update,
                    guard=cell_guard,
                    includes=includes,
                )
            )
            task.add_action_set(
                actionsets.RollOverUpdatedFace(
                    derivatives.patch_overlap_current_time_step,
                    derivatives.patch_overlap_update,
                    guard=face_guard,
                    includes=includes,
                )
            )

            # Take over what the solver waits for, then let the solver wait for us
            for dependency in solver_reconstruction.dependencies:
                task.add_dependency(dependency)
            time_step.add_task(task)
            solver_reconstruction.add_dependency(task)

        title = f"Time stepping task graph for {solver.name} with reconstructor {self._name}"
        if peano4.api.Logger().is_visualisation:
            time_step.visualise(title=title)
        if peano4.api.Logger().is_text:
            time_step.log_as_text(title=title)
