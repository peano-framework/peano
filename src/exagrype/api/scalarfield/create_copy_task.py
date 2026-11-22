""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api
import peano4.api.taskgraph
import toolbox.blockstructured.api

from ..actionsets import CopyPatchWithLinearInterpolation


def create_copy_task(name, source_patch, target_solver, auxiliary_copies):
    """!

    Create the task that fills the auxiliary variables of a solver with data of another solver

    The task copies source_patch into the auxiliary variables auxiliary_copies of
    target_solver (interpolating piecewise linearly if the patches differ in
    size), projects them onto the faces, restricts them on hanging faces, and
    rolls the faces over. The solver then reconstructs the halo of the
    auxiliary variables from the faces. The task is the one place that writes
    new face data of the auxiliary variables. It keeps the source patch alive,
    i.e. the solver that owns it can still read it afterwards. The caller adds the
    task to a graph and wires it.

    @param name: str
      Name of the task.

    @param source_patch: peano4.api.datamodel.Patch
      The data to copy, i.e. the persistent patch of the producing solver,
      or its input of a Runge-Kutta stage.

    @param target_solver: exahype2.api.Solver
      Holds the auxiliary variables. It needs a numerical scheme already.

    @param auxiliary_copies: exahype2.api.AuxiliaryVariables
      The auxiliary variables of the target solver.

    @return peano4.api.taskgraph.Task

    """
    Access = peano4.api.taskgraph.Task.DataAccess
    actionsets = exahype2.api.rkfd4.actionsets
    includes = target_solver._get_default_includes() + target_solver.user_solver_includes
    cell_guard = target_solver.grid_selector.createCellGuard()
    face_guard = target_solver.grid_selector.createFaceGuard()

    task = peano4.api.taskgraph.Task(
        name=name, entity=peano4.api.taskgraph.Task.GridEntity.CELL
    )
    task.use_cell(source_patch, Access.READ_AND_WRITE)
    task.use_cell(auxiliary_copies.patch, Access.READ_AND_WRITE)
    task.use_face(auxiliary_copies.patch_overlap_update, Access.TEMPORARY)
    task.use_face(auxiliary_copies.patch_overlap_current_time_step, Access.READ_AND_WRITE)
    task.add_action_set(
        CopyPatchWithLinearInterpolation(
            source_patch=source_patch,
            target_patch=auxiliary_copies.patch,
            guard=cell_guard,
            additional_includes=includes,
        )
    )
    task.add_action_set(
        actionsets.ProjectPatchOntoFaces(
            auxiliary_copies.patch,
            auxiliary_copies.patch_overlap_update,
            guard=cell_guard,
            includes=includes,
        )
    )
    task.add_action_set(
        toolbox.blockstructured.api.RestrictHangingFaces(
            auxiliary_copies.patch_overlap_update,
            scheme=target_solver.numerical_scheme.restriction,
            clear_guard=face_guard,
            additional_includes=includes,
        )
    )
    task.add_action_set(
        actionsets.RollOverUpdatedFace(
            auxiliary_copies.patch_overlap_current_time_step,
            auxiliary_copies.patch_overlap_update,
            guard=face_guard,
            includes=includes,
        )
    )
    return task
