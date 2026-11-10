# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import enum
import os
import re

import peano4.api
import peano4.api.dastgen2
import peano4.api.datamodel
import peano4.api.taskgraph
import toolbox.blockstructured.api
import toolbox.loadbalancing.api.actionsets
import exahype2.api.dslhype as dslhype

from ..MaxEigenvalue import MaxEigenvalue
from ..Solver import Solver
from ..AuxiliaryVariables import (
    register_auxiliary_variables,
    add_reconstruction_tasks_of_auxiliary_variables,
    use_halos_of_auxiliary_variables,
    forwarding_arguments,
)
from ..rungekutta import ButcherTableau
from ..rungekutta.LinearCombination import generate_linear_combination_kernel
from ..rungekutta.LinearCombination import include_implementation_into_header
from . import FD4Kernels
from .Quantities import resolve_quantities
from .actionsets import AddLopsidedAdvection
from .actionsets import ApplyDirichletBoundaryConditions
from ..blockstructured.BoundaryConditions import get_access_to_persistent_face_data
from ..blockstructured.BoundaryConditions import get_boundary_conditions
from ..blockstructured.SommerfeldBoundaryConditions import SommerfeldBoundaryConditions
from ..blockstructured.create_dynamic_mesh_refinement_action_set import (
    create_dynamic_mesh_refinement_action_set,
)
from .actionsets import ComputeLinearCombination
from .actionsets import ComputeMaxEigenvalue
from .actionsets import ComputeTimeStepSize
from .actionsets import EvaluateRungeKuttaRightHandSide
from .actionsets import ProjectPatchOntoFaces
from .actionsets import ReduceMaxEigenvalue
from .actionsets import RollOverUpdatedFaceWithSommerfeldBoundaryConditions


def _access_within_chain(
    access: "peano4.api.taskgraph.Task.DataAccess", position: int, length: int
) -> "peano4.api.taskgraph.Task.DataAccess":
    """!

    Qualifier of one task within a chain of tasks that share cell data

    The original step used the data with the qualifier access. If we split it
    into a chain of length tasks, the first task keeps the load part of access,
    the last task keeps the store part, and all tasks in between both load and
    store the data. The fused traversal merges the qualifiers back into access
    (@ref peano4.api.taskgraph.compiler.FuseSteps), while a compiler that puts
    the tasks into different traversals still hands the data on correctly.
    Temporary data stay temporary: such data rely on the fusion anyway.

    Do not use this for face data that several tasks of a chain access: A
    writing task prevents the compiler from fusing them with the other tasks
    that use the face data.

    @param access: peano4.api.taskgraph.Task.DataAccess
      Qualifier of the original, unsplit step.

    @param position: int
      Position of the task within the chain, starting with zero.

    @param length: int
      Number of tasks in the chain.

    """
    Access = peano4.api.taskgraph.Task.DataAccess
    if access == Access.TEMPORARY:
        return Access.TEMPORARY
    load = access in (Access.READ_AND_DISCARD, Access.READ_AND_WRITE) or position > 0
    store = access in (Access.WRITE, Access.READ_AND_WRITE) or position < length - 1
    if load and store:
        return Access.READ_AND_WRITE
    if load:
        return Access.READ_AND_DISCARD
    if store:
        return Access.WRITE
    return Access.TEMPORARY


def _readable_name(class_name: str) -> str:
    """! Turn a class name such as AddLopsidedAdvection into the task name "Add lopsided advection" """
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", class_name).split()
    return " ".join(words[:1] + [word.lower() for word in words[1:]])


class CentralDifferencesWithRungeKutta(Solver.NumericalScheme):
    """!

    Fourth-order central Finite Differences with explicit Runge-Kutta time
    stepping, realised over a block-structured mesh

    Each mesh cell hosts a patch of patch_size^d grid cells. The unknowns
    sit in the grid cells' centres. The scheme evaluates the right-hand side
    of

    @f$ \\partial _t Q = F(Q) @f$

    through fourth-order central differences (@ref
    exahype2.api.rkfd4.FD4Kernels), and integrates in time with the explicit
    Runge-Kutta scheme from @ref exahype2.api.rungekutta.ButcherTableau. The
    stencils reach two grid cells into the neighbouring patches, i.e. the
    halo is HALO_SIZE=2 grid cells wide:

    @image html central-differences-with-runge-kutta-stencil.png

    The script central-differences-with-runge-kutta-illustrations.py next to
    this file generates the illustrations of this page.

    ## Usage

    The scheme plugs into an exahype2.api.Solver like the Finite Volumes of
    exahype2.api.fv, but with the grid construction and the plotter of
    exahype2.api.rkfd4:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    solver.numerical_scheme  = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(solver, patch_size=9, rk_order=4)
    solver.grid_construction = exahype2.api.rkfd4.SetupSimulationOverRegularGrid(solver, volume_size=0.01)
    solver.plotter           = exahype2.api.rkfd4.PlotAsVTKFile(solver, time_in_between_plots=0.1)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    The solver can have a Flux, a NonconservativeProduct and a
    VolumetricSource. It needs an InitPointWisely term, a MaxEigenvalue term
    unless the time step size is fixed, and either a
    DirichletBoundaryConditions or a SommerfeldBoundaryConditions term
    unless all axes are periodic (@ref
    exahype2.api.blockstructured.BoundaryConditions.get_boundary_conditions()). Sommerfeld
    conditions also require a MaxEigenvalue term, and a Butcher tableau whose
    stage time stamps do not decrease. @ref
    benchmarks_exahype2_euler_smooth_fd4 demonstrates fourth-order
    convergence for the Euler equations.

    ## Extending the scheme

    The scheme has no artificial dissipation. Subclasses can add further
    terms to the right-hand side:

    - _right_hand_side_action_sets() returns the action sets of the task
      which evaluates the right-hand side of one stage. Subclasses can
      append action sets which add their terms to the stage's estimate.
      They run after the right-hand side kernel, within the same task.
    - Subclasses which need wider stencils override HALO_SIZE.

    @ref exagrype.api.FD4WithKODissipation adds the Kreiss-Oliger
    dissipation this way.

    ## Right-hand side kernels

    The constructor's argument right_hand_side_kernel selects how the
    scheme evaluates the right-hand side
    (CentralDifferencesWithRungeKutta.RightHandSideKernel):

    - CentralDifferences: all derivatives through fourth-order central
      differences (@ref exahype2.api.rkfd4.FD4Kernels). This is the
      default.
    - CentralDifferencesWithLopsidedAdvection: the central differences
      plus advection terms @f$ v_d \\partial _d Q_q @f$, which a separate
      kernel discretises through lopsided (upwind-biased) fourth-order
      stencils (@ref exahype2::fd::addLopsidedAdvection()). The kernel
      runs after the central differences within each Runge-Kutta stage.
      You specify the velocity v, which has to be part of Q, and the
      advected quantities q. The PDE terms of the solver must not contain
      these advection terms themselves. As the lopsided stencils reach
      three grid cells into the neighbouring patches, the halo is three
      grid cells wide.

    ## Data flow

    __create_data_structures() documents the data per mesh cell and
    face. Per time step, the scheme runs through the Runge-Kutta stages
    k=0,...,s-1. Stage k

    1. reconstructs its input @f$ Y_k @f$ with a halo
       (solver._patch_with_halo). The interior is solver._patch for the
       first stage and solver._stage_input for all others, the halo is
       taken from the faces (solver._patch_overlap_current_time_step);
    2. evaluates the right-hand side @f$ K_k = F(Y_k) @f$ and stores it in
       the stage's slot of solver._rhs_estimates;
    3. computes the input of the next stage
       @f$ Y_{k+1} = Q^n + \\Delta t \\sum _{j \\leq k} a_{k+1,j} K_j @f$
       into solver._stage_input and projects it onto the faces, so the
       neighbours can reconstruct their halo from it in the next stage.

    The last stage computes
    @f$ Q^{n+1} = Q^n + \\Delta t \\sum _k b_k K_k @f$ instead, i.e. it
    overwrites solver._patch, and projects the new solution onto the faces,
    where the first stage of the next time step picks it up.

    As the reconstruction of a stage reads the face data the previous stage
    has written, the task graph compiler maps each stage onto a mesh
    traversal of its own (@ref peano4.api.taskgraph.compiler.FuseSteps).
    For a second-order Runge-Kutta scheme with Dirichlet boundary
    conditions, the time step's task graph reads:

    @image html central-differences-with-runge-kutta-time-step-task-graph.png

    The boundary values are evaluated at the time stamp of the respective
    stage, i.e. at @f$ t^n + c_k \\Delta t @f$. Otherwise, the scheme would
    drop to first order along the boundary.

    ## Time step sizes

    If you pass a time_step_size, the scheme uses this fixed time step size.
    Otherwise, it derives the time step size from the maximum eigenvalue of
    the solution after each time step through the CFL condition with
    cfl_constant (@ref exahype2.api.rkfd4.actionsets.ComputeTimeStepSize).
    This requires a MaxEigenvalue PDE term, with or without state.

    """

    ## Width of the halo. Subclasses with wider stencils override it.
    HALO_SIZE = 2

    @property
    def halo_size(self) -> int:
        """! Width of the halo. Alias of HALO_SIZE, as the generic parts of exahype2.api expect this name."""
        return self.HALO_SIZE

    class RightHandSideKernel(enum.Enum):
        """!
        The kernels which evaluate the right-hand side, @see the class documentation
        """

        CentralDifferences = 0
        CentralDifferencesWithLopsidedAdvection = 1

    def __init__(
        self,
        solver: Solver,
        patch_size: int,
        rk_order: int,
        time_step_size: float = None,
        cfl_constant: float = 0.5,
        right_hand_side_kernel: "CentralDifferencesWithRungeKutta.RightHandSideKernel" = RightHandSideKernel.CentralDifferences,
        advection_velocity=None,
        advected_quantities=None,
        min_volume_size: float = None,
        max_volume_size: float = None,
        interpolation: "toolbox.blockstructured.api.InterpolateHangingFaces.Scheme" = toolbox.blockstructured.api.InterpolateHangingFaces.Scheme.Linear,
        restriction: "toolbox.blockstructured.api.RestrictHangingFaces.Scheme" = toolbox.blockstructured.api.RestrictHangingFaces.Scheme.Averaging,
    ):
        """!

        The constructor memorises the attributes and creates the data
        structures. It does not add the data to the solver's Peano 4
        project yet, but it assigns them to the solver, so anybody else
        working against this solver, such as the grid construction or the
        plotter, finds them there.

        @param solver: exahype2.api.Solver
          The solver this numerical scheme realises.

        @param patch_size: int
          Number of grid cells per axis within one patch, i.e. within one
          mesh cell. Has to be at least HALO_SIZE, as the projection onto
          the faces takes the HALO_SIZE outermost layers of the patch.

        @param rk_order: int
          Order of the Runge-Kutta scheme, i.e. its number of stages. Has
          to be between one and four, @ref
          exahype2.api.rungekutta.ButcherTableau.

        @param time_step_size: float or None
          Fixed time step size. If None, the time step size is adaptive.

        @param cfl_constant: float
          CFL constant for adaptive time step sizes. It scales the time step
          size h / (d lambda_max), with h the grid cell size, d the
          dimension and lambda_max the maximum eigenvalue. Ignored if
          time_step_size is set. Fourth-order central differences with the
          classic fourth-order Runge-Kutta scheme are stable up to roughly
          two for linear advection without dissipation, so 0.5 is a
          conservative default.

        @param right_hand_side_kernel: CentralDifferencesWithRungeKutta.RightHandSideKernel
          The kernel which evaluates the right-hand side, @see the class
          documentation.

        @param advection_velocity: str or List[str or int] or None
          Only for CentralDifferencesWithLopsidedAdvection: the entries of Q
          which hold the advection velocity, one per dimension. Pass the name
          of a variable with one entry per dimension (e.g. "shift" for
          exagrype.api.FOCCZ4Solver), or a list of names or indices. The
          velocity may be an auxiliary variable.

        @param advected_quantities: List[str or int] or None
          Only for CentralDifferencesWithLopsidedAdvection: the unknowns
          which are advected, as names of variables or indices of entries,
          @see exahype2.api.rkfd4.Quantities.resolve_quantities().

        @param min_volume_size: float
          The time step asks for a refinement of a cell only if its grid
          cells are at least of this size (@ref exahype2.api.blockstructured.DynamicMeshRefinement).
          None (the default) does not bound the refinement.

        @param max_volume_size: float
          The time step asks for an erase, i.e. coarsening, of a cell only
          if the grid cells of the coarser cell would be strictly smaller
          than this size. None (the default) does not bound the coarsening.

        @param interpolation: toolbox.blockstructured.api.InterpolateHangingFaces.Scheme
          How the scheme initialises the halo of a fine patch next to a
          coarser patch on an adaptive mesh. Whenever a Runge-Kutta stage
          reconstructs the halo of a fine cell, it interpolates the data of
          the adjacent coarse face onto the fine cell's hanging face
          (toolbox.blockstructured.api.InterpolateHangingFaces). The
          default is linear interpolation.

        @param restriction: toolbox.blockstructured.api.RestrictHangingFaces.Scheme
          How the scheme hands the data of fine patches over to an adjacent
          coarser patch. Whenever the fine cells project their data onto
          their faces, the scheme restricts the data of the hanging faces
          onto the coarse face (toolbox.blockstructured.api.RestrictHangingFaces).
          The default is averaging.

        On a regular mesh, there are no hanging faces, and the two schemes
        are never used. At resolution transitions, the halo of a fine patch
        is only as accurate as the interpolation, i.e. the scheme drops to
        low order there.

        """
        super().__init__(solver)
        self.interpolation = interpolation
        self.restriction = restriction
        if patch_size < self.HALO_SIZE:
            raise ValueError(
                f"Solver {solver.name}: patch_size has to be at least {self.HALO_SIZE}, "
                f"as the stencils reach {self.HALO_SIZE} grid cells into "
                f"the neighbouring patches, but got {patch_size}"
            )
        if time_step_size is not None and time_step_size <= 0.0:
            raise ValueError(
                f"Solver {solver.name}: time_step_size has to be positive, "
                f"but got {time_step_size}"
            )
        if cfl_constant <= 0.0:
            raise ValueError(
                f"Solver {solver.name}: cfl_constant has to be positive, "
                f"but got {cfl_constant}"
            )
        if not isinstance(
            right_hand_side_kernel, CentralDifferencesWithRungeKutta.RightHandSideKernel
        ):
            raise TypeError(
                f"Solver {solver.name}: right_hand_side_kernel has to be a "
                f"CentralDifferencesWithRungeKutta.RightHandSideKernel, but got "
                f"{right_hand_side_kernel!r}"
            )
        self.right_hand_side_kernel = right_hand_side_kernel
        Kernel = CentralDifferencesWithRungeKutta.RightHandSideKernel
        if right_hand_side_kernel == Kernel.CentralDifferencesWithLopsidedAdvection:
            if advection_velocity is None or advected_quantities is None:
                raise ValueError(
                    f"Solver {solver.name}: the kernel "
                    f"CentralDifferencesWithLopsidedAdvection needs an "
                    f"advection_velocity and advected_quantities"
                )
            if isinstance(advection_velocity, (str, int)):
                advection_velocity = [advection_velocity]
            self.advection_velocity = resolve_quantities(solver, advection_velocity)
            self.advected_quantities = resolve_quantities(solver, advected_quantities)
            # The lopsided stencils reach three grid cells into the neighbours
            self.HALO_SIZE = max(self.HALO_SIZE, 3)
            if patch_size < self.HALO_SIZE:
                raise ValueError(
                    f"Solver {solver.name}: patch_size has to be at least "
                    f"{self.HALO_SIZE} for lopsided advection, but got {patch_size}"
                )
        else:
            if advection_velocity is not None or advected_quantities is not None:
                raise ValueError(
                    f"Solver {solver.name}: advection_velocity and "
                    f"advected_quantities are only used by the kernel "
                    f"CentralDifferencesWithLopsidedAdvection"
                )
            self.advection_velocity = None
            self.advected_quantities = None
        self.patch_size = patch_size
        assert (
            min_volume_size is None
            or max_volume_size is None
            or min_volume_size <= max_volume_size
        ), f"min_volume_size={min_volume_size} has to be smaller than or equal to max_volume_size={max_volume_size}"
        self.min_volume_size = min_volume_size
        self.max_volume_size = max_volume_size
        self.butcher_tableau = ButcherTableau(rk_order)
        self.time_step_size = time_step_size
        self.cfl_constant = cfl_constant
        # Set in construct_output(): the generated kernels live in the project
        # namespace
        self.kernel_namespace = None
        self.max_eigenvalue_kernel_call = None
        self.__create_data_structures()

    @property
    def overlap(self) -> int:
        """!
        Width of the halo, i.e. number of grid cells each face holds per
        adjacent patch. Always HALO_SIZE.
        """
        return self.HALO_SIZE

    @property
    def rk_order(self) -> int:
        return self.butcher_tableau.order

    @property
    def has_adaptive_time_step_size(self) -> bool:
        return self.time_step_size is None

    @property
    def right_hand_side_kernel_name(self) -> str:
        """!
        Name of the generated right-hand side kernel. It carries the solver
        name, as do the kernels/{SolverName}_fd4*.h files holding it, so
        two solvers' kernels do not clash.
        """
        return self._solver.name + "_rightHandSideWithFD4"

    @property
    def max_eigenvalue_kernel_name(self) -> str:
        return self._solver.name + "_maxEigenvalueOverPatch"

    def _right_hand_side_action_sets(self, stage, cell_guard):
        """!

        Action sets of the task which evaluates the right-hand side of one stage

        The task reads solver._patch_with_halo, i.e. the stage's input with
        its halo, and writes the stage's slot of solver._rhs_estimates. By
        default, it holds one action set, which invokes the fourth-order
        Finite Differences kernel. Subclasses can append action sets which
        add further terms to the estimate. The action sets of one task run
        one after the other, in the order of the list. For the kernel
        CentralDifferencesWithLopsidedAdvection, the list holds the
        lopsided advection as second action set.

        @param stage: int
          Number of the Runge-Kutta stage, starting with zero.

        @param cell_guard: str
          C++ boolean expression, evaluated per cell.

        @return List[peano4.api.solversteps.ActionSet]

        """
        action_sets = [
            EvaluateRungeKuttaRightHandSide(self._solver, stage, guard=cell_guard)
        ]
        if (
            self.right_hand_side_kernel
            == CentralDifferencesWithRungeKutta.RightHandSideKernel.CentralDifferencesWithLopsidedAdvection
        ):
            action_sets.append(
                AddLopsidedAdvection(
                    self._solver,
                    stage,
                    self.advection_velocity,
                    self.advected_quantities,
                    guard=cell_guard,
                )
            )
        return action_sets

    def __create_data_structures(self):
        """!

        Create the mesh data structures this numerical scheme requires

        All of them are assigned to self._solver rather than self, so the
        solver's grid construction and plotter, as well as the action sets,
        find them there:

        - _patch: the solution @f$ Q^n @f$, i.e. patch_size^d grid cells per
          mesh cell, with the unknowns and the auxiliary variables per grid
          cell.
        - _stage_input: the input @f$ Y_k @f$ of the next Runge-Kutta stage,
          with the same layout as _patch. It is written at the end of one
          stage and read by the next one.
        - _patch_with_halo: the input of the current Runge-Kutta stage
          with its halo, i.e. (patch_size+2*HALO_SIZE)^d grid cells,
          reconstructed from _patch or _stage_input and the faces (see
          toolbox.blockstructured.api.ReconstructPatchWithHalo). It is
          logically temporary.
        - _rhs_estimates: the right-hand side estimates @f$ K_k @f$ of all
          rk_order stages. They hold the unknowns only, as the auxiliary
          variables do not evolve in time. The estimates of the stages are
          stored one after the other, i.e. as rk_order patches of
          patch_size^d grid cells each. This is why the first dimension is
          rk_order*patch_size.
        - _patch_overlap_current_time_step: the face data the halo is
          reconstructed from, i.e. 2*HALO_SIZE layers along the face's
          normal. The face data are exchanged between ranks.
        - _patch_overlap_update: the projection of the patches onto the
          faces which the current sweep accumulates (@ref
          exahype2.api.rkfd4.actionsets.ProjectPatchOntoFaces). It is
          rolled over into _patch_overlap_current_time_step once complete
          (@ref exahype2.api.rkfd4.actionsets.RollOverUpdatedFace).
        - _max_eigenvalue: only with adaptive time step sizes. One double
          per cell, named after the solver followed by MaxEigenvalue. It
          holds the maximum eigenvalue of the patch after the last time step.
          It is modelled via DaStGen2 as array of cardinality one on the
          heap (peano4.api.dastgen2.Peano4SmartPointerDoubleArray). A task of
          its own reduces it into the generated abstract solver class
          (reduceMaxEigenvalue()), and the global time step size computation
          reads and resets the outcome (getAndResetMaxEigenvalue()). It
          thus works on this global datum only, i.e. it is a task that acts
          at the end of the time step
          (@ref peano4.api.taskgraph.Task.Task.GridEntity).

        All patches use smart pointers (@ref
        peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer), like the
        Finite Volumes of exahype2.api.fv. The legacy
        exahype2.solvers.rkfd.CellCenteredFiniteDifferences held a third
        face overlap with the data of the previous time step. It is
        required for AMR and local time stepping only, so we do not create
        it.

        """
        total_unknowns = self._solver.unknowns
        identifier = self._solver._unknown_identifier()
        n = self.patch_size
        h = self.HALO_SIZE

        self._solver._patch = peano4.api.datamodel.Patch(
            (n, n, n),
            total_unknowns,
            identifier,
        )
        self._solver._stage_input = peano4.api.datamodel.Patch(
            (n, n, n),
            total_unknowns,
            identifier + "StageInput",
        )
        self._solver._patch_with_halo = peano4.api.datamodel.Patch(
            (n + 2 * h, n + 2 * h, n + 2 * h),
            total_unknowns,
            identifier + "WithHalo",
        )
        self._solver._rhs_estimates = peano4.api.datamodel.Patch(
            (self.rk_order * n, n, n),
            self._solver.unknowns,
            identifier + "RhsEstimates",
        )
        self._solver._patch_overlap_current_time_step = peano4.api.datamodel.Patch(
            (2 * h, n, n),
            total_unknowns,
            identifier + "CurrentTimeStep",
        )
        self._solver._patch_overlap_update = peano4.api.datamodel.Patch(
            (2 * h, n, n),
            total_unknowns,
            identifier + "Update",
        )
        if self.has_adaptive_time_step_size:
            self._solver._max_eigenvalue = peano4.api.datamodel.DaStGen2(
                self._solver.name + "MaxEigenvalue"
            )
            self._solver._max_eigenvalue.data.add_attribute(
                peano4.api.dastgen2.Peano4SmartPointerDoubleArray("maxEigenvalue", 1)
            )
            self.__add_max_eigenvalue_reduction()
        else:
            self._solver._max_eigenvalue = None

    def __add_max_eigenvalue_reduction(self):
        """!

        Let the generated abstract solver class host the maximum eigenvalue of a time step

        The class collects the maximum over all patches and threads. This is
        a reduction over the whole mesh, hence it belongs to the solver and
        not to a cell. Reading and resetting it is one operation, as the
        patches contribute in a different traversal than the one that
        derives the time step size from it.

        """
        solver = self._solver
        solver.add_user_solver_includes(
            """
#include "tarch/multicore/BooleanSemaphore.h"
#include "tarch/multicore/Lock.h"
"""
        )
        solver.add_abstract_solver_attribute("double _maxEigenvalueOfTimeStep{0.0};")
        solver.add_abstract_solver_attribute(
            "tarch::multicore::BooleanSemaphore _maxEigenvalueSemaphore;"
        )
        solver.add_abstract_solver_method(
            "void reduceMaxEigenvalue(double eigenvalue);",
            lambda qualified_class_name: "void "
            + qualified_class_name
            + "::reduceMaxEigenvalue(double eigenvalue) {\n"
            + "  tarch::multicore::Lock lock(_maxEigenvalueSemaphore);\n"
            + "  _maxEigenvalueOfTimeStep = std::max(_maxEigenvalueOfTimeStep, eigenvalue);\n"
            + "}\n",
        )
        solver.add_abstract_solver_method(
            "double getAndResetMaxEigenvalue();",
            lambda qualified_class_name: "double "
            + qualified_class_name
            + "::getAndResetMaxEigenvalue() {\n"
            + "  tarch::multicore::Lock lock(_maxEigenvalueSemaphore);\n"
            + "  const double result       = _maxEigenvalueOfTimeStep;\n"
            + "  _maxEigenvalueOfTimeStep  = 0.0;\n"
            + "  return result;\n"
            + "}\n",
        )

    def _patches(self):
        """!
        All patches of this numerical scheme, i.e. all data besides the
        maximum eigenvalue.
        """
        return [
            self._solver._patch,
            self._solver._stage_input,
            self._solver._patch_with_halo,
            self._solver._rhs_estimates,
            self._solver._patch_overlap_current_time_step,
            self._solver._patch_overlap_update,
        ]

    def __add_data_to_repository(self, peano_project):
        """!

        Add the data from __create_data_structures() to the Peano 4 project
        and configure their generators

        Only _patch_overlap_current_time_step is exchanged between ranks,
        and only on the faces the solver's grid selector holds for. The
        task graph compiler uses, loads and stores data only where the
        solver's grid selector holds, too (@ref peano4.api.datamodel.DoF).

        """
        peano4.api.Logger().log(
            """
Patch data
----------
{}
Right-hand side estimates
-------------------------
{}
Patch overlap data
------------------
{}
{}""".format(
                str(self._solver._patch),
                str(self._solver._rhs_estimates),
                str(self._solver._patch_overlap_current_time_step),
                str(self._solver._patch_overlap_update),
            )
        )

        peano_project.datamodel.add_cell(self._solver._patch)
        peano_project.datamodel.add_cell(self._solver._stage_input)
        peano_project.datamodel.add_cell(self._solver._patch_with_halo)
        peano_project.datamodel.add_cell(self._solver._rhs_estimates)
        peano_project.datamodel.add_face(self._solver._patch_overlap_current_time_step)
        peano_project.datamodel.add_face(self._solver._patch_overlap_update)
        register_auxiliary_variables(
            self._solver, peano_project, self.patch_size, self.HALO_SIZE
        )

        data = self._patches()
        if self.has_adaptive_time_step_size:
            peano_project.datamodel.add_cell(self._solver._max_eigenvalue)
            data.append(self._solver._max_eigenvalue)
        for entry in data:
            entry.grid_selector = self._solver.grid_selector

        for patch in self._patches():
            patch.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
                patch, "double"
            )

        current_time_step = self._solver._patch_overlap_current_time_step
        current_time_step.generator.merge_method_definition = (
            toolbox.blockstructured.api.get_face_merge_implementation(current_time_step)
        )
        current_time_step.generator.includes += """
#include "peano4/utils/Loop.h"
"""
        current_time_step.generator.send_condition = (
            self._solver.grid_selector.createFaceGuard()
        )
        current_time_step.generator.receive_and_merge_condition = (
            self._solver.grid_selector.createFaceGuard()
        )

        self._solver._patch_overlap_update.generator.send_condition = "false"
        self._solver._patch_overlap_update.generator.receive_and_merge_condition = (
            "false"
        )

    def __create_kernels(self, output_directory):
        """!

        Translate the DSLHyPE kernels into C++

        The kernels end up in the kernels/ subdirectory of output_directory:

        - {SolverName}_fd4_2d.h/.cpph and {SolverName}_fd4_3d.h/.cpph hold
          the right-hand side (@ref exahype2.api.rkfd4.FD4Kernels).
        - {SolverName}_fd4_eigenvalue_2d.h/.cpph and
          {SolverName}_fd4_eigenvalue_3d.h/.cpph hold the maximum
          eigenvalue over a patch. They exist only with adaptive time step
          sizes.
        - {SolverName}_fd4.h includes the variants for the dimension the
          code is compiled for.
        - rungekutta_linear_combination.h/.cpph holds the linear combination
          of the Runge-Kutta estimates (@ref
          exahype2.api.rungekutta.LinearCombination).

        Calling this function is delayed until all PDE term information is
        provided, as the kernels contain only the PDE terms the solver has.

        """
        template_parameters = [
            dslhype.SyntaxTree.Argument("NC", dslhype.SyntaxTree.TInteger()),
            dslhype.SyntaxTree.Argument("H", dslhype.SyntaxTree.TInteger()),
            dslhype.SyntaxTree.Argument(
                "NumberOfUnknowns", dslhype.SyntaxTree.TInteger()
            ),
            dslhype.SyntaxTree.Argument(
                "NumberOfAuxiliaryVariables", dslhype.SyntaxTree.TInteger()
            ),
        ]
        functor_parameters = [
            dslhype.SyntaxTree.Argument(
                "flux",
                dslhype.SyntaxTree.TCustom(
                    "const ::exahype2::fv::rusanov::FluxFunctor&"
                ),
                "(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()",
            ),
            dslhype.SyntaxTree.Argument(
                "nonconservativeProduct",
                dslhype.SyntaxTree.TCustom(
                    "const ::exahype2::fv::rusanov::NonconservativeProductFunctor&"
                ),
                "(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()",
            ),
            dslhype.SyntaxTree.Argument(
                "volumetricSource",
                dslhype.SyntaxTree.TCustom(
                    "const ::exahype2::fv::rusanov::VolumetricSourceFunctor&"
                ),
                "(!llvm.ptr, !llvm.ptr, f64, !llvm.ptr) -> ()",
            ),
            dslhype.SyntaxTree.Argument(
                "maxEigenvalue",
                dslhype.SyntaxTree.TCustom(
                    "const ::exahype2::fv::rusanov::MaxEigenvalueFunctor&"
                ),
                "(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()",
            ),
        ]

        kernels = [
            (
                FD4Kernels.rightHandSideWithFD42D,
                "_fd4_2d",
                self.right_hand_side_kernel_name,
            ),
            (
                FD4Kernels.rightHandSideWithFD43D,
                "_fd4_3d",
                self.right_hand_side_kernel_name,
            ),
        ]
        if self.has_adaptive_time_step_size:
            kernels += [
                (
                    FD4Kernels.maxEigenvalueOverPatch2D,
                    "_fd4_eigenvalue_2d",
                    self.max_eigenvalue_kernel_name,
                ),
                (
                    FD4Kernels.maxEigenvalueOverPatch3D,
                    "_fd4_eigenvalue_3d",
                    self.max_eigenvalue_kernel_name,
                ),
            ]
        for function, file_suffix, function_name in kernels:
            dslhype.translate_dslhype_function(
                function=function,
                function_namespace=self.kernel_namespace,
                template_parameters=template_parameters,
                functor_parameters=functor_parameters,
                pde_terms=self._solver.pde_terms,
                output_file_name=self._solver.name + file_suffix,
                target=dslhype.Target.Serial,
                stateless=False,
                function_name=function_name,
                output_directory=output_directory,
            )

        for _, file_suffix, _ in kernels:
            include_implementation_into_header(
                os.path.join(output_directory, "kernels"),
                f"{self._solver.name}{file_suffix}",
            )

        includes_2d = "\n".join(
            f'    #include "{self._solver.name}{file_suffix}.h"'
            for _, file_suffix, _ in kernels
            if file_suffix.endswith("2d")
        )
        includes_3d = "\n".join(
            f'    #include "{self._solver.name}{file_suffix}.h"'
            for _, file_suffix, _ in kernels
            if file_suffix.endswith("3d")
        )
        with open(
            os.path.join(output_directory, "kernels", f"{self._solver.name}_fd4.h"), "w"
        ) as file:
            file.write(
                f"""#pragma once
#if Dimensions == 2
{includes_2d}
#else
{includes_3d}
#endif
"""
            )

        generate_linear_combination_kernel(self.kernel_namespace, output_directory)

    def __construct_kernel_call(self, function_name, additional_arguments):
        """!

        Construct the call of a generated kernel

        The kernel takes a ::exahype2::CellData called patchData, the
        additional arguments and one functor per PDE term the solver has,
        in the order flux, nonconservativeProduct, volumetricSource,
        maxEigenvalue (see __create_kernels()). Each functor forwards to
        the solver instance.

        If PDE terms are registered both with and without state, the
        generated code asks cellCanUseStatelessPDETerms() before it invokes
        the kernel, and hands over the stateless variants if it returns
        true. This logic is a copy of the one of exahype2.api.fv, as rkfd4
        does not depend on the Finite Volume package.

        @param function_name: str
          Unqualified name of the kernel.

        @param additional_arguments: [str]
          C++ expressions passed after patchData.

        """
        available_terms = [
            "flux",
            "nonconservativeProduct",
            "volumetricSource",
            "maxEigenvalue",
        ]

        # Variants per term the kernel takes: {name: {stateless: PDETerm}}
        variants = {}
        for application_term in self._solver.pde_terms:
            if application_term.NAME in available_terms:
                variants.setdefault(application_term.NAME, {})[
                    application_term.stateless
                ] = application_term

        terms_with_both_variants = [
            name for name in variants if len(variants[name]) == 2
        ]
        terms_without_stateless_variant = [
            name for name in variants if True not in variants[name]
        ]
        if terms_with_both_variants and terms_without_stateless_variant:
            raise Exception(
                f"Solver {self._solver.name}: the PDE terms {terms_with_both_variants} are "
                f"registered both with and without state. In this case, the generated code "
                f"asks cellCanUseStatelessPDETerms() per cell before it invokes the kernel: "
                f"if it returns true, the kernel is called with the stateless variants of all "
                f"PDE terms, otherwise with the variants with state. However, the PDE terms "
                f"{terms_without_stateless_variant} have no stateless variant, so a kernel "
                f"call would mix stateless and stateful routines. Either add stateless "
                f"variants of {terms_without_stateless_variant}, or register "
                f"{terms_with_both_variants} only once."
            )

        def kernel_call(stateless: bool) -> str:
            arguments = ["patchData"] + list(additional_arguments)
            for term in available_terms:
                if term not in variants:
                    continue
                # A term registered only once is used in either case
                application_term = variants[term].get(
                    stateless, next(iter(variants[term].values()))
                )
                # Commas inside template brackets, e.g. Vector<Dimensions, double>,
                # do not separate parameters
                term_arguments = forwarding_arguments(
                    application_term.SIGNATURE, self._solver
                )
                # Stateless terms are static and take a dummy tag (@see PDETerm)
                if application_term.stateless:
                    term_arguments.append("::exahype2::Solver::Offloadable::Yes")
                arguments.append(
                    f"""
    [&](
        {application_term.SIGNATURE}
    )->void {{
        repositories::{self._solver.get_name_of_global_instance()}.{application_term.name}({", ".join(term_arguments)});
    }}"""
                )
            return f"""::{"::".join(self.kernel_namespace)}::{function_name}<
    {self.patch_size},
    {self.HALO_SIZE},
    {self._solver.unknowns},
    {self._solver.number_of_auxiliary_variables}
>(
    {",".join(arguments)}
);"""

        if not terms_with_both_variants:
            return kernel_call(stateless=False)
        return f"""if (repositories::{self._solver.get_name_of_global_instance()}.cellCanUseStatelessPDETerms(
    patchData.cellCentre[0],
    patchData.cellSize[0],
    patchData.t[0],
    patchData.dt[0]
)) {{
{kernel_call(stateless=True)}
}} else {{
{kernel_call(stateless=False)}
}}"""

    def create_default_mesh_adaptation(self) -> Solver.MeshAdaptation:
        """!

        The finite differences adapt the mesh and balance the load through
        exahype2.api.rkfd4.AdaptMeshAndBalanceLoad by default

        """
        from .AdaptMeshAndBalanceLoad import AdaptMeshAndBalanceLoad

        return AdaptMeshAndBalanceLoad(self._solver)

    def create_restriction_of_hanging_faces(
        self,
    ) -> "toolbox.blockstructured.api.RestrictHangingFaces":
        """!

        Action set which restricts the projected fine data from the hanging
        faces onto the coarse faces

        Every task which projects the solution onto the faces has to host
        this action set, so the faces between coarse and fine patches hold
        consistent data afterwards. The task has to write all of
        _patch_overlap_update, as averaging clears it first.

        """
        return toolbox.blockstructured.api.RestrictHangingFaces(
            self._solver._patch_overlap_update,
            scheme=self.restriction,
            clear_guard=self._solver.grid_selector.createFaceGuard(),
            additional_includes=self._solver._get_default_includes()
            + self._solver.user_solver_includes,
        )

    def construct_output(
        self, exahype2_project, peano_project, time_step, add_global_tasks=True
    ):
        """!

        Add the data to the Peano 4 project, generate the kernels and add
        the tasks of one Runge-Kutta step to time_step

        @param add_global_tasks: bool
          If False, the scheme adds neither the task that switches the load
          balancing off nor the last task that evaluates the adaptivity
          criterion and switches the load balancing on again,
          @see exahype2.api.Solver.Solver.NumericalScheme.construct_output().

        Per stage k, the time step holds the following tasks (see the class
        documentation for the data flow). Every task holds exactly one
        action set, with the exception of the reconstruction of the
        unknowns, which also holds the interpolation onto the hanging faces,
        and the reconstruction of several sets of auxiliary variables. This is deliberate: the task graph compiler
        runs the action sets of independent tasks in parallel only if each
        task holds one action set (@ref peano4.api.taskgraph.compiler.FuseSteps).
        Where the scheme needs several action sets in a row, it hence creates
        a chain of tasks. If several solvers share a task graph, their chains
        overlap.

        - {SolverName}_Disable load balancing: switches the load balancing
          off throughout the time step. Added last by
          wrap_task_graph_in_load_balancing_and_amr_tasks(), which makes all
          tasks without predecessor wait for it.
        - {SolverName}_Apply Dirichlet BCs stage k: sets the ghost grid
          cells along the domain boundary for Dirichlet boundary conditions.
          The task does not exist if all axes are periodic, or for Sommerfeld
          boundary conditions, which advance the ghost grid cells within the
          roll over of the projection onto the faces.
        - {SolverName}_Reconstruct stage k: interpolates the coarse face
          data onto the hanging faces and reconstructs the stage's input with
          its halo. Interpolation is no task of its own.
        - {SolverName}_Reconstruct auxiliary variables stage k: reconstructs
          the halos of the auxiliary variables. Exists only if the solver has
          auxiliary variables. Sets which are stationary throughout the time
          step (exahype2.api.AuxiliaryVariables.Mode.STATIONARY_THROUGHOUT_TIME_STEPS_ON_FACES)
          have this task in the first stage only: The right-hand-side tasks
          hold their halo as persistent cell data until the last stage
          discards it.
        - {SolverName}_Evaluate Runge-Kutta right hand side stage k. Where
          the right-hand side has further terms
          (_right_hand_side_action_sets()), each term adds a task behind it,
          e.g. {SolverName}_Add lopsided advection stage k.
        - {SolverName}_Prepare stage k+1, or {SolverName}_Complete time step
          after the last stage: computes the linear combination of the
          estimates. Prepare stage k+1 is tagged with
          exahype2.api.Solver.IntermediateSolutionUpdateTag, Complete time
          step with exahype2.api.Solver.SolutionUpdateTaskTag, i.e.
          postprocessing such as the enforcement of constraints can be
          inserted directly after them (see @ref
          page_exahype2_solver_manipulate_solver_steps).
        - {SolverName}_Compute max eigenvalue: determines the maximum
          eigenvalue of the new solution. Exists after the last stage for
          adaptive time stepping only.
        - {SolverName}_Project stage k+1 onto faces, or {SolverName}_Project
          solution onto faces after the last stage: projects the outcome of
          the linear combination onto the faces and restricts the hanging
          faces onto the coarse faces.
        - {SolverName}_Roll over stage k+1 faces, or {SolverName}_Roll over
          solution faces. The projection (with the restriction) and the roll over
          share the update patch, which is temporary, i.e. they rely on the
          task graph compiler to fuse them into one mesh traversal. Only
          the roll over writes the face data of the current time step, but
          the projection declares it, and the roll over declares it as
          temporary. If the roll over declared it, the projection would not conflict with the reconstruction, which reads
          the faces. The compiler would then fuse them into the traversal of
          the previous stage, and only the roll over would start a new
          traversal. The update patch does not survive such a split, and the
          roll over reads uninitialised data (observed as NaNs in the next
          stage's input). The fused traversal merges the qualifiers back into
          the one of the projection.

        After the last stage, {SolverName}_Compute time step size determines
        the time step size of the next time step and advances the solver's
        time stamp. {SolverName}_Dynamic mesh refinement and load balancing finally evaluates the
        adaptivity criterion if the solver has one
        (@ref exahype2.api.blockstructured.create_dynamic_mesh_refinement_action_set.create_dynamic_mesh_refinement_action_set()),
        switches the load balancing on again and informs it that the time
        step has finished.

        """
        peano4.api.Logger().log(
            "Create time stepping task graph for fourth-order Finite Differences "
            f"with Runge-Kutta of order {self.rk_order} over block-structured meshes"
        )

        if self.advection_velocity is not None and len(self.advection_velocity) != len(
            exahype2_project.periodic_BC
        ):
            raise ValueError(
                f"Solver {self._solver.name}: the advection velocity has "
                f"{len(self.advection_velocity)} entries, but the domain has "
                f"{len(exahype2_project.periodic_BC)} dimensions"
            )

        if self.has_adaptive_time_step_size:
            assert any(
                isinstance(term, MaxEigenvalue) for term in self._solver.pde_terms
            ), (
                f"Solver {self._solver.name} uses adaptive time step sizes, but has no "
                "MaxEigenvalue PDE term. Add one via "
                "solver.add_pde_term(exahype2.api.MaxEigenvalue(...)), or pass a "
                "time_step_size to CentralDifferencesWithRungeKutta."
            )

        relative_time_step_sizes = self.butcher_tableau.relative_time_step_sizes()
        if isinstance(
            get_boundary_conditions(self._solver, exahype2_project),
            SommerfeldBoundaryConditions,
        ) and any(
            later < earlier
            for earlier, later in zip(
                relative_time_step_sizes, relative_time_step_sizes[1:]
            )
        ):
            raise ValueError(
                f"Solver {self._solver.name} uses Sommerfeld boundary conditions, but "
                f"the time stamps c={relative_time_step_sizes} of the stages of its "
                f"Runge-Kutta scheme of order {self.rk_order} decrease. The roll over "
                "advances the ghost grid cells from one stage to the next, so it "
                "requires non-decreasing time stamps "
                "(@ref exahype2::blockstructured::applySommerfeldBoundaryConditions())."
            )

        self.__add_data_to_repository(peano_project)

        self.kernel_namespace = peano_project.namespace
        # Construct the calls first, as they validate the PDE terms' variants
        self._solver.compute_kernel_call = self.__construct_kernel_call(
            self.right_hand_side_kernel_name, []
        )
        if self.has_adaptive_time_step_size:
            self.max_eigenvalue_kernel_call = self.__construct_kernel_call(
                self.max_eigenvalue_kernel_name, []
            )
        self.__create_kernels(peano_project.directory)

        cell_guard = self._solver.grid_selector.createCellGuard()
        face_guard = self._solver.grid_selector.createFaceGuard()
        includes = (
            self._solver._get_default_includes() + self._solver.user_solver_includes
        )
        name = self._solver.name
        weight_matrix = self.butcher_tableau.weight_matrix()
        Access = peano4.api.taskgraph.Task.DataAccess
        Entity = peano4.api.taskgraph.Task.GridEntity

        previous_stage_task = None
        for stage in range(self.rk_order):
            is_last_stage = stage == self.rk_order - 1
            stage_input = (
                self._solver._patch if stage == 0 else self._solver._stage_input
            )

            ## Tasks with one action set each: The task graph compiler runs the
            ## action sets of tasks which are independent of each other and
            ## hold exactly one action set in parallel (@ref
            ## peano4.api.taskgraph.compiler.FuseSteps). Where a step of the
            ## scheme needs several action sets, it therefore is a chain of tasks.
            ## This is what lets the stages of several solvers overlap.
            face_read = get_access_to_persistent_face_data(
                self._solver, exahype2_project, Access.READ_AND_DISCARD
            )

            ## Reconstruct the stage's input with its halo
            reconstruction_task = peano4.api.taskgraph.Task(
                name=f"{name}_Reconstruct stage {stage}", entity=Entity.CELL
            )
            time_step.add_task(reconstruction_task)
            if previous_stage_task is not None:
                reconstruction_task.add_dependency(previous_stage_task)
            ## The solution has to survive the reconstruction. The input of a
            ## later stage does not: The next stage's combination writes it
            ## from scratch. If we stored it here, nobody would load it again,
            ## and Peano's stacks would accumulate one patch per cell and stage.
            reconstruction_task.use_cell(
                stage_input,
                Access.READ_AND_WRITE if stage == 0 else Access.READ_AND_DISCARD,
            )
            reconstruction_task.use_cell(self._solver._patch_with_halo, Access.WRITE)
            # Sommerfeld boundary conditions read the old face data within the
            # roll over, i.e. it has to survive the reconstruction
            reconstruction_task.use_face(
                self._solver._patch_overlap_current_time_step, face_read
            )
            ## On adaptive meshes, the reconstruction interpolates the coarse
            ## face data onto the hanging faces of fine cells itself (see
            ## src/exahype2/api/adrs/0001-interpolation-and-restriction-ride-along-with-face-users.md). The
            ## interpolation acts when the hanging faces are created, i.e.
            ## before the reconstruction acts on the cell.
            reconstruction_task.add_action_set(
                toolbox.blockstructured.api.InterpolateHangingFaces(
                    self._solver._patch_overlap_current_time_step,
                    scheme=self.interpolation,
                    additional_includes=includes,
                )
            )
            reconstruction_task.add_action_set(
                toolbox.blockstructured.api.ReconstructPatchWithHalo(
                    patch=stage_input,
                    patch_overlap=self._solver._patch_overlap_current_time_step,
                    reconstructed_patch=self._solver._patch_with_halo,
                    guard=cell_guard,
                    add_assertions_to_halo_exchange=False,
                    additional_includes=includes,
                )
            )
            ## Set the ghost grid cells along the domain boundary before the
            ## reconstruction reads them, but after the previous stage has
            ## written the face data. No task is added if all axes are periodic
            ## or for Sommerfeld boundary conditions
            boundary_task = ApplyDirichletBoundaryConditions.add_task_to_time_step(
                self._solver,
                exahype2_project,
                time_step,
                reconstruction_task,
                name_suffix=f" stage {stage}",
                relative_time_step_size=self.butcher_tableau.relative_time_step_sizes()[
                    stage
                ],
            )
            if boundary_task is not None and previous_stage_task is not None:
                boundary_task.add_dependency(previous_stage_task)

            ## The halos of the auxiliary variables, if there are any. Their
            ## interpolations and reconstructions run in parallel to the ones
            ## of the unknowns. Sets which do not change within the time step
            ## are reconstructed in the first stage only; the later stages
            ## find their halo as persistent cell data.
            reconstruction_tasks = [reconstruction_task] + (
                add_reconstruction_tasks_of_auxiliary_variables(
                    self._solver,
                    time_step,
                    self.interpolation,
                    [previous_stage_task] if previous_stage_task is not None else [],
                    name_suffix=f" stage {stage}",
                    auxiliaries=[
                        auxiliary
                        for auxiliary in self._solver.auxiliary_variables
                        if stage == 0 or not auxiliary.halo_is_held_through_stages
                    ],
                )
            )

            ## Evaluate the right-hand side into the stage's estimate, one task
            ## per action set. They run one after the other, as each adds to the
            ## estimate of its predecessor.
            right_hand_side_tasks = []
            for index, action_set in enumerate(
                self._right_hand_side_action_sets(stage, cell_guard)
            ):
                task_name = (
                    "Evaluate Runge-Kutta right hand side"
                    if index == 0
                    else getattr(action_set, "TASK_NAME", None)
                    or _readable_name(type(action_set).__name__)
                )
                task = peano4.api.taskgraph.Task(
                    name=f"{name}_{task_name} stage {stage}", entity=Entity.CELL
                )
                task.add_action_set(action_set)
                for dependency in (
                    [right_hand_side_tasks[-1]]
                    if right_hand_side_tasks
                    else reconstruction_tasks
                ):
                    task.add_dependency(dependency)
                time_step.add_task(task)
                right_hand_side_tasks.append(task)
            for position, task in enumerate(right_hand_side_tasks):
                length = len(right_hand_side_tasks)
                task.use_cell(
                    self._solver._patch_with_halo,
                    _access_within_chain(Access.READ_AND_DISCARD, position, length),
                )
                ## The first task of the time step must not be write-only: The
                ## last traversal of the previous time step stores the estimates,
                ## as it fuses read-and-write tasks. A write-only first task
                ## would not load them again, and Peano's stacks accumulate
                ## one patch per cell and time step (memory grows with the
                ## simulated time).
                task.use_cell(self._solver._rhs_estimates, Access.READ_AND_WRITE)
                use_halos_of_auxiliary_variables(
                    task,
                    self._solver,
                    _access_within_chain(Access.READ_AND_DISCARD, position, length),
                    # Halos that are held through the stages survive all but the last stage
                    held_access=_access_within_chain(
                        (
                            Access.READ_AND_DISCARD
                            if is_last_stage
                            else Access.READ_AND_WRITE
                        ),
                        position,
                        length,
                    ),
                )
            right_hand_side_task = right_hand_side_tasks[-1]

            ## Combine the estimates into the input of the next stage or into
            ## the new solution. The combination is a task of its own, tagged
            ## as solution update, so postprocessing can be inserted between
            ## the combination and the projection onto the faces. Stage inputs
            ## carry the intermediate tag
            ## (@ref page_exahype2_solver_manipulate_solver_steps).
            if is_last_stage:
                weights = self.butcher_tableau.final_estimate_weights()
                output = self._solver._patch
                combination_task = peano4.api.taskgraph.Task(
                    name=f"{name}_Complete time step", entity=Entity.CELL
                )
                suffix = "solution"
            else:
                weights = weight_matrix[stage + 1]
                output = self._solver._stage_input
                combination_task = peano4.api.taskgraph.Task(
                    name=f"{name}_Prepare stage {stage + 1}", entity=Entity.CELL
                )
                suffix = f"stage {stage + 1}"
            time_step.add_task(combination_task)
            combination_task.add_dependency(right_hand_side_task)
            combination_task.use_cell(self._solver._patch, Access.READ_AND_WRITE)
            ## Nobody reads the estimates after the final combination
            combination_task.use_cell(
                self._solver._rhs_estimates,
                Access.READ_AND_DISCARD if is_last_stage else Access.READ_AND_WRITE,
            )
            if not is_last_stage:
                combination_task.use_cell(self._solver._stage_input, Access.WRITE)
            combination_task.add_action_set(
                ComputeLinearCombination(
                    self._solver, weights, output, guard=cell_guard
                )
            )
            combination_task.tags.add(
                Solver.SolutionUpdateTaskTag
                if is_last_stage
                else Solver.IntermediateSolutionUpdateTag
            )

            ## Project the outcome onto the faces, restrict it on hanging faces and
            ## roll the faces over. After the last stage, we determine the maximum
            ## eigenvalue of the new solution right after the projection, i.e. in
            ## parallel to the roll over (see adrs/0002-eigenvalue-follows-projection.md
            ## for why not in parallel to the projection). All tasks share the
            ## update patch, which is temporary, i.e. they rely on the task graph
            ## compiler to fuse them into one mesh traversal.
            eigenvalue_task = None
            if is_last_stage and self.has_adaptive_time_step_size:
                eigenvalue_task = peano4.api.taskgraph.Task(
                    name=f"{name}_Compute max eigenvalue", entity=Entity.CELL
                )
                eigenvalue_task.use_cell(output, Access.READ_AND_WRITE)
                eigenvalue_task.use_cell(self._solver._max_eigenvalue, Access.WRITE)
                for auxiliary in self._solver.auxiliary_variables:
                    eigenvalue_task.use_cell(auxiliary.patch, Access.READ_AND_WRITE)
                eigenvalue_task.add_action_set(
                    ComputeMaxEigenvalue(self._solver, guard=cell_guard)
                )
                eigenvalue_task.add_dependency(combination_task)
                time_step.add_task(eigenvalue_task)
                reduction_task = peano4.api.taskgraph.Task(
                    name=f"{name}_Reduce max eigenvalue", entity=Entity.CELL
                )
                reduction_task.use_cell(
                    self._solver._max_eigenvalue, Access.READ_AND_DISCARD
                )
                reduction_task.add_action_set(
                    ReduceMaxEigenvalue(self._solver, guard=cell_guard)
                )
                reduction_task.add_dependency(eigenvalue_task)
                time_step.add_task(reduction_task)

            projection_task = peano4.api.taskgraph.Task(
                name=f"{name}_Project {suffix} onto faces"
                if not is_last_stage
                else f"{name}_Project solution onto faces",
                entity=Entity.CELL,
            )
            projection_task.add_dependency(combination_task)
            projection_task.use_cell(output, Access.READ_AND_WRITE)
            projection_task.use_face(
                self._solver._patch_overlap_update, Access.TEMPORARY
            )
            ## See construct_output() for why the projection declares these faces
            projection_task.use_face(
                self._solver._patch_overlap_current_time_step,
                get_access_to_persistent_face_data(
                    self._solver, exahype2_project, Access.WRITE
                ),
            )
            projection_task.add_action_set(
                ProjectPatchOntoFaces(
                    output,
                    self._solver._patch_overlap_update,
                    guard=cell_guard,
                    includes=includes,
                )
            )
            ## On adaptive meshes, the projection also restricts the fine data
            ## from the hanging faces onto the coarse faces. Peano destroys a
            ## hanging face after the fine cell has projected onto it, and
            ## touches the coarse face for the last time after that, i.e. the
            ## coarse face is final when the roll over copies it (see
            ## src/exahype2/api/adrs/0001-interpolation-and-restriction-ride-along-with-face-users.md)
            projection_task.add_action_set(self.create_restriction_of_hanging_faces())
            time_step.add_task(projection_task)
            if eigenvalue_task is not None:
                eigenvalue_task.add_dependency(projection_task)

            # Sommerfeld boundary conditions advance the ghost grid cells
            # within the roll over, from the time stamp of this stage to the
            # one of the next stage (@ref exahype2.api.blockstructured.BoundaryConditions)
            next_relative_time_step_size = (
                1.0 if is_last_stage else relative_time_step_sizes[stage + 1]
            )
            roll_over_task = peano4.api.taskgraph.Task(
                name=f"{name}_Roll over {suffix} faces", entity=Entity.CELL
            )
            roll_over_task.use_face(
                self._solver._patch_overlap_update, Access.TEMPORARY
            )
            roll_over_task.use_face(
                self._solver._patch_overlap_current_time_step, Access.TEMPORARY
            )
            roll_over_task.add_action_set(
                RollOverUpdatedFaceWithSommerfeldBoundaryConditions.create_roll_over(
                    self._solver,
                    exahype2_project,
                    guard=face_guard,
                    includes=includes,
                    time_step_size=(
                        f"{next_relative_time_step_size - relative_time_step_sizes[stage]!r}"
                        f" * repositories::{self._solver.get_name_of_global_instance()}.getMinTimeStepSize()"
                    ),
                )
            )
            roll_over_task.add_dependency(projection_task)
            time_step.add_task(roll_over_task)
            previous_stage_task = roll_over_task

        ## Determine the next time step size and advance the time stamp. The task
        ## acts at the end of the time step only, i.e. it needs no traversal
        time_step_size_task = peano4.api.taskgraph.Task(
            name=f"{name}_Compute time step size",
            entity=Entity.GLOBAL_END_OF_STEP,
            may_spawn_background_tasks=False,
        )
        ## The task must not run before the final roll over of the time step:
        ## Roll overs such as the Sommerfeld boundary treatment need the valid
        ## current time step size, which this task replaces
        ## (see adrs/0001-time-step-size-after-roll-over.md).
        time_step_size_task.add_dependency(previous_stage_task)
        if self.has_adaptive_time_step_size:
            time_step_size_task.add_dependency(reduction_task)
        time_step_size_task.add_action_set(
            ComputeTimeStepSize(
                self._solver,
                self.patch_size,
                cfl_constant=self.cfl_constant
                if self.has_adaptive_time_step_size
                else None,
                time_step_size=None
                if self.has_adaptive_time_step_size
                else self.time_step_size,
            )
        )
        time_step.add_task(time_step_size_task)

        ## The load balancing is switched off throughout the time step, and
        ## switched on again only at its very end, so it may then decide on
        ## splits. Peano carries these splits out in the subsequent mesh sweep.
        if add_global_tasks:
            self.wrap_task_graph_in_load_balancing_and_amr_tasks(time_step)

        if peano4.api.Logger().is_visualisation:
            time_step.visualise(
                title="Time stepping task graph for {}{}".format(
                    name, self._solver.task_graph_title_suffix
                )
            )
        if peano4.api.Logger().is_text:
            time_step.log_as_text(
                title="Time stepping task graph for {}{}".format(
                    name, self._solver.task_graph_title_suffix
                )
            )
