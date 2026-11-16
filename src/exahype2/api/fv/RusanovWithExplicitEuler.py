# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import os
import re

import peano4.api
import peano4.api.datamodel
import peano4.api.dastgen2
import toolbox.blockstructured.api
import toolbox.loadbalancing.api.actionsets
import exahype2.api
from exahype2.api import Target, OutputLanguage, MemoryLayout
from collections import OrderedDict

# This solver is using the rewritten version
import exahype2.api.dslhype2 as dslhype
 
from ..Solver import Solver
from ..AuxiliaryVariables import (
    register_auxiliary_variables,
    add_reconstruction_tasks_of_auxiliary_variables,
    use_halos_of_auxiliary_variables,
    forwarding_arguments,
)
from ..blockstructured.BoundaryConditions import get_access_to_persistent_face_data
from ..blockstructured.create_dynamic_mesh_refinement_action_set import (
    create_dynamic_mesh_refinement_action_set,
)
from .actionsets import ApplyDirichletBoundaryConditions
from .actionsets import ProjectPatchOntoFaces
from .actionsets import RollOverUpdatedFaceWithSommerfeldBoundaryConditions
from .actionsets import UpdateCell
from .actionsets import UpdateCellGPU
from .actionsets import ComputeGlobalCFLCondition
from .actionsets import ComputeMaxEigenvalue
from .actionsets import ReduceMaxEigenvalue
from .MaxEigenvalueReduction import add_max_eigenvalue_reduction

from ..VolumetricSource import VolumetricSource
from ..Flux import Flux
from ..NonconservativeProduct import NonconservativeProduct
from ..MaxEigenvalue import MaxEigenvalue
## DSLHyPE time step kernels
#
# The two functions below are never executed as Python. DSLHyPE parses their
# source (@ref exahype2.api.dslhype.Parser) and translates them into the C++
# time step kernels (@ref RusanovWithExplicitEuler.__create_time_step_functions()).
# They hence refer to symbols that the parser sets rather than Python imports:
#
# - Template parameters: NV, H, NumberOfUnknowns, NumberOfAuxiliaryVariables.
# - Functors for the non-stateless code: flux, nonconservativeProduct,
#   volumetricSource, maxEigenvalue.
# - loopParallelism (peano4::utils::LoopPlacement) for loop parallelisation.


## Explicit Euler time step with a Rusanov flux over a 2d patch.
def timeStepWithRusanov2D(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    t = DataBlock([[0, N]], "patchData.t", "double")
    dt = DataBlock([[0, N]], "patchData.dt", "double")
    h = DataBlock([[0, N]], "patchData.cellSize", "tarch::la::Vector<2, double>")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<2, double>")

    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [-H, NV + H], [-H, NV + H], [0, N]], "patchData.QIn")
    output = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NV], [0, NV], [0, N]], "patchData.QOut")
    output = input[:, 0:NV, 0:NV]

    if volumetricSource:
       source = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [0, N]], eval("volumetricSource", input[:, 0:NV, 0:NV], x, t))
       output[0:NumberOfUnknowns, :, :] = output[0:NumberOfUnknowns, :, :] + dt * source

    # The Rusanov flux at a face needs the eigenvalues of both adjacent
    # volumes, i.e. the patch volumes plus the one halo layer across each
    # face along the normal. We never evaluate the remaining halo (edges,
    # corners), which holds no valid data.
    maxEigenvaluesX = DataBlock([[0, 1], [-1, NV + 1], [0, NV], [0, N]], eval("maxEigenvalue", input[:, -1:NV+1, 0:NV], x, 0))
    maxEigenvaluesY = DataBlock([[0, 1], [0, NV], [-1, NV + 1], [0, N]], eval("maxEigenvalue", input[:, 0:NV, -1:NV+1], x, 1))

    lambdaLeft = max(maxEigenvaluesX[0:1, -1:NV-1, 0:NV], maxEigenvaluesX[0:1, 0:NV, 0:NV])
    lambdaRight = max(maxEigenvaluesX[0:1, 0:NV, 0:NV], maxEigenvaluesX[0:1, 1:NV + 1, 0:NV])
    lambdaBottom = max(maxEigenvaluesY[0:1, 0:NV, -1:NV-1], maxEigenvaluesY[0:1, 0:NV, 0:NV])
    lambdaTop = max(maxEigenvaluesY[0:1, 0:NV, 0:NV], maxEigenvaluesY[0:1, 0:NV, 1:NV + 1])

    fluxLeft = -1.0 * lambdaLeft * (input[0:NumberOfUnknowns, 0:NV, 0:NV] - input[0:NumberOfUnknowns, -1:NV-1, 0:NV])
    fluxRight = -1.0 * lambdaRight * (input[0:NumberOfUnknowns, 1:NV+1, 0:NV] - input[0:NumberOfUnknowns, 0:NV, 0:NV])
    output[0:NumberOfUnknowns, :, :] = output[0:NumberOfUnknowns, :, :] + dt / h(0) * (fluxLeft - fluxRight) * NV * 0.5

    fluxBottom = -1.0 * lambdaBottom * (input[0:NumberOfUnknowns, 0:NV, 0:NV] - input[0:NumberOfUnknowns, 0:NV, -1:NV-1])
    fluxTop = -1.0 * lambdaTop * (input[0:NumberOfUnknowns, 0:NV, 1:NV+1] - input[0:NumberOfUnknowns, 0:NV, 0:NV])
    output[0:NumberOfUnknowns, :, :] = output[0:NumberOfUnknowns, :, :] + dt / h(1) * (fluxBottom - fluxTop) * NV * 0.5

    if flux:
      flux_x = DataBlock([[0, NumberOfUnknowns], [-H, NV + H], [0, NV], [0, N]], eval("flux", input[:, -1:NV+1, 0:NV], x, 0))
      flux_y = DataBlock([[0, NumberOfUnknowns], [0, NV], [-H, NV + H], [0, N]], eval("flux", input[:, 0:NV, -1:NV+1], x, 1))
      
      output[0:NumberOfUnknowns, :, :] = output[0:NumberOfUnknowns, :, :] + 0.5 * NV * dt / h(0) *  (flux_x[:, -1:NV-1, 0:NV] - flux_x[:, 1:NV+1, 0:NV])
      output[0:NumberOfUnknowns, :, :] = output[0:NumberOfUnknowns, :, :] + 0.5 * NV * dt / h(1) * (flux_y[:, 0:NV, -1:NV-1] - flux_y[:, 0:NV, 1:NV+1])

    if nonconservativeProduct:
       average_x = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, -1:NV, 0:NV] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV + 1, 0:NV])
       delta_x = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV + H, 0:NV] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, -H:NV, 0:NV]
       nonconservative_x = DataBlock([[0, NumberOfUnknowns], [0, NV + H], [0, NV], [0, N]], eval("nonconservativeProduct", average_x, delta_x, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, :, :] = output[0:NumberOfUnknowns, :, :] + 0.5 * NV * dt / h(0) * (-nonconservative_x[:, 0:NV, 0:NV] - nonconservative_x[:, 1:NV+1, 0:NV])
       
       average_y = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, -1:NV] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV + 1])
       delta_y = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV + H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, -H:NV]
       nonconservative_y = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV + H], [0, N]], eval("nonconservativeProduct", average_y, delta_y, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, :, :] = output[0:NumberOfUnknowns, :, :] + 0.5 * NV * dt / h(1) * (-nonconservative_y[:, 0:NV, 0:NV] - nonconservative_y[:, 0:NV, 1:NV+1])


## Explicit Euler time step with a Rusanov flux over a 3d patch.
def timeStepWithRusanov3D(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    t = DataBlock([[0, N]], "patchData.t", "double")
    dt = DataBlock([[0, N]], "patchData.dt", "double")
    h = DataBlock([[0, N]], "patchData.cellSize", "tarch::la::Vector<3, double>")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<3, double>")

    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [-H, NV + H], [-H, NV + H], [-H, NV + H], [0, N]], "patchData.QIn")
    output = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NV], [0, NV], [0, NV], [0, N]], "patchData.QOut")
    output = input[:, 0:NV, 0:NV, 0:NV]

    if volumetricSource:
       source = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [0, NV], [0, N]], eval("volumetricSource", input[:, 0:NV, 0:NV, 0:NV], x, t))
       output[0:NumberOfUnknowns, :, :, :] = output[0:NumberOfUnknowns, :, :, :] + dt * source

    # As in 2d: patch volumes plus the one halo layer across each face along
    # the normal only, never the edges and corners of the halo
    maxEigenvaluesX = DataBlock([[0, 1], [-1, NV + 1], [0, NV], [0, NV], [0, N]], eval("maxEigenvalue", input[:, -1:NV+1, 0:NV, 0:NV], x, 0))
    maxEigenvaluesY = DataBlock([[0, 1], [0, NV], [-1, NV + 1], [0, NV], [0, N]], eval("maxEigenvalue", input[:, 0:NV, -1:NV+1, 0:NV], x, 1))
    maxEigenvaluesZ = DataBlock([[0, 1], [0, NV], [0, NV], [-1, NV + 1], [0, N]], eval("maxEigenvalue", input[:, 0:NV, 0:NV, -1:NV+1], x, 2))
    
    lambdaLeft = max(maxEigenvaluesX[0:1, -1:NV-1, 0:NV, 0:NV], maxEigenvaluesX[0:1, 0:NV, 0:NV, 0:NV])
    lambdaRight = max(maxEigenvaluesX[0:1, 0:NV, 0:NV, 0:NV], maxEigenvaluesX[0:1, 1:NV + 1, 0:NV, 0:NV])
    lambdaBottom = max(maxEigenvaluesY[0:1, 0:NV, -1:NV-1, 0:NV], maxEigenvaluesY[0:1, 0:NV, 0:NV, 0:NV])
    lambdaTop = max(maxEigenvaluesY[0:1, 0:NV, 0:NV, 0:NV], maxEigenvaluesY[0:1, 0:NV, 1:NV + 1, 0:NV])
    lambdaNear = max(maxEigenvaluesZ[0:1, 0:NV, 0:NV, -1:NV-1], maxEigenvaluesZ[0:1, 0:NV, 0:NV, 0:NV])
    lambdaFar = max(maxEigenvaluesZ[0:1, 0:NV, 0:NV, 0:NV], maxEigenvaluesZ[0:1, 0:NV, 0:NV, 1:NV + 1])

    fluxLeft = -1.0 * lambdaLeft * (input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:NV] - input[0:NumberOfUnknowns, -1:NV-1, 0:NV, 0:NV])
    fluxRight = -1.0 * lambdaRight * (input[0:NumberOfUnknowns, 1:NV+1, 0:NV, 0:NV] - input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:NV])
    output[0:NumberOfUnknowns, :, :, :] = output[0:NumberOfUnknowns, :, :, :] + dt / h(0) * (fluxLeft - fluxRight) * NV * 0.5

    fluxBottom = -1.0 * lambdaBottom * (input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:NV] - input[0:NumberOfUnknowns, 0:NV, -1:NV-1, 0:NV])
    fluxTop = -1.0 * lambdaTop * (input[0:NumberOfUnknowns, 0:NV, 1:NV+1, 0:NV] - input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:NV])
    output[0:NumberOfUnknowns, :, :, :] = output[0:NumberOfUnknowns, :, :, :] + dt / h(1) * (fluxBottom - fluxTop) * NV * 0.5

    fluxNear = -1.0 * lambdaNear * (input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:NV] - input[0:NumberOfUnknowns, 0:NV, 0:NV, -1:NV-1])
    fluxFar = -1.0 * lambdaFar * (input[0:NumberOfUnknowns, 0:NV, 0:NV, 1:NV+1] - input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:NV])
    output[0:NumberOfUnknowns, :, :, :] = output[0:NumberOfUnknowns, :, :, :] + dt / h(2) * (fluxNear - fluxFar) * NV * 0.5

    if flux:
      flux_x = DataBlock([[0, NumberOfUnknowns], [-H, NV + H], [0, NV], [0, NV], [0, N]], eval("flux", input[:, -1:NV+1, 0:NV, 0:NV], x, 0))
      flux_y = DataBlock([[0, NumberOfUnknowns], [0, NV], [-H, NV + H], [0, NV], [0, N]], eval("flux", input[:, 0:NV, -1:NV+1, 0:NV], x, 1))
      flux_z = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [-H, NV + H], [0, N]], eval("flux", input[:, 0:NV, 0:NV, -1:NV+1], x, 2))
      output[0:NumberOfUnknowns, :, :, :] = output[0:NumberOfUnknowns, :, :, :] + 0.5 * NV * dt / h(0) *  (flux_x[:, -1:NV-1, 0:NV, 0:NV] - flux_x[:, 1:NV+1, 0:NV, 0:NV])
      output[0:NumberOfUnknowns, :, :, :] = output[0:NumberOfUnknowns, :, :, :] + 0.5 * NV * dt / h(1) * (flux_y[:, 0:NV, -1:NV-1, 0:NV] - flux_y[:, 0:NV, 1:NV+1, 0:NV])
      output[0:NumberOfUnknowns, :, :, :] = output[0:NumberOfUnknowns, :, :, :] + 0.5 * NV * dt / h(2) * (flux_z[:, 0:NV, 0:NV, -1:NV-1] - flux_z[:, 0:NV, 0:NV, 1:NV+1])

    if nonconservativeProduct:
       average_x = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, -1:NV, 0:NV, 0:NV] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV + 1, 0:NV, 0:NV])
       delta_x = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV + H, 0:NV, 0:NV] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, -H:NV, 0:NV, 0:NV]
       nonconservative_x = DataBlock([[0, NumberOfUnknowns], [0, NV + H], [0, NV], [0, NV], [0, N]], eval("nonconservativeProduct", average_x, delta_x, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, :, :, :] = output[0:NumberOfUnknowns, :, :, :] + 0.5 * NV * dt / h(0) * (-nonconservative_x[:, 0:NV, 0:NV, 0:NV] - nonconservative_x[:, 1:NV+1, 0:NV, 0:NV])
       
       average_y = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, -1:NV, 0:NV] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV + 1, 0:NV])
       delta_y = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV + H, 0:NV] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, -H:NV, 0:NV]
       nonconservative_y = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV + H], [0, NV], [0, N]], eval("nonconservativeProduct", average_y, delta_y, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, :, :, :] = output[0:NumberOfUnknowns, :, :, :] + 0.5 * NV * dt / h(1) * (-nonconservative_y[:, 0:NV, 0:NV, 0:NV] - nonconservative_y[:, 0:NV, 1:NV+1, 0:NV])

       average_z = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, -1:NV] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, 0:NV + 1])
       delta_z = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, 0:NV + H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, -H:NV]
       nonconservative_z = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [0, NV + H], [0, N]], eval("nonconservativeProduct", average_z, delta_z, x, h, t, dt, 2))
       output[0:NumberOfUnknowns, :, :, :] = output[0:NumberOfUnknowns, :, :, :] + 0.5 * NV * dt / h(2) * (-nonconservative_z[:, 0:NV, 0:NV, 0:NV] - nonconservative_z[:, 0:NV, 0:NV, 1:NV+1])


## Maximum eigenvalue over a 2d patch without halo (patchData.QIn) into patchData.maxEigenvalue
def maxEigenvalueOverPatch2D(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<2, double>")
    maxEigenvalues = DataBlock([[0, N]], "patchData.maxEigenvalue", "double")
    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NV], [0, NV], [0, N]], "patchData.QIn")

    if maxEigenvalue:
      maxEigenvaluesX = DataBlock([[0, 1], [0, NV], [0, NV], [0, N]], eval("maxEigenvalue", input, x, 0))
      maxEigenvaluesY = DataBlock([[0, 1], [0, NV], [0, NV], [0, N]], eval("maxEigenvalue", input, x, 1))
      maxEigenvaluesPerVolume = max(maxEigenvaluesX, maxEigenvaluesY)
      maxEigenvalues = max(maxEigenvaluesPerVolume)


## Maximum eigenvalue over a 3d patch without halo (patchData.QIn) into patchData.maxEigenvalue
def maxEigenvalueOverPatch3D(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<3, double>")
    maxEigenvalues = DataBlock([[0, N]], "patchData.maxEigenvalue", "double")
    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NV], [0, NV], [0, NV], [0, N]], "patchData.QIn")

    if maxEigenvalue:
      maxEigenvaluesX = DataBlock([[0, 1], [0, NV], [0, NV], [0, NV], [0, N]], eval("maxEigenvalue", input, x, 0))
      maxEigenvaluesY = DataBlock([[0, 1], [0, NV], [0, NV], [0, NV], [0, N]], eval("maxEigenvalue", input, x, 1))
      maxEigenvaluesZ = DataBlock([[0, 1], [0, NV], [0, NV], [0, NV], [0, N]], eval("maxEigenvalue", input, x, 2))
      maxEigenvaluesXY = max(maxEigenvaluesX, maxEigenvaluesY)
      maxEigenvaluesPerVolume = max(maxEigenvaluesXY, maxEigenvaluesZ)
      maxEigenvalues = max(maxEigenvaluesPerVolume)


class RusanovWithExplicitEuler(Solver.NumericalScheme):
    """!

    Finite Volumes with a Rusanov Riemann solver and explicit Euler time
    stepping, realised over a block-structured mesh.

    """

    def __init__(
        self,
        solver: Solver,
        patch_size: int,
        overlap: int = 1,
        cfl_constant: float = 0.9,
        interpolation: "toolbox.blockstructured.api.InterpolateHangingFaces.Scheme" = toolbox.blockstructured.api.InterpolateHangingFaces.Scheme.Linear,
        restriction: "toolbox.blockstructured.api.RestrictHangingFaces.Scheme" = toolbox.blockstructured.api.RestrictHangingFaces.Scheme.Averaging,
        min_volume_size: float = None,
        max_volume_size: float = None,
    ):
        """!

        The constructor memorises the attributes. However, it does not do this
        within this class, but it adds the important properties straight to the
        solver, so anybody else working against this solver knows about the
        properties introduced by the numerical scheme.

        @see @ref __create_data_structures()

        patch_size: int
          Number of Finite Volumes per axis within one patch (i.e. within
          one mesh cell).

        overlap: int
          Number of Finite Volumes the halo layer reconstructed around a
          patch - from the neighbouring patches' faces - is wide.

        cfl_constant: float
          Safety factor the admissible time step size (@see
          exahype2.api.fv.actionsets.ComputeGlobalCFLCondition,
          which this is passed straight through to) is scaled down by,
          relative to the theoretical CFL=1 stability limit
          h / (Dimensions * lambda_max). 0.9 is a
          conservative default for an explicit-Euler Finite Volume scheme;
          this codebase has no prior precedent value to match (grep for
          "cfl" outside this class finds only exaparty's unrelated SPH
          solvers, which default to 0.1).

        interpolation: toolbox.blockstructured.api.InterpolateHangingFaces.Scheme
          How the scheme initialises the halo of a fine patch next to a
          coarser patch on an adaptive mesh. Whenever the time step
          reconstructs the halo of a fine cell, it interpolates the data of
          the adjacent coarse face onto the fine cell's hanging face
          (toolbox.blockstructured.api.InterpolateHangingFaces). The
          default is linear interpolation.

        restriction: toolbox.blockstructured.api.RestrictHangingFaces.Scheme
          How the scheme hands the data of fine patches over to an adjacent
          coarser patch. Whenever the fine cells project their data onto
          their faces, the scheme restricts the data of the hanging faces
          onto the coarse face (toolbox.blockstructured.api.RestrictHangingFaces).
          The default is averaging.

        min_volume_size: float
          The time step asks for a refinement of a cell only if its volumes
          are at least of this size (@ref exahype2.api.blockstructured.DynamicMeshRefinement).
          None (the default) does not bound the refinement.

        max_volume_size: float
          The time step asks for an erase, i.e. coarsening, of a cell only
          if the volumes of the coarser cell would be strictly smaller than
          this size. None (the default) does not bound the coarsening.

        On a regular mesh, there are no hanging faces, and the two schemes
        are never used. The fluxes along a face between a coarse and a fine
        patch are not corrected, i.e. the scheme is not exactly conservative
        on adaptive meshes.

        """
        super().__init__(solver)
        assert (
            min_volume_size is None
            or max_volume_size is None
            or min_volume_size <= max_volume_size
        ), f"min_volume_size={min_volume_size} has to be smaller than or equal to max_volume_size={max_volume_size}"
        self.min_volume_size = min_volume_size
        self.max_volume_size = max_volume_size
        self.interpolation = interpolation
        self.restriction = restriction
        # patch_size and halo_size must be set for proper solver generation
        self.patch_size = patch_size
        self.halo_size = overlap
        self.overlap = overlap
        # Set in construct_output(): the generated kernels live in the project
        # namespace, as their names are solver-specific
        self.kernel_namespace = None
        self.max_eigenvalue_kernel_call = None
        self.cfl_constant = cfl_constant
        self._generator = RusanovWithExplicitEulerGenerator(solver, [(Target.Serial, True, OutputLanguage.CPP, MemoryLayout.AoS_AllNested)])
        self.__create_data_structures()

    @property
    def kernel_function_name(self) -> str:
        """!
        Name of the generated time step kernel. It carries the solver name,
        as do the kernels/<SolverName>_rusanov*.h files holding it, so two
        solvers' kernels do not clash.
        """
        return self._solver.name + "_timeStepWithRusanov"

    @property
    def max_eigenvalue_kernel_function_name(self) -> str:
        """!
        Name of the generated kernel which determines the maximum eigenvalue
        over a patch. It is a kernel of its own, i.e. the time step kernel
        does not return any eigenvalue (@ref exahype2.api.fv.actionsets.ComputeMaxEigenvalue).
        """
        return self._solver.name + "_maxEigenvalueOverPatch"

    def __create_data_structures(self):
        """!

        Create the mesh data structures (a Finite Volume patch per cell,
        plus the face-based overlap patches used to reconstruct its halo)
        this numerical scheme requires.

        Ported from the legacy exahype2.solvers.fv.FV.create_data_structures():
        one @ref peano4.api.datamodel.Patch of patch_size^d Finite Volumes
        per mesh cell (_patch), plus a halo-augmented copy of it
        (_patch_with_halo, (patch_size+2*overlap)^d, @see
        toolbox.blockstructured.api.ReconstructPatchWithHalo) and two overlap
        patches of width 2*overlap attached to the mesh faces:

        - _patch_overlap_current_time_step: the persistent halo data
          toolbox.blockstructured.api.ReconstructPatchWithHalo reads from to
          rebuild _patch_with_halo (@see construct_output(), which hands it
          _patch/_patch_overlap_current_time_step/_patch_with_halo directly
          rather than the whole solver).
        - _patch_overlap_update: the projection of the just-updated _patch
          accumulated by the current sweep (@see
          exahype2.api.fv.actionsets.ProjectPatchOntoFaces), rolled over
          into _patch_overlap_current_time_step by
          exahype2.api.fv.actionsets.RollOverUpdatedFace once the update is
          complete.
        - _max_eigenvalue: one double per cell, named after the solver
          followed by MaxEigenvalue (e.g. EulerFVSmoothMaxEigenvalue),
          which holds the maximum eigenvalue of
          the patch after its last update
          (@see exahype2.api.fv.actionsets.ComputeMaxEigenvalue).
          exahype2.api.fv.actionsets.ReduceMaxEigenvalue reduces these
          values into the solver, and
          exahype2.api.fv.actionsets.ComputeGlobalCFLCondition derives the
          next time step size from the outcome. It is modelled via DaStGen2
          as an array of cardinality one. The array is a
          peano4.api.dastgen2.Peano4SmartPointerDoubleArray, i.e. the value
          resides on the heap.

        All five are assigned onto self._solver (the shared
        exahype2.api.Solver instance both this numerical scheme and its
        surrounding solver's Solver.GridConstruction/Solver.Plotter
        counterparts hold a reference to as self._solver), not onto self,
        since the remaining FV action sets that need them - @see
        exahype2.api.fv.actionsets.ApplyDirichletBoundaryConditions,
        exahype2.api.fv.actionsets.UpdateCell,
        exahype2.api.fv.actionsets.ComputeGlobalCFLCondition - are handed that
        shared solver and look the patches up as solver._patch /
        solver._patch_overlap_*.

        Storage is hardwired to smart pointers (peano4.api.datamodel's
        PatchToDoubleArrayWithSmartPointer), matching the legacy default
        (Storage.SmartPointers for both cell and face data) - the legacy
        Storage enum's other two backends (CallStack, Heap) are not
        exposed here.

        Two pieces of the legacy method are deliberately not ported yet:
        - The cell/face label creation
          (exahype2.grid.create_cell_label()/create_face_label()) - that
          module has not been ported to the new API.
        - The load/store/compute guards
          (FV._provide_cell_data_to_compute_kernels_default_guard() and
          friends) - they are keyed off the legacy SolverState state
          machine, which the task-graph architecture has no equivalent of:
          a task only ever runs within its own phase's task graph in the
          first place, so "are we still constructing the grid" does not
          need to be re-checked at runtime the same way.

        """
        total_unknowns = self._solver.unknowns
        identifier = self._solver._unknown_identifier()

        self._solver._patch = peano4.api.datamodel.Patch(
            (self.patch_size, self.patch_size, self.patch_size),
            total_unknowns,
            identifier,
        )
        self._solver._patch_with_halo = peano4.api.datamodel.Patch(
            (self.patch_size+2*self.overlap, self.patch_size+2*self.overlap, self.patch_size+2*self.overlap),
            total_unknowns,
            identifier + "WithHalo",
        )
        self._solver._patch_overlap_current_time_step = peano4.api.datamodel.Patch(
            (2 * self.overlap, self.patch_size, self.patch_size),
            total_unknowns,
            identifier + "CurrentTimeStep",
        )
        self._solver._patch_overlap_update = peano4.api.datamodel.Patch(
            (2 * self.overlap, self.patch_size, self.patch_size),
            total_unknowns,
            identifier + "Update",
        )
        self._solver._max_eigenvalue = peano4.api.datamodel.DaStGen2(self._solver.name + "MaxEigenvalue")
        self._solver._max_eigenvalue.data.add_attribute(
            peano4.api.dastgen2.Peano4SmartPointerDoubleArray("maxEigenvalue", 1)
        )
        add_max_eigenvalue_reduction(self._solver)


    def __add_data_to_repository(self, peano_project):
        """!
        
        Consult @ref page_exahype_programmers_guide_new_solver for some information
        how this routine fits to the typical workflow when we introduce a new
        solver.
        
        """
        peano4.api.Logger().log("""
Patch data
----------
{}
Patch overlap data
------------------
{}
{}""".format( str(self._solver._patch), str(self._solver._patch_overlap_current_time_step), str(self._solver._patch_overlap_update) ) )

        peano_project.datamodel.add_cell(self._solver._patch)
        peano_project.datamodel.add_cell(self._solver._patch_with_halo)
        peano_project.datamodel.add_face(self._solver._patch_overlap_current_time_step)
        peano_project.datamodel.add_face(self._solver._patch_overlap_update)
        peano_project.datamodel.add_cell(self._solver._max_eigenvalue)
        register_auxiliary_variables(self._solver, peano_project, self.patch_size, self.overlap)

        # The task graph compiler only uses, loads and stores data where the
        # solver's grid selector holds, too (@see peano4.api.datamodel.DoF).
        for patch in (
            self._solver._patch,
            self._solver._patch_with_halo,
            self._solver._patch_overlap_current_time_step,
            self._solver._patch_overlap_update,
            self._solver._max_eigenvalue,
        ):
            patch.grid_selector = self._solver.grid_selector

        self._solver._patch.generator =peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
            self._solver._patch, "double"
        )
        self._solver._patch_with_halo.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
            self._solver._patch_with_halo, "double"
        )
        self._solver._patch_overlap_current_time_step.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
            self._solver._patch_overlap_current_time_step, "double"
        )
        self._solver._patch_overlap_update.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
            self._solver._patch_overlap_update, "double"
        )
        
        self._solver._patch_overlap_current_time_step.generator.merge_method_definition = (
            toolbox.blockstructured.api.get_face_merge_implementation(
                self._solver._patch_overlap_current_time_step
            )
        )
        self._solver._patch_overlap_current_time_step.generator.includes += """
#include "peano4/utils/Loop.h"
"""

        # Only faces selected by the solver's grid selector hold valid data,
        # so only those are exchanged.
        self._solver._patch_overlap_current_time_step.generator.send_condition = self._solver.grid_selector.createFaceGuard()
        self._solver._patch_overlap_current_time_step.generator.receive_and_merge_condition = self._solver.grid_selector.createFaceGuard()

        self._solver._patch_overlap_update.generator.send_condition = "false"
        self._solver._patch_overlap_update.generator.receive_and_merge_condition = "false"

    def create_default_mesh_adaptation(self) -> Solver.MeshAdaptation:
        """!

        The Finite Volume schemes adapt the mesh and balance the load
        through exahype2.api.fv.AdaptMeshAndBalanceLoad by default

        """
        from .AdaptMeshAndBalanceLoad import AdaptMeshAndBalanceLoad

        return AdaptMeshAndBalanceLoad(self._solver)

    def construct_output(self,
                         exahype2_project,
                         peano_project,
                         time_step,
                         add_global_tasks=True):
        """!

        """
        peano4.api.Logger().log("Create time stepping task graph for Finite Volume scheme over block-structured meshes with Rusanov flux")

        self.__add_data_to_repository(peano_project)
        self.kernel_namespace = peano_project.namespace
        self._generator.construct_output(exahype2_project, peano_project, time_step, add_global_tasks)

    def create_restriction_of_hanging_faces(self) -> "toolbox.blockstructured.api.RestrictHangingFaces":
        """!

        Action set which restricts the projected fine data from the hanging
        faces onto the coarse faces

        Every task which projects the solution onto the faces has to host
        this action set, so the faces between coarse and fine patches hold
        consistent data afterwards. Besides the time step, this is the
        projection at the end of the setup
        (@ref exahype2.api.fv.SetupSimulationOverRegularGrid). The task has
        to write all of _patch_overlap_update, as averaging clears it first.

        """
        return toolbox.blockstructured.api.RestrictHangingFaces(
            self._solver._patch_overlap_update,
            scheme=self.restriction,
            clear_guard=self._solver.grid_selector.createFaceGuard(),
            additional_includes=self._solver._get_default_includes() + self._solver.user_solver_includes,
        )


class RusanovWithExplicitEulerGenerator(Solver.NumericalScheme.NumericalSchemeGenerator):
    def construct_output(self, exahype2_project, peano_project, time_step, add_global_tasks=True):
        # Construct the call first, as it validates the PDE terms' variants
        # Although we may generate multiple definitions for the purpose of benchmarking, we
        # only need one kernel call
        self._solver.compute_kernel_call = self.__construct_kernel_calls()[0]
        # The eigenvalue kernel always runs on the host
        self._solver.numerical_scheme.max_eigenvalue_kernel_call = self.__construct_kernel_call(
            Target.Serial,
            self._configurations[0][1],
            self._solver.numerical_scheme.max_eigenvalue_kernel_function_name,
            MemoryLayout.AoS_AllNested,
        )

        self.__create_time_step_functions(exahype2_project, peano_project)
        
        reconstruction_task         = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Reconstruct", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        update_cell_task            = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Update cell", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        project_on_faces_task       = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Project on faces", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        compute_eigenvalue_task     = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Compute max eigenvalue", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        reduce_eigenvalue_task      = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Reduce max eigenvalue", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        compute_cfl_condition_task  = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Compute global CFL condition", entity=peano4.api.taskgraph.Task.GridEntity.GLOBAL_END_OF_STEP, may_spawn_background_tasks=False)
        
        time_step.add_task(reconstruction_task)
        time_step.add_task(update_cell_task)
        time_step.add_task(compute_eigenvalue_task)
        time_step.add_task(reduce_eigenvalue_task)
        time_step.add_task(project_on_faces_task)
        time_step.add_task(compute_cfl_condition_task)

        ## Build the reconstruction tasks befilling the patch_with_halo, which is logically temporary
        reconstruction_task.use_cell(self._solver._patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        reconstruction_task.use_cell(self._solver._patch_with_halo, peano4.api.taskgraph.Task.DataAccess.WRITE)
        ## Sommerfeld boundary conditions read the old face data within the roll
        ## over, i.e. it has to survive the reconstruction
        reconstruction_task.use_face(
            self._solver._patch_overlap_current_time_step,
            get_access_to_persistent_face_data(self._solver, exahype2_project, peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD),
        )
        ## On adaptive meshes, the reconstruction interpolates the coarse face
        ## data onto the hanging faces of fine cells itself (see
        ## src/exahype2/api/adrs/0001-interpolation-and-restriction-ride-along-with-face-users.md)
        reconstruction_task.add_action_set(
            toolbox.blockstructured.api.InterpolateHangingFaces(
                self._solver._patch_overlap_current_time_step,
                scheme=self._solver.numerical_scheme.interpolation,
                additional_includes=self._solver._get_default_includes() + self._solver.user_solver_includes,
            )
        )
        reconstruction_task.add_action_set(
            toolbox.blockstructured.api.ReconstructPatchWithHalo(
                patch=self._solver._patch,
                patch_overlap=self._solver._patch_overlap_current_time_step,
                reconstructed_patch=self._solver._patch_with_halo,
                guard=self._solver.grid_selector.createCellGuard(),
                add_assertions_to_halo_exchange=False,
                additional_includes=self._solver._get_default_includes() + self._solver.user_solver_includes,
            )
        )
        ## Set the ghost volumes along the domain boundary before the reconstruction reads them.
        ## No task is added if all axes are periodic or for Sommerfeld boundary
        ## conditions, which advance the ghost volumes within the roll over
        ApplyDirichletBoundaryConditions.add_task_to_time_step(self._solver, exahype2_project, time_step, reconstruction_task)

        ## The halos of the auxiliary variables
        auxiliary_reconstruction_tasks = add_reconstruction_tasks_of_auxiliary_variables(
            self._solver, time_step, self._solver.numerical_scheme.interpolation
        )

        ## Do the actual time step update
        update_cell_task.add_dependency(reconstruction_task)
        for task in auxiliary_reconstruction_tasks:
            update_cell_task.add_dependency(task)
        use_halos_of_auxiliary_variables(update_cell_task, self._solver)
        update_cell_task.use_cell(self._solver._patch_with_halo, peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD)
        update_cell_task.use_cell(self._solver._patch,           peano4.api.taskgraph.Task.DataAccess.WRITE)
        if self._configurations[0][0] == Target.Accelerator:
            update_cell_task.add_action_set(
                UpdateCellGPU(
                    self._solver,
                    guard=self._solver.grid_selector.createCellGuard(),
                )
            )
        else:
            update_cell_task.add_action_set(
                UpdateCell(
                    self._solver,
                    guard=self._solver.grid_selector.createCellGuard(),
                )
            )
        update_cell_task.tags.add(Solver.SolutionUpdateTaskTag)

        ## Determine the maximum eigenvalue of the updated patches, store it
        ## per cell and reduce it over all cells. The projection onto the faces
        ## depends on the update only, i.e. it runs in parallel to the
        ## eigenvalue and its reduction. The update, the eigenvalue, the
        ## reduction and the projection fold into one mesh traversal, whereas the
        ## projection writes the face data which the reconstruction reads at the
        ## beginning of that traversal
        compute_eigenvalue_task.add_dependency(update_cell_task)
        compute_eigenvalue_task.use_cell(self._solver._patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        compute_eigenvalue_task.use_cell(self._solver._max_eigenvalue, peano4.api.taskgraph.Task.DataAccess.WRITE)
        for auxiliary in self._solver.auxiliary_variables:
            compute_eigenvalue_task.use_cell(auxiliary.patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        compute_eigenvalue_task.add_action_set(
            ComputeMaxEigenvalue(
                self._solver,
                guard=self._solver.grid_selector.createCellGuard(),
            )
        )
        reduce_eigenvalue_task.add_dependency(compute_eigenvalue_task)
        reduce_eigenvalue_task.use_cell(self._solver._max_eigenvalue, peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD)
        reduce_eigenvalue_task.add_action_set(
            ReduceMaxEigenvalue(
                self._solver,
                guard=self._solver.grid_selector.createCellGuard(),
            )
        )

        ## Derive the next time step size from the reduced eigenvalue. This
        ## acts in unprepareTraversal() at the end of the time step only
        compute_cfl_condition_task.add_dependency(reduce_eigenvalue_task)
        compute_cfl_condition_task.add_action_set(
            ComputeGlobalCFLCondition(
                self._solver,
                self._solver.numerical_scheme.cfl_constant,
                self._solver.numerical_scheme.patch_size,
            )
        )
        
        ## Next project the new data onto the faces. Also roll over the solution
        project_on_faces_task.add_dependency(update_cell_task)
        project_on_faces_task.use_cell(self._solver._patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        project_on_faces_task.use_face(self._solver._patch_overlap_update, peano4.api.taskgraph.Task.DataAccess.TEMPORARY)
        project_on_faces_task.use_face(
            self._solver._patch_overlap_current_time_step,
            get_access_to_persistent_face_data(self._solver, exahype2_project, peano4.api.taskgraph.Task.DataAccess.WRITE),
        )
        project_on_faces_task.add_action_set(
            ProjectPatchOntoFaces(
                self._solver._patch,
                self._solver._patch_overlap_update,
                guard=self._solver.grid_selector.createCellGuard(),
                includes=self._solver._get_default_includes() + self._solver.user_solver_includes,
            )
        )
        ## On adaptive meshes, the projection also restricts the fine data from
        ## the hanging faces onto the coarse faces (see
        ## src/exahype2/api/adrs/0001-interpolation-and-restriction-ride-along-with-face-users.md).
        ## Peano destroys a hanging face right after the fine cell has
        ## projected onto it, and touches the coarse face for the last time
        ## after that, i.e. the coarse face is final when the roll over copies it.
        ## The projection declares the face data of the current time step, so
        ## the compiler does not fuse it into the traversal of the reconstruction.
        project_on_faces_task.add_action_set(self._solver.numerical_scheme.create_restriction_of_hanging_faces())
        ## The roll over is part of the projection, too. The face data of the
        ## projection (the update) is TEMPORARY, i.e. it lives within one mesh
        ## traversal only, and the roll over copies it. A task of its own
        ## would be fused into a later traversal by the producer-consumer
        ## compiler, where it copies uninitialised memory
        ## (see src/exahype2/api/fv/adrs/0003-roll-over-is-part-of-the-projection.md).
        ## Sommerfeld boundary conditions advance the ghost volumes within
        ## the roll over (@ref exahype2.api.blockstructured.BoundaryConditions)
        project_on_faces_task.add_action_set(
            RollOverUpdatedFaceWithSommerfeldBoundaryConditions.create_roll_over(
                self._solver,
                exahype2_project,
                guard=self._solver.grid_selector.createFaceGuard(),
                includes=self._solver._get_default_includes() + self._solver.user_solver_includes,
                time_step_size=f"repositories::{self._solver.get_name_of_global_instance()}.getMinTimeStepSize()",
            )
        )
        ## The roll over of Sommerfeld boundary conditions reads the current
        ## time step size, which the next task replaces. It therefore never
        ## runs before the roll over has terminated
        ## (see src/exahype2/api/fv/adrs/0002-time-step-size-after-roll-over.md)
        compute_cfl_condition_task.add_dependency(project_on_faces_task)

        # The load balancing is switched off throughout the time step, and
        # switched on again only at its very end, so it may then decide on
        # splits. Peano carries these splits out in the subsequent mesh sweep.
        if add_global_tasks:
            self._solver.numerical_scheme.wrap_task_graph_in_load_balancing_and_amr_tasks(time_step)

        if peano4.api.Logger().is_visualisation:
            time_step.visualise(title="Time stepping task graph for {}{}".format(self._solver.name, self._solver.task_graph_title_suffix))
        if peano4.api.Logger().is_text:
            time_step.log_as_text(title="Time stepping task graph for {}{}".format(self._solver.name, self._solver.task_graph_title_suffix))

    def __create_time_step_functions(self, exahype2_project, peano4_project):
        """!
        
        Calling this function is delayed until all PDE term information is provided.
        In this way we can avoid generating code that will not be used. The following
        information is required to translate a Python function using dslhype:
        - template_parameters: the variables that are used within the Python function
            to represent external information
        - functor_parameters: the PDE terms that can be used by the solver
        - function: the Python function itself

        TODO: Currently the MLIR types are passed manually here, this should ideally
        be inferred

        """

        #template_parameters = [
        #    dslhype.SyntaxTree.Argument("NV", dslhype.SyntaxTree.TInteger()),
        #    dslhype.SyntaxTree.Argument("H", dslhype.SyntaxTree.TInteger()),
        #    dslhype.SyntaxTree.Argument("NumberOfUnknowns", dslhype.SyntaxTree.TInteger()),
        #    dslhype.SyntaxTree.Argument("NumberOfAuxiliaryVariables", dslhype.SyntaxTree.TInteger())
        #]
        #
        #functor_parameters = [
        #    dslhype.SyntaxTree.Argument("flux", dslhype.SyntaxTree.TCustom("const ::exahype2::fv::rusanov::FluxFunctor&"), "(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()"),
        #    dslhype.SyntaxTree.Argument("nonconservativeProduct", dslhype.SyntaxTree.TCustom("const ::exahype2::fv::rusanov::NonconservativeProductFunctor&"), "(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()"),
        #    dslhype.SyntaxTree.Argument("volumetricSource", dslhype.SyntaxTree.TCustom("const ::exahype2::fv::rusanov::VolumetricSourceFunctor&"), "(!llvm.ptr, !llvm.ptr, f64, !llvm.ptr) -> ()"),
        #    dslhype.SyntaxTree.Argument("maxEigenvalue", dslhype.SyntaxTree.TCustom("const ::exahype2::fv::rusanov::MaxEigenvalueFunctor&"), "(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()")
        #]
        dimensions = exahype2_project._dimensions
        if dimensions == 2:
            time_step_dsl_function = timeStepWithRusanov2D
        else:
            time_step_dsl_function = timeStepWithRusanov3D
        
        output_directory = peano4_project.directory

        required_includes = f"""
#include "Abstract{self._solver.name}.h"
#include "repositories/SolverRepository.h"
#include "exahype2/Solver.h"
#include "exahype2/CellData.h"
#include "exahype2/VolumeIndex.h"
#include "exahype2/fv/PatchUtils.h"
#include "exahype2/fv/rusanov/Functors.h"
#include "peano4/utils/Loop.h"
#include "tarch/timing/Measurement.h"
#include "tarch/timing/Watch.h"
"""
        stateless_kernel_generated=False
        non_stateless_kernel_generated=False
        mlir_kernel_generated=False
        for (_, stateless, language, _) in self._configurations:
            if stateless:
                stateless_kernel_generated = True
            else:
                non_stateless_kernel_generated = True
            if language == OutputLanguage.MLIR:
                mlir_kernel_generated = True
        
        kernels_directory = os.path.join(output_directory, "kernels")
        os.makedirs(kernels_directory, exist_ok=True)
        #file_name = 
        #for file_name in [f"{self._solver.name}_rusanov2d.h", f"{self._solver.name}_rusanov3d.h"]:
        file = open(os.path.join(kernels_directory, f"{self._solver.name}_rusanov{dimensions}d.h"), "w")
        file.write(f"""#pragma once
{required_includes}\n""")
        file.close()
        
        if stateless_kernel_generated:
            #for file_name in [f"{self._solver.name}_rusanov2d.cpph", f"{self._solver.name}_rusanov3d.cpph"]:
            file = open(os.path.join(kernels_directory, f"{self._solver.name}_rusanov{dimensions}d.cpph"), "w")
            file.write(f"""#include "{self._solver.name}_rusanov{dimensions}d.h"\n""")
            file.close()
        
        if non_stateless_kernel_generated or mlir_kernel_generated:
            #for file_name in [f"{self._solver.name}_rusanov2d.cpp", f"{self._solver.name}_rusanov3d.cpp"]:
            file = open(os.path.join(kernels_directory, f"{self._solver.name}_rusanov{dimensions}d.cpp"), "w")
            file.write(f"""#include "{self._solver.name}_rusanov{dimensions}d.h"\n""")
            file.close()

        external_variables = {
            dslhype.Argument(dslhype.Name("NV"), dslhype.TInteger()): self._solver.numerical_scheme.patch_size,
            dslhype.Argument(dslhype.Name("H"), dslhype.TInteger()): self._solver.numerical_scheme.overlap,
            dslhype.Argument(dslhype.Name("NumberOfUnknowns"), dslhype.TInteger()): self._solver.unknowns,
            dslhype.Argument(dslhype.Name("NumberOfAuxiliaryVariables"), dslhype.TInteger()): self._solver.number_of_auxiliary_variables
        }

        function_map = {
            dslhype.Name("volumetricSource"): dslhype.TFunction("const ::exahype2::fv::rusanov::VolumetricSourceFunctor&", OrderedDict({
                dslhype.Name("Q"): dslhype.TArray(dslhype.TFloat(), 1), 
                dslhype.Name("x"): dslhype.TVector(dslhype.TFloat(), dimensions), 
                dslhype.Name("t"): dslhype.TFloat(), 
                dslhype.Name("F"): dslhype.TArray(dslhype.TFloat(), 1)})),
            dslhype.Name("flux"): dslhype.TFunction("const ::exahype2::fv::rusanov::FluxFunctor&", OrderedDict({
                dslhype.Name("Q"): dslhype.TArray(dslhype.TFloat(), 1), 
                dslhype.Name("x"): dslhype.TVector(dslhype.TFloat(), dimensions), 
                dslhype.Name("normal"): dslhype.TInteger(), 
                dslhype.Name("F"): dslhype.TArray(dslhype.TFloat(), 1)})),
            dslhype.Name("nonconservativeProduct"): dslhype.TFunction("const ::exahype2::fv::rusanov::NonconservativeProductFunctor&", OrderedDict({
                dslhype.Name("Q"): dslhype.TArray(dslhype.TFloat(), 1), 
                dslhype.Name("deltaQ"): dslhype.TArray(dslhype.TFloat(), 1), 
                dslhype.Name("x"): dslhype.TVector(dslhype.TFloat(), dimensions), 
                dslhype.Name("h"): dslhype.TVector(dslhype.TFloat(), dimensions), 
                dslhype.Name("t"): dslhype.TFloat(), 
                dslhype.Name("dt"): dslhype.TFloat(), 
                dslhype.Name("normal"): dslhype.TInteger(), 
                dslhype.Name("BTimesDeltaQ"): dslhype.TArray(dslhype.TFloat(), 1)})),
            dslhype.Name("maxEigenvalue"): dslhype.TFunction("const ::exahype2::fv::rusanov::MaxEigenvalueFunctor&", OrderedDict({
                dslhype.Name("Q"): dslhype.TArray(dslhype.TFloat(), 1), 
                dslhype.Name("x"): dslhype.TVector(dslhype.TFloat(), dimensions), 
                dslhype.Name("normal"): dslhype.TInteger(), 
                dslhype.Name("E"): dslhype.TArray(dslhype.TFloat(), 1)}))
        }

        if mlir_kernel_generated and stateless_kernel_generated:
            dslhype.generate_mlir_function_call_bridges(
                solver=self._solver,
                function_map=function_map,
                pde_terms=self._solver.pde_terms,
                stateless=True,
                output_file_name=f"{self._solver.name}_rusanov{dimensions}d",
                output_directory=output_directory,)

        if mlir_kernel_generated and non_stateless_kernel_generated:
            dslhype.generate_mlir_function_call_bridges(
                solver=self._solver,
                function_map=function_map,
                pde_terms=self._solver.pde_terms,
                stateless=False,
                output_file_name=f"{self._solver.name}_rusanov{dimensions}d",
                output_directory=output_directory,)

        for (target, stateless, language, memory_layout) in self._configurations:
            dslhype.translate_dslhype_function(
                solver=self._solver,
                function = time_step_dsl_function,
                dimensions=dimensions,
                function_map=function_map,
                external_variables=external_variables,
                pde_terms=self._solver.pde_terms,
                output_file_name=f"{self._solver.name}_rusanov{dimensions}d",
                target=target,
                stateless=stateless,
                memory_layout=memory_layout,
                function_name=self._solver.numerical_scheme.kernel_function_name,
                output_language=language,
                output_directory=output_directory,
            )
        # The maximum eigenvalue is a kernel of its own, which always runs on the host
        for stateless in sorted({stateless for (_, stateless, _, _) in self._configurations}):
            dslhype.translate_dslhype_function(
                solver=self._solver,
                function = maxEigenvalueOverPatch2D if dimensions == 2 else maxEigenvalueOverPatch3D,
                dimensions=dimensions,
                function_map=function_map,
                external_variables=external_variables,
                pde_terms=self._solver.pde_terms,
                output_file_name=f"{self._solver.name}_rusanov{dimensions}d",
                target=Target.Serial,
                stateless=stateless,
                function_name=self._solver.numerical_scheme.max_eigenvalue_kernel_function_name,
                output_language=OutputLanguage.CPP,
                output_directory=output_directory,
            )
        
        #for file_prefix in [f"{self._solver.name}_rusanov2d"]:#, f"{self._solver.name}_rusanov3d"]:
        if stateless_kernel_generated:
            file = open(os.path.join(kernels_directory, f"{self._solver.name}_rusanov{dimensions}d.h"), "a")
            file.write(f"""#include "{self._solver.name}_rusanov{dimensions}d.cpph" """)
            file.close()

        peano4_project.output.makefile.add_h_file(os.path.join(kernels_directory, f"{self._solver.name}_rusanov{dimensions}d.h"))
        if stateless_kernel_generated:
            peano4_project.output.makefile.add_h_file(os.path.join(kernels_directory, f"{self._solver.name}_rusanov{dimensions}d.cpph"))
        if non_stateless_kernel_generated or mlir_kernel_generated:
            peano4_project.output.makefile.add_cpp_file(os.path.join(kernels_directory, f"{self._solver.name}_rusanov{dimensions}d.cpp"))

        for (_, stateless, language, memory_layout) in self._configurations:
            if language == OutputLanguage.MLIR:
                peano4_project.output.makefile.add_mlir_file(os.path.join(kernels_directory, f"{self._solver.name}_rusanov{dimensions}d_{memory_layout.name.lower()}.mlir"), "mlir-opt --pass-pipeline=\"builtin.module(canonicalize,cse,loop-invariant-code-motion,canonicalize,cse,loop-invariant-code-motion,cse,canonicalize,expand-strided-metadata,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,finalize-memref-to-llvm,convert-scf-to-cf,convert-openmp-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)\"")

        file = open(os.path.join(output_directory, "kernels", f"{self._solver.name}_rusanov.h"), "w")
        file.write(f"""#include "{self._solver.name}_rusanov{dimensions}d.h" """)
        file.close()

    def __construct_kernel_calls(self):
        kernel_calls = []
        for (target, stateless, language, memory_layout) in self._configurations:
            kernel_calls.append(self.__construct_kernel_call(target, stateless, language, memory_layout))
        return kernel_calls

    def __construct_kernel_call(self, target: Target, stateless: bool, language: OutputLanguage, memory_layout: MemoryLayout, function_name: str = None):
        """!

        This function constructs the calls to the compute kernels based on the
        application PDE terms and the version chosen.

        @param function_name: str
            Unqualified name of the kernel. Defaults to the time step kernel.

        """
        if function_name is None:
            function_name = self._solver.numerical_scheme.kernel_function_name
        available_terms = ["flux", "nonconservativeProduct", "volumetricSource", "maxEigenvalue"]

        if stateless:
            # Stateless kernels require stateless functions, which are defined in the abstract solver
            return f"""::{"::".join(self._solver.numerical_scheme.kernel_namespace + dslhype.generate_namespace(language, target, memory_layout))}::{function_name}Stateless<
        Abstract{self._solver.name}
    >(
        {"0," if target==Target.Accelerator else ""}
        {"gpuPatchData" if target==Target.Accelerator else "patchData"}
    );"""
        else:
            # Variants per term the kernel takes: {name: {stateless: PDETerm}}
            variants = {}
            for application_term in self._solver.pde_terms:
                if application_term.NAME in available_terms:
                    variants.setdefault(application_term.NAME, {})[application_term.stateless] = application_term

            terms_with_both_variants = [name for name in variants if len(variants[name]) == 2]
            terms_without_stateless_variant = [name for name in variants if True not in variants[name]]
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
                pde_term_calls = []
                for term in available_terms:
                    if term not in variants:
                        continue
                    # A term registered only once is used in either case
                    application_term = variants[term].get(stateless, next(iter(variants[term].values())))
                    # Commas inside template brackets, e.g. Vector<Dimensions, double>,
                    # do not separate parameters
                    arguments = forwarding_arguments(application_term.SIGNATURE, self._solver)
                    # Stateless terms are static and take a dummy tag (@see PDETerm)
                    if application_term.stateless:
                        arguments.append("::exahype2::Solver::Offloadable::Yes")
                    pde_term_calls.append(f"""
    [&](
        {application_term.SIGNATURE}
    )->void {{
        repositories::{self._solver.get_name_of_global_instance()}.{application_term.name}({", ".join(arguments)});
    }}""")
                return f"""::{"::".join(self._solver.numerical_scheme.kernel_namespace + dslhype.generate_namespace(language, target, memory_layout))}::{function_name}(
    patchData,
{",\n".join(pde_term_calls)}
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