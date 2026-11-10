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
from exahype2.api import Target
import exahype2.api.dslhype as dslhype
import dastgen2
 
from ..Solver import Solver
from ..blockstructured.BoundaryConditions import get_access_to_persistent_face_data
from ..blockstructured.create_dynamic_mesh_refinement_action_set import (
    create_dynamic_mesh_refinement_action_set,
)
from .actionsets import ApplyDirichletBoundaryConditions
from .actionsets import ProjectPatchOntoFaces
from .actionsets import RollOverUpdatedFaceWithSommerfeldBoundaryConditions
from .actionsets import UpdateCell
from .actionsets import ComputeGlobalCFLCondition
from .actionsets import ComputeMaxEigenvalue
from .actionsets import ReduceMaxEigenvalue
from .MaxEigenvalueReduction import add_max_eigenvalue_reduction
from .RusanovWithExplicitEuler import maxEigenvalueOverPatch2D, maxEigenvalueOverPatch3D
from .actionsets import UpdateCellInterior
from .actionsets import UpdateCellShell
from .actionsets import InsertPatchInterior
from .actionsets import DiscardPatchInterior
from .actionsets import SpawnTaskToUpdateCellInterior
from .actionsets import ReadjustTimeStepSizeForOptimisticGlobalTimestepping


## DSLHyPE time step kernels on the patch interior and on the shell of a patch
#
# Variants of exahype2.api.fv.timeStepWithRusanov2D/3D that update only the
# patch interior or only the shell of the patch
# (@ref exahype2.api.fv.RusanovWithOptimisticExplicitEuler.RusanovWithOptimisticExplicitEuler).
# They are never executed as Python, but parsed by DSLHyPE and hence follow
# the conventions documented with timeStepWithRusanov2D in
# RusanovWithExplicitEuler.py. Like the original kernels, they assume that the
# Rusanov stencil reaches one volume in each direction. All volumes between H
# and NV-H along each axis form the patch interior, so the shell is H volumes
# thick.
#
# - The patch interior kernels read the input only within [H-1,NV-H+1) along
#   each axis, i.e. never the halo. Their input hence is the NV^d patch
#   without halo (_patch), so they can run before the halo data has arrived.
#   Their output is the patch interior only, i.e. a (NV-2H)^d patch
#   (_patch_interior).
# - The shell kernels tile the shell into rectangular strips (2d) or slabs
#   (3d) and run the update once per strip with its own set of temporary
#   variables, as the DSL has no loops. They write the shell of the NV^d
#   output only.
#
# None of the kernels returns an eigenvalue. The maximum eigenvalue is a kernel
# of its own (@ref exahype2.api.fv.RusanovWithExplicitEuler.maxEigenvalueOverPatch2D),
# which runs over the whole patch once it is updated.
#
# The 3d kernels test for the PDE terms by the same names as
# timeStepWithRusanov2D, i.e. volumetricSource, flux, nonconservativeProduct and
# maxEigenvalue.


## Rusanov update of the interior of a 2d patch.
def timeStepWithRusanov2DOnPatchInterior(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    t = DataBlock([[0, N]], "patchData.t", "double")
    dt = DataBlock([[0, N]], "patchData.dt", "double")
    h = DataBlock([[0, N]], "patchData.cellSize", "tarch::la::Vector<2, double>")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<2, double>")

    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NV], [0, NV], [0, N]], "patchData.QIn")
    output = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [H, NV-H], [H, NV-H], [0, N]], "patchData.QOut")

    output[:, H:NV-H, H:NV-H] = input[:, H:NV-H, H:NV-H]

    if volumetricSource:
       source = DataBlock([[0, NumberOfUnknowns], [H, NV-H], [H, NV-H], [0, N]], eval("volumetricSource", input[:, H:NV-H, H:NV-H], x, t))
       output[0:NumberOfUnknowns, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H] + dt * source

    maxEigenvaluesX = DataBlock([[0, 1], [H-1, NV-H+1], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, H-1:NV-H+1, H:NV-H], x, 0))
    maxEigenvaluesY = DataBlock([[0, 1], [H, NV-H], [H-1, NV-H+1], [0, N]], eval("maxEigenvalue", input[:, H:NV-H, H-1:NV-H+1], x, 1))

    lambdaLeft = max(maxEigenvaluesX[0:1, H-1:NV-H-1, H:NV-H], maxEigenvaluesX[0:1, H:NV-H, H:NV-H])
    lambdaRight = max(maxEigenvaluesX[0:1, H:NV-H, H:NV-H], maxEigenvaluesX[0:1, H+1:NV-H+1, H:NV-H])
    lambdaBottom = max(maxEigenvaluesY[0:1, H:NV-H, H-1:NV-H-1], maxEigenvaluesY[0:1, H:NV-H, H:NV-H])
    lambdaTop = max(maxEigenvaluesY[0:1, H:NV-H, H:NV-H], maxEigenvaluesY[0:1, H:NV-H, H+1:NV-H+1])

    fluxLeft = -1.0 * lambdaLeft * (input[0:NumberOfUnknowns, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, H-1:NV-H-1, H:NV-H])
    fluxRight = -1.0 * lambdaRight * (input[0:NumberOfUnknowns, H+1:NV-H+1, H:NV-H] - input[0:NumberOfUnknowns, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H] + dt / h(0) * (fluxLeft - fluxRight) * NV * 0.5

    fluxBottom = -1.0 * lambdaBottom * (input[0:NumberOfUnknowns, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, H:NV-H, H-1:NV-H-1])
    fluxTop = -1.0 * lambdaTop * (input[0:NumberOfUnknowns, H:NV-H, H+1:NV-H+1] - input[0:NumberOfUnknowns, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H] + dt / h(1) * (fluxBottom - fluxTop) * NV * 0.5

    if flux:
      flux_x = DataBlock([[0, NumberOfUnknowns], [H-1, NV-H+1], [H, NV-H], [0, N]], eval("flux", input[:, H-1:NV-H+1, H:NV-H], x, 0))
      flux_y = DataBlock([[0, NumberOfUnknowns], [H, NV-H], [H-1, NV-H+1], [0, N]], eval("flux", input[:, H:NV-H, H-1:NV-H+1], x, 1))
      output[0:NumberOfUnknowns, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(0) * (flux_x[:, H-1:NV-H-1, H:NV-H] - flux_x[:, H+1:NV-H+1, H:NV-H])
      output[0:NumberOfUnknowns, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(1) * (flux_y[:, H:NV-H, H-1:NV-H-1] - flux_y[:, H:NV-H, H+1:NV-H+1])

    if nonconservativeProduct:
       average_x = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H-1:NV-H, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H+1, H:NV-H])
       delta_x = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H+1, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H-1:NV-H, H:NV-H]
       nonconservative_x = DataBlock([[0, NumberOfUnknowns], [H, NV-H+1], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_x, delta_x, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(0) * (-nonconservative_x[:, H:NV-H, H:NV-H] - nonconservative_x[:, H+1:NV-H+1, H:NV-H])

       average_y = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H-1:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H:NV-H+1])
       delta_y = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H:NV-H+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H-1:NV-H]
       nonconservative_y = DataBlock([[0, NumberOfUnknowns], [H, NV-H], [H, NV-H+1], [0, N]], eval("nonconservativeProduct", average_y, delta_y, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(1) * (-nonconservative_y[:, H:NV-H, H:NV-H] - nonconservative_y[:, H:NV-H, H+1:NV-H+1])


## Rusanov update of the shell of a 2d patch.
def timeStepWithRusanov2DOnShell(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    t = DataBlock([[0, N]], "patchData.t", "double")
    dt = DataBlock([[0, N]], "patchData.dt", "double")
    h = DataBlock([[0, N]], "patchData.cellSize", "tarch::la::Vector<2, double>")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<2, double>")

    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [-H, NV + H], [-H, NV + H], [0, N]], "patchData.QIn")
    output = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NV], [0, NV], [0, N]], "patchData.QOut")

    # Shell part YLow: [0,NV) x [0,H)
    output[:, 0:NV, 0:H] = input[:, 0:NV, 0:H]

    if volumetricSource:
       sourceYLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, H], [0, N]], eval("volumetricSource", input[:, 0:NV, 0:H], x, t))
       output[0:NumberOfUnknowns, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:H] + dt * sourceYLow

    maxEigenvaluesXYLow = DataBlock([[0, 1], [0-1, NV+1], [0, H], [0, N]], eval("maxEigenvalue", input[:, 0-1:NV+1, 0:H], x, 0))
    maxEigenvaluesYYLow = DataBlock([[0, 1], [0, NV], [0-1, H+1], [0, N]], eval("maxEigenvalue", input[:, 0:NV, 0-1:H+1], x, 1))

    lambdaLeftYLow = max(maxEigenvaluesXYLow[0:1, 0-1:NV-1, 0:H], maxEigenvaluesXYLow[0:1, 0:NV, 0:H])
    lambdaRightYLow = max(maxEigenvaluesXYLow[0:1, 0:NV, 0:H], maxEigenvaluesXYLow[0:1, 0+1:NV+1, 0:H])
    lambdaBottomYLow = max(maxEigenvaluesYYLow[0:1, 0:NV, 0-1:H-1], maxEigenvaluesYYLow[0:1, 0:NV, 0:H])
    lambdaTopYLow = max(maxEigenvaluesYYLow[0:1, 0:NV, 0:H], maxEigenvaluesYYLow[0:1, 0:NV, 0+1:H+1])

    fluxLeftYLow = -1.0 * lambdaLeftYLow * (input[0:NumberOfUnknowns, 0:NV, 0:H] - input[0:NumberOfUnknowns, 0-1:NV-1, 0:H])
    fluxRightYLow = -1.0 * lambdaRightYLow * (input[0:NumberOfUnknowns, 0+1:NV+1, 0:H] - input[0:NumberOfUnknowns, 0:NV, 0:H])
    output[0:NumberOfUnknowns, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:H] + dt / h(0) * (fluxLeftYLow - fluxRightYLow) * NV * 0.5

    fluxBottomYLow = -1.0 * lambdaBottomYLow * (input[0:NumberOfUnknowns, 0:NV, 0:H] - input[0:NumberOfUnknowns, 0:NV, 0-1:H-1])
    fluxTopYLow = -1.0 * lambdaTopYLow * (input[0:NumberOfUnknowns, 0:NV, 0+1:H+1] - input[0:NumberOfUnknowns, 0:NV, 0:H])
    output[0:NumberOfUnknowns, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:H] + dt / h(1) * (fluxBottomYLow - fluxTopYLow) * NV * 0.5

    if flux:
      flux_xYLow = DataBlock([[0, NumberOfUnknowns], [0-1, NV+1], [0, H], [0, N]], eval("flux", input[:, 0-1:NV+1, 0:H], x, 0))
      flux_yYLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0-1, H+1], [0, N]], eval("flux", input[:, 0:NV, 0-1:H+1], x, 1))
      output[0:NumberOfUnknowns, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:H] + 0.5 * NV * dt / h(0) * (flux_xYLow[:, 0-1:NV-1, 0:H] - flux_xYLow[:, 0+1:NV+1, 0:H])
      output[0:NumberOfUnknowns, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:H] + 0.5 * NV * dt / h(1) * (flux_yYLow[:, 0:NV, 0-1:H-1] - flux_yYLow[:, 0:NV, 0+1:H+1])

    if nonconservativeProduct:
       average_xYLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, 0:H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, 0:H])
       delta_xYLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, 0:H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, 0:H]
       nonconservative_xYLow = DataBlock([[0, NumberOfUnknowns], [0, NV+1], [0, H], [0, N]], eval("nonconservativeProduct", average_xYLow, delta_xYLow, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:H] + 0.5 * NV * dt / h(0) * (-nonconservative_xYLow[:, 0:NV, 0:H] - nonconservative_xYLow[:, 0+1:NV+1, 0:H])

       average_yYLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0-1:H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:H+1])
       delta_yYLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:H+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0-1:H]
       nonconservative_yYLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, H+1], [0, N]], eval("nonconservativeProduct", average_yYLow, delta_yYLow, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:H] + 0.5 * NV * dt / h(1) * (-nonconservative_yYLow[:, 0:NV, 0:H] - nonconservative_yYLow[:, 0:NV, 0+1:H+1])

    # Shell part YHigh: [0,NV) x [NV-H,NV)
    output[:, 0:NV, NV-H:NV] = input[:, 0:NV, NV-H:NV]

    if volumetricSource:
       sourceYHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [NV-H, NV], [0, N]], eval("volumetricSource", input[:, 0:NV, NV-H:NV], x, t))
       output[0:NumberOfUnknowns, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV] + dt * sourceYHigh

    maxEigenvaluesXYHigh = DataBlock([[0, 1], [0-1, NV+1], [NV-H, NV], [0, N]], eval("maxEigenvalue", input[:, 0-1:NV+1, NV-H:NV], x, 0))
    maxEigenvaluesYYHigh = DataBlock([[0, 1], [0, NV], [NV-H-1, NV+1], [0, N]], eval("maxEigenvalue", input[:, 0:NV, NV-H-1:NV+1], x, 1))

    lambdaLeftYHigh = max(maxEigenvaluesXYHigh[0:1, 0-1:NV-1, NV-H:NV], maxEigenvaluesXYHigh[0:1, 0:NV, NV-H:NV])
    lambdaRightYHigh = max(maxEigenvaluesXYHigh[0:1, 0:NV, NV-H:NV], maxEigenvaluesXYHigh[0:1, 0+1:NV+1, NV-H:NV])
    lambdaBottomYHigh = max(maxEigenvaluesYYHigh[0:1, 0:NV, NV-H-1:NV-1], maxEigenvaluesYYHigh[0:1, 0:NV, NV-H:NV])
    lambdaTopYHigh = max(maxEigenvaluesYYHigh[0:1, 0:NV, NV-H:NV], maxEigenvaluesYYHigh[0:1, 0:NV, NV-H+1:NV+1])

    fluxLeftYHigh = -1.0 * lambdaLeftYHigh * (input[0:NumberOfUnknowns, 0:NV, NV-H:NV] - input[0:NumberOfUnknowns, 0-1:NV-1, NV-H:NV])
    fluxRightYHigh = -1.0 * lambdaRightYHigh * (input[0:NumberOfUnknowns, 0+1:NV+1, NV-H:NV] - input[0:NumberOfUnknowns, 0:NV, NV-H:NV])
    output[0:NumberOfUnknowns, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV] + dt / h(0) * (fluxLeftYHigh - fluxRightYHigh) * NV * 0.5

    fluxBottomYHigh = -1.0 * lambdaBottomYHigh * (input[0:NumberOfUnknowns, 0:NV, NV-H:NV] - input[0:NumberOfUnknowns, 0:NV, NV-H-1:NV-1])
    fluxTopYHigh = -1.0 * lambdaTopYHigh * (input[0:NumberOfUnknowns, 0:NV, NV-H+1:NV+1] - input[0:NumberOfUnknowns, 0:NV, NV-H:NV])
    output[0:NumberOfUnknowns, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV] + dt / h(1) * (fluxBottomYHigh - fluxTopYHigh) * NV * 0.5

    if flux:
      flux_xYHigh = DataBlock([[0, NumberOfUnknowns], [0-1, NV+1], [NV-H, NV], [0, N]], eval("flux", input[:, 0-1:NV+1, NV-H:NV], x, 0))
      flux_yYHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [NV-H-1, NV+1], [0, N]], eval("flux", input[:, 0:NV, NV-H-1:NV+1], x, 1))
      output[0:NumberOfUnknowns, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV] + 0.5 * NV * dt / h(0) * (flux_xYHigh[:, 0-1:NV-1, NV-H:NV] - flux_xYHigh[:, 0+1:NV+1, NV-H:NV])
      output[0:NumberOfUnknowns, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV] + 0.5 * NV * dt / h(1) * (flux_yYHigh[:, 0:NV, NV-H-1:NV-1] - flux_yYHigh[:, 0:NV, NV-H+1:NV+1])

    if nonconservativeProduct:
       average_xYHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, NV-H:NV] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, NV-H:NV])
       delta_xYHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, NV-H:NV] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, NV-H:NV]
       nonconservative_xYHigh = DataBlock([[0, NumberOfUnknowns], [0, NV+1], [NV-H, NV], [0, N]], eval("nonconservativeProduct", average_xYHigh, delta_xYHigh, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV] + 0.5 * NV * dt / h(0) * (-nonconservative_xYHigh[:, 0:NV, NV-H:NV] - nonconservative_xYHigh[:, 0+1:NV+1, NV-H:NV])

       average_yYHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H-1:NV] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H:NV+1])
       delta_yYHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H:NV+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H-1:NV]
       nonconservative_yYHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [NV-H, NV+1], [0, N]], eval("nonconservativeProduct", average_yYHigh, delta_yYHigh, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV] + 0.5 * NV * dt / h(1) * (-nonconservative_yYHigh[:, 0:NV, NV-H:NV] - nonconservative_yYHigh[:, 0:NV, NV-H+1:NV+1])

    # Shell part XLow: [0,H) x [H,NV-H)
    output[:, 0:H, H:NV-H] = input[:, 0:H, H:NV-H]

    if volumetricSource:
       sourceXLow = DataBlock([[0, NumberOfUnknowns], [0, H], [H, NV-H], [0, N]], eval("volumetricSource", input[:, 0:H, H:NV-H], x, t))
       output[0:NumberOfUnknowns, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H] + dt * sourceXLow

    maxEigenvaluesXXLow = DataBlock([[0, 1], [0-1, H+1], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, 0-1:H+1, H:NV-H], x, 0))
    maxEigenvaluesYXLow = DataBlock([[0, 1], [0, H], [H-1, NV-H+1], [0, N]], eval("maxEigenvalue", input[:, 0:H, H-1:NV-H+1], x, 1))

    lambdaLeftXLow = max(maxEigenvaluesXXLow[0:1, 0-1:H-1, H:NV-H], maxEigenvaluesXXLow[0:1, 0:H, H:NV-H])
    lambdaRightXLow = max(maxEigenvaluesXXLow[0:1, 0:H, H:NV-H], maxEigenvaluesXXLow[0:1, 0+1:H+1, H:NV-H])
    lambdaBottomXLow = max(maxEigenvaluesYXLow[0:1, 0:H, H-1:NV-H-1], maxEigenvaluesYXLow[0:1, 0:H, H:NV-H])
    lambdaTopXLow = max(maxEigenvaluesYXLow[0:1, 0:H, H:NV-H], maxEigenvaluesYXLow[0:1, 0:H, H+1:NV-H+1])

    fluxLeftXLow = -1.0 * lambdaLeftXLow * (input[0:NumberOfUnknowns, 0:H, H:NV-H] - input[0:NumberOfUnknowns, 0-1:H-1, H:NV-H])
    fluxRightXLow = -1.0 * lambdaRightXLow * (input[0:NumberOfUnknowns, 0+1:H+1, H:NV-H] - input[0:NumberOfUnknowns, 0:H, H:NV-H])
    output[0:NumberOfUnknowns, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H] + dt / h(0) * (fluxLeftXLow - fluxRightXLow) * NV * 0.5

    fluxBottomXLow = -1.0 * lambdaBottomXLow * (input[0:NumberOfUnknowns, 0:H, H:NV-H] - input[0:NumberOfUnknowns, 0:H, H-1:NV-H-1])
    fluxTopXLow = -1.0 * lambdaTopXLow * (input[0:NumberOfUnknowns, 0:H, H+1:NV-H+1] - input[0:NumberOfUnknowns, 0:H, H:NV-H])
    output[0:NumberOfUnknowns, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H] + dt / h(1) * (fluxBottomXLow - fluxTopXLow) * NV * 0.5

    if flux:
      flux_xXLow = DataBlock([[0, NumberOfUnknowns], [0-1, H+1], [H, NV-H], [0, N]], eval("flux", input[:, 0-1:H+1, H:NV-H], x, 0))
      flux_yXLow = DataBlock([[0, NumberOfUnknowns], [0, H], [H-1, NV-H+1], [0, N]], eval("flux", input[:, 0:H, H-1:NV-H+1], x, 1))
      output[0:NumberOfUnknowns, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H] + 0.5 * NV * dt / h(0) * (flux_xXLow[:, 0-1:H-1, H:NV-H] - flux_xXLow[:, 0+1:H+1, H:NV-H])
      output[0:NumberOfUnknowns, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H] + 0.5 * NV * dt / h(1) * (flux_yXLow[:, 0:H, H-1:NV-H-1] - flux_yXLow[:, 0:H, H+1:NV-H+1])

    if nonconservativeProduct:
       average_xXLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:H, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H+1, H:NV-H])
       delta_xXLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H+1, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:H, H:NV-H]
       nonconservative_xXLow = DataBlock([[0, NumberOfUnknowns], [0, H+1], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_xXLow, delta_xXLow, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H] + 0.5 * NV * dt / h(0) * (-nonconservative_xXLow[:, 0:H, H:NV-H] - nonconservative_xXLow[:, 0+1:H+1, H:NV-H])

       average_yXLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H-1:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H:NV-H+1])
       delta_yXLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H:NV-H+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H-1:NV-H]
       nonconservative_yXLow = DataBlock([[0, NumberOfUnknowns], [0, H], [H, NV-H+1], [0, N]], eval("nonconservativeProduct", average_yXLow, delta_yXLow, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H] + 0.5 * NV * dt / h(1) * (-nonconservative_yXLow[:, 0:H, H:NV-H] - nonconservative_yXLow[:, 0:H, H+1:NV-H+1])

    # Shell part XHigh: [NV-H,NV) x [H,NV-H)
    output[:, NV-H:NV, H:NV-H] = input[:, NV-H:NV, H:NV-H]

    if volumetricSource:
       sourceXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H, NV], [H, NV-H], [0, N]], eval("volumetricSource", input[:, NV-H:NV, H:NV-H], x, t))
       output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] + dt * sourceXHigh

    maxEigenvaluesXXHigh = DataBlock([[0, 1], [NV-H-1, NV+1], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, NV-H-1:NV+1, H:NV-H], x, 0))
    maxEigenvaluesYXHigh = DataBlock([[0, 1], [NV-H, NV], [H-1, NV-H+1], [0, N]], eval("maxEigenvalue", input[:, NV-H:NV, H-1:NV-H+1], x, 1))

    lambdaLeftXHigh = max(maxEigenvaluesXXHigh[0:1, NV-H-1:NV-1, H:NV-H], maxEigenvaluesXXHigh[0:1, NV-H:NV, H:NV-H])
    lambdaRightXHigh = max(maxEigenvaluesXXHigh[0:1, NV-H:NV, H:NV-H], maxEigenvaluesXXHigh[0:1, NV-H+1:NV+1, H:NV-H])
    lambdaBottomXHigh = max(maxEigenvaluesYXHigh[0:1, NV-H:NV, H-1:NV-H-1], maxEigenvaluesYXHigh[0:1, NV-H:NV, H:NV-H])
    lambdaTopXHigh = max(maxEigenvaluesYXHigh[0:1, NV-H:NV, H:NV-H], maxEigenvaluesYXHigh[0:1, NV-H:NV, H+1:NV-H+1])

    fluxLeftXHigh = -1.0 * lambdaLeftXHigh * (input[0:NumberOfUnknowns, NV-H:NV, H:NV-H] - input[0:NumberOfUnknowns, NV-H-1:NV-1, H:NV-H])
    fluxRightXHigh = -1.0 * lambdaRightXHigh * (input[0:NumberOfUnknowns, NV-H+1:NV+1, H:NV-H] - input[0:NumberOfUnknowns, NV-H:NV, H:NV-H])
    output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] + dt / h(0) * (fluxLeftXHigh - fluxRightXHigh) * NV * 0.5

    fluxBottomXHigh = -1.0 * lambdaBottomXHigh * (input[0:NumberOfUnknowns, NV-H:NV, H:NV-H] - input[0:NumberOfUnknowns, NV-H:NV, H-1:NV-H-1])
    fluxTopXHigh = -1.0 * lambdaTopXHigh * (input[0:NumberOfUnknowns, NV-H:NV, H+1:NV-H+1] - input[0:NumberOfUnknowns, NV-H:NV, H:NV-H])
    output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] + dt / h(1) * (fluxBottomXHigh - fluxTopXHigh) * NV * 0.5

    if flux:
      flux_xXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H-1, NV+1], [H, NV-H], [0, N]], eval("flux", input[:, NV-H-1:NV+1, H:NV-H], x, 0))
      flux_yXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H, NV], [H-1, NV-H+1], [0, N]], eval("flux", input[:, NV-H:NV, H-1:NV-H+1], x, 1))
      output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] + 0.5 * NV * dt / h(0) * (flux_xXHigh[:, NV-H-1:NV-1, H:NV-H] - flux_xXHigh[:, NV-H+1:NV+1, H:NV-H])
      output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] + 0.5 * NV * dt / h(1) * (flux_yXHigh[:, NV-H:NV, H-1:NV-H-1] - flux_yXHigh[:, NV-H:NV, H+1:NV-H+1])

    if nonconservativeProduct:
       average_xXHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H-1:NV, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV+1, H:NV-H])
       delta_xXHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV+1, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H-1:NV, H:NV-H]
       nonconservative_xXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H, NV+1], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_xXHigh, delta_xXHigh, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] + 0.5 * NV * dt / h(0) * (-nonconservative_xXHigh[:, NV-H:NV, H:NV-H] - nonconservative_xXHigh[:, NV-H+1:NV+1, H:NV-H])

       average_yXHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H-1:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H:NV-H+1])
       delta_yXHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H:NV-H+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H-1:NV-H]
       nonconservative_yXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H, NV], [H, NV-H+1], [0, N]], eval("nonconservativeProduct", average_yXHigh, delta_yXHigh, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H] + 0.5 * NV * dt / h(1) * (-nonconservative_yXHigh[:, NV-H:NV, H:NV-H] - nonconservative_yXHigh[:, NV-H:NV, H+1:NV-H+1])


## Rusanov update of the interior of a 3d patch.
def timeStepWithRusanov3DOnPatchInterior(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    t = DataBlock([[0, N]], "patchData.t", "double")
    dt = DataBlock([[0, N]], "patchData.dt", "double")
    h = DataBlock([[0, N]], "patchData.cellSize", "tarch::la::Vector<3, double>")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<3, double>")

    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NV], [0, NV], [0, NV], [0, N]], "patchData.QIn")
    output = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [H, NV-H], [H, NV-H], [H, NV-H], [0, N]], "patchData.QOut")

    output[:, H:NV-H, H:NV-H, H:NV-H] = input[:, H:NV-H, H:NV-H, H:NV-H]

    if volumetricSource:
       source = DataBlock([[0, NumberOfUnknowns], [H, NV-H], [H, NV-H], [H, NV-H], [0, N]], eval("volumetricSource", input[:, H:NV-H, H:NV-H, H:NV-H], x, t))
       output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] + dt * source

    maxEigenvaluesX = DataBlock([[0, 1], [H-1, NV-H+1], [H, NV-H], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, H-1:NV-H+1, H:NV-H, H:NV-H], x, 0))
    maxEigenvaluesY = DataBlock([[0, 1], [H, NV-H], [H-1, NV-H+1], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, H:NV-H, H-1:NV-H+1, H:NV-H], x, 1))
    maxEigenvaluesZ = DataBlock([[0, 1], [H, NV-H], [H, NV-H], [H-1, NV-H+1], [0, N]], eval("maxEigenvalue", input[:, H:NV-H, H:NV-H, H-1:NV-H+1], x, 2))

    lambdaLeft = max(maxEigenvaluesX[0:1, H-1:NV-H-1, H:NV-H, H:NV-H], maxEigenvaluesX[0:1, H:NV-H, H:NV-H, H:NV-H])
    lambdaRight = max(maxEigenvaluesX[0:1, H:NV-H, H:NV-H, H:NV-H], maxEigenvaluesX[0:1, H+1:NV-H+1, H:NV-H, H:NV-H])
    lambdaBottom = max(maxEigenvaluesY[0:1, H:NV-H, H-1:NV-H-1, H:NV-H], maxEigenvaluesY[0:1, H:NV-H, H:NV-H, H:NV-H])
    lambdaTop = max(maxEigenvaluesY[0:1, H:NV-H, H:NV-H, H:NV-H], maxEigenvaluesY[0:1, H:NV-H, H+1:NV-H+1, H:NV-H])
    lambdaNear = max(maxEigenvaluesZ[0:1, H:NV-H, H:NV-H, H-1:NV-H-1], maxEigenvaluesZ[0:1, H:NV-H, H:NV-H, H:NV-H])
    lambdaFar = max(maxEigenvaluesZ[0:1, H:NV-H, H:NV-H, H:NV-H], maxEigenvaluesZ[0:1, H:NV-H, H:NV-H, H+1:NV-H+1])

    fluxLeft = -1.0 * lambdaLeft * (input[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, H-1:NV-H-1, H:NV-H, H:NV-H])
    fluxRight = -1.0 * lambdaRight * (input[0:NumberOfUnknowns, H+1:NV-H+1, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] + dt / h(0) * (fluxLeft - fluxRight) * NV * 0.5

    fluxBottom = -1.0 * lambdaBottom * (input[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, H:NV-H, H-1:NV-H-1, H:NV-H])
    fluxTop = -1.0 * lambdaTop * (input[0:NumberOfUnknowns, H:NV-H, H+1:NV-H+1, H:NV-H] - input[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] + dt / h(1) * (fluxBottom - fluxTop) * NV * 0.5

    fluxNear = -1.0 * lambdaNear * (input[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, H:NV-H, H:NV-H, H-1:NV-H-1])
    fluxFar = -1.0 * lambdaFar * (input[0:NumberOfUnknowns, H:NV-H, H:NV-H, H+1:NV-H+1] - input[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] + dt / h(2) * (fluxNear - fluxFar) * NV * 0.5

    if flux:
      flux_x = DataBlock([[0, NumberOfUnknowns], [H-1, NV-H+1], [H, NV-H], [H, NV-H], [0, N]], eval("flux", input[:, H-1:NV-H+1, H:NV-H, H:NV-H], x, 0))
      flux_y = DataBlock([[0, NumberOfUnknowns], [H, NV-H], [H-1, NV-H+1], [H, NV-H], [0, N]], eval("flux", input[:, H:NV-H, H-1:NV-H+1, H:NV-H], x, 1))
      flux_z = DataBlock([[0, NumberOfUnknowns], [H, NV-H], [H, NV-H], [H-1, NV-H+1], [0, N]], eval("flux", input[:, H:NV-H, H:NV-H, H-1:NV-H+1], x, 2))
      output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(0) * (flux_x[:, H-1:NV-H-1, H:NV-H, H:NV-H] - flux_x[:, H+1:NV-H+1, H:NV-H, H:NV-H])
      output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(1) * (flux_y[:, H:NV-H, H-1:NV-H-1, H:NV-H] - flux_y[:, H:NV-H, H+1:NV-H+1, H:NV-H])
      output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(2) * (flux_z[:, H:NV-H, H:NV-H, H-1:NV-H-1] - flux_z[:, H:NV-H, H:NV-H, H+1:NV-H+1])

    if nonconservativeProduct:
       average_x = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H-1:NV-H, H:NV-H, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H+1, H:NV-H, H:NV-H])
       delta_x = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H+1, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H-1:NV-H, H:NV-H, H:NV-H]
       nonconservative_x = DataBlock([[0, NumberOfUnknowns], [H, NV-H+1], [H, NV-H], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_x, delta_x, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(0) * (-nonconservative_x[:, H:NV-H, H:NV-H, H:NV-H] - nonconservative_x[:, H+1:NV-H+1, H:NV-H, H:NV-H])

       average_y = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H-1:NV-H, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H:NV-H+1, H:NV-H])
       delta_y = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H:NV-H+1, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H-1:NV-H, H:NV-H]
       nonconservative_y = DataBlock([[0, NumberOfUnknowns], [H, NV-H], [H, NV-H+1], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_y, delta_y, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(1) * (-nonconservative_y[:, H:NV-H, H:NV-H, H:NV-H] - nonconservative_y[:, H:NV-H, H+1:NV-H+1, H:NV-H])

       average_z = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H:NV-H, H-1:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H:NV-H, H:NV-H+1])
       delta_z = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H:NV-H, H:NV-H+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, H:NV-H, H:NV-H, H-1:NV-H]
       nonconservative_z = DataBlock([[0, NumberOfUnknowns], [H, NV-H], [H, NV-H], [H, NV-H+1], [0, N]], eval("nonconservativeProduct", average_z, delta_z, x, h, t, dt, 2))
       output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, H:NV-H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(2) * (-nonconservative_z[:, H:NV-H, H:NV-H, H:NV-H] - nonconservative_z[:, H:NV-H, H:NV-H, H+1:NV-H+1])


## Rusanov update of the shell of a 3d patch.
def timeStepWithRusanov3DOnShell(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    t = DataBlock([[0, N]], "patchData.t", "double")
    dt = DataBlock([[0, N]], "patchData.dt", "double")
    h = DataBlock([[0, N]], "patchData.cellSize", "tarch::la::Vector<3, double>")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<3, double>")

    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [-H, NV + H], [-H, NV + H], [-H, NV + H], [0, N]], "patchData.QIn")
    output = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NV], [0, NV], [0, NV], [0, N]], "patchData.QOut")

    # Shell part ZLow: [0,NV) x [0,NV) x [0,H)
    output[:, 0:NV, 0:NV, 0:H] = input[:, 0:NV, 0:NV, 0:H]

    if volumetricSource:
       sourceZLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [0, H], [0, N]], eval("volumetricSource", input[:, 0:NV, 0:NV, 0:H], x, t))
       output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] + dt * sourceZLow

    maxEigenvaluesXZLow = DataBlock([[0, 1], [0-1, NV+1], [0, NV], [0, H], [0, N]], eval("maxEigenvalue", input[:, 0-1:NV+1, 0:NV, 0:H], x, 0))
    maxEigenvaluesYZLow = DataBlock([[0, 1], [0, NV], [0-1, NV+1], [0, H], [0, N]], eval("maxEigenvalue", input[:, 0:NV, 0-1:NV+1, 0:H], x, 1))
    maxEigenvaluesZZLow = DataBlock([[0, 1], [0, NV], [0, NV], [0-1, H+1], [0, N]], eval("maxEigenvalue", input[:, 0:NV, 0:NV, 0-1:H+1], x, 2))

    lambdaLeftZLow = max(maxEigenvaluesXZLow[0:1, 0-1:NV-1, 0:NV, 0:H], maxEigenvaluesXZLow[0:1, 0:NV, 0:NV, 0:H])
    lambdaRightZLow = max(maxEigenvaluesXZLow[0:1, 0:NV, 0:NV, 0:H], maxEigenvaluesXZLow[0:1, 0+1:NV+1, 0:NV, 0:H])
    lambdaBottomZLow = max(maxEigenvaluesYZLow[0:1, 0:NV, 0-1:NV-1, 0:H], maxEigenvaluesYZLow[0:1, 0:NV, 0:NV, 0:H])
    lambdaTopZLow = max(maxEigenvaluesYZLow[0:1, 0:NV, 0:NV, 0:H], maxEigenvaluesYZLow[0:1, 0:NV, 0+1:NV+1, 0:H])
    lambdaNearZLow = max(maxEigenvaluesZZLow[0:1, 0:NV, 0:NV, 0-1:H-1], maxEigenvaluesZZLow[0:1, 0:NV, 0:NV, 0:H])
    lambdaFarZLow = max(maxEigenvaluesZZLow[0:1, 0:NV, 0:NV, 0:H], maxEigenvaluesZZLow[0:1, 0:NV, 0:NV, 0+1:H+1])

    fluxLeftZLow = -1.0 * lambdaLeftZLow * (input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] - input[0:NumberOfUnknowns, 0-1:NV-1, 0:NV, 0:H])
    fluxRightZLow = -1.0 * lambdaRightZLow * (input[0:NumberOfUnknowns, 0+1:NV+1, 0:NV, 0:H] - input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H])
    output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] + dt / h(0) * (fluxLeftZLow - fluxRightZLow) * NV * 0.5

    fluxBottomZLow = -1.0 * lambdaBottomZLow * (input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] - input[0:NumberOfUnknowns, 0:NV, 0-1:NV-1, 0:H])
    fluxTopZLow = -1.0 * lambdaTopZLow * (input[0:NumberOfUnknowns, 0:NV, 0+1:NV+1, 0:H] - input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H])
    output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] + dt / h(1) * (fluxBottomZLow - fluxTopZLow) * NV * 0.5

    fluxNearZLow = -1.0 * lambdaNearZLow * (input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] - input[0:NumberOfUnknowns, 0:NV, 0:NV, 0-1:H-1])
    fluxFarZLow = -1.0 * lambdaFarZLow * (input[0:NumberOfUnknowns, 0:NV, 0:NV, 0+1:H+1] - input[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H])
    output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] + dt / h(2) * (fluxNearZLow - fluxFarZLow) * NV * 0.5

    if flux:
      flux_xZLow = DataBlock([[0, NumberOfUnknowns], [0-1, NV+1], [0, NV], [0, H], [0, N]], eval("flux", input[:, 0-1:NV+1, 0:NV, 0:H], x, 0))
      flux_yZLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0-1, NV+1], [0, H], [0, N]], eval("flux", input[:, 0:NV, 0-1:NV+1, 0:H], x, 1))
      flux_zZLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [0-1, H+1], [0, N]], eval("flux", input[:, 0:NV, 0:NV, 0-1:H+1], x, 2))
      output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] + 0.5 * NV * dt / h(0) * (flux_xZLow[:, 0-1:NV-1, 0:NV, 0:H] - flux_xZLow[:, 0+1:NV+1, 0:NV, 0:H])
      output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] + 0.5 * NV * dt / h(1) * (flux_yZLow[:, 0:NV, 0-1:NV-1, 0:H] - flux_yZLow[:, 0:NV, 0+1:NV+1, 0:H])
      output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] + 0.5 * NV * dt / h(2) * (flux_zZLow[:, 0:NV, 0:NV, 0-1:H-1] - flux_zZLow[:, 0:NV, 0:NV, 0+1:H+1])

    if nonconservativeProduct:
       average_xZLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, 0:NV, 0:H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, 0:NV, 0:H])
       delta_xZLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, 0:NV, 0:H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, 0:NV, 0:H]
       nonconservative_xZLow = DataBlock([[0, NumberOfUnknowns], [0, NV+1], [0, NV], [0, H], [0, N]], eval("nonconservativeProduct", average_xZLow, delta_xZLow, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] + 0.5 * NV * dt / h(0) * (-nonconservative_xZLow[:, 0:NV, 0:NV, 0:H] - nonconservative_xZLow[:, 0+1:NV+1, 0:NV, 0:H])

       average_yZLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0-1:NV, 0:H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV+1, 0:H])
       delta_yZLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV+1, 0:H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0-1:NV, 0:H]
       nonconservative_yZLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV+1], [0, H], [0, N]], eval("nonconservativeProduct", average_yZLow, delta_yZLow, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] + 0.5 * NV * dt / h(1) * (-nonconservative_yZLow[:, 0:NV, 0:NV, 0:H] - nonconservative_yZLow[:, 0:NV, 0+1:NV+1, 0:H])

       average_zZLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, 0-1:H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, 0:H+1])
       delta_zZLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, 0:H+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, 0-1:H]
       nonconservative_zZLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [0, H+1], [0, N]], eval("nonconservativeProduct", average_zZLow, delta_zZLow, x, h, t, dt, 2))
       output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] = output[0:NumberOfUnknowns, 0:NV, 0:NV, 0:H] + 0.5 * NV * dt / h(2) * (-nonconservative_zZLow[:, 0:NV, 0:NV, 0:H] - nonconservative_zZLow[:, 0:NV, 0:NV, 0+1:H+1])

    # Shell part ZHigh: [0,NV) x [0,NV) x [NV-H,NV)
    output[:, 0:NV, 0:NV, NV-H:NV] = input[:, 0:NV, 0:NV, NV-H:NV]

    if volumetricSource:
       sourceZHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [NV-H, NV], [0, N]], eval("volumetricSource", input[:, 0:NV, 0:NV, NV-H:NV], x, t))
       output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] + dt * sourceZHigh

    maxEigenvaluesXZHigh = DataBlock([[0, 1], [0-1, NV+1], [0, NV], [NV-H, NV], [0, N]], eval("maxEigenvalue", input[:, 0-1:NV+1, 0:NV, NV-H:NV], x, 0))
    maxEigenvaluesYZHigh = DataBlock([[0, 1], [0, NV], [0-1, NV+1], [NV-H, NV], [0, N]], eval("maxEigenvalue", input[:, 0:NV, 0-1:NV+1, NV-H:NV], x, 1))
    maxEigenvaluesZZHigh = DataBlock([[0, 1], [0, NV], [0, NV], [NV-H-1, NV+1], [0, N]], eval("maxEigenvalue", input[:, 0:NV, 0:NV, NV-H-1:NV+1], x, 2))

    lambdaLeftZHigh = max(maxEigenvaluesXZHigh[0:1, 0-1:NV-1, 0:NV, NV-H:NV], maxEigenvaluesXZHigh[0:1, 0:NV, 0:NV, NV-H:NV])
    lambdaRightZHigh = max(maxEigenvaluesXZHigh[0:1, 0:NV, 0:NV, NV-H:NV], maxEigenvaluesXZHigh[0:1, 0+1:NV+1, 0:NV, NV-H:NV])
    lambdaBottomZHigh = max(maxEigenvaluesYZHigh[0:1, 0:NV, 0-1:NV-1, NV-H:NV], maxEigenvaluesYZHigh[0:1, 0:NV, 0:NV, NV-H:NV])
    lambdaTopZHigh = max(maxEigenvaluesYZHigh[0:1, 0:NV, 0:NV, NV-H:NV], maxEigenvaluesYZHigh[0:1, 0:NV, 0+1:NV+1, NV-H:NV])
    lambdaNearZHigh = max(maxEigenvaluesZZHigh[0:1, 0:NV, 0:NV, NV-H-1:NV-1], maxEigenvaluesZZHigh[0:1, 0:NV, 0:NV, NV-H:NV])
    lambdaFarZHigh = max(maxEigenvaluesZZHigh[0:1, 0:NV, 0:NV, NV-H:NV], maxEigenvaluesZZHigh[0:1, 0:NV, 0:NV, NV-H+1:NV+1])

    fluxLeftZHigh = -1.0 * lambdaLeftZHigh * (input[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] - input[0:NumberOfUnknowns, 0-1:NV-1, 0:NV, NV-H:NV])
    fluxRightZHigh = -1.0 * lambdaRightZHigh * (input[0:NumberOfUnknowns, 0+1:NV+1, 0:NV, NV-H:NV] - input[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV])
    output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] + dt / h(0) * (fluxLeftZHigh - fluxRightZHigh) * NV * 0.5

    fluxBottomZHigh = -1.0 * lambdaBottomZHigh * (input[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] - input[0:NumberOfUnknowns, 0:NV, 0-1:NV-1, NV-H:NV])
    fluxTopZHigh = -1.0 * lambdaTopZHigh * (input[0:NumberOfUnknowns, 0:NV, 0+1:NV+1, NV-H:NV] - input[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV])
    output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] + dt / h(1) * (fluxBottomZHigh - fluxTopZHigh) * NV * 0.5

    fluxNearZHigh = -1.0 * lambdaNearZHigh * (input[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] - input[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H-1:NV-1])
    fluxFarZHigh = -1.0 * lambdaFarZHigh * (input[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H+1:NV+1] - input[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV])
    output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] + dt / h(2) * (fluxNearZHigh - fluxFarZHigh) * NV * 0.5

    if flux:
      flux_xZHigh = DataBlock([[0, NumberOfUnknowns], [0-1, NV+1], [0, NV], [NV-H, NV], [0, N]], eval("flux", input[:, 0-1:NV+1, 0:NV, NV-H:NV], x, 0))
      flux_yZHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [0-1, NV+1], [NV-H, NV], [0, N]], eval("flux", input[:, 0:NV, 0-1:NV+1, NV-H:NV], x, 1))
      flux_zZHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [NV-H-1, NV+1], [0, N]], eval("flux", input[:, 0:NV, 0:NV, NV-H-1:NV+1], x, 2))
      output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] + 0.5 * NV * dt / h(0) * (flux_xZHigh[:, 0-1:NV-1, 0:NV, NV-H:NV] - flux_xZHigh[:, 0+1:NV+1, 0:NV, NV-H:NV])
      output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] + 0.5 * NV * dt / h(1) * (flux_yZHigh[:, 0:NV, 0-1:NV-1, NV-H:NV] - flux_yZHigh[:, 0:NV, 0+1:NV+1, NV-H:NV])
      output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] + 0.5 * NV * dt / h(2) * (flux_zZHigh[:, 0:NV, 0:NV, NV-H-1:NV-1] - flux_zZHigh[:, 0:NV, 0:NV, NV-H+1:NV+1])

    if nonconservativeProduct:
       average_xZHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, 0:NV, NV-H:NV] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, 0:NV, NV-H:NV])
       delta_xZHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, 0:NV, NV-H:NV] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, 0:NV, NV-H:NV]
       nonconservative_xZHigh = DataBlock([[0, NumberOfUnknowns], [0, NV+1], [0, NV], [NV-H, NV], [0, N]], eval("nonconservativeProduct", average_xZHigh, delta_xZHigh, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] + 0.5 * NV * dt / h(0) * (-nonconservative_xZHigh[:, 0:NV, 0:NV, NV-H:NV] - nonconservative_xZHigh[:, 0+1:NV+1, 0:NV, NV-H:NV])

       average_yZHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0-1:NV, NV-H:NV] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV+1, NV-H:NV])
       delta_yZHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV+1, NV-H:NV] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0-1:NV, NV-H:NV]
       nonconservative_yZHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV+1], [NV-H, NV], [0, N]], eval("nonconservativeProduct", average_yZHigh, delta_yZHigh, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] + 0.5 * NV * dt / h(1) * (-nonconservative_yZHigh[:, 0:NV, 0:NV, NV-H:NV] - nonconservative_yZHigh[:, 0:NV, 0+1:NV+1, NV-H:NV])

       average_zZHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, NV-H-1:NV] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, NV-H:NV+1])
       delta_zZHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, NV-H:NV+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:NV, NV-H-1:NV]
       nonconservative_zZHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [NV-H, NV+1], [0, N]], eval("nonconservativeProduct", average_zZHigh, delta_zZHigh, x, h, t, dt, 2))
       output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] = output[0:NumberOfUnknowns, 0:NV, 0:NV, NV-H:NV] + 0.5 * NV * dt / h(2) * (-nonconservative_zZHigh[:, 0:NV, 0:NV, NV-H:NV] - nonconservative_zZHigh[:, 0:NV, 0:NV, NV-H+1:NV+1])

    # Shell part YLow: [0,NV) x [0,H) x [H,NV-H)
    output[:, 0:NV, 0:H, H:NV-H] = input[:, 0:NV, 0:H, H:NV-H]

    if volumetricSource:
       sourceYLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, H], [H, NV-H], [0, N]], eval("volumetricSource", input[:, 0:NV, 0:H, H:NV-H], x, t))
       output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] + dt * sourceYLow

    maxEigenvaluesXYLow = DataBlock([[0, 1], [0-1, NV+1], [0, H], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, 0-1:NV+1, 0:H, H:NV-H], x, 0))
    maxEigenvaluesYYLow = DataBlock([[0, 1], [0, NV], [0-1, H+1], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, 0:NV, 0-1:H+1, H:NV-H], x, 1))
    maxEigenvaluesZYLow = DataBlock([[0, 1], [0, NV], [0, H], [H-1, NV-H+1], [0, N]], eval("maxEigenvalue", input[:, 0:NV, 0:H, H-1:NV-H+1], x, 2))

    lambdaLeftYLow = max(maxEigenvaluesXYLow[0:1, 0-1:NV-1, 0:H, H:NV-H], maxEigenvaluesXYLow[0:1, 0:NV, 0:H, H:NV-H])
    lambdaRightYLow = max(maxEigenvaluesXYLow[0:1, 0:NV, 0:H, H:NV-H], maxEigenvaluesXYLow[0:1, 0+1:NV+1, 0:H, H:NV-H])
    lambdaBottomYLow = max(maxEigenvaluesYYLow[0:1, 0:NV, 0-1:H-1, H:NV-H], maxEigenvaluesYYLow[0:1, 0:NV, 0:H, H:NV-H])
    lambdaTopYLow = max(maxEigenvaluesYYLow[0:1, 0:NV, 0:H, H:NV-H], maxEigenvaluesYYLow[0:1, 0:NV, 0+1:H+1, H:NV-H])
    lambdaNearYLow = max(maxEigenvaluesZYLow[0:1, 0:NV, 0:H, H-1:NV-H-1], maxEigenvaluesZYLow[0:1, 0:NV, 0:H, H:NV-H])
    lambdaFarYLow = max(maxEigenvaluesZYLow[0:1, 0:NV, 0:H, H:NV-H], maxEigenvaluesZYLow[0:1, 0:NV, 0:H, H+1:NV-H+1])

    fluxLeftYLow = -1.0 * lambdaLeftYLow * (input[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] - input[0:NumberOfUnknowns, 0-1:NV-1, 0:H, H:NV-H])
    fluxRightYLow = -1.0 * lambdaRightYLow * (input[0:NumberOfUnknowns, 0+1:NV+1, 0:H, H:NV-H] - input[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H])
    output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] + dt / h(0) * (fluxLeftYLow - fluxRightYLow) * NV * 0.5

    fluxBottomYLow = -1.0 * lambdaBottomYLow * (input[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] - input[0:NumberOfUnknowns, 0:NV, 0-1:H-1, H:NV-H])
    fluxTopYLow = -1.0 * lambdaTopYLow * (input[0:NumberOfUnknowns, 0:NV, 0+1:H+1, H:NV-H] - input[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H])
    output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] + dt / h(1) * (fluxBottomYLow - fluxTopYLow) * NV * 0.5

    fluxNearYLow = -1.0 * lambdaNearYLow * (input[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] - input[0:NumberOfUnknowns, 0:NV, 0:H, H-1:NV-H-1])
    fluxFarYLow = -1.0 * lambdaFarYLow * (input[0:NumberOfUnknowns, 0:NV, 0:H, H+1:NV-H+1] - input[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H])
    output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] + dt / h(2) * (fluxNearYLow - fluxFarYLow) * NV * 0.5

    if flux:
      flux_xYLow = DataBlock([[0, NumberOfUnknowns], [0-1, NV+1], [0, H], [H, NV-H], [0, N]], eval("flux", input[:, 0-1:NV+1, 0:H, H:NV-H], x, 0))
      flux_yYLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0-1, H+1], [H, NV-H], [0, N]], eval("flux", input[:, 0:NV, 0-1:H+1, H:NV-H], x, 1))
      flux_zYLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, H], [H-1, NV-H+1], [0, N]], eval("flux", input[:, 0:NV, 0:H, H-1:NV-H+1], x, 2))
      output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] + 0.5 * NV * dt / h(0) * (flux_xYLow[:, 0-1:NV-1, 0:H, H:NV-H] - flux_xYLow[:, 0+1:NV+1, 0:H, H:NV-H])
      output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] + 0.5 * NV * dt / h(1) * (flux_yYLow[:, 0:NV, 0-1:H-1, H:NV-H] - flux_yYLow[:, 0:NV, 0+1:H+1, H:NV-H])
      output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] + 0.5 * NV * dt / h(2) * (flux_zYLow[:, 0:NV, 0:H, H-1:NV-H-1] - flux_zYLow[:, 0:NV, 0:H, H+1:NV-H+1])

    if nonconservativeProduct:
       average_xYLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, 0:H, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, 0:H, H:NV-H])
       delta_xYLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, 0:H, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, 0:H, H:NV-H]
       nonconservative_xYLow = DataBlock([[0, NumberOfUnknowns], [0, NV+1], [0, H], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_xYLow, delta_xYLow, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] + 0.5 * NV * dt / h(0) * (-nonconservative_xYLow[:, 0:NV, 0:H, H:NV-H] - nonconservative_xYLow[:, 0+1:NV+1, 0:H, H:NV-H])

       average_yYLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0-1:H, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:H+1, H:NV-H])
       delta_yYLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:H+1, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0-1:H, H:NV-H]
       nonconservative_yYLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, H+1], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_yYLow, delta_yYLow, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] + 0.5 * NV * dt / h(1) * (-nonconservative_yYLow[:, 0:NV, 0:H, H:NV-H] - nonconservative_yYLow[:, 0:NV, 0+1:H+1, H:NV-H])

       average_zYLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:H, H-1:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:H, H:NV-H+1])
       delta_zYLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:H, H:NV-H+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, 0:H, H-1:NV-H]
       nonconservative_zYLow = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, H], [H, NV-H+1], [0, N]], eval("nonconservativeProduct", average_zYLow, delta_zYLow, x, h, t, dt, 2))
       output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, 0:H, H:NV-H] + 0.5 * NV * dt / h(2) * (-nonconservative_zYLow[:, 0:NV, 0:H, H:NV-H] - nonconservative_zYLow[:, 0:NV, 0:H, H+1:NV-H+1])

    # Shell part YHigh: [0,NV) x [NV-H,NV) x [H,NV-H)
    output[:, 0:NV, NV-H:NV, H:NV-H] = input[:, 0:NV, NV-H:NV, H:NV-H]

    if volumetricSource:
       sourceYHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [NV-H, NV], [H, NV-H], [0, N]], eval("volumetricSource", input[:, 0:NV, NV-H:NV, H:NV-H], x, t))
       output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] + dt * sourceYHigh

    maxEigenvaluesXYHigh = DataBlock([[0, 1], [0-1, NV+1], [NV-H, NV], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, 0-1:NV+1, NV-H:NV, H:NV-H], x, 0))
    maxEigenvaluesYYHigh = DataBlock([[0, 1], [0, NV], [NV-H-1, NV+1], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, 0:NV, NV-H-1:NV+1, H:NV-H], x, 1))
    maxEigenvaluesZYHigh = DataBlock([[0, 1], [0, NV], [NV-H, NV], [H-1, NV-H+1], [0, N]], eval("maxEigenvalue", input[:, 0:NV, NV-H:NV, H-1:NV-H+1], x, 2))

    lambdaLeftYHigh = max(maxEigenvaluesXYHigh[0:1, 0-1:NV-1, NV-H:NV, H:NV-H], maxEigenvaluesXYHigh[0:1, 0:NV, NV-H:NV, H:NV-H])
    lambdaRightYHigh = max(maxEigenvaluesXYHigh[0:1, 0:NV, NV-H:NV, H:NV-H], maxEigenvaluesXYHigh[0:1, 0+1:NV+1, NV-H:NV, H:NV-H])
    lambdaBottomYHigh = max(maxEigenvaluesYYHigh[0:1, 0:NV, NV-H-1:NV-1, H:NV-H], maxEigenvaluesYYHigh[0:1, 0:NV, NV-H:NV, H:NV-H])
    lambdaTopYHigh = max(maxEigenvaluesYYHigh[0:1, 0:NV, NV-H:NV, H:NV-H], maxEigenvaluesYYHigh[0:1, 0:NV, NV-H+1:NV+1, H:NV-H])
    lambdaNearYHigh = max(maxEigenvaluesZYHigh[0:1, 0:NV, NV-H:NV, H-1:NV-H-1], maxEigenvaluesZYHigh[0:1, 0:NV, NV-H:NV, H:NV-H])
    lambdaFarYHigh = max(maxEigenvaluesZYHigh[0:1, 0:NV, NV-H:NV, H:NV-H], maxEigenvaluesZYHigh[0:1, 0:NV, NV-H:NV, H+1:NV-H+1])

    fluxLeftYHigh = -1.0 * lambdaLeftYHigh * (input[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] - input[0:NumberOfUnknowns, 0-1:NV-1, NV-H:NV, H:NV-H])
    fluxRightYHigh = -1.0 * lambdaRightYHigh * (input[0:NumberOfUnknowns, 0+1:NV+1, NV-H:NV, H:NV-H] - input[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H])
    output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] + dt / h(0) * (fluxLeftYHigh - fluxRightYHigh) * NV * 0.5

    fluxBottomYHigh = -1.0 * lambdaBottomYHigh * (input[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] - input[0:NumberOfUnknowns, 0:NV, NV-H-1:NV-1, H:NV-H])
    fluxTopYHigh = -1.0 * lambdaTopYHigh * (input[0:NumberOfUnknowns, 0:NV, NV-H+1:NV+1, H:NV-H] - input[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H])
    output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] + dt / h(1) * (fluxBottomYHigh - fluxTopYHigh) * NV * 0.5

    fluxNearYHigh = -1.0 * lambdaNearYHigh * (input[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] - input[0:NumberOfUnknowns, 0:NV, NV-H:NV, H-1:NV-H-1])
    fluxFarYHigh = -1.0 * lambdaFarYHigh * (input[0:NumberOfUnknowns, 0:NV, NV-H:NV, H+1:NV-H+1] - input[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H])
    output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] + dt / h(2) * (fluxNearYHigh - fluxFarYHigh) * NV * 0.5

    if flux:
      flux_xYHigh = DataBlock([[0, NumberOfUnknowns], [0-1, NV+1], [NV-H, NV], [H, NV-H], [0, N]], eval("flux", input[:, 0-1:NV+1, NV-H:NV, H:NV-H], x, 0))
      flux_yYHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [NV-H-1, NV+1], [H, NV-H], [0, N]], eval("flux", input[:, 0:NV, NV-H-1:NV+1, H:NV-H], x, 1))
      flux_zYHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [NV-H, NV], [H-1, NV-H+1], [0, N]], eval("flux", input[:, 0:NV, NV-H:NV, H-1:NV-H+1], x, 2))
      output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] + 0.5 * NV * dt / h(0) * (flux_xYHigh[:, 0-1:NV-1, NV-H:NV, H:NV-H] - flux_xYHigh[:, 0+1:NV+1, NV-H:NV, H:NV-H])
      output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] + 0.5 * NV * dt / h(1) * (flux_yYHigh[:, 0:NV, NV-H-1:NV-1, H:NV-H] - flux_yYHigh[:, 0:NV, NV-H+1:NV+1, H:NV-H])
      output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] + 0.5 * NV * dt / h(2) * (flux_zYHigh[:, 0:NV, NV-H:NV, H-1:NV-H-1] - flux_zYHigh[:, 0:NV, NV-H:NV, H+1:NV-H+1])

    if nonconservativeProduct:
       average_xYHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, NV-H:NV, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, NV-H:NV, H:NV-H])
       delta_xYHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV+1, NV-H:NV, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:NV, NV-H:NV, H:NV-H]
       nonconservative_xYHigh = DataBlock([[0, NumberOfUnknowns], [0, NV+1], [NV-H, NV], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_xYHigh, delta_xYHigh, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] + 0.5 * NV * dt / h(0) * (-nonconservative_xYHigh[:, 0:NV, NV-H:NV, H:NV-H] - nonconservative_xYHigh[:, 0+1:NV+1, NV-H:NV, H:NV-H])

       average_yYHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H-1:NV, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H:NV+1, H:NV-H])
       delta_yYHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H:NV+1, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H-1:NV, H:NV-H]
       nonconservative_yYHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [NV-H, NV+1], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_yYHigh, delta_yYHigh, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] + 0.5 * NV * dt / h(1) * (-nonconservative_yYHigh[:, 0:NV, NV-H:NV, H:NV-H] - nonconservative_yYHigh[:, 0:NV, NV-H+1:NV+1, H:NV-H])

       average_zYHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H:NV, H-1:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H:NV, H:NV-H+1])
       delta_zYHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H:NV, H:NV-H+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:NV, NV-H:NV, H-1:NV-H]
       nonconservative_zYHigh = DataBlock([[0, NumberOfUnknowns], [0, NV], [NV-H, NV], [H, NV-H+1], [0, N]], eval("nonconservativeProduct", average_zYHigh, delta_zYHigh, x, h, t, dt, 2))
       output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] = output[0:NumberOfUnknowns, 0:NV, NV-H:NV, H:NV-H] + 0.5 * NV * dt / h(2) * (-nonconservative_zYHigh[:, 0:NV, NV-H:NV, H:NV-H] - nonconservative_zYHigh[:, 0:NV, NV-H:NV, H+1:NV-H+1])

    # Shell part XLow: [0,H) x [H,NV-H) x [H,NV-H)
    output[:, 0:H, H:NV-H, H:NV-H] = input[:, 0:H, H:NV-H, H:NV-H]

    if volumetricSource:
       sourceXLow = DataBlock([[0, NumberOfUnknowns], [0, H], [H, NV-H], [H, NV-H], [0, N]], eval("volumetricSource", input[:, 0:H, H:NV-H, H:NV-H], x, t))
       output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] + dt * sourceXLow

    maxEigenvaluesXXLow = DataBlock([[0, 1], [0-1, H+1], [H, NV-H], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, 0-1:H+1, H:NV-H, H:NV-H], x, 0))
    maxEigenvaluesYXLow = DataBlock([[0, 1], [0, H], [H-1, NV-H+1], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, 0:H, H-1:NV-H+1, H:NV-H], x, 1))
    maxEigenvaluesZXLow = DataBlock([[0, 1], [0, H], [H, NV-H], [H-1, NV-H+1], [0, N]], eval("maxEigenvalue", input[:, 0:H, H:NV-H, H-1:NV-H+1], x, 2))

    lambdaLeftXLow = max(maxEigenvaluesXXLow[0:1, 0-1:H-1, H:NV-H, H:NV-H], maxEigenvaluesXXLow[0:1, 0:H, H:NV-H, H:NV-H])
    lambdaRightXLow = max(maxEigenvaluesXXLow[0:1, 0:H, H:NV-H, H:NV-H], maxEigenvaluesXXLow[0:1, 0+1:H+1, H:NV-H, H:NV-H])
    lambdaBottomXLow = max(maxEigenvaluesYXLow[0:1, 0:H, H-1:NV-H-1, H:NV-H], maxEigenvaluesYXLow[0:1, 0:H, H:NV-H, H:NV-H])
    lambdaTopXLow = max(maxEigenvaluesYXLow[0:1, 0:H, H:NV-H, H:NV-H], maxEigenvaluesYXLow[0:1, 0:H, H+1:NV-H+1, H:NV-H])
    lambdaNearXLow = max(maxEigenvaluesZXLow[0:1, 0:H, H:NV-H, H-1:NV-H-1], maxEigenvaluesZXLow[0:1, 0:H, H:NV-H, H:NV-H])
    lambdaFarXLow = max(maxEigenvaluesZXLow[0:1, 0:H, H:NV-H, H:NV-H], maxEigenvaluesZXLow[0:1, 0:H, H:NV-H, H+1:NV-H+1])

    fluxLeftXLow = -1.0 * lambdaLeftXLow * (input[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, 0-1:H-1, H:NV-H, H:NV-H])
    fluxRightXLow = -1.0 * lambdaRightXLow * (input[0:NumberOfUnknowns, 0+1:H+1, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] + dt / h(0) * (fluxLeftXLow - fluxRightXLow) * NV * 0.5

    fluxBottomXLow = -1.0 * lambdaBottomXLow * (input[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, 0:H, H-1:NV-H-1, H:NV-H])
    fluxTopXLow = -1.0 * lambdaTopXLow * (input[0:NumberOfUnknowns, 0:H, H+1:NV-H+1, H:NV-H] - input[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] + dt / h(1) * (fluxBottomXLow - fluxTopXLow) * NV * 0.5

    fluxNearXLow = -1.0 * lambdaNearXLow * (input[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, 0:H, H:NV-H, H-1:NV-H-1])
    fluxFarXLow = -1.0 * lambdaFarXLow * (input[0:NumberOfUnknowns, 0:H, H:NV-H, H+1:NV-H+1] - input[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] + dt / h(2) * (fluxNearXLow - fluxFarXLow) * NV * 0.5

    if flux:
      flux_xXLow = DataBlock([[0, NumberOfUnknowns], [0-1, H+1], [H, NV-H], [H, NV-H], [0, N]], eval("flux", input[:, 0-1:H+1, H:NV-H, H:NV-H], x, 0))
      flux_yXLow = DataBlock([[0, NumberOfUnknowns], [0, H], [H-1, NV-H+1], [H, NV-H], [0, N]], eval("flux", input[:, 0:H, H-1:NV-H+1, H:NV-H], x, 1))
      flux_zXLow = DataBlock([[0, NumberOfUnknowns], [0, H], [H, NV-H], [H-1, NV-H+1], [0, N]], eval("flux", input[:, 0:H, H:NV-H, H-1:NV-H+1], x, 2))
      output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(0) * (flux_xXLow[:, 0-1:H-1, H:NV-H, H:NV-H] - flux_xXLow[:, 0+1:H+1, H:NV-H, H:NV-H])
      output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(1) * (flux_yXLow[:, 0:H, H-1:NV-H-1, H:NV-H] - flux_yXLow[:, 0:H, H+1:NV-H+1, H:NV-H])
      output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(2) * (flux_zXLow[:, 0:H, H:NV-H, H-1:NV-H-1] - flux_zXLow[:, 0:H, H:NV-H, H+1:NV-H+1])

    if nonconservativeProduct:
       average_xXLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:H, H:NV-H, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H+1, H:NV-H, H:NV-H])
       delta_xXLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H+1, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0-1:H, H:NV-H, H:NV-H]
       nonconservative_xXLow = DataBlock([[0, NumberOfUnknowns], [0, H+1], [H, NV-H], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_xXLow, delta_xXLow, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(0) * (-nonconservative_xXLow[:, 0:H, H:NV-H, H:NV-H] - nonconservative_xXLow[:, 0+1:H+1, H:NV-H, H:NV-H])

       average_yXLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H-1:NV-H, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H:NV-H+1, H:NV-H])
       delta_yXLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H:NV-H+1, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H-1:NV-H, H:NV-H]
       nonconservative_yXLow = DataBlock([[0, NumberOfUnknowns], [0, H], [H, NV-H+1], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_yXLow, delta_yXLow, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(1) * (-nonconservative_yXLow[:, 0:H, H:NV-H, H:NV-H] - nonconservative_yXLow[:, 0:H, H+1:NV-H+1, H:NV-H])

       average_zXLow = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H:NV-H, H-1:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H:NV-H, H:NV-H+1])
       delta_zXLow = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H:NV-H, H:NV-H+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, 0:H, H:NV-H, H-1:NV-H]
       nonconservative_zXLow = DataBlock([[0, NumberOfUnknowns], [0, H], [H, NV-H], [H, NV-H+1], [0, N]], eval("nonconservativeProduct", average_zXLow, delta_zXLow, x, h, t, dt, 2))
       output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, 0:H, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(2) * (-nonconservative_zXLow[:, 0:H, H:NV-H, H:NV-H] - nonconservative_zXLow[:, 0:H, H:NV-H, H+1:NV-H+1])

    # Shell part XHigh: [NV-H,NV) x [H,NV-H) x [H,NV-H)
    output[:, NV-H:NV, H:NV-H, H:NV-H] = input[:, NV-H:NV, H:NV-H, H:NV-H]

    if volumetricSource:
       sourceXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H, NV], [H, NV-H], [H, NV-H], [0, N]], eval("volumetricSource", input[:, NV-H:NV, H:NV-H, H:NV-H], x, t))
       output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] + dt * sourceXHigh

    maxEigenvaluesXXHigh = DataBlock([[0, 1], [NV-H-1, NV+1], [H, NV-H], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, NV-H-1:NV+1, H:NV-H, H:NV-H], x, 0))
    maxEigenvaluesYXHigh = DataBlock([[0, 1], [NV-H, NV], [H-1, NV-H+1], [H, NV-H], [0, N]], eval("maxEigenvalue", input[:, NV-H:NV, H-1:NV-H+1, H:NV-H], x, 1))
    maxEigenvaluesZXHigh = DataBlock([[0, 1], [NV-H, NV], [H, NV-H], [H-1, NV-H+1], [0, N]], eval("maxEigenvalue", input[:, NV-H:NV, H:NV-H, H-1:NV-H+1], x, 2))

    lambdaLeftXHigh = max(maxEigenvaluesXXHigh[0:1, NV-H-1:NV-1, H:NV-H, H:NV-H], maxEigenvaluesXXHigh[0:1, NV-H:NV, H:NV-H, H:NV-H])
    lambdaRightXHigh = max(maxEigenvaluesXXHigh[0:1, NV-H:NV, H:NV-H, H:NV-H], maxEigenvaluesXXHigh[0:1, NV-H+1:NV+1, H:NV-H, H:NV-H])
    lambdaBottomXHigh = max(maxEigenvaluesYXHigh[0:1, NV-H:NV, H-1:NV-H-1, H:NV-H], maxEigenvaluesYXHigh[0:1, NV-H:NV, H:NV-H, H:NV-H])
    lambdaTopXHigh = max(maxEigenvaluesYXHigh[0:1, NV-H:NV, H:NV-H, H:NV-H], maxEigenvaluesYXHigh[0:1, NV-H:NV, H+1:NV-H+1, H:NV-H])
    lambdaNearXHigh = max(maxEigenvaluesZXHigh[0:1, NV-H:NV, H:NV-H, H-1:NV-H-1], maxEigenvaluesZXHigh[0:1, NV-H:NV, H:NV-H, H:NV-H])
    lambdaFarXHigh = max(maxEigenvaluesZXHigh[0:1, NV-H:NV, H:NV-H, H:NV-H], maxEigenvaluesZXHigh[0:1, NV-H:NV, H:NV-H, H+1:NV-H+1])

    fluxLeftXHigh = -1.0 * lambdaLeftXHigh * (input[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, NV-H-1:NV-1, H:NV-H, H:NV-H])
    fluxRightXHigh = -1.0 * lambdaRightXHigh * (input[0:NumberOfUnknowns, NV-H+1:NV+1, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] + dt / h(0) * (fluxLeftXHigh - fluxRightXHigh) * NV * 0.5

    fluxBottomXHigh = -1.0 * lambdaBottomXHigh * (input[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, NV-H:NV, H-1:NV-H-1, H:NV-H])
    fluxTopXHigh = -1.0 * lambdaTopXHigh * (input[0:NumberOfUnknowns, NV-H:NV, H+1:NV-H+1, H:NV-H] - input[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] + dt / h(1) * (fluxBottomXHigh - fluxTopXHigh) * NV * 0.5

    fluxNearXHigh = -1.0 * lambdaNearXHigh * (input[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H-1:NV-H-1])
    fluxFarXHigh = -1.0 * lambdaFarXHigh * (input[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H+1:NV-H+1] - input[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H])
    output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] + dt / h(2) * (fluxNearXHigh - fluxFarXHigh) * NV * 0.5

    if flux:
      flux_xXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H-1, NV+1], [H, NV-H], [H, NV-H], [0, N]], eval("flux", input[:, NV-H-1:NV+1, H:NV-H, H:NV-H], x, 0))
      flux_yXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H, NV], [H-1, NV-H+1], [H, NV-H], [0, N]], eval("flux", input[:, NV-H:NV, H-1:NV-H+1, H:NV-H], x, 1))
      flux_zXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H, NV], [H, NV-H], [H-1, NV-H+1], [0, N]], eval("flux", input[:, NV-H:NV, H:NV-H, H-1:NV-H+1], x, 2))
      output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(0) * (flux_xXHigh[:, NV-H-1:NV-1, H:NV-H, H:NV-H] - flux_xXHigh[:, NV-H+1:NV+1, H:NV-H, H:NV-H])
      output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(1) * (flux_yXHigh[:, NV-H:NV, H-1:NV-H-1, H:NV-H] - flux_yXHigh[:, NV-H:NV, H+1:NV-H+1, H:NV-H])
      output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(2) * (flux_zXHigh[:, NV-H:NV, H:NV-H, H-1:NV-H-1] - flux_zXHigh[:, NV-H:NV, H:NV-H, H+1:NV-H+1])

    if nonconservativeProduct:
       average_xXHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H-1:NV, H:NV-H, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV+1, H:NV-H, H:NV-H])
       delta_xXHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV+1, H:NV-H, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H-1:NV, H:NV-H, H:NV-H]
       nonconservative_xXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H, NV+1], [H, NV-H], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_xXHigh, delta_xXHigh, x, h, t, dt, 0))
       output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(0) * (-nonconservative_xXHigh[:, NV-H:NV, H:NV-H, H:NV-H] - nonconservative_xXHigh[:, NV-H+1:NV+1, H:NV-H, H:NV-H])

       average_yXHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H-1:NV-H, H:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H:NV-H+1, H:NV-H])
       delta_yXHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H:NV-H+1, H:NV-H] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H-1:NV-H, H:NV-H]
       nonconservative_yXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H, NV], [H, NV-H+1], [H, NV-H], [0, N]], eval("nonconservativeProduct", average_yXHigh, delta_yXHigh, x, h, t, dt, 1))
       output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(1) * (-nonconservative_yXHigh[:, NV-H:NV, H:NV-H, H:NV-H] - nonconservative_yXHigh[:, NV-H:NV, H+1:NV-H+1, H:NV-H])

       average_zXHigh = 0.5 * (input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H:NV-H, H-1:NV-H] + input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H:NV-H, H:NV-H+1])
       delta_zXHigh = input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H:NV-H, H:NV-H+1] - input[0:NumberOfUnknowns + NumberOfAuxiliaryVariables, NV-H:NV, H:NV-H, H-1:NV-H]
       nonconservative_zXHigh = DataBlock([[0, NumberOfUnknowns], [NV-H, NV], [H, NV-H], [H, NV-H+1], [0, N]], eval("nonconservativeProduct", average_zXHigh, delta_zXHigh, x, h, t, dt, 2))
       output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] = output[0:NumberOfUnknowns, NV-H:NV, H:NV-H, H:NV-H] + 0.5 * NV * dt / h(2) * (-nonconservative_zXHigh[:, NV-H:NV, H:NV-H, H:NV-H] - nonconservative_zXHigh[:, NV-H:NV, H:NV-H, H+1:NV-H+1])


class RusanovWithOptimisticExplicitEuler(Solver.NumericalScheme):
    """!

    Finite Volumes with a Rusanov Riemann solver and optimistic explicit Euler time
    stepping, realised over a block-structured mesh. 
    
    ## Design idea
    
    The finite volume solver operates over a block-structured mesh, where each 
    cell of the mesh holds a patch of finite volumes. This solver's fundamental
    idea is that it splits up the patch into three classes of volumes. Assume
    we have a NxNxN patch of volumes, where N is the patch_size, and this patch
    is surrounded by a halo of size H. So we have in total a (N+2H)x(N+2H)x(N+2H)
    patch that feeds into the time stepping as reconstructed patch.
    We now distinguish three classes of volumes:
    
    - The halo cells
    - The shell cells, which is the outer layer of the NxNxN arrangement.
    - The patch interior cells, which are the remaining volumes in the interior
      of the patch, so effectively a (N-2H)x(N-2H)x(N-2H) patch.
      
    The sketch below shows the three classes for a 2d patch with N=10 and H=1.
    It is generated by
    src/exahype2/api/fv/rusanov-with-optimistic-explicit-euler-illustrations.py.

    @image html rusanov-with-optimistic-explicit-euler-volume-classes.png

    For the new solver, we now define two new update kernels:
    
    - timeStepWithRusanov2DOnPatchInterior which is timeStepWithRusanov2D but acts only on 
      the patch interior.
    - timeStepWithRusanov2DOnShell which works only on the shell.
    
    The 3D variants follow equivalently.

    ## Time stepping

    Every time step either runs the traditional branch (Update cell and
    Discard patch interior) or the optimistic branch (Update cell shell and
    Insert patch interior). The traditional branch runs if the solver's time
    step size is zero, or if the estimated time step size, i.e. the second
    entry of the eigenvalue data, exceeds it. The optimistic branch runs
    otherwise. Project on faces, Compute global CFL condition and Spawn
    patch interior update follow either branch. If at least one axis of the
    domain is not periodic and the solver uses Dirichlet boundary
    conditions, Apply Dirichlet BCs sets the ghost volumes along the domain
    boundary before Reconstruct reads them (@see
    exahype2.api.fv.actionsets.ApplyDirichletBoundaryConditions). With
    Sommerfeld boundary conditions, the task does not exist, and the roll
    over within Project on faces advances the ghost volumes instead (@see
    exahype2.api.fv.actionsets.RollOverUpdatedFaceWithSommerfeldBoundaryConditions).
    If all axes are periodic, there is no boundary treatment at all. The
    task graph below is
    generated by
    src/exahype2/api/fv/rusanov-with-optimistic-explicit-euler-illustrations.py
    from this class for a domain without periodic boundaries.

    @image html rusanov-with-optimistic-explicit-euler-time-step-task-graph.png

    The estimate is the admissible time step size of the current time step,
    i.e. the one the solver could have chosen if it had not been enforced to
    use a smaller one, times damp_time_step_size. Reconstruct
    memorises the admissible time step size per cell (@see
    exahype2.api.fv.actionsets.ReadjustTimeStepSizeForOptimisticGlobalTimestepping),
    and Spawn patch interior update damps it (@see
    exahype2.api.fv.actionsets.SpawnTaskToUpdateCellInterior). As the
    admissible time step size is never damped itself, the estimates do not
    shrink from time step to time step:

    - Time step 0 runs with time step size 0, as the admissible time step
      size is not known yet. Update cell runs but does not change the
      solution, and Compute global CFL condition yields the first time step
      size dt0. There is no admissible time step size to estimate from, so
      the spawn does not update the patch interior.
    - Time step 1 runs with dt0 down the traditional branch. The spawn
      updates the patch interior with damp_time_step_size*dt0.
    - Time step 2 finds this estimate to be smaller than the admissible time
      step size dt1. It enforces the estimate as time step size and runs
      the optimistic branch.

    Compute global CFL condition does not only compute the next time step
    size, it also makes it the solver's time step size and time stamp
    straightaway (@see exahype2.api.fv.actionsets.ComputeGlobalCFLCondition).
    It therefore depends on the spawn, which still sees the time stamp and
    time step size of the current time step and starts the patch interior
    update from their sum. The order in which the tasks are added to the
    task graph does not matter.

    If spawn_optimistic_calculations_as_task is set, the spawn runs the
    patch interior update as a task. Insert patch interior waits for this
    task before it copies the patch interior into the patch, and Discard
    patch interior waits for it, too, so no task is still running when the
    next one is spawned. The task numbers are held per cell.

    """

    def __init__(self, solver: Solver, patch_size: int, overlap: int = 1, cfl_constant: float = 0.9, damp_time_step_size: float = 0.95, spawn_optimistic_calculations_as_task: bool = True, min_volume_size: float = None, max_volume_size: float = None):
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

        damp_time_step_size: float
          Factor, smaller or equal to one, which the current time step size
          is multiplied with to estimate the time step size of the patch
          interior update (@see
          exahype2.api.fv.actionsets.SpawnTaskToUpdateCellInterior).

        spawn_optimistic_calculations_as_task: bool
          If True, Spawn patch interior update spawns the patch interior
          update as a task, and Insert patch interior and Discard patch
          interior wait for this task. The task number is held per cell
          (@see __create_data_structures()). If False, Spawn patch interior
          update runs the patch interior update straightaway.

        min_volume_size: float
          The time step asks for a refinement of a cell only if its volumes
          are at least of this size (@ref exahype2.api.blockstructured.DynamicMeshRefinement).
          None (the default) does not bound the refinement.

        max_volume_size: float
          The time step asks for an erase, i.e. coarsening, of a cell only
          if the volumes of the coarser cell would be strictly smaller than
          this size. None (the default) does not bound the coarsening.

        """
        if damp_time_step_size > 1.0:
            raise Exception(f"damp_time_step_size has to be smaller or equal to 1, but is {damp_time_step_size}")
        super().__init__(solver)
        assert (
            min_volume_size is None
            or max_volume_size is None
            or min_volume_size <= max_volume_size
        ), f"min_volume_size={min_volume_size} has to be smaller than or equal to max_volume_size={max_volume_size}"
        self.min_volume_size = min_volume_size
        self.max_volume_size = max_volume_size
        self.patch_size = patch_size
        self.halo_size = overlap
        self.overlap = overlap
        # Set in construct_output(): the generated kernels live in the project
        # namespace, as their names are solver-specific
        self.kernel_namespace = None
        self.max_eigenvalue_kernel_call = None
        self.cfl_constant = cfl_constant
        self.damp_time_step_size = damp_time_step_size
        self.spawn_optimistic_calculations_as_task = spawn_optimistic_calculations_as_task
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
        over the whole patch. It is a kernel of its own, i.e. none of the
        time step kernels returns an eigenvalue (@ref exahype2.api.fv.actionsets.ComputeMaxEigenvalue).
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
        - _max_eigenvalue_and_time_step_sizes: three doubles per cell, named
          after the solver followed by MaxEigenvalueAndTimeStepSizes (e.g.
          EulerFVSmoothMaxEigenvalueAndTimeStepSizes).
          exahype2.api.fv.RusanovWithExplicitEuler calls the corresponding
          data _max_eigenvalue. exahype2.api.fv.actionsets.ComputeMaxEigenvalue,
          exahype2.api.fv.actionsets.ReduceMaxEigenvalue and
          exahype2.api.fv.actionsets.ComputeGlobalCFLCondition are shared
          with that scheme and hence are handed this data explicitly.
          It is modelled via DaStGen2 as an array of cardinality three. The
          array is a peano4.api.dastgen2.Peano4SmartPointerDoubleArray,
          i.e. the values reside on the heap. The three entries are used as
          follows:
          - Entry 0 holds the maximum eigenvalue of the whole patch, as
            written by exahype2.api.fv.actionsets.ComputeMaxEigenvalue after
            the patch has been updated, either in one go
            (exahype2.api.fv.actionsets.UpdateCell) or as shell plus patch
            interior. exahype2.api.fv.actionsets.ReduceMaxEigenvalue reduces
            entry 0 of all cells.
          - Entry 1 holds the estimated time step size of the patch
            interior update, i.e. entry 2 times damp_time_step_size, as
            written by
            exahype2.api.fv.actionsets.SpawnTaskToUpdateCellInterior.
          - Entry 2 holds the time step size the solver could have chosen
            in the current time step if it had not been enforced to use a
            smaller one, i.e. the admissible time step size, as written by
            exahype2.api.fv.actionsets.ReadjustTimeStepSizeForOptimisticGlobalTimestepping.
        - _patch_interior: the patch interior, i.e. the patch without its
          shell, so (patch_size-2*overlap)^d Finite Volumes. It matches the
          region timeStepWithRusanov2DOnPatchInterior() and
          timeStepWithRusanov3DOnPatchInterior() update.
        - _optimistic_cell_update_task_number: only if
          spawn_optimistic_calculations_as_task is set, otherwise None. One
          integer per cell, named after the solver followed by
          OptimisticCellUpdateTaskNumber (e.g.
          EulerFVSmoothOptimisticCellUpdateTaskNumber), which holds the
          number of the task updating the patch interior, or
          tarch::multicore::TaskEnumerator::NoNumber if there is none. The
          class furthermore holds a static tarch::multicore::TaskEnumerator
          (getTaskEnumerator()) handing out the task numbers.

        All six are assigned onto self._solver (the shared
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
        self._solver._patch_interior = peano4.api.datamodel.Patch(
            (self.patch_size-2*self.overlap, self.patch_size-2*self.overlap, self.patch_size-2*self.overlap),
            total_unknowns,
            identifier + "PatchInterior",
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
        self._solver._max_eigenvalue_and_time_step_sizes = peano4.api.datamodel.DaStGen2(self._solver.name + "MaxEigenvalueAndTimeStepSizes")
        self._solver._max_eigenvalue_and_time_step_sizes.data.add_attribute(
            peano4.api.dastgen2.Peano4SmartPointerDoubleArray("maxEigenvalue", 3)
        )

        add_max_eigenvalue_reduction(self._solver)

        self._solver._optimistic_cell_update_task_number = None
        if self.spawn_optimistic_calculations_as_task:
            self._solver._optimistic_cell_update_task_number = peano4.api.datamodel.DaStGen2(
                self._solver.name + "OptimisticCellUpdateTaskNumber"
            )
            self._solver._optimistic_cell_update_task_number.data.add_attribute(
                dastgen2.attributes.Integer(name="taskNumber")
            )
            self._solver._optimistic_cell_update_task_number.data.add_attribute(
                dastgen2.attributes.UserDefinedType(
                    name="taskEnumerator",
                    type="tarch::multicore::TaskEnumerator",
                    include='#include "tarch/multicore/TaskEnumerator.h"',
                    qualifier=dastgen2.attributes.Attribute.Qualifier.STATIC,
                    user_type_has_toString_method=False,
                )
            )


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
        peano_project.datamodel.add_cell(self._solver._patch_interior)
        peano_project.datamodel.add_face(self._solver._patch_overlap_current_time_step)
        peano_project.datamodel.add_face(self._solver._patch_overlap_update)
        peano_project.datamodel.add_cell(self._solver._max_eigenvalue_and_time_step_sizes)
        if self._solver._optimistic_cell_update_task_number is not None:
            peano_project.datamodel.add_cell(self._solver._optimistic_cell_update_task_number)
            self._solver._optimistic_cell_update_task_number.grid_selector = self._solver.grid_selector

        # The task graph compiler only uses, loads and stores data where the
        # solver's grid selector holds, too (@see peano4.api.datamodel.DoF).
        for patch in (
            self._solver._patch,
            self._solver._patch_with_halo,
            self._solver._patch_interior,
            self._solver._patch_overlap_current_time_step,
            self._solver._patch_overlap_update,
            self._solver._max_eigenvalue_and_time_step_sizes,
        ):
            patch.grid_selector = self._solver.grid_selector

        self._solver._patch.generator =peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
            self._solver._patch, "double"
        )
        self._solver._patch_with_halo.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
            self._solver._patch_with_halo, "double"
        )
        self._solver._patch_interior.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
            self._solver._patch_interior, "double"
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
        if self._solver.auxiliary_variables:
            raise NotImplementedError(
                f"Solver {self._solver.name}: RusanovWithOptimisticExplicitEuler does not support "
                "auxiliary variables yet. Use RusanovWithExplicitEuler."
            )
        peano4.api.Logger().log("Create time stepping task graph for Finite Volume scheme over block-structured meshes with Rusanov flux")

        self.__add_data_to_repository(peano_project)

        self.kernel_namespace = peano_project.namespace
        # Construct the call first, as it validates the PDE terms' variants
        self._solver.compute_kernel_call = self.__construct_kernel_call(Target.Serial, False)
        self._solver.compute_kernel_call_on_patch_interior = self.__construct_kernel_call(Target.Serial, False, self.kernel_function_name + "OnPatchInterior")
        self._solver.compute_kernel_call_on_shell = self.__construct_kernel_call(Target.Serial, False, self.kernel_function_name + "OnShell")
        self.max_eigenvalue_kernel_call = self.__construct_kernel_call(Target.Serial, False, self.max_eigenvalue_kernel_function_name)
        self.__create_time_step_functions(peano_project.directory)
        
        reconstruction_task           = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Reconstruct", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        update_cell_task              = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Update cell", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        update_cell_shell_task        = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Update cell shell", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        insert_patch_interior_task    = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Insert patch interior", entity=peano4.api.taskgraph.Task.GridEntity.CELL, may_spawn_background_tasks=False)
        discard_patch_interior_task   = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Discard patch interior", entity=peano4.api.taskgraph.Task.GridEntity.CELL, may_spawn_background_tasks=False)
        project_on_faces_task         = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Project on faces", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        compute_eigenvalue_task       = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Compute max eigenvalue", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        reduce_eigenvalue_task        = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Reduce max eigenvalue", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
        compute_cfl_condition_task    = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Compute global CFL condition", entity=peano4.api.taskgraph.Task.GridEntity.GLOBAL_END_OF_STEP, may_spawn_background_tasks=False)
        spawn_patch_interior_task     = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Spawn patch interior update", entity=peano4.api.taskgraph.Task.GridEntity.CELL, may_spawn_background_tasks=False)

        time_step.add_task(reconstruction_task)
        time_step.add_task(update_cell_task)
        time_step.add_task(update_cell_shell_task)
        time_step.add_task(insert_patch_interior_task)
        time_step.add_task(discard_patch_interior_task)
        time_step.add_task(compute_eigenvalue_task)
        time_step.add_task(reduce_eigenvalue_task)
        time_step.add_task(project_on_faces_task)
        time_step.add_task(compute_cfl_condition_task)
        time_step.add_task(spawn_patch_interior_task)

        ## Build the reconstruction tasks befilling the patch_with_halo, which is logically temporary
        reconstruction_task.use_cell(self._solver._patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        reconstruction_task.use_cell(self._solver._patch_with_halo, peano4.api.taskgraph.Task.DataAccess.WRITE)
        ## Sommerfeld boundary conditions read the old face data within the roll
        ## over, i.e. it has to survive the reconstruction
        reconstruction_task.use_face(
            self._solver._patch_overlap_current_time_step,
            get_access_to_persistent_face_data(self._solver, exahype2_project, peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD),
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
        ## Validate the estimated time step size of the optimistic patch
        ## interior update, and memorise the admissible time step size
        reconstruction_task.use_cell(self._solver._max_eigenvalue_and_time_step_sizes, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        reconstruction_task.add_action_set(
            ReadjustTimeStepSizeForOptimisticGlobalTimestepping(
                self._solver,
                guard=self._solver.grid_selector.createCellGuard(),
            )
        )

        ## Guards of the two branches. Both are combined with the grid
        ## selector's guard, i.e. both have to hold.
        ## - The traditional branch (Update cell, Discard patch interior) runs
        ##   if the estimated time step size, i.e. entry 1 of _max_eigenvalue_and_time_step_sizes,
        ##   is bigger than the solver's time step size (@see
        ##   ReadjustTimeStepSizeForOptimisticGlobalTimestepping, which sets
        ##   overestimated time step sizes to the maximum double value), or
        ##   if the time step size is zero, i.e. in the very first time step.
        ## - The optimistic branch (Update cell shell, Insert patch interior)
        ##   runs if and only if the traditional condition does not hold. As
        ##   ReadjustTimeStepSizeForOptimisticGlobalTimestepping enforces any
        ##   estimate that is smaller than or equal to the solver's time step
        ##   size, this means that the estimate equals the time step size.
        ## The projection onto the faces, the CFL condition and the spawn of
        ## the next patch interior update follow both branches and are guarded
        ## by the grid selector only.
        max_eigenvalue = f"fineGridCell{self._solver._max_eigenvalue_and_time_step_sizes.name}"
        # Cell data holding the number of the task which updates the patch
        # interior, or None if the patch interior is updated straightaway
        task_number = self._solver._optimistic_cell_update_task_number
        time_step_size = f"repositories::{self._solver.get_name_of_global_instance()}.getMinTimeStepSize()"
        cell_guard = self._solver.grid_selector.createCellGuard()
        ## The time step size is tested first: C++ evaluates the expressions
        ## from left to right and stops as soon as the result is known, so
        ## entry 1 is never read if the time step size is zero.
        ## The optimistic condition is the exact negation of the traditional
        ## one, so every cell runs exactly one of the two branches.
        traditional_condition = (
            f"(tarch::la::equals({time_step_size}, 0.0) or tarch::la::greater({max_eigenvalue}.getMaxEigenvalue(1), {time_step_size}))"
        )
        traditional_guard = f"({cell_guard}) and {traditional_condition}"
        optimistic_guard = f"({cell_guard}) and not {traditional_condition}"


        ## Traditional branch: update the whole patch
        update_cell_task.add_dependency(reconstruction_task)
        update_cell_task.use_cell(self._solver._patch_with_halo, peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD)
        update_cell_task.use_cell(self._solver._patch,           peano4.api.taskgraph.Task.DataAccess.WRITE)
        update_cell_task.use_cell(self._solver._max_eigenvalue_and_time_step_sizes,  peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        update_cell_task.add_action_set(
            UpdateCell(
                self._solver,
                guard=traditional_guard,
            )
        )
        update_cell_task.tags.add(Solver.SolutionUpdateTaskTag)

        ## Traditional branch: the update makes any precomputed patch interior obsolete
        discard_patch_interior_task.add_dependency(update_cell_task)
        if task_number is not None:
            discard_patch_interior_task.use_cell(task_number, peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD)
        discard_patch_interior_task.use_cell(self._solver._max_eigenvalue_and_time_step_sizes, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        discard_patch_interior_task.add_action_set(
            DiscardPatchInterior(
                self._solver,
                guard=traditional_guard,
                task_number=task_number,
            )
        )

        ## Optimistic branch: update the shell only. The patch interior of
        ## _patch and entry 1 of _max_eigenvalue_and_time_step_sizes remain unchanged, so both are
        ## read and written
        update_cell_shell_task.add_dependency(reconstruction_task)
        update_cell_shell_task.use_cell(self._solver._patch_with_halo, peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD)
        update_cell_shell_task.use_cell(self._solver._patch,           peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        update_cell_shell_task.use_cell(self._solver._max_eigenvalue_and_time_step_sizes,  peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        update_cell_shell_task.add_action_set(
            UpdateCellShell(
                self._solver,
                guard=optimistic_guard,
            )
        )

        ## Optimistic branch: insert the patch interior once the shell is updated
        insert_patch_interior_task.add_dependency(update_cell_shell_task)
        if task_number is not None:
            insert_patch_interior_task.use_cell(task_number, peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD)
        insert_patch_interior_task.use_cell(self._solver._patch_interior, peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD)
        insert_patch_interior_task.use_cell(self._solver._patch,          peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        insert_patch_interior_task.use_cell(self._solver._max_eigenvalue_and_time_step_sizes, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        insert_patch_interior_task.add_action_set(
            InsertPatchInterior(
                self._solver,
                self.patch_size,
                self.overlap,
                guard=optimistic_guard,
                task_number=task_number,
            )
        )
        insert_patch_interior_task.tags.add(Solver.SolutionUpdateTaskTag)

        ## Both branches: determine the maximum eigenvalue of the updated patch,
        ## store it per cell and reduce it over all cells. The patch is
        ## updated completely by the traditional update or, in the optimistic
        ## branch, once the patch interior is inserted. The pair runs in
        ## parallel to the spawn of the next time step's patch interior update.
        ## The projection onto the faces depends on the patch update only,
        ## i.e. it runs in parallel to the eigenvalue and its reduction. The
        ## update, the eigenvalue, the reduction and the projection fold into
        ## one mesh traversal
        compute_eigenvalue_task.add_dependency(update_cell_task)
        compute_eigenvalue_task.add_dependency(insert_patch_interior_task)
        compute_eigenvalue_task.use_cell(self._solver._patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        compute_eigenvalue_task.use_cell(self._solver._max_eigenvalue_and_time_step_sizes, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        compute_eigenvalue_task.add_action_set(
            ComputeMaxEigenvalue(
                self._solver,
                guard=cell_guard,
                max_eigenvalue=self._solver._max_eigenvalue_and_time_step_sizes,
            )
        )
        reduce_eigenvalue_task.add_dependency(compute_eigenvalue_task)
        reduce_eigenvalue_task.use_cell(self._solver._max_eigenvalue_and_time_step_sizes, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        reduce_eigenvalue_task.add_action_set(
            ReduceMaxEigenvalue(
                self._solver,
                guard=cell_guard,
                max_eigenvalue=self._solver._max_eigenvalue_and_time_step_sizes,
            )
        )

        ## Both branches: derive the next time step size from the reduced
        ## eigenvalue. This acts in unprepareTraversal() at the end of the
        ## time step only. The spawn has to see the solver's time stamp and
        ## time step size before the CFL condition advances them
        compute_cfl_condition_task.add_dependency(reduce_eigenvalue_task)
        compute_cfl_condition_task.add_dependency(spawn_patch_interior_task)
        ## The roll over of Sommerfeld boundary conditions is part of the
        ## projection and reads the current time step size, which the next task
        ## replaces. It therefore never runs before the roll over has terminated
        ## (see src/exahype2/api/fv/adrs/0002-time-step-size-after-roll-over.md)
        compute_cfl_condition_task.add_dependency(project_on_faces_task)
        compute_cfl_condition_task.add_action_set(
            ComputeGlobalCFLCondition(
                self._solver,
                self.cfl_constant,
                self.patch_size,
            )
        )

        ## Both branches: project the new data onto the faces. Also roll over the solution
        project_on_faces_task.add_dependency(update_cell_task)
        project_on_faces_task.add_dependency(insert_patch_interior_task)
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
                guard=cell_guard,
                includes=self._solver._get_default_includes() + self._solver.user_solver_includes,
            )
        )
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

        ## Both branches: spawn the next time step's patch interior update.
        ## Reads the patch and writes the patch interior plus the estimated
        ## time step size
        spawn_patch_interior_task.add_dependency(update_cell_task)
        spawn_patch_interior_task.add_dependency(insert_patch_interior_task)
        spawn_patch_interior_task.use_cell(self._solver._patch,          peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        spawn_patch_interior_task.use_cell(self._solver._max_eigenvalue_and_time_step_sizes, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        spawn_patch_interior_task.use_cell(self._solver._patch_interior, peano4.api.taskgraph.Task.DataAccess.WRITE)
        if task_number is not None:
            spawn_patch_interior_task.use_cell(task_number, peano4.api.taskgraph.Task.DataAccess.WRITE)
        spawn_patch_interior_task.add_action_set(
            SpawnTaskToUpdateCellInterior(
                self._solver,
                self.damp_time_step_size,
                guard=cell_guard,
                task_number=task_number,
            )
        )

        # The load balancing is switched off throughout the time step, and
        # switched on again only at its very end, so it may then decide on
        # splits. Peano carries these splits out in the subsequent mesh sweep.
        if add_global_tasks:
            self._solver.numerical_scheme.wrap_task_graph_in_load_balancing_and_amr_tasks(time_step)

        if peano4.api.Logger().is_visualisation:
            time_step.visualise(title="Time stepping task graph for {}{}".format(self._solver.name, self._solver.task_graph_title_suffix))
        if peano4.api.Logger().is_text:
            time_step.log_as_text(title="Time stepping task graph for {}{}".format(self._solver.name, self._solver.task_graph_title_suffix))

    def __create_time_step_functions(self, output_directory):
        """!
        
        Calling this function is delayed until all PDE term information is provided.
        In this way we can avoid generating code that will not be used. The following
        information is required to translate a Python function using DSLHyPE:
        - template_parameters: the variables that are used within the Python function
          to represent external information
        - functor_parameters: the PDE terms that can be used by the solver
        - function: the Python function itself

        TODO: Currently the MLIR types are passed manually here, this should ideally
        be inferred

        """

        template_parameters = [
            dslhype.SyntaxTree.Argument("NV", dslhype.SyntaxTree.TInteger()),
            dslhype.SyntaxTree.Argument("H", dslhype.SyntaxTree.TInteger()),
            dslhype.SyntaxTree.Argument("NumberOfUnknowns", dslhype.SyntaxTree.TInteger()),
            dslhype.SyntaxTree.Argument("NumberOfAuxiliaryVariables", dslhype.SyntaxTree.TInteger())
        ]
    
        functor_parameters = [
            dslhype.SyntaxTree.Argument("flux", dslhype.SyntaxTree.TCustom("const ::exahype2::fv::rusanov::FluxFunctor&"), "(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()"),
            dslhype.SyntaxTree.Argument("nonconservativeProduct", dslhype.SyntaxTree.TCustom("const ::exahype2::fv::rusanov::NonconservativeProductFunctor&"), "(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()"),
            dslhype.SyntaxTree.Argument("volumetricSource", dslhype.SyntaxTree.TCustom("const ::exahype2::fv::rusanov::VolumetricSourceFunctor&"), "(!llvm.ptr, !llvm.ptr, f64, !llvm.ptr) -> ()"),
            dslhype.SyntaxTree.Argument("maxEigenvalue", dslhype.SyntaxTree.TCustom("const ::exahype2::fv::rusanov::MaxEigenvalueFunctor&"), "(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()")
        ]

        time_step_dsl_function_2d = exahype2.api.fv.timeStepWithRusanov2D
        time_step_dsl_function_3d = exahype2.api.fv.timeStepWithRusanov3D
        dslhype.translate_dslhype_function(
            function = time_step_dsl_function_2d,
            function_namespace = self.kernel_namespace,
            template_parameters=template_parameters,
            functor_parameters=functor_parameters,
            pde_terms=self._solver.pde_terms,
            output_file_name=self._solver.name + "_rusanov2d",
            target=dslhype.Target.Serial,
            stateless=False,
            function_name=self.kernel_function_name,
            output_directory=output_directory,
        )
        dslhype.translate_dslhype_function(
            function = time_step_dsl_function_3d,
            function_namespace = self.kernel_namespace,
            template_parameters=template_parameters,
            functor_parameters=functor_parameters,
            pde_terms=self._solver.pde_terms,
            output_file_name=self._solver.name + "_rusanov3d",
            target=dslhype.Target.Serial,
            stateless=False,
            function_name=self.kernel_function_name,
            output_directory=output_directory,
        )
        # The maximum eigenvalue over the whole patch is a kernel of its own
        for dimensions, function in [(2, maxEigenvalueOverPatch2D), (3, maxEigenvalueOverPatch3D)]:
            dslhype.translate_dslhype_function(
                function = function,
                function_namespace = self.kernel_namespace,
                template_parameters=template_parameters,
                functor_parameters=functor_parameters,
                pde_terms=self._solver.pde_terms,
                output_file_name=f"{self._solver.name}_rusanov{dimensions}d",
                target=dslhype.Target.Serial,
                stateless=False,
                function_name=self.max_eigenvalue_kernel_function_name,
                output_directory=output_directory,
            )
        # Kernels on the patch interior and on the shell
        for function, file_suffix, function_suffix in [
            (timeStepWithRusanov2DOnPatchInterior, "_rusanov2d_patch_interior", "OnPatchInterior"),
            (timeStepWithRusanov3DOnPatchInterior, "_rusanov3d_patch_interior", "OnPatchInterior"),
            (timeStepWithRusanov2DOnShell,         "_rusanov2d_shell",          "OnShell"),
            (timeStepWithRusanov3DOnShell,         "_rusanov3d_shell",          "OnShell"),
        ]:
            dslhype.translate_dslhype_function(
                function = function,
                function_namespace = self.kernel_namespace,
                template_parameters=template_parameters,
                functor_parameters=functor_parameters,
                pde_terms=self._solver.pde_terms,
                output_file_name=self._solver.name + file_suffix,
                target=dslhype.Target.Serial,
                stateless=False,
                function_name=self.kernel_function_name + function_suffix,
                output_directory=output_directory,
            )
        file = open(os.path.join(output_directory, "kernels", f"{self._solver.name}_rusanov.h"), "w")
        file.write(f"""#if Dimensions == 2
    #include "{self._solver.name}_rusanov2d.h"
    #include "{self._solver.name}_rusanov2d_patch_interior.h"
    #include "{self._solver.name}_rusanov2d_shell.h"
#else
    #include "{self._solver.name}_rusanov3d.h"
    #include "{self._solver.name}_rusanov3d_patch_interior.h"
    #include "{self._solver.name}_rusanov3d_shell.h"
#endif""")
        file.close()

    def __construct_kernel_call(self, target: Target, stateless: bool, function_name: str = None):
        """!

        This function constructs the calls to the compute kernels based on the
        application PDE terms and the version chosen.

        @param function_name: str
          Name of the kernel to call. Defaults to kernel_function_name, i.e.
          the kernel updating the whole patch.

        """
        available_terms = ["flux", "nonconservativeProduct", "volumetricSource", "maxEigenvalue"]
        if function_name is None:
            function_name = self.kernel_function_name

        if stateless:
            return f"""::{"::".join(self.kernel_namespace)}::{function_name}<
        {self.patch_size},
        {self.overlap},
        {self._solver.unknowns},
        {self._solver.number_of_auxiliary_variables}
    >(
        patchData,
        {self._solver.name}
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
                    arguments = [
                        parameter.split()[-1]
                        for parameter in re.split(r",(?![^<]*>)", application_term.SIGNATURE)
                    ]
                    # Stateless terms are static and take a dummy tag (@see PDETerm)
                    if application_term.stateless:
                        arguments.append("::exahype2::Solver::Offloadable::Yes")
                    pde_term_calls.append(f"""
    [&](
        {application_term.SIGNATURE}
    )->void {{
        repositories::{self._solver.get_name_of_global_instance()}.{application_term.name}({", ".join(arguments)});
    }}""")
                return f"""::{"::".join(self.kernel_namespace)}::{function_name}<
    {self.patch_size},
    {self.overlap},
    {self._solver.unknowns},
    {self._solver.number_of_auxiliary_variables}
>(
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
