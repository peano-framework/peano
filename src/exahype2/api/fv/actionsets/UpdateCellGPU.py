# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from .AbstractFVActionSet import AbstractFVActionSet

from exahype2.api.Solver import Solver
from ...AuxiliaryVariables import (
    code_to_assemble_kernel_input,
    code_to_allocate_kernel_output,
    code_to_copy_back_kernel_output,
)
import peano4.api


class UpdateCellGPU(AbstractFVActionSet):
    """!

    Update a Finite Volume patch (solver._patch) from its halo-augmented
    reconstruction (solver._patch_with_halo, see
    toolbox.blockstructured.api.ReconstructPatchWithHalo) by applying the
    solver's numerical scheme (Rusanov flux, explicit Euler time stepping)
    once per volume.

    Global time stepping: all cells share one time stamp and time step
    size, which are taken from the solver (getMinTimeStamp() and
    getMinTimeStepSize()) rather than from per-cell bookkeeping.

    The kernel does not return any eigenvalue: exahype2.api.fv.actionsets.ComputeMaxEigenvalue
    determines it in a task of its own, and
    exahype2.api.fv.actionsets.ComputeGlobalCFLCondition derives the next
    time step size from it.

    """

    def __init__(self, solver: Solver, guard):
        """!

        @param solver: exahype2.api.Solver
          Has to have solver._patch and
          solver._patch_with_halo set - @see
          exahype2.api.fv.RusanovWithExplicitEuler.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime,
          guarding whether that cell's patch is updated. Typically the
          output of the solver's grid_selector (@see
          peano4.api.toolbox.GridSelector).

        """
        AbstractFVActionSet.__init__(self, solver)
        self.guard = guard

    def get_body_of_operation(self, operation_name):
        result = ""

        stride_per_dof = self._solver.unknowns + self._solver.number_of_auxiliary_variables
        input_dofs_per_axis = self._solver.numerical_scheme.patch_size + 2 * self._solver.numerical_scheme.halo_size
        output_dofs_per_axis = self._solver.numerical_scheme.patch_size


        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            solver = self._solver
            unknowns_in  = f"fineGridCell{solver._patch_with_halo.name}.value"
            unknowns_out = f"fineGridCell{solver._patch.name}.value"
            has_parameters = len(solver.auxiliary_variables) > 0
            kernel_in  = "kernelIn.data()" if has_parameters else unknowns_in
            kernel_out = "kernelOut.data()" if has_parameters else unknowns_out
            result = f"""
  #if Dimensions == 2
    int inputStride = {stride_per_dof} * {input_dofs_per_axis} * {input_dofs_per_axis};
    int outputStride = {stride_per_dof} * {output_dofs_per_axis} * {output_dofs_per_axis};
  #else
    int inputStride = {stride_per_dof} * {input_dofs_per_axis} * {input_dofs_per_axis} * {input_dofs_per_axis};
    int outputStride = {stride_per_dof} * {output_dofs_per_axis} * {output_dofs_per_axis} * {output_dofs_per_axis};
  #endif

  double timeStamp    = repositories::{self._solver.get_name_of_global_instance()}.getMinTimeStamp();
  double timeStepSize = repositories::{self._solver.get_name_of_global_instance()}.getMinTimeStepSize();
  {code_to_assemble_kernel_input(solver, "kernelIn", unknowns_in, solver.numerical_scheme.halo_size)}
  {code_to_allocate_kernel_output(solver, "kernelOut")}
  ::exahype2::CellData patchData({kernel_in}, marker.x(), marker.h(), timeStamp, timeStepSize, {kernel_out});
  ::exahype2::CopyCellDataGPU gpuPatchData(patchData, inputStride, outputStride);
  {self._solver.compute_kernel_call}
  gpuPatchData.copyBackAndDestroy();
  {code_to_copy_back_kernel_output(solver, "kernelOut", unknowns_out)}

"""
        return result

    def get_guard_of_operation(self, operation_name):
        if operation_name == peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self.guard
        return "true"

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")
