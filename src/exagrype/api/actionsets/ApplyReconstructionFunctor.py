""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

from peano4.api.solversteps.ActionSet import ActionSet


class ApplyReconstructionFunctor(ActionSet):
    """!

    Apply a functor to the reconstructed halos of a cell and write the result into the cell's patch

    The action set hooks into touchCellFirstTime(). It does not reconstruct
    anything: the halos have been filled by earlier tasks (@ref
    toolbox.blockstructured.api.ReconstructPatchWithHalo). The functor, a
    C++ snippet, may use

    - oldQWithHalo: a double* to the reconstructed primary variables;
    - oldDerivativesWithHalo: a double* to the reconstructed derivative
      variables (the previous values of newQ, with halo);
    - newQ: a double* to the patch which the functor writes;
    - marker: the cell marker.

    """

    def __init__(
        self,
        patch,
        primary_variables_with_halo,
        derivative_variables_with_halo,
        functor_implementation: str,
        guard: str = "true",
        additional_includes: str = "",
    ):
        """!

        @param patch: peano4.api.datamodel.Patch
          The patch which the functor writes (newQ).

        @param primary_variables_with_halo: peano4.api.datamodel.Patch
          The reconstructed patch behind oldQWithHalo.

        @param derivative_variables_with_halo: peano4.api.datamodel.Patch
          The reconstructed patch behind oldDerivativesWithHalo.

        @param functor_implementation: str
          C++ snippet.

        @param guard: str
          C++ boolean expression, evaluated per cell in touchCellFirstTime.

        @param additional_includes: str
          Includes that the snippet needs.

        """
        super().__init__(descend_invocation_order=1, parallel=False)
        self._patch = patch
        self._primary = primary_variables_with_halo
        self._derivatives = derivative_variables_with_halo
        self._functor = functor_implementation
        self._guard = guard
        self.additional_includes = additional_includes

    def get_constructor_body(self):
        return ""

    def get_destructor_body(self):
        return ""

    def get_body_of_createGridControlEventsForNextTraversal(self):
        return "  return std::vector< peano4::grid::GridControlEvent >();\n"

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")

    def user_should_modify_template(self):
        return False

    def get_body_of_operation(self, operation_name):
        if operation_name != ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return "\n"
        return f"""
  [[maybe_unused]] double* oldQWithHalo = fineGridCell{self._primary.name}.value;
  [[maybe_unused]] double* oldDerivativesWithHalo = fineGridCell{self._derivatives.name}.value;
  double* newQ = fineGridCell{self._patch.name}.value;
  {self._functor}
"""

    def get_guard_of_operation(self, operation_name):
        if operation_name == ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self._guard
        return "true"

    def get_attributes(self):
        return ""

    def get_includes(self):
        return self.additional_includes
