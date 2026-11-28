# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import enum

import jinja2

from peano4.api.solversteps.ActionSet import ActionSet


class RestrictHangingFaces(ActionSet):
    """!

    Restrict the overlap data of hanging faces onto the coarser mesh

    A hanging face is a face of a fine cell which sits on the face of a
    coarser, unrefined cell (toolbox.blockstructured.api.InterpolateHangingFaces).
    The action set plugs into destroyHangingFace(): It takes the half of the
    hanging face which belongs to the fine cell and restricts it onto the
    corresponding half of the coarse face
    (::toolbox::blockstructured::restrictInnerHalfOfHaloLayer_AoS_...).
    Peano destroys a hanging face after touchCellLastTime() of its fine
    cell, i.e. after the fine cell has projected its data onto its faces.
    Peano touches the coarse face for the last time after all of these
    hanging faces have been destroyed. Afterwards, the coarse face holds the
    data of the coarse cell on one side and the restricted fine data on the
    other side.

    The face overlap has to be a 2MxNxN patch: M is the overlap, N the
    number of volumes per patch axis.


    ## Averaging and clearing

    Averaging accumulates the fine grid values within the coarse face. The
    face data have to be cleared before the projection writes them. The
    action set does not do this itself, as it would act in
    touchFaceFirstTime() and thus never share a mesh traversal with a
    preceding cell task. The task graph gets a task of its own which clears
    the faces
    (@ref toolbox.blockstructured.api.ClearFaceProjections.add_clearing_tasks()).
    Injection overwrites the coarse values and hence does not clear
    anything.

    """

    class Scheme(enum.Enum):
        """!

        Restriction schemes

        The value of each entry is the suffix of the C++ routine
        ::toolbox::blockstructured::restrictInnerHalfOfHaloLayer_AoS_<suffix>().

        - Averaging sets each coarse volume to the average of the fine
          volumes along the face which it covers.
        - Inject copies the value of the fine volume in the centre of each
          coarse volume.

        """

        Averaging = "averaging"
        Inject = "inject"

    TemplateDestroyHangingFace = jinja2.Template(
        """
  logTraceInWith1Argument( "destroyHangingFace(...)---RestrictHangingFaces", marker.toString() );
  ::toolbox::blockstructured::restrictInnerHalfOfHaloLayer_AoS_{{SCHEME}}(
    marker,
    {{DOFS_PER_AXIS}},
    {{OVERLAP}},
    {{UNKNOWNS}},
    fineGridFace{{FACE_NAME}}.value,
    coarseGridFaces{{FACE_NAME}}(marker.getSelectedFaceNumber()).value
  );
  logTraceOut( "destroyHangingFace(...)---RestrictHangingFaces" );
"""
    )

    def __init__(
        self,
        patch_overlap,
        scheme: "RestrictHangingFaces.Scheme" = None,
        guard: str = "true",
        clear_guard: str = "true",
        additional_includes: str = "",
    ):
        """!

        @param patch_overlap: peano4.api.datamodel.Patch
          The face data which the action set restricts. The data of the
          hanging face are read, the data of the coarse face are written.

        @param scheme: RestrictHangingFaces.Scheme
          Restriction scheme. Defaults to Scheme.Averaging. Injection
          supports an overlap of one only.

        @param guard: str
          C++ boolean expression over the hanging face's
          peano4::datamanagement::FaceMarker marker.

        @param clear_guard: str
          C++ boolean expression over a persistent face's
          peano4::datamanagement::FaceMarker marker. Only used for averaging:
          The clearing of the face data by
          @ref toolbox.blockstructured.api.ClearFaceProjections uses it.

        @param additional_includes: str
          Further include statements, e.g. required by the guards.

        """
        super().__init__(descend_invocation_order=1, parallel=False)
        assert patch_overlap.dim[0] % 2 == 0, (
            "Patch associated to face has to have even number of volumes along "
            "the normal. Otherwise, it is not a symmetric overlap."
        )
        self.scheme = (
            RestrictHangingFaces.Scheme.Averaging if scheme is None else scheme
        )
        assert isinstance(self.scheme, RestrictHangingFaces.Scheme), self.scheme
        assert (
            self.scheme != RestrictHangingFaces.Scheme.Inject
            or patch_overlap.dim[0] == 2
        ), (
            "Injection supports an overlap of one only, but the face data "
            f"{patch_overlap.name} have an overlap of {patch_overlap.dim[0] // 2}. "
            "Use averaging instead."
        )
        self.d = {
            "UNKNOWNS": patch_overlap.no_of_unknowns,
            "DOFS_PER_AXIS": patch_overlap.dim[1],
            "OVERLAP": patch_overlap.dim[0] // 2,
            "FACE_NAME": patch_overlap.name,
            "SCHEME": self.scheme.value,
        }
        self.patch_overlap = patch_overlap
        self.guard = guard
        self.clear_guard = clear_guard
        self.additional_includes = additional_includes

    @property
    def clears_faces(self) -> bool:
        """! Does the scheme accumulate, i.e. does it have to clear the faces first?"""
        return self.scheme == RestrictHangingFaces.Scheme.Averaging

    def get_constructor_body(self):
        return ""

    def get_destructor_body(self):
        return ""

    def get_body_of_createGridControlEventsForNextTraversal(self):
        return "  return std::vector< peano4::grid::GridControlEvent >();\n"

    def get_body_of_prepareTraversal(self):
        return "\n"

    def get_body_of_unprepareTraversal(self):
        return "\n"

    def get_body_of_operation(self, operation_name):
        if operation_name == ActionSet.OPERATION_DESTROY_HANGING_FACE:
            return self.TemplateDestroyHangingFace.render(**self.d)
        return ""

    def get_guard_of_operation(self, operation_name):
        if operation_name == ActionSet.OPERATION_DESTROY_HANGING_FACE:
            return self.guard
        return "true"

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")

    def user_should_modify_template(self):
        return False

    def get_attributes(self):
        return ""

    def get_includes(self):
        return (
            """
#include "toolbox/blockstructured/Restriction.h"
"""
            + self.additional_includes
        )
