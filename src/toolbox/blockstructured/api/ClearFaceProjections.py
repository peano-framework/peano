# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import jinja2

import peano4.api
import peano4.api.taskgraph
from peano4.api.solversteps.ActionSet import ActionSet

from .RestrictHangingFaces import RestrictHangingFaces


class ClearFaceProjections(ActionSet):
    """!

    Clear the face data which a projection writes and a restriction accumulates

    The action set plugs into touchFaceFirstTime() and clears both halves of
    each face which passes the guard
    (::toolbox::blockstructured::clearHaloLayerAoS()). It belongs to
    toolbox.blockstructured.api.RestrictHangingFaces with averaging, which
    adds the fine data onto the coarse faces. Use
    @ref toolbox.blockstructured.api.ClearFaceProjections.add_clearing_tasks()
    to add the task which hosts the action set.

    """

    Template = jinja2.Template(
        """
  logTraceInWith1Argument( "touchFaceFirstTime(...)---ClearFaceProjections", marker.toString() );
  ::toolbox::blockstructured::clearHaloLayerAoS(
    marker,
    {{DOFS_PER_AXIS}},
    {{OVERLAP}},
    {{UNKNOWNS}},
    fineGridFace{{FACE_NAME}}.value
  );
  logTraceOut( "touchFaceFirstTime(...)---ClearFaceProjections" );
"""
    )

    def __init__(self, restriction: RestrictHangingFaces):
        """!

        @param restriction: toolbox.blockstructured.api.RestrictHangingFaces
          The restriction whose face data (and clear guard) the action set takes over.

        """
        super().__init__(descend_invocation_order=1, parallel=False)
        self.restriction = restriction

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
        if operation_name == ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME:
            return self.Template.render(**self.restriction.d)
        return ""

    def get_guard_of_operation(self, operation_name):
        if operation_name == ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME:
            return self.restriction.clear_guard
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
            + self.restriction.additional_includes
        )


TASK_NAME_SUFFIX = "Clear face projections"
"""! The task which clears the faces for task T is called T's name, an underscore and this suffix """


def add_clearing_tasks(graph: "peano4.api.taskgraph.Graph") -> list:
    """!

    Add a task which clears the face data to each task that restricts by averaging

    Every task of the graph which holds a RestrictHangingFaces action set with averaging gets
    a task of its own (ClearFaceProjections, working on the faces) which it depends on. The
    face data which the projection task writes are TEMPORARY, i.e. clearing and projection have to
    share one mesh traversal. The clearing task therefore waits for the same tasks as the projection
    task, and the projection task waits for it. Placing it earlier is not safe: if
    a task between them is split into producer and consumer, the compiler puts clearing and projection into
    different traversals, and the projection accumulates onto uninitialised memory (see
    src/exahype2/api/rkfd4/adrs/0002-eigenvalue-follows-projection.md).

    @param graph: peano4.api.taskgraph.Graph
      The graph is altered.

    @return [peano4.api.taskgraph.Task]
      The tasks added.

    """
    result = []
    for task in list(graph.tasks):
        restrictions = [
            action_set
            for action_set in task.action_sets
            if isinstance(action_set, RestrictHangingFaces) and action_set.clears_faces
        ]
        if not restrictions:
            continue
        clearing = peano4.api.taskgraph.Task(
            name=f"{task.name}_{TASK_NAME_SUFFIX}",
            entity=peano4.api.taskgraph.Task.GridEntity.FACE,
        )
        for restriction in restrictions:
            assert restriction.patch_overlap in task.face_data_access, (
                f"Task {task.name} restricts {restriction.patch_overlap.name} but does not declare the face data"
            )
            if restriction.patch_overlap not in clearing.face_data_access:
                clearing.use_face(restriction.patch_overlap, task.face_data_access[restriction.patch_overlap])
            clearing.add_action_set(ClearFaceProjections(restriction))
        for dependency in task.dependencies:
            clearing.add_dependency(dependency)
        task.add_dependency(clearing)
        graph.add_task(clearing)
        result.append(clearing)
    return result
