# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import dataclasses
from typing import Dict, List, Optional

from peano4.api.solversteps.ActionSet import ActionSet

VERTEX = "vertex"
FACE = "face"
CELL = "cell"
ENTITIES = (VERTEX, FACE, CELL)
"""! The entities in the order in which Peano touches them for the first time"""


@dataclasses.dataclass(frozen=True)
class EntitySpec:
    """!

    What the wrapper has to know about one entity kind (vertex, face or cell)

    """

    kind: str
    first_event: str
    last_event: str
    destroy_events: tuple
    own_label: str
    """! Name of the entity's own label within the events of this entity, with the placeholder {labels} for the label type"""


SPECS: Dict[str, EntitySpec] = {
    VERTEX: EntitySpec(
        VERTEX,
        ActionSet.OPERATION_TOUCH_VERTEX_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_VERTEX_LAST_TIME,
        (ActionSet.OPERATION_DESTROY_PERSISTENT_VERTEX,),
        "fineGridVertex{labels}",
    ),
    FACE: EntitySpec(
        FACE,
        ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_FACE_LAST_TIME,
        (ActionSet.OPERATION_DESTROY_PERSISTENT_FACE,),
        "fineGridFace{labels}",
    ),
    CELL: EntitySpec(
        CELL,
        ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME,
        ActionSet.OPERATION_TOUCH_CELL_LAST_TIME,
        (ActionSet.OPERATION_DESTROY_CELL,),
        "fineGridCell{labels}",
    ),
}

BARRIER_OPERATIONS = (ActionSet.OPERATION_END_TRAVERSAL,)
"""! Events which have to wait for all tasks. The unprepareTraversal() is handled separately, as it is no operation of the action set. """


@dataclasses.dataclass
class EntityAttributes:
    """!

    Names of the attributes of one wrapper within the task labels of one entity

    first and last hold the numbers of the tasks of the first and the last event
    (NoNumber if the wrapper spawns none). around holds the numbers of the tasks of
    the cells around a vertex or a face, written by the cells. children holds the numbers of
    the last tasks of the fine entities within the entity (an array for faces and
    cells, a single number for vertices). The names are None if the wrapper does not need
    them.

    """

    first: str
    last: str
    around: Optional[str] = None
    children: Optional[str] = None


def accessor(attribute_name: str) -> str:
    """! Name of the getter or setter without the get or set prefix"""
    return attribute_name[0].upper() + attribute_name[1:]


class SpawnTaskActionSet(ActionSet):
    """!

    Producer-side wrapper around an action set, which spawns all six touch events as tasks

    The wrapper turns each of touchVertexFirstTime(), touchVertexLastTime(),
    touchFaceFirstTime(), touchFaceLastTime(), touchCellFirstTime() and
    touchCellLastTime() into a task, provided the wrapped action set has a
    non-empty body for the event. An entity holds two task numbers per
    wrapped action set, one for the first and one for the last event. They
    are independent. The task graph compiler
    @ref peano4.api.taskgraph.compiler.TaskGraphAssembly describes the dependencies.

    The wrapper delegates all other events to the wrapped action set with
    these exceptions:

    - The touch...FirstTime() events additionally reset the arrays of the
      entity (cells around it, children), reserve the number of the last
      task, announce it to the coarse entity, and (cells only) announce
      the cell's number to the vertices and faces around the cell.
    - The destroy events of the entities wait for the tasks of the entity
      before they delegate, as the entity is gone afterwards.
    - endTraversal() and unprepareTraversal() wait for all tasks before they
      delegate, if the wrapped action set implements them. They realise
      global activities which might use anything the tasks produce.

    The spawned tasks capture their environment by value. The comments in
    @ref peano4.api.taskgraph.compiler.SpawnCellTaskActionSet.SpawnCellTaskActionSet
    on smart pointers and enumerators hold here, too, for vertices and faces.

    """

    def __init__(
        self,
        action_set: ActionSet,
        labels: Dict[str, str],
        attributes: Dict[str, EntityAttributes],
        earlier: List["SpawnTaskActionSet"],
        needs_around: Dict[str, bool],
        face_enumerators: List[str],
        vertex_enumerators: List[str],
        task=None,
    ):
        """!

        @param action_set: peano4.api.solversteps.ActionSet
          The wrapped action set.

        @param labels: {str: str}
          Name of the task label type per entity (VERTEX, FACE, CELL). Only
          entities that exist at all in this compilation are keys.

        @param attributes: {str: EntityAttributes}
          Names of the attributes of this wrapper per entity. Only entities for
          which the wrapper spawns a task are keys.

        @param earlier: [SpawnTaskActionSet]
          The wrappers whose tasks come before the ones of this wrapper in the
          traversal, in the order of their execution. They are the wrappers
          of the same task and of all tasks that this task depends on.

        @param needs_around: {str: bool}
          Per entity (VERTEX, FACE), whether the cells around the entity have
          to announce their tasks to this wrapper's attributes.

        @param face_enumerators: [str]
          Names of the face enumerators, e.g. fineGridFacesQ, that the tasks capture
          as peano4::datamanagement::FaceHeapDataEnumerator.

        @param vertex_enumerators: [str]
          Names of the vertex enumerators, e.g. fineGridVerticesQ, that the tasks capture
          as peano4::datamanagement::VertexHeapDataEnumerator.

        @param task: object
          Identifies the task which the wrapper belongs to. Wrappers of the same
          task reset and fill each other's arrays within a traversal. Numbers of
          the wrappers of other tasks are only read as plain numbers.

        """
        super().__init__(
            descend_invocation_order=action_set.descend_invocation_order,
            parallel=action_set.parallel,
        )
        self.action_set = action_set
        self.labels = dict(labels)
        self.attributes = dict(attributes)
        self.earlier = list(earlier)
        self.needs_around = dict(needs_around)
        self.task = task
        self._face_enumerators = list(face_enumerators)
        self._vertex_enumerators = list(vertex_enumerators)

    # Queries ------------------------------------------------------------

    def has_body_of(self, operation_name: str) -> bool:
        return self.action_set.get_body_of_operation(operation_name).strip() != ""

    def spawns(self, kind: str, first: bool) -> bool:
        spec = SPECS[kind]
        return self.has_body_of(spec.first_event if first else spec.last_event)

    def _me(self, kind: str) -> str:
        return SPECS[kind].own_label.format(labels=self.labels[kind])

    def _lambda_captures(self) -> str:
        captures = ["="]
        captures += [
            f"{name} = peano4::datamanagement::FaceHeapDataEnumerator({name})"
            for name in self._face_enumerators
        ]
        captures += [
            f"{name} = peano4::datamanagement::VertexHeapDataEnumerator({name})"
            for name in self._vertex_enumerators
        ]
        return ", ".join(captures)

    def _chain(
        self, kind: str, which: str, include_self: bool, scope: str = "all"
    ) -> List["SpawnTaskActionSet"]:
        """!

        Wrappers with an attribute on the entity, latest first

        @param scope: str
          all, or ancestors for the wrappers of the tasks which this task depends on, or own for the wrappers of this task.

        """
        wrappers = self.earlier + ([self] if include_self else [])
        if scope == "ancestors":
            wrappers = [
                wrapper for wrapper in wrappers if wrapper.task is not self.task
            ]
        elif scope == "own":
            wrappers = [wrapper for wrapper in wrappers if wrapper.task is self.task]
        return [wrapper for wrapper in reversed(wrappers) if kind in wrapper.attributes]

    def _latest(
        self,
        kind: str,
        which: str,
        entity: str,
        include_self: bool,
        scope: str = "all",
    ) -> str:
        """!

        C++ expression for the number of the latest first or last task on an entity

        @param entity: str
          C++ expression of the entity's label object, e.g. fineGridCellX or fineGridVerticesX(i)

        """
        chain = self._chain(kind, which, include_self, scope)
        if not chain:
            return "tarch::multicore::TaskEnumerator::NoNumber"
        getters = ", ".join(
            f"{entity}.get{accessor(getattr(wrapper.attributes[kind], which))}()"
            for wrapper in chain
        )
        return f"peano4::datamanagement::taskassembly::latestTask({{ {getters} }})"

    @staticmethod
    def _add(expression: str, indentation: str = "  ") -> str:
        """! Statement that adds the number to inDependencies, empty if the number is known to be NoNumber"""
        if expression == "tarch::multicore::TaskEnumerator::NoNumber":
            return ""
        return f"{indentation}peano4::datamanagement::taskassembly::addTask( inDependencies, {expression} );\n"

    def _all_arrays(
        self, kind: str, include_self: bool = True
    ) -> List["SpawnTaskActionSet"]:
        wrappers = self.earlier + ([self] if include_self else [])
        return [w for w in wrappers if kind in w.attributes and w.task is self.task]

    # Code generation -----------------------------------------------------

    def _reset(self, kind: str) -> str:
        attributes = self.attributes.get(kind)
        if attributes is None:
            return ""
        me = self._me(kind)
        result = f"  {me}.set{accessor(attributes.first)}( tarch::multicore::TaskEnumerator::NoNumber );\n"
        result += f"  {me}.set{accessor(attributes.last)}( tarch::multicore::TaskEnumerator::NoNumber );\n"
        if attributes.around is not None:
            result += f"  for (int i=0; i<{'TwoPowerD' if kind == VERTEX else '2'}; i++) {me}.set{accessor(attributes.around)}( i, tarch::multicore::TaskEnumerator::NoNumber );\n"
        if attributes.children is not None:
            if kind == VERTEX:
                result += f"  {me}.set{accessor(attributes.children)}( tarch::multicore::TaskEnumerator::NoNumber );\n"
            else:
                size = "ThreePowerD" if kind == CELL else "(ThreePowerD/3)"
                result += f"  for (int i=0; i<{size}; i++) {me}.set{accessor(attributes.children)}( i, tarch::multicore::TaskEnumerator::NoNumber );\n"
        return result

    def _announce_to_parent(self, kind: str) -> str:
        """! Write the number of the last task into the coarse entity"""
        attributes = self.attributes[kind]
        me = self._me(kind)
        last = f"{me}.get{accessor(attributes.last)}()"
        labels = self.labels[kind]
        child = accessor(attributes.children)
        if kind == CELL:
            return f"""
  {{
    const int index = peano4::datamanagement::taskassembly::getIndexWithinParentCell( marker.getRelativePositionWithinFatherCell() );
    if (index>=0) {{
      coarseGridCell{labels}.set{child}( index, {last} );
    }}
  }}
"""
        if kind == FACE:
            return f"""
  if ( not marker.isInteriorFaceWithinPatch() ) {{
    const int index = peano4::datamanagement::taskassembly::getIndexWithinParentFace( marker.getRelativePositionWithinFatherFace() );
    if (index>=0) {{
      coarseGridFaces{labels}( marker.getSelectedFaceNumber() ).set{child}( index, {last} );
    }}
  }}
"""
        return f"""
  if ( marker.coincidesWithCoarseGridVertex() ) {{
    coarseGridVertices{labels}( peano4::datamanagement::taskassembly::getIndexOfCoincidingCoarseVertex( marker.getRelativePositionWithinFatherCell() ) ).set{child}( {last} );
  }}
"""

    def _announce_cell_to_neighbours(self) -> str:
        """! The cell writes its number into the vertices and faces around it"""
        attributes = self.attributes.get(CELL)
        if attributes is None:
            return ""
        me = self._me(CELL)
        head = f"peano4::datamanagement::taskassembly::latestTask({{ {me}.get{accessor(attributes.last)}(), {me}.get{accessor(attributes.first)}() }})"
        result = ""
        for kind, count, index in (
            (
                VERTEX,
                "TwoPowerD",
                "peano4::datamanagement::taskassembly::getIndexOfCellAroundVertex(i)",
            ),
            (
                FACE,
                "2*Dimensions",
                "peano4::datamanagement::taskassembly::getIndexOfCellAroundFace(i)",
            ),
        ):
            if kind not in self.labels or not self.needs_around.get(kind, False):
                continue
            enumerator = "fineGridVertices" if kind == VERTEX else "fineGridFaces"
            if result == "":
                result = f"  const int cellTask = {head};\n"
            result += f"""  for (int i=0; i<{count}; i++) {{
    {enumerator}{self.labels[kind]}(i).set{accessor(self.attributes[kind].around)}( {index}, cellTask );
  }}
"""
        return result

    def _dependencies_of_first(self, kind: str) -> str:
        me = self._me(kind)
        result = ""
        helper = "peano4::datamanagement::taskassembly"
        result += self._add(self._latest(kind, "first", me, include_self=False))
        # Tasks in front whose consumer is gone: Their last tasks are still running
        result += self._add(
            self._latest(kind, "last", me, include_self=False, scope="ancestors")
        )
        # Parent to child
        if kind == CELL:
            result += self._add(
                self._latest(
                    CELL,
                    "first",
                    "coarseGridCell" + self.labels[CELL],
                    include_self=True,
                )
            )
        for coarse, enumerator, count in (
            (VERTEX, "coarseGridVertices", "TwoPowerD"),
            (FACE, "coarseGridFaces", "2*Dimensions"),
        ):
            if coarse == kind and coarse in self.labels:
                expression = self._latest(
                    coarse,
                    "first",
                    f"{enumerator}{self.labels[coarse]}(i)",
                    include_self=True,
                )
                if "latestTask" in expression:
                    result += f"  for (int i=0; i<{count}; i++) {helper}::addTask( inDependencies, {expression} );\n"
        result += self._dependencies_on_lower_entities(kind)
        return result

    def _dependencies_on_lower_entities(self, kind: str) -> str:
        """!

        Statements which add the first tasks of the vertices and faces of the entity

        The first and the last task of an entity both work on the data of the
        vertices and faces of the entity, i.e. they both wait for the first tasks
        of these. Peano touches them for the first time before the entity itself.
        If the last task did not wait, the clearing of the face data by a first
        task of the faces could run after the projection of the cell in its last
        task, and delete it.

        """
        helper = "peano4::datamanagement::taskassembly"
        result = ""
        for lower, enumerator, count in (
            (VERTEX, "fineGridVertices", "TwoPowerD"),
            (FACE, "fineGridFaces", "2*Dimensions"),
        ):
            if ENTITIES.index(lower) < ENTITIES.index(kind) and lower in self.labels:
                expression = self._latest(
                    lower,
                    "first",
                    f"{enumerator}{self.labels[lower]}(i)",
                    include_self=True,
                )
                if "latestTask" in expression:
                    result += f"  for (int i=0; i<{count}; i++) {helper}::addTask( inDependencies, {expression} );\n"
                expression = self._latest(
                    lower,
                    "last",
                    f"{enumerator}{self.labels[lower]}(i)",
                    include_self=False,
                    scope="ancestors",
                )
                if "latestTask" in expression:
                    result += f"  for (int i=0; i<{count}; i++) {helper}::addTask( inDependencies, {expression} );\n"
        return result

    def _dependencies_of_last(self, kind: str) -> str:
        me = self._me(kind)
        helper = "peano4::datamanagement::taskassembly"
        result = "  std::set<tarch::multicore::TaskNumber> inDependencies;\n"
        attributes = self.attributes[kind]
        result += self._add(self._latest(kind, "first", me, include_self=True))
        result += self._add(self._latest(kind, "last", me, include_self=False))
        result += self._dependencies_on_lower_entities(kind)
        if attributes.children is not None:
            if kind == VERTEX:
                result += f"  {helper}::addTask( inDependencies, {me}.get{accessor(attributes.children)}() );\n"
            else:
                size = "ThreePowerD" if kind == CELL else "(ThreePowerD/3)"
                result += f"  for (int i=0; i<{size}; i++) {helper}::addTask( inDependencies, {me}.get{accessor(attributes.children)}(i) );\n"
        if kind in (VERTEX, FACE):
            size = "TwoPowerD" if kind == VERTEX else "2"
            for wrapper in self._all_arrays(kind):
                around = wrapper.attributes[kind].around
                if around is not None:
                    result += f"  for (int i=0; i<{size}; i++) {helper}::addTask( inDependencies, {me}.get{accessor(around)}(i) );\n"
        return result

    @staticmethod
    def _enclave_condition(kind: str) -> str:
        """!

        C++ predicate: May the entity's task run in the background?

        This is the enclave tasking of Charrier, Hazelwood and Weinzierl. Entities along boundaries
        have to follow a strict sequential order, and all others are real background tasks.

        - A cell belongs to exactly one tree, i.e. a non-overlapping domain decomposition does not
          share its data. Only cells next to a resolution transition are skeleton cells: Their tasks work
          on hanging faces and vertices, which the traversal interpolates and restricts on the main
          thread right before and after the cell
          (@ref peano4::datamanagement::CellMarker::isAdjacentToResolutionTransition()).
        - Faces and vertices are shared between trees. Their data are exchanged when the entity is stored, i.e.
          right after the last event. Entities next to a parallel boundary, next to refined cells or hanging, are skeleton entities.

        """
        if kind == CELL:
            return "not marker.isAdjacentToResolutionTransition()"
        if kind == FACE:
            return "not marker.isAdjacentToParallelBoundary() and not marker.hasBeenRefined() and not marker.willBeRefined()"
        return "not marker.isAdjacentToParallelDomainBoundary() and not marker.isHanging() and not marker.hasBeenRefined() and not marker.willBeRefined()"

    def _wait_if_skeleton(self, kind: str, number: str) -> str:
        """!

        Wait for the task right after its spawn if the entity is a skeleton entity

        The task is still a task with all its dependencies, but the traversal cannot continue before it
        has finished, i.e. the tasks along tree and AMR boundaries are executed sequentially.

        """
        return f"    if ( not ({self._enclave_condition(kind)}) ) {{ tarch::multicore::waitForTask( {number} ); }}\n"

    def _spawn(self, operation_name: str, number: str, kind: str) -> str:
        return f"""
    tarch::multicore::spawnTask(
      new tarch::multicore::TaskWithCopyOfFunctor(
        tarch::multicore::Task::DontFuse,
        tarch::multicore::Task::DefaultPriority,
        [{self._lambda_captures()}]() mutable {{
{self.action_set.get_body_of_operation(operation_name)}
        }}
      ),
      inDependencies,
      {number}
    );
{self._wait_if_skeleton(kind, number)}"""

    def _touch_first(self, kind: str) -> str:
        spec = SPECS[kind]
        attributes = self.attributes.get(kind)
        if attributes is None:
            return self.action_set.get_implementation_of_operation(spec.first_event)
        me = self._me(kind)
        result = self._reset(kind)
        enumerator = "peano4::datamanagement::taskassembly::getTaskEnumerator()"
        if self.spawns(kind, first=False):
            result += (
                f"  {me}.set{accessor(attributes.last)}( {enumerator}.getNumber() );\n"
            )
            result += self._announce_to_parent(kind)
        if self.spawns(kind, first=True):
            guard = self.action_set.get_guard_of_operation(spec.first_event)
            result += f"""
  std::set<tarch::multicore::TaskNumber> inDependencies;
  int firstTaskNumber = tarch::multicore::TaskEnumerator::NoNumber;
  if ({guard}) {{
{self._dependencies_of_first(kind)}
    firstTaskNumber = {enumerator}.getNumber();
    {me}.set{accessor(attributes.first)}( firstTaskNumber );
  }}
"""
        if kind == CELL:
            result += self._announce_cell_to_neighbours()
        if self.spawns(kind, first=True):
            result += f"""
  if (firstTaskNumber>=0) {{
{self._spawn(spec.first_event, 'firstTaskNumber', kind)}
  }}
"""
        return result

    def _touch_last(self, kind: str) -> str:
        spec = SPECS[kind]
        attributes = self.attributes.get(kind)
        if attributes is None or not self.spawns(kind, first=False):
            return self.action_set.get_implementation_of_operation(spec.last_event)
        me = self._me(kind)
        guard = self.action_set.get_guard_of_operation(spec.last_event)
        number = f"{me}.get{accessor(attributes.last)}()"
        run = self._spawn(spec.last_event, number, kind)
        return f"""
{self._dependencies_of_last(kind)}
  if ({guard}) {{
{run}
  }}
  else {{
    peano4::datamanagement::taskassembly::spawnEmptyTask( inDependencies, {number} );
{self._wait_if_skeleton(kind, number)}  }}
"""

    def _wait_before_destroy(self, kind: str) -> str:
        attributes = self.attributes.get(kind)
        if attributes is None:
            return ""
        me = self._me(kind)
        result = ""
        for name in (attributes.last, attributes.first):
            result += f"""
  {{
    int number = {me}.get{accessor(name)}();
    peano4::datamanagement::taskassembly::waitForAndReleaseTask( peano4::datamanagement::taskassembly::getTaskEnumerator(), number );
    {me}.set{accessor(name)}( number );
  }}
"""
        return result

    def _wait_for_coarse_entities(self, kind: str, hanging: bool) -> str:
        """!

        Wait for the tasks of the coarse entities before an entity is created or destroyed

        The create and destroy events run on the main thread. They interpolate from
        and restrict into the coarse entities, i.e. they read and write data which the
        tasks of the coarse entity might still use or produce. Nothing in the task graph
        orders them. Hanging entities see no touch events and never hold task numbers, and a
        new cell does not hold any yet. We therefore wait for the first tasks of the
        coarse entity, and for the last tasks of tasks that this task depends on.

        @param hanging: bool
          If True, the coarse face is the one the marker selects. Otherwise, we wait for all of them.

        """
        if kind not in self.labels or not self._chain(kind, "first", True):
            return ""
        labels = self.labels[kind]
        if kind == CELL:
            entity, loop = f"coarseGridCell{labels}", None
        elif kind == FACE and hanging:
            entity, loop = (
                f"coarseGridFaces{labels}( marker.getSelectedFaceNumber() )",
                None,
            )
        elif kind == FACE:
            entity, loop = f"coarseGridFaces{labels}(i)", "2*Dimensions"
        else:
            entity, loop = f"coarseGridVertices{labels}(i)", "TwoPowerD"
        result = ""
        for which, scope in (("first", "all"), ("last", "ancestors")):
            expression = self._latest(
                kind, which, entity, include_self=True, scope=scope
            )
            if "latestTask" not in expression:
                continue
            statement = f"""    const int number = {expression};
    if (number>=0) tarch::multicore::waitForTask( number );
"""
            if loop is None:
                result += f"  {{\n{statement}  }}\n"
            else:
                result += f"  for (int i=0; i<{loop}; i++) {{\n{statement}  }}\n"
        return result

    # ActionSet interface -------------------------------------------------

    def get_constructor_body(self):
        return self.action_set.get_constructor_body()

    def get_destructor_body(self):
        return self.action_set.get_destructor_body()

    def get_static_initialisations(self, full_qualified_classname):
        return self.action_set.get_static_initialisations(full_qualified_classname)

    def get_body_of_createGridControlEventsForNextTraversal(self):
        return (
            self.action_set.get_implementation_of_createGridControlEventsForNextTraversal()
        )

    def get_body_of_prepareTraversal(self):
        return self.action_set.get_implementation_of_prepareTraversal()

    def get_body_of_unprepareTraversal(self):
        body = self.action_set.get_implementation_of_unprepareTraversal()
        if body.strip() == "":
            return body
        return "  tarch::multicore::waitForAllTasks();\n" + body

    def get_body_of_operation(self, operation_name):
        # Not used, as get_implementation_of_operation() is overwritten
        return ""

    def get_implementation_of_operation(self, operation_name):
        for kind in ENTITIES:
            spec = SPECS[kind]
            if kind in self.labels:
                if operation_name == spec.first_event:
                    return self._touch_first(kind)
                if operation_name == spec.last_event:
                    return self._touch_last(kind)
        waits = {
            ActionSet.OPERATION_CREATE_CELL: (CELL, False, False),
            ActionSet.OPERATION_DESTROY_CELL: (CELL, False, True),
            ActionSet.OPERATION_CREATE_PERSISTENT_FACE: (FACE, False, False),
            ActionSet.OPERATION_DESTROY_PERSISTENT_FACE: (FACE, False, True),
            ActionSet.OPERATION_CREATE_HANGING_FACE: (FACE, True, False),
            ActionSet.OPERATION_DESTROY_HANGING_FACE: (FACE, True, False),
            ActionSet.OPERATION_CREATE_PERSISTENT_VERTEX: (VERTEX, False, False),
            ActionSet.OPERATION_DESTROY_PERSISTENT_VERTEX: (VERTEX, False, True),
            ActionSet.OPERATION_CREATE_HANGING_VERTEX: (VERTEX, True, False),
            ActionSet.OPERATION_DESTROY_HANGING_VERTEX: (VERTEX, True, False),
        }
        if operation_name in waits:
            kind, hanging, persistent_destroy = waits[operation_name]
            body = self.action_set.get_implementation_of_operation(operation_name)
            if body.strip() == "" or kind not in self.labels:
                return body
            prefix = self._wait_for_coarse_entities(kind, hanging)
            if persistent_destroy:
                prefix += self._wait_before_destroy(kind)
            return prefix + body
        if operation_name == ActionSet.OPERATION_END_TRAVERSAL:
            body = self.action_set.get_implementation_of_operation(operation_name)
            if body.strip() == "":
                return body
            return "  tarch::multicore::waitForAllTasks();\n" + body
        return self.action_set.get_implementation_of_operation(operation_name)

    def get_action_set_name(self):
        return self.action_set.get_action_set_name()

    def user_should_modify_template(self):
        return False

    def get_attributes(self):
        return self.action_set.get_attributes()

    def get_includes(self):
        return (
            self.action_set.get_includes()
            + """
#include <set>
#include "tarch/multicore/multicore.h"
#include "tarch/multicore/Task.h"
#include "tarch/multicore/TaskEnumerator.h"
#include "peano4/datamanagement/FaceHeapDataEnumerator.h"
#include "peano4/datamanagement/VertexHeapDataEnumerator.h"
#include "peano4/datamanagement/TaskAssemblyDependencies.h"
"""
        )
