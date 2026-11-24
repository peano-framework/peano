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
        writes_shared_data: bool = True,
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

        @param writes_shared_data: bool
          Whether the task writes face or vertex data. Peano exchanges these data
          between the trees of a domain decomposition when a traversal ends, i.e.
          the tasks of such a task must have finished by then
          (@ref peano4.api.taskgraph.compiler.SpawnTaskActionSet.SpawnTaskActionSet._record()).

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
        self.writes_shared_data = writes_shared_data
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
        self, kind: str, which: str, include_self: bool
    ) -> List["SpawnTaskActionSet"]:
        """! Wrappers with an attribute on the entity, latest first"""
        wrappers = self.earlier + ([self] if include_self else [])
        return [wrapper for wrapper in reversed(wrappers) if kind in wrapper.attributes]

    def _latest(self, kind: str, which: str, entity: str, include_self: bool) -> str:
        """!

        C++ expression for the number of the latest first or last task on an entity

        @param entity: str
          C++ expression of the entity's label object, e.g. fineGridCellX or fineGridVerticesX(i)

        """
        chain = self._chain(kind, which, include_self)
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
        # Entities below
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
        return result

    def _dependencies_of_last(self, kind: str) -> str:
        me = self._me(kind)
        helper = "peano4::datamanagement::taskassembly"
        result = "  std::set<tarch::multicore::TaskNumber> inDependencies;\n"
        attributes = self.attributes[kind]
        result += self._add(self._latest(kind, "first", me, include_self=True))
        result += self._add(self._latest(kind, "last", me, include_self=False))
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

    def _record(self, kind: str, number: str) -> str:
        """!

        Remember the task number if the task has to finish before the traversal ends

        Peano exchanges face and vertex data between the trees of a domain
        decomposition at the end of the traversal, i.e. before the consumer
        waits for the tasks. A task that might still run then modifies data
        which is sent to the neighbour. We therefore record the numbers of
        all vertex tasks, of the face tasks of faces next to a parallel
        boundary, and of the cell tasks of a task that writes face or vertex
        data. endTraversal() waits for them. Cell tasks which write cell data
        only are free to outlive the traversal.

        """
        statement = f"_tasksToFinishBeforeDataExchange.insert( {number} );"
        if kind == VERTEX:
            return f"    {statement}\n"
        if kind == FACE:
            return (
                f"    if ( marker.isAdjacentToParallelBoundary() ) {{ {statement} }}\n"
            )
        return f"    {statement}\n" if self.writes_shared_data else ""

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
{self._record(kind, number)}"""

    def _touch_first(self, kind: str) -> str:
        spec = SPECS[kind]
        attributes = self.attributes.get(kind)
        if attributes is None:
            return self.action_set.get_implementation_of_operation(spec.first_event)
        me = self._me(kind)
        result = self._reset(kind)
        enumerator = f"{me}.getTaskEnumerator()"
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
        return f"""
{self._dependencies_of_last(kind)}
  if ({guard}) {{
{self._spawn(spec.last_event, f'{me}.get{accessor(attributes.last)}()', kind)}
  }}
  else {{
    peano4::datamanagement::taskassembly::spawnEmptyTask( inDependencies, {me}.get{accessor(attributes.last)}() );
{self._record(kind, f'{me}.get{accessor(attributes.last)}()')}  }}
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
    peano4::datamanagement::taskassembly::waitForAndReleaseTask( {me}.getTaskEnumerator(), number );
    {me}.set{accessor(name)}( number );
  }}
"""
        return result

    def _wait_for_coarse_entities(self, kind: str) -> str:
        """!

        Wait for the first tasks of the coarse entities before a hanging entity is created or destroyed

        Hanging entities are not persistent and only see create and destroy events,
        i.e. they never hold task numbers, and nothing in the task graph
        orders the main thread's interpolation or restriction against the tasks of
        the coarse entity. We therefore wait for the coarse entity's first tasks.

        """
        if kind not in self.labels or not self._chain(kind, "first", True):
            return ""
        labels = self.labels[kind]
        if kind == FACE:
            entities = f"coarseGridFaces{labels}( marker.getSelectedFaceNumber() )"
            expression = self._latest(kind, "first", entities, include_self=True)
            return f"""
  {{
    const int number = {expression};
    if (number>=0) tarch::multicore::waitForTask( number );
  }}
"""
        expression = self._latest(
            kind, "first", f"coarseGridVertices{labels}(i)", include_self=True
        )
        return f"""
  for (int i=0; i<TwoPowerD; i++) {{
    const int number = {expression};
    if (number>=0) tarch::multicore::waitForTask( number );
  }}
"""

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
                if operation_name in spec.destroy_events:
                    return self._wait_before_destroy(
                        kind
                    ) + self.action_set.get_implementation_of_operation(operation_name)
        hanging_kind = {
            ActionSet.OPERATION_CREATE_HANGING_FACE: FACE,
            ActionSet.OPERATION_DESTROY_HANGING_FACE: FACE,
            ActionSet.OPERATION_CREATE_HANGING_VERTEX: VERTEX,
            ActionSet.OPERATION_DESTROY_HANGING_VERTEX: VERTEX,
        }
        if operation_name in hanging_kind:
            body = self.action_set.get_implementation_of_operation(operation_name)
            if body.strip() == "":
                return body
            return self._wait_for_coarse_entities(hanging_kind[operation_name]) + body
        if operation_name == ActionSet.OPERATION_END_TRAVERSAL:
            body = self.action_set.get_implementation_of_operation(operation_name)
            barrier = """
  tarch::multicore::waitForTasks( _tasksToFinishBeforeDataExchange );
  _tasksToFinishBeforeDataExchange.clear();
"""
            if body.strip() != "":
                barrier += "  tarch::multicore::waitForAllTasks();\n"
            return barrier + body
        return self.action_set.get_implementation_of_operation(operation_name)

    def get_action_set_name(self):
        return self.action_set.get_action_set_name()

    def user_should_modify_template(self):
        return False

    def get_attributes(self):
        return (
            self.action_set.get_attributes()
            + """
  std::set<tarch::multicore::TaskNumber> _tasksToFinishBeforeDataExchange;
"""
        )

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
