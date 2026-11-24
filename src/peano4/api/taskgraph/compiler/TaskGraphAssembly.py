# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import copy
import re
from typing import Dict, List, Optional, Tuple

import dastgen2

import peano4.api
import peano4.api.datamodel
import peano4.api.solversteps

from ..Graph import Graph
from ..Task import Task
from . import FuseSteps
from . import Sequential
from .CellTaskProducerConsumer import (
    _generator_name,
    _is_held_via_smart_pointer,
    _non_static_attribute_names,
    _SMART_POINTER_HINT,
)
from .SpawnTaskActionSet import (
    CELL,
    ENTITIES,
    FACE,
    SPECS,
    VERTEX,
    EntityAttributes,
    SpawnTaskActionSet,
)
from .ValidateTaskGraph import derive_marks_and_events, validate
from .WaitForTasksActionSet import WaitForTasksActionSet

PRODUCER_SUFFIX = " Producer"
CONSUMER_SUFFIX = " Consumer"
"""! Appended to the name of a task which is split """


def translate_with_sequential(
    graph: Graph,
    identifier: str,
    peano_project: "peano4.api.Project",
) -> List["peano4.api.solversteps.Step"]:
    """!

    Split every task into producer and consumer, spawn all six touch events as tasks, and map each task onto its own mesh traversal

    You can use this task graph compiler in your ExaHyPE 2 project through

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    project.compiler = peano4.api.taskgraph.compiler.TaskGraphAssembly.translate_with_sequential
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    See @ref peano4.api.taskgraph.compiler.TaskGraphAssembly._translate()

    """
    return _translate(graph, identifier, peano_project, use_fusion=False)


def translate_with_fusion(
    graph: Graph,
    identifier: str,
    peano_project: "peano4.api.Project",
) -> List["peano4.api.solversteps.Step"]:
    """!

    Split every task into producer and consumer, spawn all six touch events as tasks, and fuse tasks into as few mesh traversals as possible

    You can use this task graph compiler in your ExaHyPE 2 project through

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    project.compiler = peano4.api.taskgraph.compiler.TaskGraphAssembly.translate_with_fusion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    The rewritten graph is handed over to @ref peano4.api.taskgraph.compiler.FuseSteps.translate(),
    which never fuses a producer with its consumer. See
    @ref peano4.api.taskgraph.compiler.TaskGraphAssembly._translate()

    """
    return _translate(graph, identifier, peano_project, use_fusion=True)


def _translate(
    graph: Graph,
    identifier: str,
    peano_project: "peano4.api.Project",
    use_fusion: bool,
) -> List["peano4.api.solversteps.Step"]:
    """!

    Task graph compiler which assembles the task graph of a mesh traversal dynamically (work in progress)

    Do not use this routine directly. Pick
    @ref peano4.api.taskgraph.compiler.TaskGraphAssembly.translate_with_sequential()
    or
    @ref peano4.api.taskgraph.compiler.TaskGraphAssembly.translate_with_fusion()
    instead.

    ## Producer and consumer

    Every task which acts on the mesh (it has events besides the global ones)
    is replaced by two tasks, unless the task opts out
    (@ref peano4.api.taskgraph.Task.Task.may_spawn_background_tasks is False), repeats, or holds action sets which the user
    has to modify. The producer holds the original action sets. Its
    non-empty touchVertexFirstTime(), touchVertexLastTime(), touchFaceFirstTime(),
    touchFaceLastTime(), touchCellFirstTime() and touchCellLastTime() events
    become tasks
    (@ref peano4.api.taskgraph.compiler.SpawnTaskActionSet.SpawnTaskActionSet).
    The consumer visits the entities in a mesh traversal of its own and waits for
    the tasks
    (@ref peano4.api.taskgraph.compiler.WaitForTasksActionSet.WaitForTasksActionSet).
    Tasks depending on the original task depend on the consumer. Producer and
    consumer are never fused. If a task has an action set which has to run on the main thread but acts in a
    touch event, the task is not split and the compiler writes a warning.

    The producer inherits the events of the original action sets, i.e. the
    fusion works with the events which the task graph states. The barriers
    which the producer adds (endTraversal, unprepareTraversal) do not alter
    them.

    ## Task numbers

    The compiler adds up to three data types to the data model: One for the
    vertices, one for the faces and one for the cells (<identifier>_VertexTaskLabels,
    <identifier>_FaceTaskLabels, <identifier>_CellTaskLabels). They hold, per
    wrapped action set and entity, two task numbers: One for the touch...FirstTime()
    task and one for the touch...LastTime() task. The two are independent, i.e.
    an action set might have only one of them. The labels are used by the
    steps that hold producers (WRITE) or consumers (READ_AND_DISCARD),
    or both (READ_AND_WRITE). They are
    added to the steps after the fusion, i.e. the face and vertex labels never keep the
    fusion from putting two producers into one traversal.

    ## Dependencies

    All of them follow from the order of the events within a traversal. See
    @ref peano4_datamanagement_TaskAssemblyDependencies for a description.

    @param graph: peano4.api.taskgraph.Graph
      The task graph to compile. Assumed to be acyclic.

    @param identifier: str
      @ref peano4.api.taskgraph.compiler.FuseSteps.translate()

    @param peano_project: peano4.api.Project
      Project the task labels are added to. If None, the labels are created
      but not added to any project (unit tests).

    @param use_fusion: bool
      If True, hand the rewritten graph over to FuseSteps, otherwise to
      Sequential.

    @return [peano4.api.solversteps.Step]
      @ref peano4.api.taskgraph.compiler.FuseSteps.translate()

    """
    validate(graph)
    assembly = _Assembly(graph, identifier)
    assembly.build()

    if peano_project is not None:
        for kind, labels in assembly.labels.items():
            if assembly.has_attributes[kind]:
                {
                    VERTEX: peano_project.datamodel.add_vertex,
                    FACE: peano_project.datamodel.add_face,
                    CELL: peano_project.datamodel.add_cell,
                }[kind](labels)

    non_fusable_task_pairs = [
        (producer, consumer) for producer, consumer in assembly.producer_consumer_pairs
    ]
    if use_fusion:
        steps = FuseSteps.translate(
            assembly.result, identifier, peano_project, non_fusable_task_pairs
        )
    else:
        steps = Sequential.translate(assembly.result, identifier, peano_project)
    assembly.add_labels_to_steps(steps)
    return steps


class _ProducerTask(Task):
    """!

    A task whose action sets are wrapped, but whose events are the ones of the original action sets

    The wrappers reset the arrays of the entities in the first touch of every entity, and thus
    seem to act in events in which the original action set does nothing. The fusion would
    treat the producer as if it acted in these events, too, and not fuse it with the tasks
    that it can be fused with.

    """

    def _derived(self):
        key = tuple(id(action_set) for action_set in self.action_sets)
        if self._derived_cache is None or self._derived_cache[0] != key:
            originals = [
                getattr(action_set, "action_set", action_set)
                for action_set in self.action_sets
            ]
            self._derived_cache = (key, *derive_marks_and_events(originals))
        return self._derived_cache[1], self._derived_cache[2]


def _copy_task(task: Task, task_class=None) -> Task:
    """!

    Shallow copy of a task with its own lists and dictionaries, so we can alter the copy
    without touching the original graph. The action sets and the submodels are shared.

    """
    result = copy.copy(task)
    if task_class is not None:
        result.__class__ = task_class
    result.dependencies = []
    result.action_sets = list(task.action_sets)
    result.cell_data = list(task.cell_data)
    result.face_data = list(task.face_data)
    result.vertex_data = list(task.vertex_data)
    result.cell_data_access = dict(task.cell_data_access)
    result.face_data_access = dict(task.face_data_access)
    result.vertex_data_access = dict(task.vertex_data_access)
    result._derived_cache = None
    return result


_TOUCH_OPERATIONS = [
    operation
    for kind in ENTITIES
    for operation in (SPECS[kind].first_event, SPECS[kind].last_event)
]


def _touch_bodies(action_set) -> str:
    """! The bodies of the six touch events, concatenated"""
    return "\n".join(
        action_set.get_body_of_operation(operation) for operation in _TOUCH_OPERATIONS
    )


def _has_touch_body(action_set) -> bool:
    return any(
        action_set.get_body_of_operation(operation).strip() != ""
        for operation in _TOUCH_OPERATIONS
    )


def _is_spawnable(action_set) -> bool:
    """! True if one of the six touch events has a body and the action set may run in a task"""
    return (
        _has_touch_body(action_set)
        and action_set.may_run_in_background_task()
        and not action_set.user_should_modify_template()
    )


def _reason_not_to_split(task: Task) -> Optional[str]:
    """!

    Why the task is not split into producer and consumer, or None if it is split

    A task is not split silently if it does not act on the mesh, repeats, opts out, or has
    nothing to spawn. The compiler complains if an action set has to run on the main thread
    though it acts in a touch event: It would run while the tasks of the other action
    sets still run.

    """
    if task.entity.is_global or not task.entity.mesh_marks:
        return ""
    if task.may_spawn_background_tasks is False or task.has_to_repeat is not None:
        return ""
    if not any(_is_spawnable(action_set) for action_set in task.action_sets):
        return ""
    main_thread_action_sets = [
        action_set.get_action_set_name()
        for action_set in task.action_sets
        if _has_touch_body(action_set) and not _is_spawnable(action_set)
    ]
    if main_thread_action_sets:
        return (
            "the action sets "
            + ", ".join(main_thread_action_sets)
            + " have to run on the main thread but act in touch events"
        )
    return None


def _enumerators_used_by(task: Task, task_body: str) -> Tuple[List[str], List[str]]:
    """!

    Names of the vertex and face enumerators which a task body might use, as
    @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._enumerators_used_by()

    @return ([str], [str])
      Names of the face enumerators and of the vertex enumerators.

    """
    faces = [
        f"{prefix}{submodel.name}"
        for prefix in ("fineGridFaces", "coarseGridFaces")
        for submodel in task.face_data
        if re.search(rf"\b{prefix}{submodel.name}\b", task_body)
    ]
    vertices = [
        f"{prefix}{submodel.name}"
        for prefix in ("fineGridVertices", "coarseGridVertices")
        for submodel in task.vertex_data
        if re.search(rf"\b{prefix}{submodel.name}\b", task_body)
    ]
    return faces, vertices


def _warn_about_unsafe_task(task: Task, action_set, task_body: str):
    """!

    Issue the warnings about task bodies which might not be safe

    The tasks outlive the event. Data of the entity or the cell which is not
    held via a smart pointer is copied, and writes are lost. We do not inspect
    the code but only search the text for the names.
    @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._warn_about_unsafe_task()

    """
    logger = peano4.api.Logger()
    description = f"action set {action_set.get_action_set_name()} of task {task.name}"
    for accessors, submodels in (
        (("fineGridCell", "coarseGridCell"), task.cell_data),
        (("fineGridFace",), task.face_data),
        (("fineGridVertex", "coarseGridVertex"), task.vertex_data),
    ):
        for accessor_name in accessors:
            for submodel in submodels:
                if re.search(
                    rf"\b{accessor_name}{submodel.name}\b", task_body
                ) and not _is_held_via_smart_pointer(submodel):
                    logger.warn_and_confirm(
                        f"The touch task of {description} might access {accessor_name}{submodel.name}. {submodel.name} is realised via the generator "
                        f"{_generator_name(submodel)}, which does not hold its data via a smart pointer. The task captures the data by value, "
                        f"i.e. it works on a copy, and all writes are lost. "
                        + _SMART_POINTER_HINT
                    )
    face_enumerators, vertex_enumerators = _enumerators_used_by(task, task_body)
    for enumerators, submodels, prefixes in (
        (face_enumerators, task.face_data, ("fineGridFaces", "coarseGridFaces")),
        (
            vertex_enumerators,
            task.vertex_data,
            ("fineGridVertices", "coarseGridVertices"),
        ),
    ):
        for submodel in submodels:
            for prefix in prefixes:
                if (
                    f"{prefix}{submodel.name}" in enumerators
                    and not _is_held_via_smart_pointer(submodel)
                ):
                    logger.warn_and_confirm(
                        f"The touch task of {description} might access {submodel.name} via {prefix}{submodel.name}. The task captures the enumerator "
                        f"as a copy of the data. {submodel.name} is realised via the generator {_generator_name(submodel)}, which does not hold its data "
                        f"via a smart pointer, i.e. the copies are deep and all writes are lost. "
                        + _SMART_POINTER_HINT
                    )
    for attribute_name in _non_static_attribute_names(action_set.get_attributes()):
        if re.search(rf"\b{attribute_name}\b", task_body):
            logger.warn_and_confirm(
                f"The touch task of {description} might use the non-static attribute {attribute_name}. The task then "
                f"accesses an action set object which is destroyed after the producer's mesh traversal."
            )


def _writes_shared_data(task: Task) -> bool:
    """! True if the task writes a face or a vertex submodel, i.e. data which Peano exchanges between trees"""
    writing = (Task.DataAccess.WRITE, Task.DataAccess.READ_AND_WRITE)
    return any(
        access in writing
        for access in list(task.face_data_access.values())
        + list(task.vertex_data_access.values())
    )


def _stem(task: Task, action_set_index: int) -> str:
    """!

    CamelCase version of the task's name followed by ActionSet and the action set's index

    """
    words = re.split(r"[^A-Za-z0-9]+", task.name)
    return (
        "".join(word[:1].upper() + word[1:] for word in words if word)
        + f"ActionSet{action_set_index}"
    )


class _Assembly:
    """!

    The rewrite of the task graph and the creation of the task labels

    """

    def __init__(self, graph: Graph, identifier: str):
        self.graph = graph
        self.identifier = identifier
        self.result = Graph()
        self.producer_consumer_pairs: List[Tuple[Task, Task]] = []
        self.labels: Dict[str, "peano4.api.datamodel.DaStGen2"] = {}
        self.has_attributes: Dict[str, bool] = {kind: False for kind in ENTITIES}
        self._wrappers_of_task: Dict[Task, List[SpawnTaskActionSet]] = {}
        self._consumer_of_task: Dict[Task, Task] = {}
        self._producer_of_task: Dict[Task, Task] = {}

    def _create_labels(self):
        for kind, name in (
            (VERTEX, "VertexTaskLabels"),
            (FACE, "FaceTaskLabels"),
            (CELL, "CellTaskLabels"),
        ):
            labels = peano4.api.datamodel.DaStGen2(f"{self.identifier}_{name}")
            labels.data.add_attribute(
                dastgen2.attributes.UserDefinedType(
                    name="taskEnumerator",
                    type="tarch::multicore::TaskEnumerator",
                    include='#include "tarch/multicore/TaskEnumerator.h"',
                    qualifier=dastgen2.attributes.Attribute.Qualifier.STATIC,
                    user_type_has_toString_method=False,
                )
            )
            self.labels[kind] = labels

    @staticmethod
    def _ancestors(task: Task) -> List[Task]:
        result = []
        for dependency in task.dependencies:
            for ancestor in _Assembly._ancestors(dependency) + [dependency]:
                if ancestor not in result:
                    result.append(ancestor)
        return result

    def build(self):
        self._create_labels()
        ordered = self.graph.ordered_tasks()
        split = {}
        for task in ordered:
            reason = _reason_not_to_split(task)
            if reason is None:
                split[task] = True
            elif reason != "":
                peano4.api.Logger().log(
                    f"Warning: Task {task.name} is not split into a producer and a consumer task, as {reason}.",
                    peano4.api.LoggerPrintOption.ALWAYS_PRINT,
                )

        for task in ordered:
            producer = _copy_task(task, _ProducerTask if task in split else None)
            self._producer_of_task[task] = producer
            if task in split:
                producer.name = f"{task.name}{PRODUCER_SUFFIX}"

        for task in ordered:
            if task in split:
                self._wrap(task, ordered)

        for task in ordered:
            producer = self._producer_of_task[task]
            for dependency in task.dependencies:
                producer.add_dependency(
                    self._consumer_of_task.get(
                        dependency, self._producer_of_task[dependency]
                    )
                )
            self.result.add_task(producer)
            if task in self._consumer_of_task:
                consumer = self._consumer_of_task[task]
                consumer.add_dependency(producer)
                self.result.add_task(consumer)
                self.producer_consumer_pairs.append((producer, consumer))

    def _wrap(self, task: Task, ordered: List[Task]):
        """! Replace the spawnable action sets of the producer by wrappers and create the consumer"""
        producer = self._producer_of_task[task]
        action_sets = task.action_sets
        spawnable = [_is_spawnable(action_set) for action_set in action_sets]
        indices = [index for index, flag in enumerate(spawnable) if flag]

        def spawns(index: int, kind: str, first: bool) -> bool:
            spec = SPECS[kind]
            return (
                action_sets[index]
                .get_body_of_operation(spec.first_event if first else spec.last_event)
                .strip()
                != ""
            )

        earlier: List[SpawnTaskActionSet] = []
        for ancestor in self._ancestors(task):
            earlier += self._wrappers_of_task.get(ancestor, [])

        own_wrappers: List[SpawnTaskActionSet] = []
        wrapped = []
        for index, action_set in enumerate(action_sets):
            if not spawnable[index]:
                wrapped.append(action_set)
                continue
            stem = _stem(task, index)
            attributes: Dict[str, EntityAttributes] = {}
            needs_around: Dict[str, bool] = {}
            spawns_cell = spawns(index, CELL, True) or spawns(index, CELL, False)
            for kind in ENTITIES:
                spawn_any = spawns(index, kind, True) or spawns(index, kind, False)
                last_later = any(
                    spawns(later, kind, False) for later in indices if later >= index
                )
                around = kind != CELL and last_later and spawns_cell
                needs_around[kind] = around
                if not (spawn_any or around):
                    continue
                attributes[kind] = EntityAttributes(
                    first=f"{stem}FirstTaskNumber",
                    last=f"{stem}LastTaskNumber",
                    around=f"{stem}TasksAround" if around else None,
                    children=f"{stem}ChildrenTaskNumbers"
                    if spawns(index, kind, False)
                    else None,
                )
            body = _touch_bodies(action_set)
            _warn_about_unsafe_task(task, action_set, body)
            face_enumerators, vertex_enumerators = _enumerators_used_by(task, body)
            wrapper = SpawnTaskActionSet(
                action_set,
                {kind: labels.name for kind, labels in self.labels.items()},
                attributes,
                earlier + own_wrappers,
                needs_around,
                face_enumerators,
                vertex_enumerators,
                task=task,
                writes_shared_data=_writes_shared_data(task),
            )
            own_wrappers.append(wrapper)
            wrapped.append(wrapper)
            self._add_attributes(attributes, needs_around)

        self._wrappers_of_task[task] = own_wrappers
        producer.action_sets = wrapped

        consumer_action_set = WaitForTasksActionSet(
            re.sub(
                r"[^0-9A-Za-z_]+",
                "",
                f"{_stem(task, 0)[:-len('ActionSet0')]}WaitForTasks",
            ),
            own_wrappers,
        )
        entity = Task.GridEntity(0)
        for kind in consumer_action_set.kinds():
            entity |= {
                VERTEX: Task.GridEntity.VERTEX,
                FACE: Task.GridEntity.FACE,
                CELL: Task.GridEntity.CELL,
            }[kind]
        if entity == Task.GridEntity(0):
            # Only multiscale or neighbour bookkeeping, nothing to wait for
            entity = Task.GridEntity.CELL
        consumer = Task(name=f"{task.name}{CONSUMER_SUFFIX}", entity=entity)
        consumer.add_action_set(consumer_action_set)
        # The consumer stands for the tasks of the producer. FuseSteps has to see the
        # conflicts on shared data between these tasks and the tasks which follow:
        # The consumer waits for the tasks of a cell only when it enters the cell, i.e. after
        # a fused producer has already spawned tasks on the faces and vertices of the cell.
        # READ_AND_WRITE does not alter the data, but never loses it.
        for submodel in task.face_data:
            if task.face_data_access[submodel] != Task.DataAccess.TEMPORARY:
                consumer.use_face(submodel, Task.DataAccess.READ_AND_WRITE)
        for submodel in task.vertex_data:
            if task.vertex_data_access[submodel] != Task.DataAccess.TEMPORARY:
                consumer.use_vertex(submodel, Task.DataAccess.READ_AND_WRITE)
        self._consumer_of_task[task] = consumer

    def _add_attributes(
        self, attributes: Dict[str, EntityAttributes], needs_around: Dict[str, bool]
    ):
        for kind, attribute in attributes.items():
            labels = self.labels[kind]
            self.has_attributes[kind] = True
            labels.data.add_attribute(
                dastgen2.attributes.Integer(name=attribute.first, initval="-1")
            )
            labels.data.add_attribute(
                dastgen2.attributes.Integer(name=attribute.last, initval="-1")
            )
            if attribute.around is not None:
                labels.data.add_attribute(
                    dastgen2.attributes.IntegerArray(
                        name=attribute.around,
                        cardinality="TwoPowerD" if kind == VERTEX else "2",
                    )
                )
            if attribute.children is not None:
                if kind == VERTEX:
                    labels.data.add_attribute(
                        dastgen2.attributes.Integer(
                            name=attribute.children, initval="-1"
                        )
                    )
                else:
                    labels.data.add_attribute(
                        dastgen2.attributes.IntegerArray(
                            name=attribute.children,
                            cardinality="ThreePowerD"
                            if kind == CELL
                            else "(ThreePowerD/3)",
                        )
                    )

    def add_labels_to_steps(self, steps: List["peano4.api.solversteps.Step"]):
        """!

        Let each step that holds a producer or a consumer use the task labels

        Producers write the labels, consumers read and discard them. A step which
        holds a consumer of one task and a producer of another one both reads
        and writes them: It has to load the numbers which it waits for, and
        store the numbers which its producers create.

        """
        Access = Task.DataAccess
        for step in steps:
            produces = any(
                isinstance(action_set, SpawnTaskActionSet)
                for action_set in step.action_sets
            )
            consumes = any(
                isinstance(action_set, WaitForTasksActionSet)
                for action_set in step.action_sets
            )
            if not (produces or consumes):
                continue
            access = (
                Access.READ_AND_WRITE
                if produces and consumes
                else (Access.WRITE if produces else Access.READ_AND_DISCARD)
            )
            for kind in ENTITIES:
                if not self.has_attributes[kind]:
                    continue
                labels = self.labels[kind]
                if kind == VERTEX:
                    step.use_vertex(labels)
                    step.vertex_data_access[labels] = access
                elif kind == FACE:
                    step.use_face(labels)
                    step.face_data_access[labels] = access
                else:
                    step.use_cell(labels)
                    step.cell_data_access[labels] = access
