# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import copy
import re
from typing import List

import dastgen2

import peano4.api
import peano4.api.datamodel
import peano4.api.solversteps

from ..Graph import Graph
from ..Task import Task
from . import FuseSteps
from . import Sequential
from .SpawnCellTaskActionSet import SpawnCellTaskActionSet
from .WaitForCellTaskActionSet import WaitForCellTaskActionSet


def translate_with_sequential(
    graph: Graph,
    identifier: str,
    peano_project: "peano4.api.Project",
) -> List["peano4.api.solversteps.Step"]:
    """!

    Split tasks writing cell data into producer and consumer, and map each task onto its own mesh traversal

    You can use this task graph compiler in your ExaHyPE 2 project through

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    project.compiler = peano4.api.taskgraph.compiler.CellTaskProducerConsumer.translate_with_sequential
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    The rewritten graph is handed over to
    @ref peano4.api.taskgraph.compiler.Sequential.translate(). See
    @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._translate()
    for a description of the rewrite and of the parameters.

    """
    return _translate(graph, identifier, peano_project, use_fusion=False)


def translate_with_fusion(
    graph: Graph,
    identifier: str,
    peano_project: "peano4.api.Project",
) -> List["peano4.api.solversteps.Step"]:
    """!

    Split tasks writing cell data into producer and consumer, and fuse tasks into as few mesh traversals as possible

    You can use this task graph compiler in your ExaHyPE 2 project through

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    project.compiler = peano4.api.taskgraph.compiler.CellTaskProducerConsumer.translate_with_fusion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    The rewritten graph is handed over to
    @ref peano4.api.taskgraph.compiler.FuseSteps.translate(), which never
    fuses a producer with its consumer. See
    @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._translate()
    for a description of the rewrite and of the parameters.

    """
    return _translate(graph, identifier, peano_project, use_fusion=True)


def _translate(
    graph: Graph,
    identifier: str,
    peano_project: "peano4.api.Project",
    use_fusion: bool,
) -> List["peano4.api.solversteps.Step"]:
    """!

    Task graph compiler which splits tasks writing cell data into a producer spawning tasks and a consumer waiting for them (work in progress)

    Do not use this routine directly. Pick
    @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer.translate_with_sequential()
    or
    @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer.translate_with_fusion()
    instead.

    This compiler adds a preamble which rewrites the task graph, and then
    hands the rewritten graph over to either
    @ref peano4.api.taskgraph.compiler.Sequential.translate() or
    @ref peano4.api.taskgraph.compiler.FuseSteps.translate().

    ## Preamble

    Every task which exclusively writes cell data is replaced by two tasks.
    A task exclusively writes cell data if it declares at least one cell
    submodel as WRITE or READ_AND_WRITE, but no face or vertex submodel as
    WRITE or READ_AND_WRITE (@ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._exclusively_writes_cell_data()). It may
    read face or vertex data. All other tasks are kept as they are.

    A task is only split if this is safe
    (@ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._has_events_after_touch_cell_first_time()):
    The producer spawns tasks which might still run after the producer's
    mesh traversal. If any action set of the task yields code for
    unprepareTraversal(), endTraversal() or one of the touch...LastTime()
    events, these events might be invoked before the spawned tasks have
    terminated. Such a task is hence not split, and the compiler writes a
    warning. You can enforce the split by setting the task's flag
    @ref peano4.api.taskgraph.Task.Task.may_spawn_background_tasks to True,
    which skips this check. If the flag is False, the task is never split,
    and the compiler writes no warning.

    The two tasks are:

    1. The producer, named after the original task plus "Producer". It
       inherits all action sets and data declarations of the original
       task.
    2. The consumer, named after the original task plus "Consumer". It
       works on cells, too, and depends on the producer. It declares every
       cell submodel that the producer writes (WRITE or READ_AND_WRITE) as
       READ_AND_WRITE.

    Every task that depended on the original task depends on the consumer
    instead. Producer and consumer are marked as non-fusable, i.e. if we
    use FuseSteps, it never puts them into the same mesh traversal. A
    consumer however might be fused with subsequent tasks. In an ExaHyPE 2
    setup, for example, the initial condition yields a producer and a
    consumer, and the latter is fused with the subsequent projection onto
    the faces.

    The preamble does not alter the graph passed in. It works on a copy of
    the graph's tasks.

    ## Spawning tasks

    If a peano_project is given, the producer spawns the touchCellFirstTime()
    events of its action sets as tasks, and the consumer waits for these
    tasks (@ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._spawn_tasks()):

    - Every action set of the producer whose touchCellFirstTime() has a
      non-empty body is wrapped into a
      @ref peano4.api.taskgraph.compiler.SpawnCellTaskActionSet.SpawnCellTaskActionSet.
      The wrapper evaluates the guard. If it holds, the wrapper takes a new
      task number from the tarch, stores it in the cell's task label (see
      below), and spawns the body as a task with this number. The task
      depends on the task spawned by the previous wrapped action set of the
      same producer, so the bodies are executed in their original order.
    - Action sets whose touchCellFirstTime() has an empty body, such as
      @ref exahype2.api.fv.actionsets.ApplyDirichletBoundaryConditions.ApplyDirichletBoundaryConditions, are not
      part of the producer-consumer pattern. They remain plain action sets
      of the producer, i.e. all their events are invoked by the producer.
    - The same holds for action sets which do not want to run in a task
      (@ref peano4.api.solversteps.ActionSet.ActionSet.may_run_in_background_task()),
      such as
      @ref exahype2.api.fv.actionsets.ReadjustTimeStepSizeForOptimisticGlobalTimestepping.ReadjustTimeStepSizeForOptimisticGlobalTimestepping.
      Their touchCellFirstTime() is invoked directly by the producer, i.e.
      possibly while tasks spawned by previous action sets of the same
      producer still run. They hence must not depend on these tasks.
    - The consumer is given a cousin
      @ref peano4.api.taskgraph.compiler.WaitForCellTaskActionSet.WaitForCellTaskActionSet
      per wrapped action set. If the cell's task label holds a valid task
      number, it waits for this task, returns the number to the task
      enumerator and resets the label to NoNumber=-1.
    - All other events of the wrapped action sets remain within the
      producer.

    The spawned tasks outlive touchCellFirstTime(). The compiler therefore
    warns about constructs which might not be safe within a task. It does
    not inspect the code but only searches the generated C++ text for the
    names of data and attributes, i.e. the warnings state that a task
    might access something. Every warning names the generator which
    realises the data in question and suggests a fix. It has to be
    accepted or the generation is aborted
    (@ref peano4.api.Logger.Logger.warn_and_confirm()):

    - The task body might access fine or coarse grid cell data which is not
      held via a smart pointer, e.g. via @ref peano4.api.datamodel.PatchToDoubleArray or
      @ref peano4.api.datamodel.PatchToDoubleArrayOnHeap. The task copies
      the data, and writes to the copy are lost. The fix is a smart-pointer
      generator such as
      @ref peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer.
      Heap-based generators do not suffice, as they deep-copy their data
      whenever Peano copies an object.
    - The task body might access vertex or face data via an enumerator
      whose data is not held via a smart pointer. The enumerators hold
      pointers to the objects on Peano's stacks, which are invalid once
      touchCellFirstTime() has terminated. The task therefore always
      captures them as peano4::datamanagement::FaceHeapDataEnumerator or
      peano4::datamanagement::VertexHeapDataEnumerator, which copy the
      faces or vertices. Without smart pointers, these copies are deep,
      i.e. writes are lost. With smart pointers, no warning is issued.
    - The task body might use a non-static attribute of the action set.
      The task accesses an object which is destroyed after the producer's
      mesh traversal.
    - An action set of the producer has to be modified by the user
      (@ref peano4.api.solversteps.ActionSet.ActionSet.user_should_modify_template()).
      Such a producer is not altered, i.e. it spawns no tasks.

    ## Task labels

    The compiler adds a new cell data type <identifier>_TaskLabels to the
    project's data model. It is modelled via DaStGen2 and holds one integer
    per producer action set with a non-empty touchCellFirstTime(). The
    integer's name is the CamelCase version of the original task's name
    followed by ActionSet, the action set's index within the task and
    TaskNumber, e.g. EulerFVSmoothInitialConditionActionSet0TaskNumber
    (@ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._task_number_attribute_name()).
    The labels furthermore hold a static tarch::multicore::TaskEnumerator
    (getTaskEnumerator()), from which the producers take their task numbers
    and to which the consumers return them. The producer declares the task
    labels as WRITE, the consumer as READ_AND_DISCARD: The producer sets
    the label in every sweep, whether it spawns a task or not, so the
    consumer's reset does not have to be kept. If the consumer declared
    READ_AND_WRITE, a fused producer step that directly follows a consumer
    would read the labels, too
    (@ref peano4.api.taskgraph.compiler.PropagateUseStatements), and the
    labels would be relayed through steps of other phases which run before
    the labels are ever stored.

    @param graph: peano4.api.taskgraph.Graph
      The task graph to compile. Assumed to be acyclic.

    @param identifier: str
      @ref peano4.api.taskgraph.compiler.FuseSteps.translate()

    @param peano_project: peano4.api.Project
      Project the task labels are added to. If None, no task labels are
      added and the producers spawn no tasks.

    @param use_fusion: bool
      If True, hand the rewritten graph over to FuseSteps, otherwise to
      Sequential.

    @return [peano4.api.solversteps.Step]
      @ref peano4.api.taskgraph.compiler.FuseSteps.translate()

    """
    producer_consumer_graph, split_tasks = _split_cell_tasks(graph)

    if peano_project is not None and split_tasks:
        task_labels = peano4.api.datamodel.DaStGen2(f"{identifier}_TaskLabels")
        task_labels.data.add_attribute(
            dastgen2.attributes.UserDefinedType(
                name="taskEnumerator",
                type="tarch::multicore::TaskEnumerator",
                include='#include "tarch/multicore/TaskEnumerator.h"',
                qualifier=dastgen2.attributes.Attribute.Qualifier.STATIC,
                user_type_has_toString_method=False,
            )
        )
        if _spawn_tasks(split_tasks, task_labels) > 0:
            peano_project.datamodel.add_cell(task_labels)

    non_fusable_task_pairs = [(producer, consumer) for _, producer, consumer in split_tasks]
    if use_fusion:
        return FuseSteps.translate(producer_consumer_graph, identifier, peano_project, non_fusable_task_pairs)
    else:
        return Sequential.translate(producer_consumer_graph, identifier, peano_project)


def _task_number_attribute_name(task: Task, action_set_index: int) -> str:
    """!

    CamelCase version of the task's name followed by ActionSet, the action
    set index and TaskNumber. Words are separated by whitespace, underscores
    or any other character that is not allowed in a C++ identifier (such as
    the hyphen in "Runge-Kutta"), and only the first letter of each word is capitalised, so
    existing camel case and acronyms are preserved. "EulerFVSmooth_Update
    cell" with action set index 0 for example becomes
    EulerFVSmoothUpdateCellActionSet0TaskNumber.

    """
    words = re.split(r"[^A-Za-z0-9]+", task.name)
    return "".join(word[:1].upper() + word[1:] for word in words if word) + f"ActionSet{action_set_index}TaskNumber"


def _copy_task(task: Task) -> Task:
    """!

    Shallow copy of a task with its own lists and dictionaries, so we can
    alter the copy's dependencies without touching the original graph. The
    action sets and the submodels themselves are shared.

    """
    result = copy.copy(task)
    result.dependencies = []
    result.action_sets = list(task.action_sets)
    result.cell_data = list(task.cell_data)
    result.face_data = list(task.face_data)
    result.vertex_data = list(task.vertex_data)
    result.cell_data_access = dict(task.cell_data_access)
    result.face_data_access = dict(task.face_data_access)
    result.vertex_data_access = dict(task.vertex_data_access)
    return result


def _exclusively_writes_cell_data(task: Task) -> bool:
    """!

    True if task declares at least one cell submodel as WRITE or
    READ_AND_WRITE, and no face or vertex submodel as WRITE or
    READ_AND_WRITE. @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._translate()

    """
    writing_accesses = (Task.DataAccess.WRITE, Task.DataAccess.READ_AND_WRITE)
    writes_cell_data = any(access in writing_accesses for access in task.cell_data_access.values())
    writes_other_data = any(
        access in writing_accesses
        for access in list(task.face_data_access.values()) + list(task.vertex_data_access.values())
    )
    return writes_cell_data and not writes_other_data


def _has_events_after_touch_cell_first_time(task: Task) -> List[str]:
    """!

    List the events which the task's action sets realise and which a
    producer would invoke before its spawned tasks have necessarily
    terminated: unprepareTraversal(), endTraversal() and the
    touch...LastTime() events. An event counts if the action set yields
    code for it.

    @return [str]
      One entry "<action set name>::<event>" per such event. The list is
      empty if the task can be split safely.

    """
    late_operations = (
        peano4.api.solversteps.ActionSet.OPERATION_END_TRAVERSAL,
        peano4.api.solversteps.ActionSet.OPERATION_TOUCH_VERTEX_LAST_TIME,
        peano4.api.solversteps.ActionSet.OPERATION_TOUCH_FACE_LAST_TIME,
        peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_LAST_TIME,
    )
    result = []
    for action_set in task.action_sets:
        if action_set.get_implementation_of_unprepareTraversal().strip() != "":
            result.append(f"{action_set.get_action_set_name()}::unprepareTraversal")
        for operation_name in late_operations:
            if action_set.get_implementation_of_operation(operation_name).strip() != "":
                result.append(f"{action_set.get_action_set_name()}::{operation_name}")
    return result


def _split_cell_tasks(graph: Graph):
    """!

    Build the producer-consumer graph. @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._translate()

    @return (peano4.api.taskgraph.Graph, [(Task, Task, Task)])
      The new graph and the list of (original task, producer, consumer)
      triples. Producer and consumer must not be fused.

    """
    producer_of_task = {}
    consumer_of_task = {}
    for task in graph.tasks:
        producer_of_task[task] = _copy_task(task)
        if task.may_spawn_background_tasks is False or not _exclusively_writes_cell_data(task):
            continue
        late_events = _has_events_after_touch_cell_first_time(task)
        if late_events and task.may_spawn_background_tasks is not True:
            peano4.api.Logger().log(
                f"Warning: Task {task.name} is not split into a producer and a consumer task, as a split-up might be unsafe: "
                f"The producer's spawned tasks might still run when the following events are invoked: "
                + ", ".join(late_events)
                + f". Set may_spawn_background_tasks=True on task {task.name} to enforce the split.",
                peano4.api.LoggerPrintOption.ALWAYS_PRINT,
            )
            continue
        producer_of_task[task].name = f"{task.name} Producer"
        consumer = Task(name=f"{task.name} Consumer", entity=Task.GridEntity.CELL)
        for submodel in task.cell_data:
            if task.cell_data_access[submodel] in (Task.DataAccess.WRITE, Task.DataAccess.READ_AND_WRITE):
                consumer.use_cell(submodel, Task.DataAccess.READ_AND_WRITE)
        consumer_of_task[task] = consumer

    result = Graph()
    for task in graph.tasks:
        producer = producer_of_task[task]
        for dependency in task.dependencies:
            # Depend on whatever finishes the dependency: its consumer if
            # it has been split, otherwise the (copied) task itself
            producer.add_dependency(consumer_of_task.get(dependency, producer_of_task[dependency]))
        result.add_task(producer)
        if task in consumer_of_task:
            consumer = consumer_of_task[task]
            consumer.add_dependency(producer)
            result.add_task(consumer)

    split_tasks = [(task, producer_of_task[task], consumer_of_task[task]) for task in graph.tasks if task in consumer_of_task]
    return result, split_tasks


def _non_static_attribute_names(attributes: str) -> List[str]:
    """!

    Names of the non-static attributes declared in the C++ snippet returned
    by an action set's get_attributes(). This is a heuristic: We split the
    snippet into statements, skip all statements which are static or
    contain a bracket (functions), and take the last identifier before an
    optional array size or initialisation.

    """
    attributes = re.sub(r"/\*.*?\*/", "", attributes, flags=re.S)
    attributes = re.sub(r"//[^\n]*", "", attributes)
    result = []
    for statement in attributes.split(";"):
        words = statement.split()
        if not words or "static" in words or "(" in statement:
            continue
        match = re.search(r"([A-Za-z_]\w*)\s*(\[[^\]]*\]\s*)*(=.*)?$", statement.strip(), re.S)
        if match:
            result.append(match.group(1))
    return result


def _generator_name(submodel) -> str:
    """!

    Name of the generator class which realises the submodel in C++.

    """
    generator = getattr(submodel, "generator", None)
    return "no generator" if generator is None else type(generator).__name__


def _is_held_via_smart_pointer(submodel) -> bool:
    """!

    True if the submodel's generator holds the data via a smart pointer,
    i.e. if copies of a C++ object share the data rather than cloning it.

    """
    return "SmartPointer" in _generator_name(submodel)


_SMART_POINTER_HINT = (
    "Realise the data via a smart-pointer generator, e.g. peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer. "
    "Heap-based generators such as peano4.api.datamodel.PatchToDoubleArrayOnHeap do not suffice, as they deep-copy "
    "their data whenever an object is copied, and Peano copies the objects when it moves them between its stacks."
)


def _enumerators_used_by(task: Task, task_body: str):
    """!

    Names of the vertex and face enumerators which a task body might use,
    i.e. which occur in the generated C++ text. They are realised as
    peano4::datamanagement::FaceEnumerator or
    peano4::datamanagement::VertexEnumerator and hold pointers into Peano's
    stacks, so the spawned task has to capture copies
    (@ref peano4.api.taskgraph.compiler.SpawnCellTaskActionSet.SpawnCellTaskActionSet).

    @return ([str], [str])
      Names of the face enumerators and of the vertex enumerators.

    """
    face_enumerators = [
        f"{prefix}{submodel.name}"
        for prefix in ("fineGridFaces", "coarseGridFaces")
        for submodel in task.face_data
        if re.search(rf"\b{prefix}{submodel.name}\b", task_body)
    ]
    vertex_enumerators = [
        f"{prefix}{submodel.name}"
        for prefix in ("fineGridVertices", "coarseGridVertices")
        for submodel in task.vertex_data
        if re.search(rf"\b{prefix}{submodel.name}\b", task_body)
    ]
    return face_enumerators, vertex_enumerators


def _warn_about_unsafe_task(task: Task, action_set_index: int, action_set: "peano4.api.solversteps.ActionSet", task_body: str):
    """!

    Issue the warnings about task bodies which might not
    be safe. @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._translate()

    We do not inspect the code. We only search the generated C++ text for
    the names of data and attributes, i.e. the warnings say that a task
    might access something.

    """
    logger = peano4.api.Logger()
    description = f"action set {action_set_index} ({action_set.get_action_set_name()}) of task {task.name}"

    for accessor in ("fineGridCell", "coarseGridCell"):
        for submodel in task.cell_data:
            if f"{accessor}{submodel.name}" in task_body and not _is_held_via_smart_pointer(submodel):
                logger.warn_and_confirm(
                    f"The touchCellFirstTime() task of {description} might access cell data {submodel.name} ({accessor}{submodel.name}). "
                    f"{submodel.name} is realised via the generator {_generator_name(submodel)}, which does not hold its data via a smart "
                    f"pointer. The task captures the cell data by value, i.e. it works on a copy, and all writes are lost. "
                    + _SMART_POINTER_HINT
                )

    face_enumerators, vertex_enumerators = _enumerators_used_by(task, task_body)
    for kind, submodels, enumerators, prefixes, enumerator_type in (
        ("face", task.face_data, face_enumerators, ("fineGridFaces", "coarseGridFaces"), "peano4::datamanagement::FaceHeapDataEnumerator"),
        ("vertex", task.vertex_data, vertex_enumerators, ("fineGridVertices", "coarseGridVertices"), "peano4::datamanagement::VertexHeapDataEnumerator"),
    ):
        for submodel in submodels:
            for enumerator in [f"{prefix}{submodel.name}" for prefix in prefixes if f"{prefix}{submodel.name}" in enumerators]:
                if not _is_held_via_smart_pointer(submodel):
                    logger.warn_and_confirm(
                        f"The touchCellFirstTime() task of {description} might access {kind} data {submodel.name} via {enumerator}. "
                        f"The task therefore captures {enumerator} as {enumerator_type}, which copies the {kind}s. {submodel.name} is "
                        f"realised via the generator {_generator_name(submodel)}, which does not hold its data via a smart pointer, i.e. "
                        f"the copies are deep: The task reads the data as it was when the task has been spawned, and all writes are lost. "
                        + _SMART_POINTER_HINT
                    )

    for attribute_name in _non_static_attribute_names(action_set.get_attributes()):
        if re.search(rf"\b{attribute_name}\b", task_body):
            logger.warn_and_confirm(
                f"The touchCellFirstTime() task of {description} might use the non-static attribute {attribute_name}. The task then "
                f"accesses an action set object which is destroyed after the producer's mesh traversal."
            )


def _spawn_tasks(split_tasks, task_labels: "peano4.api.datamodel.DaStGen2"):
    """!

    Make the producers spawn tasks and the consumers wait for them, and add
    one task number per spawning action set to the task labels.
    @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer._translate()

    @param split_tasks: [(Task, Task, Task)]
      Triples of original task, producer and consumer.

    @param task_labels: peano4.api.datamodel.DaStGen2
      The task labels. This routine adds the attributes.

    @return int
      Number of attributes added to the task labels.

    """
    logger = peano4.api.Logger()
    number_of_task_numbers = 0
    for task, producer, consumer in split_tasks:
        user_action_sets = [action_set for action_set in task.action_sets if action_set.user_should_modify_template()]
        if user_action_sets:
            logger.warn_and_confirm(
                f"Task {task.name} holds action sets which the user has to modify ("
                + ", ".join(action_set.get_action_set_name() for action_set in user_action_sets)
                + "). The task is split into producer and consumer, but spawns no tasks."
            )
            continue

        producer_action_sets = []
        consumer_action_sets = []
        previous_task_number_attribute_name = None
        for action_set_index, action_set in enumerate(task.action_sets):
            task_body = action_set.get_body_of_operation(peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)
            if task_body.strip() == "" or not action_set.may_run_in_background_task():
                # Nothing to spawn, or not worth spawning: The action set
                # remains a plain action set of the producer and is not part
                # of the producer-consumer pattern
                producer_action_sets.append(action_set)
                continue

            _warn_about_unsafe_task(task, action_set_index, action_set, task_body)
            task_number_attribute_name = _task_number_attribute_name(task, action_set_index)
            task_labels.data.add_attribute(dastgen2.attributes.Integer(name=task_number_attribute_name))
            number_of_task_numbers += 1

            face_enumerators, vertex_enumerators = _enumerators_used_by(task, task_body)
            producer_action_set = SpawnCellTaskActionSet(
                action_set,
                task_labels.name,
                task_number_attribute_name,
                previous_task_number_attribute_name,
                face_enumerators,
                vertex_enumerators,
            )
            producer_action_sets.append(producer_action_set)
            consumer_action_sets.append(WaitForCellTaskActionSet(producer_action_set))
            previous_task_number_attribute_name = task_number_attribute_name

        producer.action_sets = producer_action_sets
        consumer.action_sets = consumer_action_sets

        if previous_task_number_attribute_name is not None:
            producer.use_cell(task_labels, Task.DataAccess.WRITE)
            # The consumer only reads the label. Its reset to NoNumber is not
            # needed downstream, as every producer sets the label in each
            # sweep. Declaring READ_AND_WRITE would make the labels persist
            # into the next producer and, through it, into unrelated steps.
            consumer.use_cell(task_labels, Task.DataAccess.READ_AND_DISCARD)

    return number_of_task_numbers
