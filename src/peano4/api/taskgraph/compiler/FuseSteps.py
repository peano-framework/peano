# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import re
from typing import List, Optional, Tuple

import peano4.api
import peano4.api.solversteps

from ..Graph import Graph
from ..Task import Task
from .ReportCompiledStepSequence import report_compiled_step_sequence


def translate(
    graph: Graph,
    identifier: str,
    peano_project: "peano4.api.Project",
    non_fusable_task_pairs: Optional[List[Tuple[Task, Task]]] = None,
) -> List["peano4.api.solversteps.Step"]:
    """!

    Task graph compiler which fuses tasks into as few mesh traversals as possible

    You can use this task graph compiler in your ExaHyPE 2 project through

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    project.compiler = peano4.api.taskgraph.compiler.FuseSteps.translate
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Where peano4.api.taskgraph.compiler.Sequential.translate() maps every
    task onto a mesh traversal (Observer) of its own, this compiler packs
    several tasks into one traversal whenever this is safe.

    ## When can two tasks be fused?

    Two tasks end up in the same traversal if and only if all of the
    following constraints hold:

    1. **Entity order.** Each grid sweep first does something with the
       vertices, then with the faces, then with the cells (@ref
       peano4.api.taskgraph.Task.Task.GridEntity, whose values follow exactly this order). A task that
       works on a face can therefore be followed by a task working on a face
       or a cell within the same traversal, but a task working on a vertex
       has to go into the next traversal. In general, a node can be appended
       to the traversal built up so far if and only if the lowest grid
       entity it works on is not lower than the highest grid entity the
       traversal works on so far. Tasks of one dependency layer (see
       below) are independent of each other, so this constraint does not
       apply among them.
    2. **No conflicting face or vertex data.** A face or a vertex is shared
       between several cells. If one task of a traversal writes a face or
       vertex submodel (WRITE or READ_AND_WRITE) while another task of the
       same traversal reads or writes the very same submodel, a cell might
       see the submodel before or after a neighbouring cell has already
       altered it, depending on the traversal order. Such tasks are never
       fused (@ref peano4.api.taskgraph.compiler.FuseSteps._have_conflicting_data()). Cell data are not shared
       between cells and hence do not constrain the fusion. TEMPORARY uses
       are ignored. There is one exception which follows from the entity
       order: Peano touches every vertex and face of a cell for the first
       time before it enters the cell, and every vertex of a face before the
       face. A FACE task which reads or writes face data therefore has
       finished with a face before any subsequent face or cell task of the
       same traversal sees this face. The same holds for a VERTEX task and
       vertex data. A conflict on face (vertex) data hence is ignored if the
       earlier of the two tasks is a FACE (VERTEX) task. Two FACE tasks with
       conflicting face data thus can share one traversal, but they never
       run in parallel (constraint 2 of the algorithm below). Apart from
       this exception, the constraint is conservative: it relies only on
       the Task.DataAccess qualifiers and the tasks' grid entities, and does
       not know that a cell task might for example write the face data only
       when it touches a face for the last time.
    3. **No repetition.** Tasks which have to repeat (Task.has_to_repeat)
       are never fused. They always yield a traversal of their own.
    4. **No annotation.** The caller can annotate the fusion through
       non_fusable_task_pairs: two tasks forming such a pair never end up
       in the same traversal.
    5. **Global tasks.** A GLOBAL task (@ref peano4.api.taskgraph.Task.Task.GridEntity)
       does not touch any mesh entity. It does not take part in the
       algorithm below. Instead, it is appended to the traversal holding
       its last dependency, even if this traversal repeats. A GLOBAL task
       without dependencies goes into the very last traversal.
    6. **Start and end of the step.** GLOBAL_START_OF_STEP and
       GLOBAL_END_OF_STEP tasks do not take part in the algorithm below,
       either, and they never yield a traversal of their own.
       The action sets of the former are the first ones of the first
       traversal, those of the latter are the last ones of the last
       traversal, behind the other GLOBAL tasks. Among the end-of-step tasks,
       the dependencies decide the order. Why: The tasks act in
       prepareTraversal() and unprepareTraversal() only, i.e. they have to be
       the first and last thing that happens to the mesh in a step, and
       fusing them into the existing traversals saves a mesh sweep. A start
       task may not depend on any task. No task but another end-of-step task
       may depend on an end-of-step task, as the end-of-step tasks run when
       everything else has finished. If the first (last) traversal repeats,
       the compiler places the tasks into a traversal of their own in front
       of (behind) it, as they would otherwise run in every repetition.

<br /><br />
<div style="background-color: #fcc ; padding: 10px; border: 1px solid green;"> 
A task does not tell the compiler if its action sets plug into touch-first or touch-last
events. The compiler therefore assumes that a VERTEX (FACE) task works on its vertex
(face) data within touchVertexFirstTime() (touchFaceFirstTime()), i.e. before any
adjacent face or cell is entered. A VERTEX or FACE task which writes its data in
touchVertexLastTime() or touchFaceLastTime() violates this assumption, and the
compiler might fuse it with a subsequent task that then reads stale data. Conversely,
a CELL task A might only write face data in touchFaceLastTime(), while its successor
CELL task B reads this face data in touchCellFirstTime(). We could fuse A and B, but
this option is hidden from the task compiler.
</div>


    ## Algorithm

    1. Split the graph into layers by dependency depth (a task without
       dependencies has depth 0, every other task 1 + the deepest of its
       dependencies - the same notion of depth Graph.visualise() uses).
       Tasks within one layer can never depend on one another, i.e. they
       can run in parallel. Within a layer, tasks keep the order in which
       Graph.ordered_tasks() lists them.
    2. The tasks of one layer are combined into intermediate nodes. A
       repeating task forms a node of its own, placed in front of the
       layer's remaining nodes. The remaining tasks are added one by one to
       the layer's current node unless they have conflicting data with it
       or form a non-fusable pair with one of its tasks (constraint 4). In
       this case, they start a new node. Different to constraint 2, cell
       data counts as conflicting data here, too: Tasks of one node may be
       invoked concurrently on the same cell, i.e. they must not write
       cell data which another task of the node reads or writes. A node
       might work on several grid entities. A node is parallel if it holds
       more than one task and each of these tasks holds exactly one action
       set (@ref peano4.api.taskgraph.compiler.FuseSteps._is_parallel_node()).
       The action sets of a parallel node are labelled as parallel (@ref
       peano4.api.solversteps.ActionSet.ActionSet.parallel), i.e. they may
       be invoked concurrently. All other action sets are labelled as not
       parallel. A task with several action sets never is part of a
       parallel node: Its action sets have to be invoked one after the
       other, and the parallel flags of individual action sets can not
       express that a sequence of action sets runs in parallel to another
       task.
    3. The resulting sequence of nodes is fused greedily into traversals:
       a node is appended to the current traversal if constraints 1 to 4
       hold for the node and all tasks already in the traversal. Otherwise,
       it starts a new traversal. The fusion never reorders nodes and never
       looks further ahead.

    ## Examples

    The illustrations below are generated by
    src/peano4/api/taskgraph/compiler/fuse-steps-illustrations.py. Each
    shows the input task graph on top and the compiled sequence of
    traversals underneath.

    ### Entity order

    A chain of tasks working on a vertex, a face, a cell, a vertex and a
    cell. The vertex, face and cell tasks are fused into one traversal.
    The second vertex task has to start a new traversal, which also takes
    the subsequent cell task.

    @image html fuse-steps-entity-order.png

    ### Parallel tasks

    Two independent vertex tasks form one layer and hence one parallel
    node. They are fused with the two independent cell tasks of the
    subsequent layer. Action sets which may run in parallel share a number
    and are told apart by letters, e.g. 1.a and 1.b.

    @image html fuse-steps-parallel.png

    ### Conflicting face data

    The time step of an ExaHyPE 2 Finite Volume solver: the reconstruction
    reads the face data that the projection eventually writes. Therefore,
    the projection cannot be fused with the reconstruction and the update,
    and starts a traversal of its own.

    @image html fuse-steps-data-conflict.png

    ### Face task before cell task

    A FACE task sets boundary data on the faces, which the subsequent
    reconstruction reads. As the face task is done with a face before
    either adjacent cell is entered, all three tasks share one traversal
    even though the first two share face data.

    @image html fuse-steps-face-before-cell.png

    ## Constructing the fused step

    A fused step's name concatenates the names of its tasks, prefixed with
    identifier. Its action sets are the tasks' action sets one after the
    other, so the evaluation order of the tasks is preserved. We do not use
    peano4.api.solversteps.Step.add_action_set() for this, as it reorders
    the action sets according to their descend_invocation_order: a
    reconstruction with a higher descend_invocation_order would then
    overtake the update of the subsequent task. The descend_invocation_order
    is applied (stably) only among the action sets of one task.

    The generated observer spawns consecutive action sets which are labelled
    as parallel as one task group
    (@ref peano4.api.solversteps.SynchroniseActionSets.get_task_groups()).
    If two parallel nodes follow each other directly within a step, their
    groups would merge. We therefore insert a
    @ref peano4.api.solversteps.SynchroniseActionSets.SynchroniseActionSets
    between them - @ref peano4.api.taskgraph.compiler.FuseSteps._create_step().

    A submodel used by several tasks of a fused step is declared once on
    the step. Its qualifier is merged from the tasks' qualifiers: TEMPORARY
    uses are ignored, the step loads the submodel if the first remaining
    use reads it, and stores it if the last remaining use writes it -
    @ref peano4.api.taskgraph.compiler.FuseSteps._merge_data_access().

    propagate_use_statements() is not invoked here. It is a postprocessing
    step called outside of the translation.

    @param graph: peano4.api.taskgraph.Graph
      The task graph to compile. Assumed to be acyclic.

    @param identifier: str
      Identifies which of the surrounding project's algorithmic phases this
      graph belongs to (e.g. "SetupSimulation", "TimeStep", "Plot") and
      prefixes every generated Observer's name.

    @param peano_project: peano4.api.Project
      Not used by this compiler. @ref peano4.api.taskgraph.compiler

    @param non_fusable_task_pairs: [(peano4.api.taskgraph.Task, peano4.api.taskgraph.Task)]
      Pairs of tasks of graph which must not end up in the same traversal.
      The order within a pair does not matter. Defaults to none.

    @return [peano4.api.solversteps.Step]
      One Observer per fused group of tasks, in an order that respects
      every dependency.

    """
    non_fusable = set()
    for first_task, second_task in non_fusable_task_pairs or []:
        non_fusable.add((first_task, second_task))
        non_fusable.add((second_task, first_task))

    def can_be_combined(
        tasks: List[Task], further_tasks: List[Task], include_cell_data: bool = False, respect_entity_order: bool = False
    ) -> bool:
        """Constraints 2 and 4 between two lists of tasks, optionally taking cell data and the entity order into account"""
        return not _have_conflicting_data(tasks, further_tasks, include_cell_data, respect_entity_order) and not any(
            (task, further_task) in non_fusable for task in tasks for further_task in further_tasks
        )

    Entity = Task.GridEntity
    all_tasks = graph.ordered_tasks()
    start_tasks = [task for task in all_tasks if task.entity == Entity.GLOBAL_START_OF_STEP]
    end_tasks = [task for task in all_tasks if task.entity == Entity.GLOBAL_END_OF_STEP]
    global_tasks = [task for task in all_tasks if task.entity == Entity.GLOBAL]
    order = [task for task in all_tasks if not task.entity.is_global]
    for task in start_tasks:
        assert not task.dependencies, f"Task {task.name} is GLOBAL_START_OF_STEP and hence may not depend on any task"
    for task in all_tasks:
        for dependency in task.dependencies:
            assert dependency not in global_tasks, f"Task {task.name} depends on GLOBAL task {dependency.name}, which is not supported"
            assert dependency not in end_tasks or task in end_tasks, (
                f"Task {task.name} depends on GLOBAL_END_OF_STEP task {dependency.name}, "
                "but only other GLOBAL_END_OF_STEP tasks may do so"
            )

    depth_of_task = {}

    def depth(task: Task) -> int:
        # A start task has run before everything else, i.e. it does not delay anybody
        dependencies = [dependency for dependency in task.dependencies if dependency not in start_tasks]
        if task not in depth_of_task:
            depth_of_task[task] = 0 if not dependencies else 1 + max(depth(dependency) for dependency in dependencies)
        return depth_of_task[task]

    layers = {}
    for task in order:
        layers.setdefault(depth(task), []).append(task)

    # Step 1 and 2: nodes (lists of tasks) per layer
    nodes = []
    for layer in sorted(layers.keys()):
        tasks_in_layer = layers[layer]
        for task in tasks_in_layer:
            if task.has_to_repeat:
                nodes.append([task])
        nodes_of_layer = []
        for task in tasks_in_layer:
            if task.has_to_repeat:
                continue
            if nodes_of_layer and can_be_combined(nodes_of_layer[-1], [task], include_cell_data=True):
                nodes_of_layer[-1].append(task)
            else:
                nodes_of_layer.append([task])
        for node in nodes_of_layer:
            for task in node:
                for action_set in task.action_sets:
                    action_set.parallel = _is_parallel_node(node)
        nodes += nodes_of_layer

    # Step 3: greedy fusion along the sequence of nodes
    groups = []
    current_group = None
    highest_entity_of_current_group = None
    for node in nodes:
        lowest_entity = min(task.entity.value for task in node)
        highest_entity = max(task.entity.value for task in node)
        is_repeating = len(node) == 1 and node[0].has_to_repeat
        if (
            not is_repeating
            and current_group is not None
            and lowest_entity >= highest_entity_of_current_group
            and can_be_combined(
                [task for group_node in current_group for task in group_node], node, respect_entity_order=True
            )
        ):
            current_group.append(node)
            highest_entity_of_current_group = max(highest_entity_of_current_group, highest_entity)
        else:
            current_group = [node]
            highest_entity_of_current_group = highest_entity
            groups.append(current_group)
        if is_repeating:
            current_group = None

    # Constraint 5: append each global task to the group of its last dependency
    for global_task in global_tasks:
        groups_with_dependency = [
            index
            for index, group in enumerate(groups)
            if any(dependency in node for node in group for dependency in global_task.dependencies)
        ]
        if groups_with_dependency:
            groups[max(groups_with_dependency)].append([global_task])
        elif groups:
            groups[-1].append([global_task])
        else:
            groups.append([[global_task]])

    # Constraint 6: The tasks of the start and the end of the step
    def repeats(group):
        return any(task.has_to_repeat for node in group for task in node)

    if start_tasks:
        if not groups or repeats(groups[0]):
            groups.insert(0, [])
        groups[0] = [[task] for task in start_tasks] + groups[0]
    if end_tasks:
        if not groups or repeats(groups[-1]):
            groups.append([])
        groups[-1] += [[task] for task in end_tasks]

    steps = [_create_step(group, identifier) for group in groups]

    peano4.api.Logger().new_section(f"Compile task graph '{identifier}'")
    report_compiled_step_sequence(steps, title=f"Compiled task graph for {identifier}")

    return steps


def _have_conflicting_data(
    tasks: List[Task],
    further_tasks: List[Task],
    include_cell_data: bool = False,
    respect_entity_order: bool = False,
) -> bool:
    """!

    Check whether two sets of tasks can not share one mesh traversal due to
    their face or vertex data, or can not run in parallel

    True if one of the tasks writes a face or vertex submodel (WRITE or
    READ_AND_WRITE) which one of the further_tasks reads or writes
    (READ_AND_DISCARD, WRITE or READ_AND_WRITE), or vice versa. TEMPORARY
    uses are ignored. @ref peano4.api.taskgraph.compiler.FuseSteps.translate()

    @param include_cell_data: bool
      If False, cell data is ignored. This is the check whether tasks can
      share one mesh traversal. If True, conflicting cell data counts,
      too. This is the check whether tasks can run in parallel.

    @param respect_entity_order: bool
      If True, tasks precede further_tasks within one traversal, and a
      conflict on a face (vertex) submodel is ignored if the involved task
      of tasks is a FACE (VERTEX) task: Such a task has finished with a
      face (vertex) before any subsequent task sees it. This is the
      exception of constraint 2 in
      @ref peano4.api.taskgraph.compiler.FuseSteps.translate(). If False,
      tasks and further_tasks are treated symmetrically.

    """
    writing_accesses = (Task.DataAccess.WRITE, Task.DataAccess.READ_AND_WRITE)
    using_accesses = (Task.DataAccess.READ_AND_DISCARD, Task.DataAccess.WRITE, Task.DataAccess.READ_AND_WRITE)
    entity_of_kind = {"face": Task.GridEntity.FACE, "vertex": Task.GridEntity.VERTEX}

    def submodels(task, accesses):
        return {
            (kind, submodel)
            for kind, access_of_submodel in (
                (("cell", task.cell_data_access),) if include_cell_data else ()
            ) + (("face", task.face_data_access), ("vertex", task.vertex_data_access))
            for submodel, access in access_of_submodel.items()
            if access in accesses
        }

    for task in tasks:
        for further_task in further_tasks:
            conflicts = (submodels(task, writing_accesses) & submodels(further_task, using_accesses)) | (
                submodels(further_task, writing_accesses) & submodels(task, using_accesses)
            )
            if respect_entity_order:
                conflicts = {(kind, submodel) for kind, submodel in conflicts if entity_of_kind.get(kind) != task.entity}
            if conflicts:
                return True
    return False


def _task_name(task: Task) -> str:
    """!

    Turn every word into its own name segment, capitalising only its first
    letter - a blind .title() would also lower-case the rest of the word,
    which mangles existing camel case and acronyms (e.g. "EulerFVSmooth"
    would become "Eulerfvsmooth"). Words are separated by whitespace or by
    any other character that C++ identifiers do not allow, such as the
    hyphen in "Runge-Kutta", as the step name ends up in an enumerator.

    """
    words = re.split(r"[^0-9A-Za-z_]+", task.name)
    return "_".join(word[:1].upper() + word[1:] for word in words if word)


def _merge_data_access(accesses: List["Task.DataAccess"]) -> "Task.DataAccess":
    """!

    Merge the qualifiers several fused tasks attach to one submodel into one

    TEMPORARY uses are ignored. The fused step has to load the submodel if
    the first remaining use reads it (READ_AND_DISCARD or READ_AND_WRITE),
    and it has to store the submodel if the last remaining use writes it
    (WRITE or READ_AND_WRITE):

    | load | store | merged qualifier |
    |------|-------|------------------|
    | yes  | yes   | READ_AND_WRITE   |
    | yes  | no    | READ_AND_DISCARD |
    | no   | yes   | WRITE            |
    | no   | no    | TEMPORARY        |

    A submodel produced by one fused task and consumed by a later one
    (WRITE followed by READ_AND_DISCARD) hence becomes TEMPORARY. For a
    single access, the merged qualifier equals the access itself.

    """
    relevant_accesses = [access for access in accesses if access != Task.DataAccess.TEMPORARY]
    if not relevant_accesses:
        return Task.DataAccess.TEMPORARY
    load = relevant_accesses[0] in (Task.DataAccess.READ_AND_DISCARD, Task.DataAccess.READ_AND_WRITE)
    store = relevant_accesses[-1] in (Task.DataAccess.WRITE, Task.DataAccess.READ_AND_WRITE)
    if load and store:
        return Task.DataAccess.READ_AND_WRITE
    if load:
        return Task.DataAccess.READ_AND_DISCARD
    if store:
        return Task.DataAccess.WRITE
    return Task.DataAccess.TEMPORARY


def _create_step(nodes: List[List[Task]], identifier: str) -> "peano4.api.solversteps.Step":
    """!

    Create one step from a group of fused nodes. @ref peano4.api.taskgraph.compiler.FuseSteps.translate()

    Every node is a list of tasks: either a single task, or several tasks
    from one layer of the task graph which can run in parallel.

    Whenever a parallel node directly follows another parallel node, we
    insert a
    @ref peano4.api.solversteps.SynchroniseActionSets.SynchroniseActionSets,
    so the two nodes yield two task groups rather than one.

    """
    tasks = [task for node in nodes for task in node]
    step = peano4.api.solversteps.Step(
        name=f"{identifier}_" + "__".join(_task_name(task) for task in tasks),
        add_user_defined_actions=False,
    )
    # Repeating tasks are never fused with other non-global tasks, i.e. a
    # group holds at most one repeating task
    repeating_tasks = [task for task in tasks if task.has_to_repeat]
    assert len(repeating_tasks) <= 1
    step.has_to_repeat = repeating_tasks[0].has_to_repeat if repeating_tasks else None

    for data_attribute, access_attribute, step_access, use in (
        ("cell_data", "cell_data_access", step.cell_data_access, step.use_cell),
        ("face_data", "face_data_access", step.face_data_access, step.use_face),
        ("vertex_data", "vertex_data_access", step.vertex_data_access, step.use_vertex),
    ):
        accesses_of_submodel = {}
        for task in tasks:
            for submodel in getattr(task, data_attribute):
                accesses_of_submodel.setdefault(submodel, []).append(getattr(task, access_attribute)[submodel])
        for submodel, accesses in accesses_of_submodel.items():
            use(submodel)
            step_access[submodel] = _merge_data_access(accesses)

    previous_node_is_parallel = False
    for node in nodes:
        node_is_parallel = _is_parallel_node(node)
        if node_is_parallel and previous_node_is_parallel:
            step.action_sets.append(peano4.api.solversteps.SynchroniseActionSets())
        for task in node:
            for action_set in sorted(task.action_sets, key=lambda action_set: action_set.descend_invocation_order):
                step.action_sets.append(action_set)
        previous_node_is_parallel = node_is_parallel

    return step


def _is_parallel_node(node: List[Task]) -> bool:
    """!

    A node is parallel if it holds more than one task and each task holds
    exactly one action set. @ref peano4.api.taskgraph.compiler.FuseSteps.translate()

    """
    return len(node) > 1 and all(len(task.action_sets) == 1 for task in node)
