# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org

import enum
from typing import FrozenSet, Iterable, List, Optional, Set


class Task:
    """!

    One task within a peano4.api.taskgraph.Graph

    Every task works on exactly one kind of mesh entity - a vertex, a face
    or a cell - which has to be specified through entity when the task is
    constructed. GLOBAL tasks are the exception: they do not work on any
    mesh entity (@ref peano4.api.taskgraph.Task.Task.GridEntity).
    
    ## Task dependencies

    Every task also holds dependencies, i.e. other Task instances that have
    to be completed before this task can run. This sequence is empty by
    default; use add_dependency() to add further tasks to it.

    ## Task action sets

    Every task furthermore holds an arbitrary number of Peano 4 action sets
    (peano4.api.solversteps.ActionSet), i.e. the actual grid-sweep logic the
    task realises once it is compiled into an Observer. This sequence is
    empty by default; use add_action_set() to add further action sets to it.
    Their order matters - see peano4.api.solversteps.Step.add_action_set().

    ## Task data usage

    A task's action sets are handed mesh data (fineGridCell..., fineGridFace...,
    fineGridVertices...) as arguments once they are compiled into an Observer -
    but only for the vertex/face/cell submodels the surrounding
    peano4.api.solversteps.Step was told about via its own use_cell()/
    use_face()/use_vertex(). An action set cannot make that declaration
    itself (several tasks, and hence several Steps, can share the same
    action set), so every task has to declare which submodels it needs
    through use_cell()/use_face()/use_vertex() below. The task graph
    compiler then forwards these declarations to the Step(s) it compiles
    this task into.

    ## Task repetitions
    
    Finally, a task can be marked as repeating via task_has_to_repeat. By
    default (None), a task runs once. Passing a non-None string works if 
    and only if it is a valid boolean C++ expression. Peano will evaluate
    it on the global runner after the task has finished and then decide if
    to rerun this step. An all time classic is the following expression
    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    not peano4::grid::isGridStationary( peano4::parallel::SpacetreeSet::getInstance().getGridStatistics() )
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    which means this task will repeat until the mesh becomes stationary.

    ## Background tasks

    Some task graph compilers can run parts of a task in the background,
    i.e. as tasks that outlive the mesh traversal event which spawns them
    (@ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer.translate_with_sequential(), @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer.translate_with_fusion()).
    Such compilers check whether this is safe, and leave the task as it is
    if they cannot be sure. The flag may_spawn_background_tasks steers this
    behaviour:

    - None (default): The compilers decide whether running the task in the
      background is safe.
    - True: You know that running the task in the background is fine. The
      compilers skip their checks.
    - False: The task must never run in the background, e.g. as it
      reduces data in unprepareTraversal() such as a global CFL condition.
      The compilers leave the task as it is without any further check or
      warning.

    ## Task tags

    Every task carries a set of tags, i.e. of plain strings. A task can carry
    arbitrarily many tags at the same time. The task graph compilers ignore
    them. Tags allow the code which creates a task graph to tell other code
    what a task does, without that code having to know how the graph is
    built. Tags are global: Add a tag via task.tags.add(tag) and find
    all tasks with a tag via peano4.api.taskgraph.Graph.find_tasks().
    ExaHyPE's numerical schemes, for example, tag all tasks which produce a
    new solution with exahype2.api.Solver.Solver.SolutionUpdateTaskTag, such
    that others can add postprocessing steps after them (@ref
    page_exahype2_solver_manipulate_solver_steps). Use names with a prefix,
    e.g. the name of the defining class, so tags of different parties
    cannot clash by accident.

    ## Task processing

    The tasks will later on be taken by the graph compiler which maps them
    onto a sequence of @ref peano4.api.solversteps.Step instances. How many
    of these logical steps (grid traversals) are required depends on the 
    structure of the task graph but also the type of compiler chosen. 
    Contraray, each action set that is added to a task will remain to be a 
    real @ref peano4.api.solversteps.ActionSet, and subsequently be translated
    into exactly one @ref peano4.api.output.ActionSet, i.e. one pair of header
    and implementation file called MyFancyObserverOfAGridTraversal2ThisActionSet.
    
    ## Things to read to get the full picture:    

    - @ref peano4.api.taskgraph This is the overall description of the task 
           graph API and should be the starting point for understanding how to use it.
           It notably describes some design decisions re dependencies, grid
           entities and action sets.
    - @ref page_peano_action_sets    
    - @ref peano4.api.taskgraph

    """

    class GridEntity(enum.Flag):
        """!

        Which kind of mesh entities a Task works on.

        The entity is a set of marks: VERTEX, FACE and CELL for the mesh
        entities and GLOBAL for the activities per rank. They combine
        freely, e.g. FACE_CELL (the same as FACE | CELL), which results in
        2x2x2x2 = 16 variants. The compiler synthesises the marks from the
        events which the task's action sets use
        (@ref peano4.api.taskgraph.Task.Task.entity). The projection of the
        cell data onto the faces, for example, acts in touchCellFirstTime() and
        in the touchFace...() events and thus becomes a FACE_CELL task.

        Tasks which carry the GLOBAL mark besides mesh marks (e.g.
        CELL | GLOBAL) are mesh tasks which also act in the traversal
        events.

        For the entity order of the fusion
        (@ref peano4.api.taskgraph.compiler.FuseSteps.translate()), a task
        counts as its primary entity, i.e. its highest mesh mark:
        a FACE_CELL task is a cell task.

        GLOBAL tasks do not work on any mesh entity at all, i.e. they carry no
        mesh mark. They only plug into prepareTraversal()/unprepareTraversal(), i.e. they realise
        global (per rank) activities such as triggering the load balancing.
        They therefore may not declare any cell, face or vertex data, may
        not repeat. Task graph compilers which fuse tasks append a GLOBAL
        task to the traversal holding its last dependency
        (@ref peano4.api.taskgraph.compiler.FuseSteps.translate()). Other
        tasks may depend on a GLOBAL task. The compiler then does not fuse
        them into the traversal of the GLOBAL task, as the latter acts in
        the prepare and unprepare events of this traversal: they run in a
        later mesh sweep.

        GLOBAL_START_OF_STEP and GLOBAL_END_OF_STEP are GLOBAL tasks that are
        tied to the beginning and the end of the whole task graph rather than
        to the traversal of their dependencies. Both do not require a mesh
        traversal of their own:

        - A GLOBAL_START_OF_STEP task runs first, i.e. its action sets are the
          first ones of the first traversal. It may not depend on any task.
          Other tasks may depend on it, but this is trivial.
        - A GLOBAL_END_OF_STEP task runs last, i.e. its action sets are the
          last ones of the last traversal, after all other tasks including
          the other GLOBAL tasks. Other GLOBAL_END_OF_STEP tasks may depend
          on it, which orders them among each other. A typical
          example is the global CFL condition, which reduces data in
          unprepareTraversal() only. If a task that is not an end-of-step
          task depends on it, it cannot run at the very end of the step.
          It then ends the traversal of its last dependency, as a GLOBAL
          task does, and the dependent task runs in a later traversal
          (@ref peano4.api.taskgraph.Graph.Graph.anchored_end_of_step_tasks()).

        Task graph compilers that do not fuse tasks guarantee the same order
        (@ref peano4.api.taskgraph.Graph.Graph.ordered_tasks()).

        """

        VERTEX = 1
        FACE = 2
        CELL = 4
        GLOBAL = 8
        GLOBAL_START_OF_STEP = 8 | 16
        GLOBAL_END_OF_STEP = 8 | 32
        FACE_VERTEX = 2 | 1
        VERTEX_CELL = 1 | 4
        FACE_CELL = 2 | 4
        FACE_VERTEX_CELL = 2 | 1 | 4

        @property
        def mesh_marks(self) -> "Task.GridEntity":
            """! The VERTEX, FACE and CELL marks of the entity """
            return self & (Task.GridEntity.FACE_VERTEX_CELL)

        @property
        def is_global(self) -> bool:
            """! True for GLOBAL, GLOBAL_START_OF_STEP and GLOBAL_END_OF_STEP, i.e. if there is no mesh mark """
            return bool(self & Task.GridEntity.GLOBAL) and not self.mesh_marks

        @property
        def primary(self) -> "Task.GridEntity":
            """! The highest mesh mark, i.e. CELL, FACE or VERTEX. Undefined for global tasks """
            for mark in (Task.GridEntity.CELL, Task.GridEntity.FACE, Task.GridEntity.VERTEX):
                if self & mark:
                    return mark
            raise ValueError(f"{self} has no mesh mark")

        @property
        def label(self) -> str:
            """! Name of the entity, e.g. FACE_CELL, also for combinations without a name of their own """
            return self.name if self.name is not None else str(self)

    class TraversalEvent(enum.IntEnum):
        """!

        The events of a mesh traversal in the order in which they occur

        A traversal begins with the first touch of the vertices, then of the
        faces, then of the cells, and it ends with the last touch in the
        opposite order. The creation of vertices, faces and cells belongs to the
        first touch, the destruction to the last touch. A task's first
        and last event (@ref peano4.api.taskgraph.Task.Task.first_event)
        decide which tasks can share a traversal.

        """

        PREPARE_TRAVERSAL = 0
        BEGIN_TRAVERSAL = 1
        VERTEX_FIRST = 2
        FACE_FIRST = 3
        CELL_FIRST = 4
        CELL_LAST = 5
        FACE_LAST = 6
        VERTEX_LAST = 7
        END_TRAVERSAL = 8
        UNPREPARE_TRAVERSAL = 9

    class DataAccess(enum.Enum):
        """!

        How a task's action sets actually access a submodel it declares via
        use_cell()/use_face()/use_vertex() - i.e. the qualifier that
        accompanies every such declaration.

        - READ_AND_DISCARD: the action set only reads the submodel, never
          writes it, and does not need its value kept around afterwards -
          this task is a final consumer, not a link in a longer chain.
          The submodel is loaded before the step, but it is not stored
          after the step: Its value is lost unless another task of the
          same step writes it.
        - WRITE: the action sets only write the submodel (e.g. fill in an
          initial condition), without relying on whatever value it held
          before. The submodel is stored after the step, but it is not
          loaded before the step: The action sets see uninitialised data,
          not the value the submodel held before.
        - READ_AND_WRITE: the action sets both read the submodel's
          existing value and write a new one (e.g. an in-place update).
          The WRITE component doesn't mean necessarily that we alter the
          content. It simply tells the task that the data should not be
          discarded afterwards. The submodel is loaded before and stored
          after the step.
        - TEMPORARY: the submodel is neither meaningfully read nor written
          by this task for its own sake - available for a task graph
          author to say so explicitly, though nothing currently derives
          this qualifier automatically. The submodel is neither loaded nor
          stored.

        The qualifiers matter at two levels. On the task-graph level, WRITE
        and READ_AND_WRITE are treated identically by
        peano4.api.taskgraph.compiler.PropagateUseStatements.propagate_use_statements():
        both mean this task produces a genuinely new value for the
        submodel, which someone downstream then has to actually pick up
        (read) - see that function's documentation for the precise rule,
        and note that the READ_AND_WRITE declarations *it* inserts to
        satisfy that rule are indistinguishable from ones a task graph
        author wrote by hand. READ_AND_DISCARD and TEMPORARY impose no
        such obligation on whatever comes after.

        On the level of the generated code, every qualifier translates
        into the same use_cell()/use_face()/use_vertex() call on the
        peano4.api.solversteps.Step this task compiles into (@ref
        peano4.api.taskgraph.compiler.Sequential.translate()). However,
        the qualifiers decide whether the generated code loads the
        submodel before and stores it after a step: @ref
        peano4.api.taskgraph.compiler.DeriveLoadStoreComputeFlag.derive_load_store_compute_flag()
        derives these flags from the qualifiers as described per qualifier
        above, and exahype2.api.Project.generate_Peano4_project() applies
        it to all steps. If several tasks share a step,
        peano4.api.taskgraph.compiler.FuseSteps merges their qualifiers
        first.

        Pick the qualifiers with care if a submodel's value has to survive
        several steps: A task which reads data that an earlier task of
        another step has declared READ_AND_DISCARD, or a task declaring
        WRITE which still reads the previous value, reads garbage. The
        Sommerfeld boundary conditions of exahype2.api are an example
        (@ref exahype2.api.blockstructured.BoundaryConditions.get_access_to_persistent_face_data()).

        """

        READ_AND_DISCARD = 1
        WRITE = 2
        READ_AND_WRITE = 3
        TEMPORARY = 4

    def __init__(
        self,
        name: str,
        entity: "Task.GridEntity",
        task_has_to_repeat: Optional[str] = None,
        may_spawn_background_tasks: Optional[bool] = None,
        tags: Optional[Iterable[str]] = None,
    ):
        self.name = name
        self.tags: Set[str] = set(tags) if tags is not None else set()
        self.declared_entity = entity
        self._derived_cache = None
        self.dependencies: List["Task"] = []
        self.action_sets: List["peano4.api.solversteps.ActionSet"] = []
        self.cell_data: List["peano4.api.datamodel.DoF"] = []
        self.face_data: List["peano4.api.datamodel.DoF"] = []
        self.vertex_data: List["peano4.api.datamodel.DoF"] = []
        self.cell_data_access: dict = {}
        self.face_data_access: dict = {}
        self.vertex_data_access: dict = {}
        self.has_to_repeat = task_has_to_repeat
        self.may_spawn_background_tasks = may_spawn_background_tasks
        assert not (entity.is_global and task_has_to_repeat is not None), (
            f"Task {name} is global and hence may not repeat"
        )

    def _derived(self):
        """! Marks and events which the action sets imply, cached as long as the action sets do not change """
        key = tuple(id(action_set) for action_set in self.action_sets)
        if self._derived_cache is None or self._derived_cache[0] != key:
            from .compiler.ValidateTaskGraph import derive_marks_and_events

            self._derived_cache = (key, *derive_marks_and_events(self.action_sets))
        return self._derived_cache[1], self._derived_cache[2]

    @property
    def entity(self) -> "Task.GridEntity":
        """!

        The entity of the task: the marks which the action sets' events imply
        united with the declared entity

        The task graph compiler synthesises the entity from the action sets,
        i.e. nobody has to find the right one of the 16 combinations.
        Which event implies which mark is defined by
        @ref peano4.api.taskgraph.compiler.ValidateTaskGraph.MARKS_OF_EVENTS.
        The declared entity (constructor) is a lower bound. A task declared as
        GLOBAL_START_OF_STEP or GLOBAL_END_OF_STEP keeps this entity, as
        this is a statement on when the task runs, which the action sets cannot tell.

        """
        if self.declared_entity in (Task.GridEntity.GLOBAL_START_OF_STEP, Task.GridEntity.GLOBAL_END_OF_STEP):
            return self.declared_entity
        return self.declared_entity | self._derived()[0]

    @property
    def events(self) -> FrozenSet["Task.TraversalEvent"]:
        """!

        The traversal events in which the task's action sets are not empty

        If the action sets use no event at all (they might not exist yet), the
        task acts in the first event of its primary declared entity, i.e. a CELL task in
        CELL_FIRST. Global tasks without events act in none.

        """
        events = self._derived()[1]
        if not events and not self.declared_entity.is_global:
            first_event_of_mark = {
                Task.GridEntity.VERTEX: Task.TraversalEvent.VERTEX_FIRST,
                Task.GridEntity.FACE: Task.TraversalEvent.FACE_FIRST,
                Task.GridEntity.CELL: Task.TraversalEvent.CELL_FIRST,
            }
            return frozenset({first_event_of_mark[self.declared_entity.primary]})
        return events

    @property
    def first_event(self) -> Optional["Task.TraversalEvent"]:
        """! The earliest event of the task, or None if it has none """
        return min(self.events) if self.events else None

    @property
    def last_event(self) -> Optional["Task.TraversalEvent"]:
        """! The latest event of the task, or None if it has none """
        return max(self.events) if self.events else None

    @property
    def event_label(self) -> str:
        """!

        The first and the last event of the task as label

        VERTEX_FIRST_CELL_LAST says that the first non-empty event is
        touchVertexFirstTime() and the last one touchCellLastTime(). The label has one
        part only if first and last event coincide, e.g. CELL_FIRST. Events in between
        might or might not be used. The label is NONE for tasks without events.

        """
        if not self.events:
            return "NONE"
        if self.first_event == self.last_event:
            return self.first_event.name
        return f"{self.first_event.name}_{self.last_event.name}"

    def add_dependency(self, task: "Task"):
        self.dependencies.append(task)

    def use_cell(self, submodel: "peano4.api.datamodel.DoF", qualifier: "Task.DataAccess"):
        """!

        Declare that this task's action sets access a cell submodel (e.g. a
        peano4.api.datamodel.Patch registered via
        peano_project.datamodel.add_cell()), and how - @see Task.DataAccess
        for what qualifier to pass. Forwarded to the
        peano4.api.solversteps.Step this task is compiled into, i.e. to its
        own use_cell(), so the generated Observer passes fineGridCell<name>
        into every touch-cell/create-cell/... operation.

        """
        assert not self.declared_entity.is_global, f"Task {self.name} is global and hence may not use cell data"
        self.cell_data.append(submodel)
        self.cell_data_access[submodel] = qualifier

    def use_face(self, submodel: "peano4.api.datamodel.DoF", qualifier: "Task.DataAccess"):
        """!

        Declare that this task's action sets access a face submodel, and
        how. @see use_cell() for the general mechanism and Task.DataAccess
        for what qualifier to pass.

        """
        assert not self.declared_entity.is_global, f"Task {self.name} is global and hence may not use face data"
        self.face_data.append(submodel)
        self.face_data_access[submodel] = qualifier

    def use_vertex(self, submodel: "peano4.api.datamodel.DoF", qualifier: "Task.DataAccess"):
        """!

        Declare that this task's action sets access a vertex submodel, and
        how. @see use_cell() for the general mechanism and Task.DataAccess
        for what qualifier to pass.

        """
        assert not self.declared_entity.is_global, f"Task {self.name} is global and hence may not use vertex data"
        self.vertex_data.append(submodel)
        self.vertex_data_access[submodel] = qualifier

    def add_action_set(self, action_set: "peano4.api.solversteps.ActionSet"):
        """!
        
        A task can hold an arbitrary number of Peano 4 action sets. Eventually
        the overarching task graph will be mapped onto a sequence of steps
        (observers) and those steps then grab those action sets from the tasks and execute them. 
        
        The events the action set uses determine the task's entity
        (@ref peano4.api.taskgraph.Task.Task.entity).
        
        """
        self.action_sets.append(action_set)
