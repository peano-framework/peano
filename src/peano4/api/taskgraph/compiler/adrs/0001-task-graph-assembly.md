# ADR 0001: The task graph assembly spawns the six touch events as tasks and builds the DAG while it traverses

Status: proposed (implemented: sequential, fusion and concatenation; open issue: adaptive meshes with several trees)
Date: 2026-11-24

## Context

`CellTaskProducerConsumer` turns `touchCellFirstTime()` and `touchCellLastTime()` of tasks that only write
cell data into background tasks. Vertices and faces are not covered, a task that writes face or vertex data
cannot be split, and every consumer waits for its tasks in a mesh traversal of its own, i.e. the
traversal waits for a running task to quit and spawns the next one immediately. We want
to move the decision "what runs when" from the compiler into the traversal: Each mesh traversal inserts tasks and their dependencies
into a task graph, and nobody waits (outside of barriers).

## Decision

`TaskGraphAssembly` (`peano4.api.taskgraph.compiler.TaskGraphAssembly`) splits every task that acts on the mesh
into a producer and a consumer.

- **Six event tasks.** The producer wraps each action set into
  `SpawnTaskActionSet`. Its non-empty `touchVertexFirstTime/LastTime`, `touchFaceFirstTime/LastTime`
  and `touchCellFirstTime/LastTime` are spawned as tasks, one per entity and event. A vertex, a face and a cell hold
  two task numbers per wrapped action set, one for the first and one for the last event. They are independent:
  an action set that only has a last event only has a last task.
- **One enumerator.** All task numbers come from one `tarch::multicore::TaskEnumerator`
  (`peano4::datamanagement::taskassembly::getTaskEnumerator()`), and not from one per label type: Different
  enumerator instances do not hand out disjoint numbers. With one per label type, a face task and a cell task shared a
  number and waited for each other (the run produced NaNs in multi-threaded runs only, as a single thread runs
  tasks as soon as they are spawned).
- **Numbers are reserved early.** The number of the last task is reserved in the first touch, as
  the coarse entity and the entities around have to know it before the entity's last touch spawns the task.
- **Dependencies** (`peano4/datamanagement/TaskAssemblyDependencies.h`, all follow from the event order):
  - Within an entity: the tasks of the action sets follow each other; the last task depends on the first task.
  - Parent to child: the first task depends on the first tasks of the coarse entities (all coarse
    vertices, all coarse faces, the parent cell).
  - Child to parent: a fine cell, face (not interior of its patch) or vertex (coinciding with a coarse
    vertex) writes the number of its last task into an array of its coarse entity in its first touch
    (3^d entries per cell, 3^(d-1) per face, one per vertex). The coarse entity's last task depends on them.
    The array is reset in the first touch.
  - Across entities: the first task of a cell depends on the first tasks of its vertices and faces, the one of a face on
    the first tasks of its vertices. The cells announce their number on the vertices and faces around them
    (`2^d` and 2 slots), and the last tasks of the vertex and the face wait for them. The slots exist only if an
    action set of the same task has a last event on the entity.
  - Between tasks: the first task depends on the latest numbers of all wrapped action sets of tasks it depends on.
- **Hanging entities** are not persistent and see no touch events, i.e. they never hold a number.
  The create and destroy events of hanging faces and vertices run on the main thread and interpolate from or restrict into coarse data.
  They wait for the first tasks of the coarse entities. Persistent destroy events (`destroyCell`, `destroyPersistentFace`,
  `destroyPersistentVertex`) wait for the entity's own tasks and release their numbers.
- **Barriers.** `endTraversal()` and `unprepareTraversal()` of a wrapped action set start with
  `waitForAllTasks()`. The last event of the task is then global, and the global tasks follow the consumer anyway.
- **Domain decomposition.** Peano exchanges face and vertex data between the trees when a traversal ends, i.e.
  before the consumer sweep. The producer therefore records the numbers of all vertex tasks, of the face tasks next to a
  parallel boundary (`marker.isAdjacentToParallelBoundary()`), and of the cell tasks of a task that writes face or vertex
  data. Its `endTraversal()` waits for them (per tree, no global barrier). Cell tasks which write cell data only
  may outlive the traversal.
- **Consumer.** The consumer visits every persistent entity in its first touch and waits for the entity's tasks, releases
  the numbers and resets them. As the last task of a cell waits for the whole subtree, and the last task of a vertex or face
  for the cells around it, nothing is left once the traversal has terminated.
- **The consumer declares no data.** The consumer waits for the tasks of a cell only when it enters the cell. A task which follows
  and which is fused into the consumer's traversal has spawned tasks on the faces and vertices of the cell by then. If both tasks
  work on the same face or vertex submodel and one writes it, they race. FuseSteps cannot see this for a consumer which
  declares no data, i.e. the compiler hands it the pairs (consumer, follower with conflicting face or vertex data) as non-fusable
  pairs (`_add_hazard_pairs()`). The consumer must not declare the data of its producer instead: Peano relays data from
  one traversal to the next in a strict alternation (`PropagateUseStatements`). A declaration stores data that the producer
  has discarded, and the exchange stack of the next traversal is not empty anymore (assertion in `SpacetreeSet.cpph`).
- **Tasks that share TEMPORARY data are merged.** Peano creates TEMPORARY data at the beginning of a traversal and destroys
  them at the end. A producer and its consumer never share a traversal. The clearing of the face projections and the
  projection onto the faces share the TEMPORARY update of the faces: split, the projection accumulates onto
  uninitialised memory. `_merge_tasks_sharing_temporary_data()` merges such tasks into one task before the split.
- **A last task waits for the first tasks of the vertices and faces of its entity, too.** Otherwise the clearing of
  the faces, a face first task, runs after the projection of the cell, which is a cell last task, and deletes it.
- **Labels.** The compiler adds `<identifier>_VertexTaskLabels`, `_FaceTaskLabels` and `_CellTaskLabels`. They are added
  to the *steps* after the fusion (producers WRITE, consumers READ_AND_DISCARD, a fused step with both READ_AND_WRITE),
  and not to the tasks. Otherwise two producers which both write the face labels would conflict on face data and
  never share a traversal.
- **Events of the producer.** The producer task reports the events of the original action sets (`_ProducerTask`). The
  wrappers act in more events (they reset the arrays), and FuseSteps would otherwise place the producers too late.
- **A task is not split** if it does not act on the mesh, repeats, opts out (`may_spawn_background_tasks=False`), has
  nothing to spawn, or has an action set which has to run on the main thread (`may_run_in_background_task()`) but acts in a touch event.
- **Concatenation** (`translate_with_concatenation_and_sequential/fusion`) turns the consumer's wait into a task
  dependency of the next producer and removes the consumer. The task numbers of the removed consumer stay in the
  labels (READ_AND_WRITE in the steps that carry them), the first and last tasks of the follower depend on the last
  tasks of the tasks in front (own entity, vertices and faces of the cell), and the consumer at the end of the chain
  waits for the numbers of the whole chain. The compiler decides autonomously: A consumer is removed if its task has
  exactly one follower, the follower is split and has no main-thread action set on the mesh, and no task that follows
  conflicts on face or vertex data. A follower that is not split (global tasks, mesh refinement, load balancing)
  keeps the consumer in front of it, i.e. it acts as a barrier. Conservative on purpose: A conflicting follower would
  have to wait for the tasks of the neighbouring cells, and the entities do not know them. Only cell tasks
  that write cell data thus run across traversals; in the Euler time step, a consumer fewer is needed.

## Consequences

- The task graph of a traversal is a DAG over all entities, not only cells. Tasks of different levels run concurrently
  as far as the data allow.
- The mesh traversal still processes entities sequentially. A task writing data of a neighbour (shared face) relies
  on the dependencies above. A write-after-read hazard between a cell task and the face task of a later action set
  only holds if both belong to the same task (the arrays of the other task are not read).
- Face and vertex tasks read their data after the entity has been stored. Data that is not held via a smart pointer is
  copied. `TaskGraphAssembly` warns like `CellTaskProducerConsumer` does.
- The vertex last task does not wait for the faces around the vertex, only for the cells around it.
- The assembly uses data which `TaskGraphAssembly` adds to every entity, i.e. the memory per cell, face and vertex grows with the number of wrapped action sets.
- Only the sequential and the fusion variant exist so far.

## Validation and open issues

- Euler entropy wave (Runge-Kutta, rk4fd4, 8 trees): the errors and the orders are the ones of `FuseSteps`, bit by bit.
- Euler point source on a regular grid, debug build (assertions on), 2, 4 and 8 threads: the same time steps as `FuseSteps`.
- Euler point source with AMR: identical to `FuseSteps` with one tree (one thread). With two or more trees the
  solution deviates after the second mesh adaptation (the density criterion then wants to refine more cells), and the run
  reaches dt=inf some time steps later. The cause is not known. `FuseSteps` itself trips `SpacetreeSet.cpph:212`
  (exchange stack not empty) in a debug build of this setup, i.e. data relayed in the mesh adaptation phase is not consistent
  in either compiler. Hanging faces and vertices wait for everything which the traversal has spawned so far, which is correct but slow.
- `sequential` does not work on benchmarks whose setup repeats a traversal until the grid is stationary, as the load balancing
  task then sits in its own traversal (the plain `Sequential` compiler hangs in the same way).
