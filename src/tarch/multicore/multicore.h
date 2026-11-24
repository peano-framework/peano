// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org

#include <string>
#include <set>
#include <vector>
#include <list>
#include <limits>

#include "config.h"
#include "tarch/compiler/CompilerSpecificSettings.h"
#include "tarch/multicore/orchestration/Strategy.h"

#if defined(SharedOMP) || defined(SharedTBB) || defined(SharedCPP) || defined(SharedSYCL)
#define SharedMemoryParallelisation
#endif

#pragma once

namespace tarch {
  /**

   @namespace tarch::multicore


   This page describes Peano 4's multithreading namespace. A high-level
   overview is provided through @ref page_tarch_multicore. This high-level
   page describes the multicore component from a user perspective, whereas
   the present namespace documentation focuses on the implementation of all
   multithreading within Peano.


   # Multicore architecture

   The multithreading environment is realised through a small set of classes
   and functions within the namespace @ref tarch::multicore. User
   code programmes against these classes and functions. Each type/function has
   a vanilla implementation within the directory
   src/multicore. This implementation is a dummy that ensures that all code works
   properly without any multithreading support.


   Subdirectories hold alternative
   implementations (backends) which are enabled once the user selects a certain multithreading
   implementation variant, i.e. depending on the ifdefs set, one of the
   subdirectories is used. Some implementations introduce further headers, but user
   code is never supposed to work against functions or classes held within
   subdirectories.

   @image html src/tarch/multicore/multicore_architecture.png


   # Where to find documentation

   As most backend-specific subdirectories solely contain specific implementation
   files, i.e. no headers, all back-end specific documentation resides directly
   within the actual routines. That is, many multicore routines have different
   sections within the documentation (in the header) which discusses how the
   routine semantics are mapped onto specific back-ends.

   This implies that documentation about a specific backend (e.g. "how is the
   multithreading done with OpenMP") is not consolidated within a specific file.
   It is scattered among various routines.


   # Task execution patterns and orchestration

   Peano works with arbitrary dynamic task graphs where individual tasks are
   identified by a unique number (cmp discussion on Task graphs on @ref page_tarch_multicore, which
   provides details how to use this feature from a programmer's point of view).
   Within the task graph, the code makes no assumptions on

   - how many tasks are created at most at one point;
   - if a task A that has spawned another task B is still alive when B becomes
     ready;
   - how many of the tasks have the same type.

   The second variant is particularly important: It implies that Peano makes
   no assumption at all about a certain well-structuredness (tree structure)
   of the task graph. Many task graphs will be constructed along a tree pattern,
   where tasks fork into more and more children tasks. However, it is absolutely
   not necessary that a task waits for all children that is has created before it
   terminates itself, i.e. we do not require the task graph to unfold and collapse
   along a tree pattern.

   @image html src/tarch/multicore/typical-task-graphs.png

   Notably, each mesh
   traversal creates one task per subdomain on a rank when we trigger a mesh
   traversal. These mesh traversals might span tasks with complex dependencies,
   while we wait for them to terminate in a subsequent mesh traversal. We allow
   for producer-consumer patterns without strong synchronisation.

   In the example above, the mesh traversal is a task (bright red) which in
   turn creates four traversal tasks, as four local subdomains reside on this
   rank (dark green). These traversals assemble a rather complex task graph
   (blue) and then terminate. But they do not wait for the nodes of this task
   graph to complete, too. Later on, we issue another mesh traversal (dark red)
   spawning four children in a fork-join manner (bright green). These child
   tasks now might wait for tasks from the task graph (bright blue).

   Peano supports various task execution patterns which are guided through the
   task orchestrations. Users can switch the orchestration for their code or
   even write bespoke strategies how to arrange the orchestration.

   The orchestration determines, for example, which tasks end up on which GPU,
   or if a code wants to fuse multiple tasks into big meta tasks.


   ## Orchestration details

   The information how tasks can be orchestrated but also how the orchestration
   is programmed is all consolidated within the namespace tarch::multicore::orchestration.
   You can swap the orchestration through calls of @ref tarch::multicore::swapOrchestration().

   Bespoke orchestrations have to inherit from @ref tarch::multicore::orchestration::Strategy.
   The signature of this class showcases how much control the user has
   eventually over the task orchestration, as all orchestration is guided
   via callbacks to the strategy in control.

   ## Task fusion

   The canonical starting point to learn more about Peano's task fusion is the page
   @ref documentation_tarch_task_fusion_strategies.

   An overview of the most important strategy decisions - in particular when it
   comes to fusion - is available through @ref tarch::multicore:orchestration::FusionImplementation.

   # Backends

   The backend of choice is set when we compile Peano, i.e. through the
   autotools or CMake. Details are provided on
   @ref page_tarch_multicore documentation.
   In principle, all programming should be completely backend-agnostic. However,
   there are some high-level issues with have to be considered. For some back-ends,
   we also have to provide workarounds and provide some additional utility
   features.

   ## OpenMP

   <img src="https://upload.wikimedia.org/wikipedia/commons/e/eb/OpenMP_logo.png" width="400px" />

   ### High-level usage (main function)

   If you want to use the OpenMP backend, you have to embed your whole main loop
   within an

  ~~~~~~~~~~~~~~~~~~~~~~~~~
  #pragma omp parallel
  #pragma omp single
  {
  ~~~~~~~~~~~~~~~~~~~~~~~~~

   environment.

   ### Using OpenMP code

   Furthermore, you will have to use

  ~~~~~~~~~~~~~~~~~~~~~~~~~
    export OMP_NESTED=true
  ~~~~~~~~~~~~~~~~~~~~~~~~~

   on some systems, as we rely heavily on nested parallelism.


   At the moment,
   OpenMP does not allow you to set the number of used threads within the
   application, i.e. through @ref tarch::multicore::Core. Instead, you have to
   set ```OMP_NUM_THREADS``` accordingly.


   ### Transparent tasks

   The realisation of task patterns as described above requires transparent
   task dependencies as they become available with OpenMP 6. As many production
   systems do not support OpenMP 6 yet, Peano has a workaround to mirror
   transparent task dependencies. Consequently, the documentation of individual
   routines and their implementation often distinguishes which OpenMP version
   is actually supported and maps onto the implementation. There is a routine
   tarch::multicore::omp::majorOpenMPVersion() which allows code to switch
   depending on the version. This routine is for the implementation internal
   only, so there is no Doxygen documentation for it.

   The documentation of the OpenMP 5 workarounds, but also some details on
   OpenMP 6 can be found in the documentation of the namespace
   @ref tarch::multicore::omp::internal.


   ## OneTBB

   <img src="https://oneapi.io/wp-content/uploads/2019/11/cropped-oneAPI-rgb-3000@2x.png" width="400px" />

   Peano's usage pattern of OneTBB


   ## SYCL

   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/SYCL_logo.svg/2560px-SYCL_logo.svg.png" width="400px" />


   */
  namespace multicore {
     /**
      * Forward declaration
      */
     class Task;

     using TaskNumber = int;

     /**
      * Should be the same as tarch::Enumerator::NoNumber, but I cannot use it
      * here due to cyclic includes. The reason why it should be NoNumber is
      * explained in spawnTask().
      */
     constexpr TaskNumber ExecuteImmediately = -1;
     constexpr TaskNumber NoOutDependencies =  -2;

     const std::set<TaskNumber> NoInDependencies = std::set<TaskNumber>();

     /**
      * Statistics identifier for spawned tasks
      *
      * If you translate in statistics mode, Peano's tarch will yield a
      * statistics file, where it tracks the number of tasks that we spawn
      * over time.
      *
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      * python3 ~/git/Peano/python/peano4/postprocessing/inspect-statistics.py statistics-rank-0.csv
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      *
      * over the output file (with the .csv extension), the script tells
      * you in which column it dumps data of this flag.
      *
      * If you visualise this through
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      * python3 ~/git/Peano/python/peano4/postprocessing/plot-statistics.py -c 3 -pt scatter -o pdf statistics-rank-0.csv
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      *
      * with the correct column number, you get a plot similar to
      *
      * @image html src/tarch/multicore/statistics-spawned-tasks.png
      *
      * This scatter plot shows us how many tasks have been spawned over a
      * certain time frame. The scatter plot is not that easy to ready, but
      * the system probes the system regularly for the accumulated value of
      * all spawned tasks and dumps it. After that, it resets the accumulated
      * value to zero.
      *
      * In the example above, it does not mean that we get only up to 1,000
      * tasks. To make statements on the total number of tasks at any point
      * in the system, we would have to take the density (x-spacing) of the
      * points into account. What it tells us however is that we have a
      * relatively high task spawn frequency. In only very rare cases, we
      * encounter only a few tasks.
      *
      * Many applications do not plot only this quantity, but plot
      * SpawnedFusableTasksStatisticsIdentifier at the same time, so
      * users can compare the number of spawned tasks against the number
      * of fusable tasks at any point.
      *
      * @ref tarch_logging
      */
     extern const std::string SpawnedTasksStatisticsIdentifier;

     /**
      * Number of fusable tasks over time
      * Please consult std::string SpawnedTasksStatisticsIdentifier
      * for a description. This identifier does not probe all tasks,
      * but only those for which canFuse() holds.
      */
     extern const std::string SpawnedFusableTasksStatisticsIdentifier;


     /**
      * Statistics identifier for active BSP sections
      *
      * If you translate in statistics mode, Peano's tarch will yield a
      * statistics file, where it tracks the number of fork-join (BSP)
      * partitions over time. If you invoke
      *
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      * python3 ~/git/Peano/python/peano4/postprocessing/inspect-statistics.py statistics-rank-0.csv
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      *
      * over the output file (with the .csv extension), the script tells
      * you in which column it dumps data of this flag.
      *
      * If you visualise this through
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      * python3 ~/git/Peano/python/peano4/postprocessing/plot-statistics.py -c 3 -pt scatter -o pdf statistics-rank-0.csv
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      *
      * with the correct column number, you get a plot similar to
      *
      * @image html src/tarch/multicore/statistics-bsp.png
      *
      * In this example, we have a warm-up (e.g. grid construction) phase up to
      * t=340. After that, the actual computation starts and the BSP level, i.e.
      * number of forked tasks in a fork-join sense, is either 0 or 8. For most
      * of the time, we seem to have 8 threads busy processing 8 rank-local
      * subpartitions.
      *
      * @ref tarch_logging
      */
     extern const std::string BSPConcurrencyLevelStatisticsIdentifier;

     /**
      * Construct string representation
      *
      * Returns a string representation of the task set which either is
      *
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      * {<empty>}
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      *
      * or it resembles
      *
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      * {1,2,3,4}
      * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      *
      * We use this routine in logging statements internally mainly. The
      * routine has to be within the namespace, as we overload a general
      * C++ std lib, and I want to keep this definition of toString()
      * specific to task numbers. It is also fine to have it in the
      * namespace, as it is only used internally anyway.
      */
     std::string toString( const std::set<TaskNumber>& taskNumbers );

    /**
     * Switch on SmartMPI
     *
     * If you use SmartMPI, then the bookkeeping registers the the local scheduling.
     * If you don't use SmartMPI, this operation becomes nop, i.e. you can always
     * call it and configure will decide whether it does something useful.
     */
    void initSmartMPI();
    void shutdownSmartMPI();

    void setOrchestration(tarch::multicore::orchestration::Strategy* realisation);

    /**
     * Swap the active orchestration
     *
     * Different to setOrchestration(), this operation does not delete
     * the current orchestration. It swaps them, so you can use setOrchestration()
     * with the result afterwards and re-obtain the original strategy.
     */
    tarch::multicore::orchestration::Strategy* swapOrchestration(tarch::multicore::orchestration::Strategy* realisation);

    tarch::multicore::orchestration::Strategy& getOrchestration();

    /**
     * Spawns a single task in a non-blocking fashion
     *
     * Ownership goes over to Peano's job namespace, i.e. you don't have
     * to delete the pointer.
     *
     * Spawn a task that depends on one other task. Alternatively, pass in
     * NoDependency. In this case, the task can kick off immediately. You have
     * to specify a task number. This number allows other, follow-up tasks to
     * become dependent on this very task. Please note that the tasks have to
     * be spawned in order, i.e. if B depends on A, then A has to be spawned
     * before B. Otherwise, you introduce a so-called anti-dependency. This is
     * OpenMP jargon which we adopted ruthlessly.
     *
     * You may pass NoDependency as taskNumber. In this case, you have a
     * fire-and-forget task which is just pushed out there without anybody
     * ever waiting for it later on (at least not via task dependencies).
     *
     * There is also the opportunity to pass in ExecuteImmediately as task
     * number. This option is there as place-holder: A lot of tasking
     * front-ends work with a final number of task numbers. If they run out
     * of numbers, they will get an "error code" telling the code "I don't have
     * any free numbers left". In this case, we can still pass on the task to
     * this routine. ExecuteImmediately is the same as NoNumber left, so we
     * just run this task immediately and crack on.
     *
     *
     * ## Data ownership
     *
     * The very moment you hand over a task to spawnTask(), Peano's tasking
     * backhand has ownership of the pointer. That is, you are not allowed to
     * use the task object anymore. It might have been destroyed right after
     * the spawnTask() call.
     *
     *
     * ## Usage with functors
     *
     * If you have a task object, you can directly call it. If you have only a
     * functor, use tarch::multicore::TaskWithCopyOfFunctor.
     *
     *
     * ## OpenMP variant
     *
     * The OpenMP variant has to jump through a number of hoops:
     *
     * 1. We check if we have enough addresses to administer the task
     *    dependencies. We basically ask: should we grow this big table
     *    that holds the dependencies (dependencyTrackingBaseAddress).
     * 2. If that analysis says "yes", we acquire a write lock on the
     *    whole tasking. Otherwise, we acquire a read lock over a
     *    MultiReadSingleWriteSemaphore instance.
     * 3. We grow the task dependency management memory region if we
     *    have got a write lock.
     * 4. We look for each involved task number if this can be modelled
     *    via an OpenMP dependency, i.e. if the task number is one that
     *    we can manage. Otherwise, we move this dependency (we know
     *    that the number of tasks managed is monotonously growing, i.e.
     *    we can assume that this dependency's task has never been used).
     * 5. If there are no out dependencies, we can just spawn this task.
     *    It goes straight into the background.
     * 6. If we have an out dependency (task number) and if we can serve
     *    this one, we spawn a proper task.
     * 7. Otherwise, we complete the task immediately.
     *
     * @see The page @ref page_tarch_multicore contains some high level
     *      overview of the multithreading.
     * @see The documentation of tarch::multicore::taskfusion for details
     *      how we handle task fusion.
     * @see tarch::multicore::taskfusion::handleFusableTask()
     *      for details how we react if task->canFuse() holds.
     * @see tarch::native::spawnTask() for implementation details if a task
     *      cannot be fused.
     *
     * @param task Pointer to a task. The responsibility for this task is
     *   handed over to the tasking system, i.e. you are not allowed to delete
     *   it.
     * @param inDependencies Set of incoming tasks that have to finish before
     *   the present task is allowed to run. You can pass the alias
     *   tarch::multicore::Tasks::NoInDependencies to make clear what's
     *   going on.
     * @param taskNumber Allow the runtime to track out dependencies. Only
     *   numbers handed in here may be in inDependencies in an upcoming call.
     *   If you do not expect to construct any follow-up in-dependencies, you
     *   can pass in the default, i.e. NoOutDependencies.
     */
    void spawnTask(
      Task*                        task,
      const std::set<TaskNumber>&  inDependencies = tarch::multicore::NoInDependencies,
      const TaskNumber&            taskNumber = tarch::multicore::NoOutDependencies
    );

    /**
     * This is a hint type which can be used to accelerate any taskwaits
     */
    enum class TaskWaitType {
      Mixed,
      ExclusivelyFusedTasks,
      OnlyNativeTasks
    };

    /**
     * Wait for set of tasks
     *
     * Entries in inDependencies can be NoDependency. This is a trivial
     * implementation, as we basically run through each task in
     * inDependencies and invoke waitForTask() for it. We don't have to rely on
     * some backend-specific implementation.
     *
     * ## Usage
     *
     * You can obviously construct a task set explicitly. If you know the
     * number of tasks, you can however directly use the bracket notation
     * to invoke this function:
     *
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     *    int taskNumber = ...;
     *    tarch::multicore::waitForTasks({taskNumber+0,taskNumber+1});
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     *
     * ## Serial code
     *
     * This routine degenerates to nop, as no task can be pending. spawnTask()
     * always executed the task straightaway.
     *
     *
     *
     * ## Optimisation
     *
     * If you know that you have exclusively native tasks, then we can skip
     * the check if there are any fused tasks still pending that we have to map
     * onto native tasks first.
     *
     * If you have exclusively fused tasks, then we can hand the baton on to
     * the fusion subpackage, but in return ask it if there's a chance that
     * some native tasks arise from the fusion. This is not the case for all
     * fusion strategies. If so, we can skip the subsequent test for native
     * tasks. See tarch::multicore::taskfusion::ensureHeldBackTasksAreMappedOntoNativeTasks().
     *
     *
     * @param taskWaitType This is an optimisation hint. See description above.
     */
    void waitForTasks(const std::set<TaskNumber>& inDependencies, TaskWaitType taskWaitType = TaskWaitType::Mixed);

    /**
     * Wrapper around waitForTasks() with a single-element set.
     */
    void waitForTask(const int taskNumber, TaskWaitType taskWaitType = TaskWaitType::Mixed);

    /**
     * Fork-join task submission pattern
     *
     * The realisation is relatively straightforward:
     *
     * - Maintain nestedSpawnAndWaits which is incremented for every fork-join
     *   section that we enter.
     * - Tell the orchestration that a BSP section starts.
     * - Ask the orchestration which realisation to pick.
     * - Either run through the task set sequentially or invoke the native
     *   parallel implementation.
     * - If there are task pending and the orchestration instructs us to do so,
     *   map them onto native tasks.
     * - Tell the orchestration that the BSP section has terminated
     * - Tell the orchestration that a BSP section ends.
     * - Maintain nestedSpawnAndWaits which is decremented whenever we leave a
     *   fork-join section.
     *
     *
     * ## Scheduling variants
     *
     * The precise behaviour of the implementation is controlled through the
     * orchestration. At the moment, we support three different variants:
     *
     * 1. The serial variant tarch::multicore::orchestration::Strategy::ExecutionPolicy::RunSerially runs
     *    through all the tasks one by one. Our rationale is that a good
     *    orchestration picks this variant for very small task sets where the
     *    overhead of the join-fork makes a parallelisation counterproductive.
     *
     * 2. The parallel variant tarch::multicore::orchestration::Strategy::ExecutionPolicy::RunParallel runs
     *    through all the tasks in parallel. Once all tasks are completed, the
     *    code commits all the further tasks that have been spawned into a
     *    global queue and then studies if to fuse them further or if to map
     *    them onto native tasks. This behaviour has to be studied in the
     *    context of tarch::multicore::spawnTask() which might already have
     *    mapped tasks onto native tasks or GPU tasks, i.e. at this point no
     *    free subtasks might be left over in the local queues even though
     *    there had been some. It is important to be careful with this "commit
     *    all tasks after the traversal" approach: In OpenMP, it can lead to
     *    deadlocks if the taskwait is realised via busy polling. See the bug
     *    description below.
     *
     * 3. The parallel variant
     * tarch::multicore::orchestration::Strategy::ExecutionPolicy::RunParallelAndIgnoreWithholdSubtasks runs through all
     * the tasks in parallel. Different to tarch::multicore::orchestration::Strategy::ExecutionPolicy::RunParallel, it
     * does not try to commit any further subtasks or to fuse them. This variant allows the scheduler to run task sets
     * in parallel but to avoid the overhead introduced by the postprocessing.
     *
     * I would appreciate if we could distinguish busy polling from task
     * scheduling in the taskwait, but such a feature is not available within
     * OpenMP, and we haven't studied TBB in this context yet.
     *
     *
     * ## Implementation flaws in OpenMP and bugs burried within the sketch
     *
     * In OpenMP, the taskwait pragma allows the scheduler to process other
     * tasks as it is a scheduling point. This way, it should keep cores busy
     * all the time as long as there are enough tasks in the system. If a
     * fork-join task spawns a lot of additional subtasks, and if the
     * orchestration does not tell Peano to hold them back, the OpenMP runtime
     * might switch to the free tasks rather than continue with the actual
     * fork-join tasks. Which is not what we want and introduces runtime flaws
     * later down the line. This phenomenon is described in our
     * 2021 IWOMP paper by H. Schulz et al.
     *
     * A more severe problem arises the other way round: Several groups have
     * reported that the taskwait does not continue with other tasks. See in
     * particular
     *
     * Jones, Christopher Duncan (Fermilab): Using OpenMP for HEP Framework Algorithm Scheduling.
     * http://cds.cern.ch/record/2712271
     *
     * Their presentation slides can be found at https://zenodo.org/record/3598796#.X6eVv8fgqV4.
     *
     * This paper clarifies that some OpenMP runtimes do (busy) waits within
     * the taskwait construct to be able to continue immediately. They do not
     * process other tasks meanwhile. Our own ExaHyPE 2 POP review came to the
     * same conclusion.
     *
     * This can lead to a deadlock in applications such as ExaHyPE which spawn
     * bursts of enclave tasks and then later on wait for their results to drop
     * in. The consuming tasks will issue a taskyield() but this will not help,
     * if the taskyield() now permutes through all the other traversal tasks.
     *
     * If you suffer from that, you have to ensure that all enclave tasks
     * have finished prior to the next traversal.
     *
     *
     * ## Statistics
     *
     * It is important to know how many BSP sections are active at a point. I
     * therefore use the stats interface to maintain the BSP counters. However,
     * I disable any statistics sampling, so I get a spot-on overview of the
     * number of forked subtasks at any point.
     */
    void spawnAndWait(const std::vector<Task*>& tasks);


    namespace native {
      /**
       * Map onto native tasking
       *
       * Run over the tasks and issue native tasks. If the tasks spawn, in
       * return, further subtasks, these will either end up in a thread-local
       * queue or will be mapped onto native tasks. The behaviour here depends
       * on decisions within tarch::multicore::spawnTask() guided by the
       * orchestration. If the subtasks enqueue tasks into the thread-local
       * queue, they will remain there. This routine does not touch the
       * thread-local queue.
       *
       * The responsibility for the pointers in tasks is handed over to the
       * runtime of choice, i.e. you don't have to delete them.
       *
       *
       * ## OpenMP mapping of a task loop onto a task loop
       *
       * In OpenMP 6 and newer, we should use transparent tasks, as we want to
       * expose task dependencies which do not stem from siblings. If OpenMP 6
       * is not supported, then we should disable the corresponding syntax.
       *
       *
       * ### NVIDIA
       *
       * NVIDIA's nvc++ does not support taskloops. We have contacted their R&D and they
       * confirmed that they only implement a subset of OpenMP, and the taskloop is not
       * on the list of features that they want to support.
       *
       *
       * ## Implementation and bugs
       *
       * See discussion in the context of the other taskloop: It is important to
       * label tasks as shared even though I think copying should be fine as a copy
       * degenerates to a shallow copy.
       *
       * You are not allowed to call this routine for an empty task
       * set. The empty task set means that the taskloop construct becomes nop. The
       * subsequent taskwait hence synchronises on one logical level above within the
       * task hierarchy, which is not what we want.
       *
       *
       * @param tasks Set of tasks. Is guaranteed to be non-empty.
       */
      void spawnAndWaitAsTaskLoop(const std::vector<tarch::multicore::Task*>& tasks);

      /**
       * Spawn a new task into the tasking backend
       *
       * This routine is invoked by tarch::multicore::spawnTask(). The invoking
       * method is not a sole forwarding. It takes care of the categorisation
       * of ready tasks into pending tasks, and it also handles all the fusion
       * for GPUs. Therefore, if it delegates calls to this class, we really
       * can directly map the task 1:1 onto the native task system used.
       *
       *
       * ## Serial code
       *
       * This is a trivial case in a serial run: As we have already completed
       * all incoming tasks (we work serially and all incoming tasks have to
       * be done by this time), we can simply process the task and return. As a
       * direct consequence, all dependencies of future tasks are always
       * fulfilled.
       *
       *
       * ## C++ variant
       *
       * 1. We first check if a task with this number is among the submitted
       *    tasks. This is possible if a previous algorithm phase has submitted
       *    a task with this number which was a sink, but noone ever waited for
       *    it. This should usually not happen and is a sign of bad code
       *    design. We issue a warning, free all locks and wait for the
       *    previous task with the same number to terminate.
       * 2. Next, we spawn the actual task via a std::async call. The core task
       *    code consists of three parts:
       *    - We first run through all the dependencies and ensure they all are
       *      fulfilled. waitForTasks() realises check.
       *    - Execute the task object.
       *    - Free its memory.
       * 3. We store the shared_future returned by the task. If the task number
       *    equals NoOutDependencies, we know that noone will ever be able to wait
       *    for this task. In this case, we skip the storing.
       *
       *
       * ## OpenMP
       *
       * OpenMP models dependencies through addresses. While it does not use
       * the data to which these addresses point to, it is important that the
       * addresses are valid, i.e. the system owns these memory locations.
       * I use an internal array to administer the byte array which I
       * use for dependencies. So the first thing is that we try to expand the
       * memory region so we can accommodate the dependencies. Therefore, we
       * call
       * tarch::multicore::omp::internal::reserveDependencyTrackingMemory()
       * first.
       *
       * Next we increment numberOfPendingTasks, so noone else assumes that no
       * tasks are pending. While we continue, we further block the memory
       * region with a read semaphore, so noone reallocates it.
       *
       * OpenMP now has iterators in taskwait and dependency statements. Yet,
       * the set of in-dependencies is not iterable. So we create a vector into
       * which we paste the task numbers. This also contains the outgoing task
       * number, i.e. if there had been a task with that number before, we
       * basically impose an inout dependency and ensure this one terminates
       * first.
       *
       * We also use iterators to administer the out dependencies. This "set"
       * is either empty or has exactly one entry.
       *
       * Finally, we distinguish OpenMP 5 and OpenMP 6 realisations. They
       * sit in different source files, TasksWithOpenMP6.cpp (see also
       * tarch::multicore::omp::internal::spawnTaskWithOpenMP6()) and
       * TasksWithOpenMP5.cpp. The latter does not use the dependency memory
       * at all. See the documentation of tarch::multicore::omp::internal.
       *
       *
       * ### Realisation pitfalls
       *
       * I originally had
       *
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       * #pragma omp atomic
       * numberOfSpawnedTasks++;
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       *
       * outside of the if/else cascade. However, that means that the
       * subsequent canServeDependencyEntry() calls will not increase the
       * memory used. They assume that the task were already out. Therefore,
       * it is important to increment this counter as late as possible.
       *
       *
       * ### Task dependencies
       *
       * OpenMP now supports iterators and accessors over arrays. I tried
       *
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       * DependencyTrackingDataType* outDependencyPointer = dependencyTrackingBaseAddress + taskNumber;
       * [...]
       * #pragma omp task [...] depend(out: *outDependencyPointer) untied
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       *
       * but that didn't work. So it seems that OpenMP struggles with the
       * dereferencing. However, it seems that
       *
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       * #pragma omp task [...] depend(out: dependencyTrackingBaseAddress[taskNumber]) untied
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       *
       * does work.
       *
       *
       * ## TBB
       *
       * With TBB, we maintain two versions: one for a new (preview) TBB
       * version where dynamic task graph construction is supported, and one
       * for an older version, where such a feature is not yet available, and
       * we need our own TBB extension.
       *
       * In the one version, we create a task handle via a defer() call. In
       * the other version, we create an instance of dynamic_task_graph_node.
       * If the task has no outgoing dependencies, we don't have to register
       * it in some way, i.e. to memorise, as noone will be able to refer to
       * it later anyway. However, we have to add an additional incoming
       * dependency in case that such a task has existed before.
       *
       * Next, we add the user-defined incoming dependencies to this task.
       *
       * Finally, we either move the handle into the global task group, making
       * it a proper task (new TBB version), or we put it into our own task
       * graph object which then translates it into a proper task.
       *
       *
       * ## No multithreading
       *
       * We just execute this task. We know that tasks may not have anti-dependencies.
       * That is, we know that any incoming task has been spawned before. By
       * induction, it therefore has already completed. We also don't have any
       * competition, as there are no competing tasks. So we actually can ignore
       * all the parameters and just execute the task.
       */
      void spawnTask(
        Task*                        task,
        const std::set<TaskNumber>&  inDependencies,
        const TaskNumber&            taskNumber
      );

      /**
       * Wait for other task
       *
       *
       * ## Serial code
       *
       * Degenerates to nop here, as all incoming tasks are completed. We just
       * have to ensure that the task we are waiting for is not(!) registered,
       * i.e. has actually been submitted.
       *
       * ## C++ code
       *
       * waitForTasks() is quite tricky, as we have to avoid any deadlocks
       * here. We therefore use a quite defensive programming.
       *
       * 1. We lock the set of spawned tasks. If a dependency task is in there
       *    and is valid, we memorise it in a pointer checkFuture and release
       *    the lock.
       * 2. Should we have found a dependency, we call the C++ wait. From
       *    hereon, the core task is complete, but there's still an entry in
       *    the global lookup table of spawned tasks. At this point, we
       *    might actually have finished a lot of tasks recursively.
       * 3. We briefly lock the data structure again and check if the task is
       *    bookmarked in there and is already complete. In this case, we
       *    remove the entry. We kind of run an on-the-fly immediate garbage
       *    collection to slim down our map of spawned tasks.
       *
       *
       * ## OpenMP
       *
       * As we model task dependencies over pointers on a huge memory region,
       * we first of all issue a read lock over this memory region. After all,
       * we don't want to code to free this memory region while we work on it.
       *
       * After that, we move those task dependencies that we can map onto
       * OpenMP dependencies (there might be some numbers which are so high
       * that they have never been used so far and hence cannot be served by
       * our memory region) into an std::vector. It has to be a vector, as
       * OpenMP's iterators want to access the iterator with a [...] accessor.
       *
       * If this set (list) of task dependencies is not empty, we issue an
       * OpenMP taskwait. See remarks below on flaws with some OpenMP versions
       * (which we currently do not accommodate). I think this part would not
       * require an if, as it degenerates if the set of dependencies is empty,
       * but then I am afraid that it might degenerate into a taskwait, which
       * is not what we want.
       *
       * Finally, we check which OpenMP version we support. If the version is
       * OpenMP 5 or lower, then task dependencies cannot be transparent.
       * That is, task dependencies in OpenMP work only between siblings. We
       * however have task dependencies between tasks that are not spawned by
       * the same parent task. Therefore, we have to manually ensure that all
       * tasks have terminated. We do so by polling the status of the tasks
       * (TasksWithOpenMP5.cpp). The documentation of
       * tarch::multicore::omp::internal provides further information on the
       * underlying mechanisms.
       *
       *
       * ## OpenMP flaws
       *
       * Joachim Jenke has flagged that there are some OpenMP versions that
       *
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       * #pragma omp taskwait depend(iterator(j=0:iterableExistingInDepencencies.size()), in: dependencyTrackingBaseAddress[iterableExistingInDepencencies[j]])
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       *
       * might fail with some (older) OpenMP versions. A workaround is the
       * construct through an empty task:
       *
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       * #pragma omp task depend(iterator(j=0:iterableExistingInDepencencies.size()), in: dependencyTrackingBaseAddress[iterableExistingInDepencencies[j]]) if (0)
       * {}
       * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       *
       * Here, we model a task which is actually never spawned as task due to
       * if statement that evaluates to false. However, the in-dependencies
       * will be resolved nevertheless.
       * 
       * In OpenMP 5, a thread that waits yields while it polls. This is safe, as
       * no OpenMP task waits for another one (see tarch::multicore::omp::internal).
       */
      void waitForTasks(const std::set<TaskNumber>& inDependencies);

      /**
       * Slightly different than the umbrella version in the general namespace.
       * See comments there. This version only waits for tasks which have been
       * submitted with a valid task number.
       *
       * ## Serial code
       *
       * This routine degenerates to nop, as no task can be pending. spawnTask()
       * always executed the task straightaway.
       *
       * ## C++
       *
       * There are different ways how to implement this wait: The most
       * straight-forward implementation takes all the keys and the invokes
       * waitForTasks() for all keys. We do not implement this pattern, as it
       * would mean that we potentially copy quite a lot of keys. Instead, we
       * iteratively wait for one task after another to finish. This implies
       * that we give the runtime system a lot of freedom in which order and
       * how parallel to tidy up all tasks.
       *
       * ## OpenMP
       *
       * In OpenMP, we do a generic taskwait, but then we use the opportunity
       * as well to grow the memory used for dependency tracking by calling
       * reserveDependencyTrackingMemory(). We still need the generic taskwait
       * first, as there's no guarantee that reserveDependencyTrackingMemory()
       * will actually expand the memory and wait for all other tasks.
       */
      void waitForAllTasks();
    } // namespace native


    /**
     * Wait for all tasks notably has to take fused tasks into account.
     */
    void waitForAllTasks();
  } // namespace multicore
} // namespace tarch
