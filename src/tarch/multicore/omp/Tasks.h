// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once


namespace tarch {
  namespace multicore {
    namespace omp {
      /**
       * OpenMP-internal namespace
       *
       * Peano's task fusion pattern requires transparent tasks. An explanation
       * of this can be found in @ref tarch::multicore. Just scroll down to the
       * caption "Task execution patterns and orchestration" in the detailed
       * description.
       *
       * OpenMP supports transparent tasks with version 6. Peano therefore
       * has an internal helper routine tarch::multicore::omp::majorOpenMPVersion(),
       * through which we can check if this major version is already supported.
       * Depending on the OpenMP version, we can either directly map our task
       * graphs onto tasks, or we have to work around the whole transparency
       * thing. No matter if we have OpenMP 6 or not, the fundamental idea of
       * how we map Peano's task numbers, which are all integers, onto OpenMP
       * task dependencies.
       *
       *
       * # Task dependencies: mapping integer task numbers onto addresses
       *
       * Dependency tracking in OpenMP is organised via addresses. These
       * addresses are sole helpers, i.e. the underlying memory region doesn't
       * hold any data with semantics. However, it has to be owned by the
       * application code. To ensure this is the case, we allocate some chunk
       * only for the dependency tracking and hold its baseline address in a
       * variable dependencyTrackingBaseAddress.
       *
       * The initial default size of this memory tracking is specified by
       * DefaultSizeOfMemoryRegionUsedForDependencyTracking. This is implicitly
       * the highest task number that we can handle. However, if the code
       * requests a higher number, then we pause and grow the memory region
       * before we continue.
       *
       * We maintain two integers, sizeOfMemoryRegionUsedForDependencyTracking
       * and maxSizeOfMemoryRegionUsedForDependencyTracking, to always keep
       * track how big the memory is.
       *
       *
       * ## Lesson learned: Not handling tasks
       *
       * I originally thought that we could refrain from always immediately
       * growing this memory dependency region. We could, in theory, look up
       * if we can handle a dependency (we do that anyway) and then execute
       * tasks immediately if we cannot map them onto an OpenMP dependency as
       * we don't have the memory reserved yet.
       *
       * Is a bad idea. The task fusion relies on working taskwaits to yield
       * correct results. That is, if a thread issues a taskwait on a specific
       * task, it has to trust the OpenMP layer that this task actually has
       * terminated. If the task is currently executed immediately by another
       * thread, as we don't have dependency memory, there's no way for the
       * taskwait to actually wait for the task to terminate. All tasks are
       * therefore mapped onto actual tasks as a matter of principle.
       *
       * If we don't have the memory to administer a task, we have to grow
       * it before we continue handling it. The process how to handle a new
       * task thus is always the same:
       *
       * - Invoke @ref tarch::multicore::omp::internal::reserveDependencyTrackingMemory()
       *   to ensure that memory is available to handle a task number.
       * - For all dependencies that we handle, check with @ref tarch::multicore::omp::internal::canServeDependencyEntry()
       *   if these had been mapped onto proper tasks before. If not, we can
       *   just ignore these dependencies (this also holds for task waits).
       * - Check via tarch::multicore::omp::majorOpenMPVersion() if we can
       *   map a task onto a transparent task or need to do all depencency
       *   tracking manually.
       *
       * You find this pattern in @ref tarch::multicore::native::spawnTask()
       * for example.
       *
       *
       * ## Tasks in OpenMP before version 5
       *
       * If we have OpenMP 5 or earlier, we don't have transparent tasks, so
       * have to provide a workaround. What we do it that we do not only
       * allocate some memory for dependency tracking, we also set each entry
       * to one of the following values:
       *
       * - TaskNumberNotUsedYet
       * - TaskNumberSpawned
       * - TaskNumberRunning
       * - TaskNumberTerminated
       *
       * Task waits are then realised through atomics and polling over these
       * variables.
       *
       */
      namespace internal {
        using DependencyTrackingDataType = unsigned short int;

        /**
         * Translate task marker into string
         *
         * Used only for debugging reasons.
         */
        std::string OpenMP5TaskStatus(DependencyTrackingDataType);

        /**
         * Grow the memory region used for the dependency tracking
         *
         * Addresses that represent dependencies in OpenMP have to correspond to
         * allocated, user-owned memory. To ensure this is the case, I hold one
         * big memory block for all the dependencies.
         *
         * Unfortunately, I do not know how big this memory block has to be. So I
         * use any check if an out condition can be served as an
         * opportunity to grow the memory region which arises always if
         * there are no more tasks pending that might use the dependency tracking.
         * This happens in shouldGrowDependencyMemory().
         *
         * ## Thread safety
         *
         * No new tasks with dependencies can not be spawned while I grow the memory
         * region. This means we need a critical section in the routine around this
         * one: multiple threads can read, but if one decides to write, no other
         * thread should do anything related.
         *
         * ## Dependencies
         *
         * I originally had this idea that each task could increment/decrement a
         * counter, so I can check if tasks are still pending. If so, I'd increment.
         * But this does not work: If a task decrements the counter and signals
         * "I'm done", OpenMP might not have tidied up the underlying memory region.
         * So if I now free the memory (as I think no tasks are there anymore), this
         * might happen at exactly the point in time where OpenMP still updates its
         * out-dependencies. Therefore I loop over all potential in-dependencies
         * (many of them will be antidependencies) to ensure that OpenMP really has
         * done all of its work. A simple wait will btw not work: It only works on
         * directly children.
         *
         * ## Remark
         *
         * At this point, I do not use the routine. For some applications, it might
         * be reasonable to invoke it, but it has to happen outside.
         */
        void reserveDependencyTrackingMemory();

        /**
         * Spawn task using OpenMP 6 syntax
         *
         * With OpenMP 6, we get transparent tasking. That is, we can have tasks with
         * dependencies which are not sibling tasks. Therefore, we can directly map
         * all dependencies on memory references in dependencyTrackingBaseAddress.
         *
         *
         * ## Assumptions
         *
         * This routine makes a couple of assumptions:
         *
         * - The caller has already incremented numberOfPendingTasks by one;
         * - The dependencyTrackingBaseAddress array hosts all indices specified
         *   within inDependencies and outDependency;
         * - The semaphore dependencyTrackingSemaphore is locked with a read lock.
         *
         *
         * ## OpenMP < 6 and the GNU compiler
         *
         * If we use an "older" version of OpenMP, then the transparent keyword is
         * not supported. That means, the pragma with a transparent will create a
         * translation error or at least a warning. I therefore tried to embed the
         * function into a
         *
         * ~~~~~~~~~~~~~~~~~~~~~~~~~
         * if constexpr( tarch::multicore::omp::majorOpenMPVersion()>=6 ) {
         *    ...
         * }
         * ~~~~~~~~~~~~~~~~~~~~~~~~~
         *
         * section. Unfortunately, this does not help. C++ might ignore the if
         * branch, but the standard also says that the loop body has to be
         * syntactically correct, which is kind of violated. So you'll still
         * get a translation error/warning for this code if you use older
         * OpenMP versions.
         *
         * I therefore decided to remove the if and work with the direct
         * OpenMP version pragma. Please note that this is only required for
         * GCC. With LLVM, it would just ignore the unknown transparent.
         *
         *
         * @param inDependencies Sequence of task numbers which can be empty. We need
         *  a vector here (even if the udnerlying type is a set), as we use OpenMP's
         *  iterator.
         *
         * @param outDependency Is either empty or contains exactly one element. As
         *  we use a vector, we can work with OpenMP's iterator supporting both an
         *  outgoing dependency and no dependencies at all.
         */
        void spawnTaskWithOpenMP6(
          tarch::multicore::Task*                           job,
          const std::vector<tarch::multicore::TaskNumber>&  inDependencies,
          const std::vector<tarch::multicore::TaskNumber>&  outDependency
        );

        /**
         * Variant of spawn for all OpenMP versions under 6
         *
         * The name is slightly misleading, as we support any OpenMP version
         * between 3.x and 5 here. The routine is very similar to the version
         * 6, but this time we keep track of the tasks manually.
         *
         * We manually keep track in a map what the status of a task is. This
         * has a couple of serious implications:
         *
         * 1. A task might start even though it is not ready yet. That means,
         *    we have to manually check once we are fired up if we are really
         *    ready. The reason why a task is starting "prematurely" is that
         *    we cannot use transparent tasks, i.e. a neighbour non-sibling
         *    task might have spawned the task and we might fire up even though
         *    we are not ready yet.
         * 2. We cannot handle self-dependencies, i.e. in-out ones. If we spot
         *    an in-out dependency, we remove it from inDependencies (therefore
         *    it is not const), wait manually for our predecessor task to
         *    terminate by polling the status, and then we carry on. The manual
         *    wait is important, as we will overwrite the status with pending
         *    immediately after.
         *
         * As we do all of our dependency management manually, we don't have to
         * rely on any OpenMP dependencies at all. But it might actually make
         * sense to employ them too (see remarks below).
         *
         *
         * ## Assumptions
         *
         * This routine makes a couple of assumptions:
         *
         * - The caller has already incremented numberOfPendingTasks by one;
         * - The dependencyTrackingBaseAddress array hosts all indices specified
         *   within inDependencies and outDependency;
         * - The semaphore dependencyTrackingSemaphore is locked with a read lock.
         * 
         *
         * ## Issues we found
         *
         * It is possible that many threads spawn tasks at the same time that have
         * to wait for other tasks that are currently stored in the ready queue.
         * As we poll the ready queue state, this can lead to starvation.
         *
         * To avoid this we add a taskyield if an
         * in-dependency isn't ready. This moves the thread onto a different task,
         * so it will eventually end up with one that is ready to process. This
         * can also occur in waitForTasks(...), and the problem is dealt with in the
         * same way. 
         *
         *
         * ## Implementation decision
         *
         * Modelling the out dependencies as vector is not particularly elegant.
         * As each task as at most one out dependency (its "number"), we could
         * have worked with an integer making one particular value, e.g. -1,
         * mean "no dependency". As I work with a vector however, I can use
         * OpenMP's iterators.
         *
         *
         * ## Flaws and known issues
         *
         * Logically, there is no need for the spawned task to have any
         * dependencies, as all dependency management is done manually through
         * polling. This is not very elegant. Indeed, I'm afraid that it can
         * lead to starvation. Therefore, I would like to use dependencies.
         * If the lack of transparent dependencies leads to antidependencies,
         * i.e. tasks start prematurely, I don't care, as we still have the
         * manual handling.
         *
         * If the dependency management works however, the task runtime ensures
         * that we do not stress the system through polling too much.
         * Therefore, I originally had this splendid idea to use
         *
         * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         * #pragma omp task priority(job->getPriority()) depend(iterator(j=0:inDependencies.size()), in: dependencyTrackingMemory[inDependencies[j]]) depend(iterator(j=0:outDependency.size()), out: dependencyTrackingMemory[outDependency[j]]) shared(numberOfPendingTasks) untied
         * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         *
         * for the tasks. However, this made the code crash in the OpenMP
         * runtime non-deterministically.
         *
         * @see tarch::multicore::omp::internal::hasTaskFinishedInOpenMP5() as counterpart how to
         *   manually wait for tasks.
         *
         * @param job Pointer to the actual task. Is never a nullptr.
         * @param inDependencies Number of in-dependencies. All of these
         *   indices can be served by the underlying memory region. Set might
         *   be empty.
         * @param outDependency Out-dependency which is either an empty set
         *   meaning that there is no out-dependency and noone will ever wait
         *   for this specific task, or it holds exactly one number.
         */
        void spawnTaskWithOpenMP5(
          tarch::multicore::Task*                           job,
          std::vector<tarch::multicore::TaskNumber>         inDependencies,
          const std::vector<tarch::multicore::TaskNumber>&  outDependency
        );

        /**
         * Reserve memory for dependency tracking
         *
         * OpenMP manages dependencies through memory addresses. However, to be able
         * to do so, the code has to own these addresses. This routine checks if the
         * space for the dependency tracking is big enough. If not, it grows it.
         *
         * We cannot grow the memory region while anybody still refers to them.
         * Therefore, the growing process first polls numberOfPendingTasks to be sure
         * that everybody has finished before we continue. Due to this polling and
         * the semaphores employed, this routine is thread-safe.
         */
        void reserveDependencyTrackingMemory( int taskNumber );

        /**
         * Can I serve a certain dependency
         *
         * This routine exclusively checks if an in-dependency could exist.
         * This does not mean anything re if depending upon taskNumber might be
         * an anti-dependency. It just means that the underlying memory region
         * that we use to administer task dependencies could host the task
         * number.
         *
         * As we know that sizeOfMemoryRegionUsedForDependencyTracking is
         * monotonously growing, we can work with a simple read atomic.
         */
        bool canServeDependencyEntry(int taskNumber);

        /**
         * This is the counterpart to hasTaskFinishedInOpenMP6
         *
         * The same assumption as for hasTaskFinishedInOpenMP6() holds, but
         * this one takes into account that we manually keep track of which
         * tasks are still running and which not.
         *
         * Therefore, we loop over all tasks identified by inDependencies and
         * check if all entries are either TaskNumberNotUsedYet or
         * TaskNumberTerminated.
         */
        bool hasTaskFinishedInOpenMP5( const std::vector<tarch::multicore::TaskNumber>&  inDependencies );

        /**
         * Poll until all in-dependencies have terminated
         *
         * All OpenMP 5 waits (a task waiting for its predecessors, a spawn
         * waiting for the previous use of its own number, a consumer in
         * waitForTasks()) go through this routine, i.e. we have one place
         * which yields while we wait and which detects tasks that wait
         * suspiciously long. See reportStuckTask().
         *
         * @param inDependencies Numbers to wait for.
         * @param waiter Number of the task which waits, or
         *   tarch::multicore::NoOutDependencies if a caller waits which is not
         *   a task with a number.
         * @param location Short description who waits. Only used for the
         *   report.
         */
        void pollUntilTasksHaveFinishedInOpenMP5(
          const std::vector<tarch::multicore::TaskNumber>&  inDependencies,
          tarch::multicore::TaskNumber                      waiter,
          const char*                                       location
        );

        /**
         * Report a task that waits suspiciously long
         *
         * Only has an effect if you compile with -DTrackStatistics. Called
         * by pollUntilTasksHaveFinishedInOpenMP5() once per wait after
         * NumberOfPollsBeforeTaskIsReportedAsStuck unsuccessful polls. It
         * does not abort and does not change the wait. The report holds
         *
         * - which wait is stuck (location) and the OpenMP thread;
         * - the number of the waiting task, if it has one;
         * - for each in-dependency its number and status, e.g.
         *   "{36:task-spawned, 37:task-running}"; and
         * - the number of pending tasks.
         *
         * If all in-dependencies are in state task-spawned and the number of
         * pending tasks is small, no task is left that could ever terminate
         * them, i.e. we have a cyclic or lost dependency rather than a
         * scheduling problem.
         *
         * The report is written as warning and logged through
         * tarch::logging::Statistics under the identifier
         * "tarch::multicore::omp::stuck-task". As the code might never
         * terminate, the first report in a run also dumps all statistics,
         * including the task events of trackTaskEvent(), into the file
         * statistics-stuck-task(-rank-<n>).csv. This dump empties the
         * statistics, i.e. later statistics only hold what happens after the
         * report.
         */
        void reportStuckTask(
          const std::vector<tarch::multicore::TaskNumber>&  inDependencies,
          tarch::multicore::TaskNumber                      waiter,
          const char*                                       location
        );

        /**
         * Track the life cycle of one task
         *
         * Only has an effect if you compile with both -DTrackStatistics and
         * -DTrackTaskDependencies. The flag TrackTaskDependencies is separate
         * from TrackStatistics, as we create several entries per task, all
         * written under one lock. It slows down the code and eats memory.
         * Use it to find out what a hanging run did before it hung.
         *
         * The events are written through
         * tarch::logging::Statistics under the identifier
         * "tarch::multicore::omp::task-event". Each entry is a string of the
         * form "<event> <task number> in={<numbers>}":
         *
         * - spawn: spawnTask() has been called (the in-set holds the
         *   dependencies that we can serve);
         * - start: the task has passed its in-dependencies and starts to run;
         * - end: the task has terminated and the status is set to terminated;
         * - wait: waitForTasks() starts to wait for the numbers in the set.
         *   The task number is the one of no task (NoOutDependencies).
         */
        void trackTaskEvent(
          const char*                                       event,
          tarch::multicore::TaskNumber                      taskNumber,
          const std::vector<tarch::multicore::TaskNumber>&  inDependencies
        );
      }
    }
  }
}
