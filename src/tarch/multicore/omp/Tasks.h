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
       * ## Tasks in OpenMP before version 6: release on completion
       *
       * If we have OpenMP 5 or earlier, we don't have transparent tasks, so
       * we have to provide a workaround. The two implementations are so
       * different that they live in different files: TasksWithOpenMP6.cpp is
       * only translated if the compiler claims OpenMP 6, TasksWithOpenMP5.cpp
       * otherwise. Tasks.cpp holds what both have in common.
       *
       * The workaround does not use OpenMP dependencies at all. Every task
       * that is spawned with a number is entered into a map from task
       * number to a task node. A node holds the job, the number of
       * predecessors that have not terminated yet, and the list of
       * successors that wait for this task. Only a task whose predecessors
       * have all terminated is turned into an OpenMP task. This is done by
       * the task that terminates last among the predecessors (or by the
       * spawn itself if all predecessors have terminated already). A
       * terminating task removes itself from the map. A number that is not in
       * the map is either not used or has terminated, i.e. nobody has to
       * wait for it.
       *
       * @dot
       * digraph ReleaseOnCompletion {
       *   rankdir=LR; node [shape=box, fontsize=10];
       *   spawn    [label="spawnTask()"];
       *   map      [label="map: number -> node\n(unfinished predecessors,\nsuccessors)"];
       *   omp      [label="OpenMP task\n(runs job)"];
       *   terminate[label="terminate:\nremove from map,\ncount down successors"];
       *   spawn -> map [label="register at\nunfinished\npredecessors"];
       *   spawn -> omp [label="no unfinished\npredecessor"];
       *   map -> omp [label="last predecessor\nterminated"];
       *   omp -> terminate;
       *   terminate -> omp [label="releases ready\nsuccessors", style=dashed];
       * }
       * @enddot
       *
       * If a task is spawned with a number that is still alive, it replaces
       * the old task in the map. If the number is also one of its
       * in-dependencies, it has registered at the old task, i.e. it is an
       * inout dependency and runs after it. Tasks spawned later find the new
       * task. A terminating task only removes itself from the map if it has
       * not been replaced.
       *
       * ### Waits do the work themselves
       *
       * A thread that waits for a task (tarch::multicore::native::waitForTasks())
       * does not wait for the OpenMP runtime to run it. Each node knows its
       * unfinished predecessors. Starting at the task that we wait for, the
       * waiting thread follows these predecessors backwards until it finds a
       * task without unfinished predecessors, runs it itself, and then works
       * its way forward again. Whoever sets the claim flag of a task first runs
       * it: the OpenMP task that has been created for it, or a waiting thread.
       * If the OpenMP task comes late, it finds the claim taken and
       * returns. The wait only polls and yields if the task (or a predecessor)
       * is already running on another thread. In this case, it ends once
       * that thread has finished the task.
       *
       * If you compile with -DTrackStatistics, the statistics count how many
       * tasks waiting threads ran
       * ("tarch::multicore::omp::tasks-run-by-waiting-thread") and how often
       * they yielded ("tarch::multicore::omp::yields-while-waiting").
       *
       * ### Lesson learned: Do not poll in tasks
       *
       * We used to spawn all tasks immediately as OpenMP tasks, and each of
       * them polled the status of its predecessors in a loop with a
       * taskyield. A yielding thread runs further tasks on top of its stack.
       * If it picks up a successor of a task that it has suspended earlier,
       * this successor waits for a predecessor which cannot continue before
       * the successor returns, i.e. we deadlock. We observed this with
       * chains of dependent tasks on more than two threads: threads idle,
       * tasks whose predecessors had all terminated never started, and each
       * stuck thread held two nested polling tasks.
       * As nothing but runnable tasks becomes an OpenMP task now, no OpenMP
       * task ever waits for another task.
       *
       */
      namespace internal {
        using DependencyTrackingDataType = unsigned short int;

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
