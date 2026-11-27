// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "tarch/multicore/Task.h"

#include "multicore.h"
#include "Tasks.h"
#include "config.h"
#include "tarch/Assertions.h"
#include "tarch/logging/Statistics.h"
#include "tarch/mpi/Rank.h"
#include "tarch/multicore/Core.h"
#include "tarch/multicore/multicore.h"
#include "tarch/multicore/otter.h"
#include "tarch/multicore/MultiReadSingleWriteSemaphore.h"
#include "tarch/multicore/MultiReadSingleWriteLock.h"
#include "tarch/services/ServiceRepository.h"
#include "tarch/compiler/CompilerSpecificSettings.h"


#if defined(SharedOMP) and _OPENMP>=202411

#include <omp.h>


namespace {
  static tarch::logging::Log _log( "tarch::multicore::native(omp6)" );

  #ifndef DefaultSizeOfMemoryRegionUsedForDependencyTracking
  constexpr int DefaultSizeOfMemoryRegionUsedForDependencyTracking = 16000;
  #endif

  /**
   * Initialisation value for entries in dependencyTrackingMemory
   *
   * Consult docu of tarch::multicore::omp::internal for details.
   */
  constexpr tarch::multicore::omp::internal::DependencyTrackingDataType TaskNumberNotUsedYet = 0;

  /**
   * Memory region to administer dependencies
   *
   * Consult docu of tarch::multicore::omp::internal for details.
   */
  tarch::multicore::omp::internal::DependencyTrackingDataType*  dependencyTrackingMemory = new tarch::multicore::omp::internal::DependencyTrackingDataType[DefaultSizeOfMemoryRegionUsedForDependencyTracking](TaskNumberNotUsedYet);

  int dependencyTrackingMemorySize{DefaultSizeOfMemoryRegionUsedForDependencyTracking};

  /**
   * We need a multi-read, single-write semaphores around the dependency
   * tracking. Arbitrary many tasks can be read and modify the dependency
   * tracking memory region at the same time, but we may only change it while
   * noone else uses it at all.
   */
  tarch::multicore::MultiReadSingleWriteSemaphore dependencyTrackingSemaphore;

  /**
   * We need to keep track of the number of tasks in flight, so we can identify
   * points where we can grow the dependency memory region.
   */
  int numberOfPendingTasks{0};
} // empty (internal) namespace


void tarch::multicore::omp::internal::spawnTaskWithOpenMP6(
  tarch::multicore::Task*                           job,
  const std::vector<tarch::multicore::TaskNumber>&  inDependencies,
  const std::vector<tarch::multicore::TaskNumber>&  outDependency
) {
  #pragma omp task priority(job->getPriority()) depend(iterator(j=0:inDependencies.size()), in: dependencyTrackingMemory[inDependencies[j]]) depend(iterator(j=0:outDependency.size()), inout: dependencyTrackingMemory[outDependency[j]]) shared(numberOfPendingTasks) untied transparent
  {
    trackTaskEvent( "start", outDependency.empty() ? tarch::multicore::NoOutDependencies : outDependency[0], inDependencies );
    OTTER_TASK_START(task);
    job->run();
    delete job;
    OTTER_TASK_END();
    trackTaskEvent( "end", outDependency.empty() ? tarch::multicore::NoOutDependencies : outDependency[0], inDependencies );
    #pragma omp atomic
    numberOfPendingTasks--;
  }
}


void tarch::multicore::omp::internal::reserveDependencyTrackingMemory( int taskNumber ) {
  tarch::multicore::MultiReadSingleWriteSemaphore::Lock optimisticLock(dependencyTrackingSemaphore, tarch::multicore::MultiReadSingleWriteSemaphore::Lock::Read);
  bool mightHaveToGrow = taskNumber>=dependencyTrackingMemorySize;
  optimisticLock.free();

  if (mightHaveToGrow) {
    tarch::multicore::MultiReadSingleWriteSemaphore::Lock conservativeLock(dependencyTrackingSemaphore, tarch::multicore::MultiReadSingleWriteSemaphore::Lock::Write);
    if (taskNumber>=dependencyTrackingMemorySize) {
      int volatile currentNumberOfPendingTasks = 1;

      while (currentNumberOfPendingTasks>0) {
        #pragma omp taskyield

        #pragma omp atomic read
        currentNumberOfPendingTasks = numberOfPendingTasks;
      }

      int oldSize = dependencyTrackingMemorySize;

      #pragma omp atomic write
      dependencyTrackingMemorySize = (taskNumber+1)*2;

      logInfo( "reserveDependencyTrackingMemory(int)", "allocated memory for task dependency management for up to " << (dependencyTrackingMemorySize+1) << " tasks (previously max " << oldSize << ")" );

      dependencyTrackingMemory = new tarch::multicore::omp::internal::DependencyTrackingDataType[dependencyTrackingMemorySize](TaskNumberNotUsedYet);
    }
  }
}



void tarch::multicore::native::spawnTask(
  Task*                        job,
  const std::set<TaskNumber>&  inDependencies,
  const TaskNumber&            taskNumber
) {
  tarch::multicore::omp::internal::reserveDependencyTrackingMemory(taskNumber);

  #pragma omp atomic
  numberOfPendingTasks++;

  tarch::multicore::MultiReadSingleWriteSemaphore::Lock optimisticLock(dependencyTrackingSemaphore, tarch::multicore::MultiReadSingleWriteSemaphore::Lock::Read);

  // Administer in-dependencies
  std::vector<TaskNumber>  iterableExistingInDepencencies;
  iterableExistingInDepencencies.reserve( inDependencies.size()+1 );
  for (auto inTask: inDependencies) {
    if ( tarch::multicore::omp::internal::canServeDependencyEntry(inTask) ) {
      iterableExistingInDepencencies.push_back(inTask);
    }
  }

  std::vector<TaskNumber>  iterableOutDepencencies( taskNumber == NoOutDependencies ? 0 : 1 );
  if (taskNumber != NoOutDependencies) iterableOutDepencencies[0] = taskNumber;

  logDebug(
    "spawnTask(...)",
    "fire task " << taskNumber <<
    " with " << iterableOutDepencencies.size() << " outgoing task dependencies and incoming dependencies " << toString(inDependencies) <<
    " out of which " << iterableExistingInDepencencies.size() << " can be served"
  );

  tarch::multicore::omp::internal::trackTaskEvent( "spawn", taskNumber, iterableExistingInDepencencies );

  OTTER_DEFINE_TASK(task, otterGetActiveTask(), otter_no_add_to_pool, __func__);

  tarch::multicore::omp::internal::spawnTaskWithOpenMP6(
    job,
    iterableExistingInDepencencies,
    iterableOutDepencencies
  );
}


void tarch::multicore::native::waitForTasks(const std::set<TaskNumber>& inDependencies) {
  // ensure we do not alter dependency memory while we spawn
  tarch::multicore::MultiReadSingleWriteSemaphore::Lock optimisticLock(dependencyTrackingSemaphore, tarch::multicore::MultiReadSingleWriteSemaphore::Lock::Read);

  // Administer in-dependencies
  std::vector<TaskNumber>  iterableExistingInDepencencies;
  iterableExistingInDepencencies.reserve( inDependencies.size() );
  for (auto inTask: inDependencies) {
    if ( tarch::multicore::omp::internal::canServeDependencyEntry(inTask) ) {
      iterableExistingInDepencencies.push_back(inTask);
    }
  }

  if (not iterableExistingInDepencencies.empty()) {
    logDebug( "waitForTasks(...)", "wait for "<< iterableExistingInDepencencies.size() << " tasks to terminate. Original in-set=" << toString(inDependencies) );

    tarch::multicore::omp::internal::trackTaskEvent( "wait", tarch::multicore::NoOutDependencies, iterableExistingInDepencencies );

    OTTER_TASK_WAIT_START(children); //! (AT) Note - otter doesn't currently support taskwait with dependencies

    #pragma omp taskwait depend(iterator(j=0:iterableExistingInDepencencies.size()), in: dependencyTrackingMemory[iterableExistingInDepencencies[j]])

    OTTER_TASK_WAIT_END();
  }
}


bool tarch::multicore::omp::internal::canServeDependencyEntry(int taskNumber) {
  return taskNumber < dependencyTrackingMemorySize;
}


#endif
