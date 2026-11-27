// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "tarch/multicore/Task.h"

#include "multicore.h"
#include "Tasks.h"
#include "config.h"
#include "tarch/Assertions.h"
#include "tarch/logging/Statistics.h"
#include "tarch/multicore/Core.h"
#include "tarch/multicore/multicore.h"
#include "tarch/multicore/otter.h"
#include "tarch/compiler/CompilerSpecificSettings.h"


#if defined(SharedOMP)

#include <atomic>
#include <omp.h>
#include <sstream>


void tarch::multicore::omp::internal::trackTaskEvent(
  [[maybe_unused]] const char*                                       event,
  [[maybe_unused]] tarch::multicore::TaskNumber                      taskNumber,
  [[maybe_unused]] const std::vector<tarch::multicore::TaskNumber>&  inDependencies
) {
  #if defined(TrackStatistics) and defined(TrackTaskDependencies)
  std::ostringstream msg;
  msg << event << " " << taskNumber << " in={";
  bool first = true;
  for (auto i: inDependencies) {
    msg << (first ? "" : ",") << i;
    first = false;
  }
  msg << "}";
  tarch::logging::Statistics::getInstance().log( "tarch::multicore::omp::task-event", msg.str() );
  #endif
}


void tarch::multicore::native::spawnAndWaitAsTaskLoop(const std::vector<tarch::multicore::Task*>& tasks) {
  assertion(not tasks.empty());

#if defined(UseOtter)
  aacuto otter_tasks = std::vector<otter_task_context*>();
  for (int i = 0; i < static_cast<int>(tasks.size()); i++) {
    otter_tasks.push_back(otterTaskInitialise(OTTER_NULL_TASK, -1, otter_no_add_to_pool, true, OTTER_SOURCE_LOCATION(), __func__));
  }
#endif

  #if defined(OpenMPTaskGroup) and _OPENMP>=202411
  #pragma omp taskloop nogroup shared(tasks) grainsize(1) priority(tarch::multicore::Task::DefaultPriority) untied transparent
  #elif defined(OpenMPTaskGroup)
  #pragma omp taskloop nogroup shared(tasks) grainsize(1) priority(tarch::multicore::Task::DefaultPriority) untied
  #endif
  for (int i = 0; i < static_cast<int>(tasks.size()); i++) {
    #if !defined(OpenMPTaskGroup) and _OPENMP>=202411
    #pragma omp task priority(tarch::multicore::Task::DefaultPriority) transparent
    #elif !defined(OpenMPTaskGroup)
    #pragma omp task priority(tarch::multicore::Task::DefaultPriority)
    #endif
    {
      OTTER_TASK_START(otter_tasks[i]);
      tasks[i]->run();
      delete tasks[i];
      OTTER_TASK_END();
    }
  }

  OTTER_TASK_WAIT_START(children);
  #pragma omp taskwait // Wait for all elements from tasks to complete
                       // do not wait for the children of tasks
  OTTER_TASK_WAIT_END();
}


void tarch::multicore::native::waitForAllTasks() {
  #pragma omp taskwait
}



#endif
