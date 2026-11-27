// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "tarch/multicore/Task.h"

#include "multicore.h"
#include "Tasks.h"
#include "config.h"
#include "tarch/Assertions.h"
#include "tarch/logging/Log.h"
#include "tarch/multicore/BooleanSemaphore.h"
#include "tarch/multicore/Lock.h"
#include "tarch/multicore/multicore.h"
#include "tarch/multicore/otter.h"


#if defined(SharedOMP) and _OPENMP<202411

#include <atomic>
#include <memory>
#include <omp.h>
#include <sstream>
#include <unordered_map>
#include <vector>


namespace {
  static tarch::logging::Log _log( "tarch::multicore::native(omp5)" );

  /**
   * Life cycle of a task. Consult tarch::multicore::omp::internal for details.
   */
  enum class Status {
    WaitingForPredecessors,
    Released,
    Running,
    Terminated
  };

  std::string toString(Status status) {
    switch (status) {
      case Status::WaitingForPredecessors: return "waiting-for-predecessors";
      case Status::Released:               return "released-but-not-started";
      case Status::Running:                return "running";
      case Status::Terminated:             return "terminated";
    }
    return "<undef>";
  }

  /**
   * One task which has been spawned and not terminated yet. The members
   * unfinishedPredecessors and successors are only changed while we hold
   * the semaphore. The other fields are written once or atomic.
   */
  struct TaskNode {
    TaskNode(tarch::multicore::Task* job_, tarch::multicore::TaskNumber number_):
      job(job_),
      number(number_),
      priority(job_->getPriority()),
      status(Status::WaitingForPredecessors),
      unfinishedPredecessors(0) {}

    tarch::multicore::Task*                       job;
    const tarch::multicore::TaskNumber            number;
    const int                                     priority;
    std::atomic<Status>                           status;
    std::atomic<int>                              unfinishedPredecessors;
    std::vector< std::shared_ptr<TaskNode> >      successors;
  };

  using TaskNodePointer = std::shared_ptr<TaskNode>;

  /**
   * Guards liveTasks and the successors of all nodes.
   *
   * The critical sections are short and never contain a task scheduling
   * point: Neither do we create a task nor do we yield or wait while we hold
   * the semaphore.
   */
  tarch::multicore::BooleanSemaphore  semaphore;

  /**
   * All tasks that have been spawned with a number and have not terminated.
   * A number which is not in the map is not in use or has terminated, i.e.
   * nobody has to wait for it.
   */
  std::unordered_map< tarch::multicore::TaskNumber, TaskNodePointer >  liveTasks;

  void release( TaskNodePointer node );

  /**
   * Mark a task as terminated and release all successors whose last
   * predecessor it was.
   *
   * The successors are released outside of the critical section: Creating
   * an OpenMP task is a task scheduling point, i.e. the runtime might
   * start another task right away.
   */
  void terminateTask( TaskNodePointer node ) {
    std::vector<TaskNodePointer> readySuccessors;

    {
      tarch::multicore::Lock lock(semaphore);
      node->status = Status::Terminated;

      if (node->number!=tarch::multicore::NoOutDependencies) {
        auto p = liveTasks.find(node->number);
        if (p!=liveTasks.end() and p->second==node) {
          liveTasks.erase(p);
        }
      }

      for (auto& successor: node->successors) {
        if ( --successor->unfinishedPredecessors == 0 ) {
          readySuccessors.push_back(successor);
        }
      }
      node->successors.clear();
    }

    for (auto& successor: readySuccessors) {
      release(successor);
    }
  }

  /**
   * Turn a task whose predecessors have all terminated into an OpenMP task.
   * This is the only place where we create OpenMP tasks.
   */
  void release( TaskNodePointer node ) {
    node->status = Status::Released;
    const int priority = node->priority;

    OTTER_DEFINE_TASK(task, otterGetActiveTask(), otter_no_add_to_pool, __func__);

    #pragma omp task priority(priority) firstprivate(node) untied
    {
      node->status = Status::Running;
      tarch::multicore::omp::internal::trackTaskEvent( "start", node->number, {} );

      OTTER_TASK_START(task);
      node->job->run();
      delete node->job;
      OTTER_TASK_END();

      tarch::multicore::omp::internal::trackTaskEvent( "end", node->number, {} );
      terminateTask(node);
    }
  }
} // empty (internal) namespace


void tarch::multicore::native::spawnTask(
  Task*                        job,
  const std::set<TaskNumber>&  inDependencies,
  const TaskNumber&            taskNumber
) {
  auto node = std::make_shared<TaskNode>(job, taskNumber);

  std::vector<TaskNumber> predecessors;
  {
    tarch::multicore::Lock lock(semaphore);

    for (auto inTask: inDependencies) {
      auto p = liveTasks.find(inTask);
      if ( p!=liveTasks.end() and p->second->status.load()!=Status::Terminated ) {
        p->second->successors.push_back(node);
        predecessors.push_back(inTask);
      }
    }
    node->unfinishedPredecessors = static_cast<int>( predecessors.size() );

    if (taskNumber!=NoOutDependencies) {
      liveTasks[taskNumber] = node;
    }
  }

  logDebug(
    "spawnTask(...)",
    "spawn task " << taskNumber << " with in-dependencies " << toString(inDependencies) <<
    " out of which " << predecessors.size() << " have not terminated"
  );

  tarch::multicore::omp::internal::trackTaskEvent( "spawn", taskNumber, predecessors );

  if (predecessors.empty()) {
    release(node);
  }
}


void tarch::multicore::native::waitForTasks(const std::set<TaskNumber>& inDependencies) {
  std::vector<TaskNodePointer> nodes;
  std::vector<TaskNumber>      numbers;
  {
    tarch::multicore::Lock lock(semaphore);
    for (auto inTask: inDependencies) {
      auto p = liveTasks.find(inTask);
      if ( p!=liveTasks.end() and p->second->status.load()!=Status::Terminated ) {
        nodes.push_back( p->second );
        numbers.push_back( inTask );
      }
    }
  }

  if (not nodes.empty()) {
    tarch::multicore::omp::internal::trackTaskEvent( "wait", tarch::multicore::NoOutDependencies, numbers );
  }

  for (auto& node: nodes) {
    while ( node->status.load()!=Status::Terminated ) {
      #pragma omp taskyield
    }
  }
}

#endif
