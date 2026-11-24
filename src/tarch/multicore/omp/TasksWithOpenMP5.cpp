// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "tarch/multicore/Task.h"

#include "multicore.h"
#include "Tasks.h"
#include "config.h"
#include "tarch/Assertions.h"
#include "tarch/logging/Log.h"
#include "tarch/logging/Statistics.h"
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

  #ifndef NumberOfPollsBeforeTaskIsReportedAsStuck
  /**
   * See tarch::multicore::omp::internal::reportStuckTask(). Overwrite it via
   * -DNumberOfPollsBeforeTaskIsReportedAsStuck=... A poll is one yield, i.e.
   * the default means a few seconds on a modern machine.
   */
  constexpr int NumberOfPollsBeforeTaskIsReportedAsStuck = 1000000;
  #endif

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
   * predecessors and successors are only changed while we hold the semaphore.
   * The other fields are written once or atomic.
   *
   * Whoever sets claimed first runs the job: either the OpenMP task which
   * release() creates or a thread that waits for the task in waitFor().
   * The predecessors are cleared once the node has terminated. Otherwise,
   * nodes would keep their whole history alive through the cycle between
   * predecessors and successors.
   */
  struct TaskNode {
    TaskNode(tarch::multicore::Task* job_, tarch::multicore::TaskNumber number_):
      job(job_),
      number(number_),
      priority(job_->getPriority()),
      status(Status::WaitingForPredecessors),
      claimed(false),
      unfinishedPredecessors(0) {}

    tarch::multicore::Task*                       job;
    const tarch::multicore::TaskNumber            number;
    const int                                     priority;
    std::atomic<Status>                           status;
    std::atomic<bool>                             claimed;
    std::atomic<int>                              unfinishedPredecessors;
    std::vector< std::shared_ptr<TaskNode> >      predecessors;
    std::vector< std::shared_ptr<TaskNode> >      successors;
  };

  using TaskNodePointer = std::shared_ptr<TaskNode>;

  /**
   * Guards liveTasks and the predecessors and successors of all nodes.
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

  std::string describe( const TaskNodePointer& node ) {
    std::ostringstream msg;
    msg << node->number << ":" << toString(node->status.load()) << "(" << node->unfinishedPredecessors.load() << " unfinished predecessors)";
    return msg.str();
  }

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
      node->predecessors.clear();
    }

    for (auto& successor: readySuccessors) {
      release(successor);
    }
  }

  /**
   * Try to become the one that runs the task. Exactly one caller succeeds.
   */
  bool claim( const TaskNodePointer& node ) {
    return not node->claimed.exchange(true);
  }

  /**
   * Run a task that the caller has claimed and release its successors. The
   * caller has to ensure that all predecessors have terminated.
   */
  void runClaimedTask( TaskNodePointer node ) {
    node->status = Status::Running;
    tarch::multicore::omp::internal::trackTaskEvent( "start", node->number, {} );

    node->job->run();
    delete node->job;

    tarch::multicore::omp::internal::trackTaskEvent( "end", node->number, {} );
    terminateTask(node);
  }

  /**
   * Turn a task whose predecessors have all terminated into an OpenMP task.
   * This is the only place where we create OpenMP tasks.
   *
   * A thread that waits for the task might already have claimed it, as it
   * does not need the OpenMP task. We then do not create it. If it
   * claims it later, the OpenMP task finds the claim taken and does nothing.
   */
  void release( TaskNodePointer node ) {
    Status expected = Status::WaitingForPredecessors;
    if ( not node->status.compare_exchange_strong(expected, Status::Released) ) {
      return;
    }

    const int priority = node->priority;

    OTTER_DEFINE_TASK(task, otterGetActiveTask(), otter_no_add_to_pool, __func__);

    #pragma omp task priority(priority) firstprivate(node) untied
    {
      if ( claim(node) ) {
        OTTER_TASK_START(task);
        runClaimedTask(node);
        OTTER_TASK_END();
      }
    }
  }

  /**
   * Yield until a task that somebody else has claimed terminates. Without the
   * yield, the thread would spin and block other threads from executing
   * tasks. We only wait for tasks that run, i.e. the wait ends.
   */
  void pollUntilTerminated( const TaskNodePointer& node ) {
    int polls = 0;
    while ( node->status.load()!=Status::Terminated ) {
      #pragma omp taskyield

      polls++;
      if (polls==NumberOfPollsBeforeTaskIsReportedAsStuck) {
        tarch::multicore::omp::internal::reportStuckTask(
          "waitForTasks(...) (OpenMP thread " + std::to_string(omp_get_thread_num()) + ") waits for task " + describe(node) + " after " + std::to_string(polls) + " polls"
        );
      }
    }
    if (polls>0) {
      tarch::logging::Statistics::getInstance().inc( "tarch::multicore::omp::yields-while-waiting", polls );
    }
  }

  /**
   * Wait for a task by doing the work that nobody else has started
   *
   * We follow the predecessors backwards from the task that we wait for to
   * the first one that has no unfinished predecessors, run it on this thread
   * and then work our way forward again. We never create or wait for an
   * OpenMP task unless somebody else runs the task already. The stack is an
   * explicit one, as chains of tasks might be long.
   */
  void waitFor( const TaskNodePointer& target ) {
    std::vector<TaskNodePointer> stack;
    stack.push_back(target);

    while ( not stack.empty() ) {
      TaskNodePointer node = stack.back();

      if ( node->status.load()==Status::Terminated ) {
        stack.pop_back();
      }
      else if ( node->claimed.load() ) {
        pollUntilTerminated(node);
        stack.pop_back();
      }
      else {
        TaskNodePointer unfinishedPredecessor;
        {
          tarch::multicore::Lock lock(semaphore);
          for (auto& predecessor: node->predecessors) {
            if ( predecessor->status.load()!=Status::Terminated ) {
              unfinishedPredecessor = predecessor;
              break;
            }
          }
        }

        if (unfinishedPredecessor) {
          stack.push_back(unfinishedPredecessor);
        }
        else {
          if ( claim(node) ) {
            tarch::logging::Statistics::getInstance().inc( "tarch::multicore::omp::tasks-run-by-waiting-thread" );
            runClaimedTask(node);
          }
          else {
            pollUntilTerminated(node);
          }
          stack.pop_back();
        }
      }
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
        node->predecessors.push_back(p->second);
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
    waitFor(node);
  }
}

#endif
