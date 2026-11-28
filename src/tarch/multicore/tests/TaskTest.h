// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once


#include "tarch/tests/TestCase.h"
#include "tarch/logging/Log.h"


namespace tarch {
  namespace multicore {
    namespace tests {
      class TaskTest;
    }
  }
}


/**
 * Tests the task dependencies of the native tasking backend
 *
 * The tests only do something for the OpenMP backend. They run within a
 * parallel region, as Peano's main does, and they always use all threads
 * OpenMP offers, i.e. they test on as many threads as you configure
 * (OMP_NUM_THREADS). If task dependencies deadlock, the test does not
 * terminate.
 */
class tarch::multicore::tests::TaskTest: public tarch::tests::TestCase {
  private:
    static tarch::logging::Log _log;

    /**
     * Many cells spawn chains of dependent tasks concurrently
     *
     * Each cell spawns a chain of tasks A0 -> A1 -> ... Every task takes the
     * time stamp of its start and its end from a global counter. After all
     * cells have spawned their chains, we wait for the last task of each
     * chain. Each task has to run exactly once and has to start after its
     * predecessor has terminated.
     *
     * This is the pattern of the producer-consumer compiler with chains.
     * With OpenMP 5, we once spawned all tasks right away and let them poll
     * their predecessors. This deadlocks as soon as a thread picks up a
     * successor while it has suspended the predecessor.
     */
    void testChainsOfDependentTasks();

    /**
     * Spawn a task with a number that is still in use
     *
     * If the number is also in the in-set, the new task is an inout
     * dependency, i.e. it has to run after the old one. A wait for the
     * number has to wait for the new task.
     */
    void testTaskNumberReusedAsInoutDependency();

  public:
    TaskTest();

    /**
     * This routine is triggered by the TestCaseCollection
     */
    virtual void run() override;
};
