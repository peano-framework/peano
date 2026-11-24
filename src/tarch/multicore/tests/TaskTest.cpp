#include "TaskTest.h"

#include "tarch/multicore/multicore.h"
#include "tarch/multicore/Task.h"

#if defined(SharedOMP)
#include <atomic>
#include <omp.h>
#include <set>
#include <vector>
#endif


tarch::logging::Log tarch::multicore::tests::TaskTest::_log( "tarch::multicore::tests::TaskTest" );


tarch::multicore::tests::TaskTest::TaskTest():
  TestCase( "tarch::multicore::tests::TaskTest" ) {
}


void tarch::multicore::tests::TaskTest::run() {
  testMethod( testChainsOfDependentTasks );
  testMethod( testTaskNumberReusedAsInoutDependency );
  testMethod( testWaitRunsTasksThatHaveNotStarted );
}


void tarch::multicore::tests::TaskTest::testChainsOfDependentTasks() {
  #if defined(SharedOMP)
  constexpr int NumberOfProducers = 8;
  constexpr int CellsPerProducer  = 250;
  constexpr int ChainLength       = 7;
  constexpr int NumberOfTasks     = NumberOfProducers * CellsPerProducer * ChainLength;

  std::atomic<int>               clock(0);
  std::vector< std::atomic<int> > started(NumberOfTasks);
  std::vector< std::atomic<int> > finished(NumberOfTasks);
  std::vector< std::atomic<int> > runs(NumberOfTasks);
  for (int i=0; i<NumberOfTasks; i++) {
    started[i]  = -1;
    finished[i] = -1;
    runs[i]     = 0;
  }

  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp taskloop grainsize(1) nogroup untied
    for (int producer=0; producer<NumberOfProducers; producer++) {
      for (int cell=0; cell<CellsPerProducer; cell++) {
        for (int link=0; link<ChainLength; link++) {
          const int number = (producer*CellsPerProducer+cell)*ChainLength + link;
          std::set<tarch::multicore::TaskNumber> inDependencies;
          if (link>0) inDependencies.insert( number-1 );
          tarch::multicore::native::spawnTask(
            new tarch::multicore::TaskWithCopyOfFunctor(
              tarch::multicore::Task::DontFuse,
              tarch::multicore::Task::DefaultPriority,
              [&started,&finished,&runs,&clock,number]() {
                started[number] = clock++;
                runs[number]++;
                volatile double x = 1.0;
                for (int i=0; i<1000; i++) x = x*1.0000001;
                finished[number] = clock++;
              }
            ),
            inDependencies,
            number
          );
        }
      }
    }
    #pragma omp taskwait

    // Consumers: wait for the last task of each chain
    #pragma omp taskloop grainsize(1) nogroup untied
    for (int producer=0; producer<NumberOfProducers; producer++) {
      for (int cell=0; cell<CellsPerProducer; cell++) {
        const int last = (producer*CellsPerProducer+cell)*ChainLength + ChainLength-1;
        tarch::multicore::native::waitForTasks( {last} );
      }
    }
    #pragma omp taskwait
  }

  for (int number=0; number<NumberOfTasks; number++) {
    validateWithParams2( runs[number]==1, number, runs[number] );
    validateWithParams1( finished[number]>started[number], number );
    if (number%ChainLength!=0) {
      validateWithParams3( started[number]>finished[number-1], number, started[number], finished[number-1] );
    }
  }
  #endif
}


void tarch::multicore::tests::TaskTest::testTaskNumberReusedAsInoutDependency() {
  #if defined(SharedOMP)
  std::atomic<int> clock(0);
  std::atomic<int> firstFinished(-1);
  std::atomic<int> secondStarted(-1);
  std::atomic<int> secondFinished(-1);
  bool             waitedSuccessfully = false;

  constexpr int Number = 17;

  #pragma omp parallel
  #pragma omp single
  {
    tarch::multicore::native::spawnTask(
      new tarch::multicore::TaskWithCopyOfFunctor(
        tarch::multicore::Task::DontFuse,
        tarch::multicore::Task::DefaultPriority,
        [&firstFinished,&clock]() {
          volatile double x = 1.0;
          for (int i=0; i<100000; i++) x = x*1.0000001;
          firstFinished = clock++;
        }
      ),
      {},
      Number
    );

    tarch::multicore::native::spawnTask(
      new tarch::multicore::TaskWithCopyOfFunctor(
        tarch::multicore::Task::DontFuse,
        tarch::multicore::Task::DefaultPriority,
        [&secondStarted,&secondFinished,&clock]() {
          secondStarted = clock++;
          volatile double x = 1.0;
          for (int i=0; i<100000; i++) x = x*1.0000001;
          secondFinished = clock++;
        }
      ),
      {Number},
      Number
    );

    tarch::multicore::native::waitForTasks( {Number} );
    // The wait has to wait for the second task. We check the outcome below.
    waitedSuccessfully = secondFinished>=0;
    #pragma omp taskwait
  }

  validate( waitedSuccessfully );
  validateWithParams2( firstFinished>=0, firstFinished, secondStarted );
  validateWithParams2( secondStarted>firstFinished, secondStarted, firstFinished );
  validateWithParams2( secondFinished>secondStarted, secondStarted, secondFinished );
  #endif
}


void tarch::multicore::tests::TaskTest::testWaitRunsTasksThatHaveNotStarted() {
  #if defined(SharedOMP)
  constexpr int A = 1;
  constexpr int B = 2;
  constexpr int C = 3;
  constexpr int D = 4;

  std::atomic<int>                clock(0);
  std::vector< std::atomic<int> > started(5);
  std::vector< std::atomic<int> > finished(5);
  std::vector< std::atomic<int> > threadOfTask(5);
  for (int i=0; i<5; i++) {
    started[i]  = -1;
    finished[i] = -1;
    threadOfTask[i] = -1;
  }

  auto spawn = [&](int number, std::set<tarch::multicore::TaskNumber> inDependencies) {
    tarch::multicore::native::spawnTask(
      new tarch::multicore::TaskWithCopyOfFunctor(
        tarch::multicore::Task::DontFuse,
        tarch::multicore::Task::DefaultPriority,
        [&started,&finished,&threadOfTask,&clock,number]() {
          started[number]      = clock++;
          threadOfTask[number] = omp_get_thread_num();
          finished[number]     = clock++;
        }
      ),
      inDependencies,
      number
    );
  };

  // The second thread spins. It never reaches a task scheduling point, i.e.
  // it does not run any of the tasks. Only the thread that waits can.
  std::atomic<bool> waitIsOver(false);
  bool              hadTwoThreads = false;

  #pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num()==0) {
      hadTwoThreads = omp_get_num_threads()==2;

      spawn( A, {} );
      spawn( B, {} );
      spawn( C, {A,B} );
      spawn( D, {C} );

      tarch::multicore::native::waitForTasks( {D} );
      waitIsOver = true;
    }
    else {
      while ( not waitIsOver.load() ) {}
    }
  }

  if (not hadTwoThreads) return;

  for (int number=A; number<=D; number++) {
    validateWithParams2( started[number]>=0, number, started[number] );
    validateWithParams2( threadOfTask[number]==0, number, threadOfTask[number] );
  }
  validateWithParams2( started[C]>finished[A], started[C], finished[A] );
  validateWithParams2( started[C]>finished[B], started[C], finished[B] );
  validateWithParams2( started[D]>finished[C], started[D], finished[C] );
  #endif
}
