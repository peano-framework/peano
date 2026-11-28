// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include <string>
#include <map>
#include <vector>
#include <tuple>

#include "Log.h"
#include "tarch/timing/Watch.h"
#include "tarch/multicore/BooleanSemaphore.h"

namespace tarch {
  namespace logging {
    class Statistics;
  }
}

/** Global statistics interface
 *
 * This interface only logs data if you have translated with
 *
 * ~~~~~~~~~~~~~~~~~~~~~
 * CXXFLAGS="... -DTrackStatistics"
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * To get the stats right, you might want to invoke the clear() operation 
 * explicitly when you start up your code. This way, you ensure that the 
 * internal timer (_globalWatch) is properly initialised.
 *
 * You can control if you want to track data every time it is submitted
 * or sample a quantity every now and then.
 *
 *
 * ## CSV format
 *
 * The first column in the csv file will hold time stamps. Each further
 * column presents one type of measurement. The name for these entries is
 * determined by the source code logging the data. As we sample, not all
 * columns will hold data for all entries, as the rows are ordered
 * temporary and we only write entries if there are entries updated.
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * t   mpi wait times  tarch::multicore::bsp-concurrency-level   tarch::multicore::fuse-tasks  tarch::multicore::global-pending-tasks  tarch::multicore::thread-local-pending-tasks  grid-control-events
 * t, mpi wait times, tarch::multicore::bsp-concurrency-level, tarch::multicore::fuse-tasks, tarch::multicore::global-pending-tasks, tarch::multicore::thread-local-pending-tasks, grid-control-events
 * 0.00130565, (0.150548/0/0.157291/#10), (1/0/1/#1), , , , "((refinementControl=Refine,offset=[-9.03,-9.03,-9.03],width=[18.06,18.06,18.06],h=[2.02,2.02,2.02]))" ,
 * 0.00131057, , (2/1/2/#1), , , , "((refinementControl=Refine,offset=[-9.03,-9.03,-9.03],width=[18.06,18.06,18.06],h=[2.02,2.02,2.02]))" ,
 * 0.00134099, , (3/2/3/#1), , , , "((refinementControl=Refine,offset=[-9.03,-9.03,-9.03],width=[18.06,18.06,18.06],h=[2.02,2.02,2.02]))" ,
 * 0.00135324, , (4/3/4/#1), , , , "((refinementControl=Refine,offset=[-9.03,-9.03,-9.03],width=[18.06,18.06,18.06],h=[2.02,2.02,2.02]))" ,
 * 0.00135806, , (5/4/5/#1), , , , "((refinementControl=Refine,offset=[-9.03,-9.03,-9.03],width=[18.06,18.06,18.06],h=[2.02,2.02,2.02]))" ,
 * 0.00137025, , (6/5/6/#1), , , , "((refinementControl=Refine,offset=[-9.03,-9.03,-9.03],width=[18.06,18.06,18.06],h=[2.02,2.02,2.02]))" ,
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Each entry follows a format of its own. The time stamps are given in
 * seconds. Some columns (like the refinement strings) hold strings. Most
 * comments will hold three-tuples:
 *
 * 1. The first entry is the actual value of the numeric value at the time
 *    when we make the snapshot.
 * 2. The second entry is the minimum over the whole sample period, i.e. since
 *    the last data dump.
 * 3. The third entry is the maximum.
 * 4. The fourth entry holds the number of samples taken over the sampling
 *    period. If you sample every 1000 steps, it should be 1000 always. If
 *    you sample every x seconds, it will be a non-constant value.
 *
 * There is a Python script peano4.postprocessing.plot-statistics.py which
 * helps you to extract the 4-tuple data.
 *
 *
 * ## Event logs
 *
 * The routine log() with a string value stores a time-stamped event under an
 * identifier. All events of one identifier end up in one csv column. The task
 * runtime uses this (with the additional flag -DTrackTaskDependencies)
 * for the identifier "tarch::multicore::omp::task-event", see
 * tarch::multicore::omp::internal::trackTaskEvent(). The events are held in
 * memory until writeToCSV() is called, i.e. they are lost if the code hangs
 * or crashes before.
 *
 */
class tarch::logging::Statistics {
  public:
    /**
     * This is not the canonical realisation of singletons as I use it usually
     * for stats in Peano.
     */
    static Statistics& getInstance();

    #ifdef TrackStatistics
      /**
       * Log one particular value
       *
       * A log means that you basically report a snapshot. The value is
       * memorised. It is up to the database logic to decide if to store this
       * value or not.
       *
       * @param identifier Unique name (string) for this event
       * @see updateDataSnapshot() for the data base logic.
       */
      void log( const std::string& identifier, double value, bool disableSampling = false );

      /**
       * Log an event
       */
      void log( const std::string& identifier, const std::string& value );

      /**
       * Increment an internal stats counter
       *
       * There is no decrement, as you can always pass in a negative value as well.
       */
      void inc( const std::string& identifier, double value = 1.0, bool disableSampling = false, bool clearAfterDatabaseDump = false );
    #else
      inline void log( [[maybe_unused]] const std::string& identifier, [[maybe_unused]] double value, [[maybe_unused]] bool disableSampling = false ) {}
      inline void log( [[maybe_unused]] const std::string& identifier, [[maybe_unused]] const std::string& value ) {}
      inline void inc( [[maybe_unused]] const std::string& identifier, [[maybe_unused]] double value = 1.0, [[maybe_unused]] bool disableSampling = false, [[maybe_unused]] bool clearAfterDatabaseDump = false ) {}
    #endif

    /**
     * Write data to csv file
     *
     * I do append the extension (csv) and a rank identifier myself, i.e. each
     * rank will write a csv file of its own. The format of the csv file is
     * discussed above. The name is not 100% correct, as the routine also
     * erases snapshot data which is not required anymore, as we have dumped
     * it on disk.
     *
     * ## Realisation
     *
     * Each data entry hosts a long list of snapshots which are chronologically
     * ordered. That is, we have
     *
     *
     *       first-stats:  0.4xa, 0.7b, 0.8c, 1.2d
     *       second-stats: 0.3xe, 0.5f, 0.9g
     *
     * In the map. The code runs through all stats iteratively, until no more
     * stats are in the file, i.e. until we have dumped all data. In a first
     * step per iteration, we find the minimum time stamp that has to be
     * dumped. If there are no data left, this one is max and therefore we
     * can terminate.
     *
     * As long as there are entries, now have a valid cut off time stamp t.
     * We run over all entries in our database next. In the example above, that
     * would be two of them. We only ever have to study the first entry within
     * a sequence of events, as they are chronologically ordered. If the time
     * stamp of this first entry is smaller or equal to our cut-off, we dump
     * it and then remove it from the data set.
     *
     */
    void writeToCSV( std::string  filename = "statistics" );

    void clear();

  private:
    static Statistics   _singleton;

    static Log          _log;

    static tarch::multicore::BooleanSemaphore  _semaphore;

    int                   _maxCountInBetweenTwoMeasurements;
    double                _maxTimeInBetweenTwoMeasurements;

    tarch::timing::Watch  _globalWatch;

    /**
     * One data set for one type (identifier) of statistics
     *
     * The watch is used to track time (obviously) and the counter counts the
     * number of hits. If you enable sampling, the counter can trigger writes,
     * i.e. you can specify after how many hits you wanna dump data. The actual
     * data is a five-tuple of time stamp, current value, minimal value over
     * the tracked period and maximum value. The last entry holds the number of
     * writes to the bucket.
     */
    struct DataSet {
      /**
       * Not used for time stamps but only for sampling.
       */
      tarch::timing::Watch  _watch;
      int                   _counter;
      std::vector< std::tuple<double,double,double,double,int> >   _data;
      /**
       * As we hold data sets within a map, we need a default constructor.
       * However, this one yields an invalid data set.
       */
      DataSet();
      DataSet(double time);
      void createNewSnapshot(double t, bool clearAfterDatabaseDump);

      std::string toString() const;
    };

    struct LogMessage {
      std::vector< std::tuple<double,std::string> >   _data;
      LogMessage() = default;
    };

    /**
     * Mapping from identifier who wrote stats (key) onto DataSet.
     */
    std::map< std::string, DataSet >     _dataSetMap;
    std::map< std::string, LogMessage >  _logMessageMap;

    Statistics();

    /**
     * Not const, as it also updates the internal counters.
     *
     * A new snapshot is written if the number of writes for this data set
     * exceeds _maxCountInBetweenTwoMeasurements or if the time since the
     * last write exceeds _maxTimeInBetweenTwoMeasurements. Furthermore,
     * a new data set is written if the sampling is disabled. In this case,
     * all the local counters are still reset, as you might want to combine
     * manual disabled stats with sampled ones.
     *
     * If the data has not changed since the last snapshot, I don't write
     * anything.
     *
     * If data has changed by more than 10%, I always write.
     *
     * @return Write a new snapshot.
     */
    bool acceptNewData(const std::string& identifier, bool disableSampling, bool clearAfterDatabaseDump);

    /**
     * Ensures that dataset is there. If not, it creates a new entry.
     * If there's already an entry for identifier, it degenerates to
     * nop.
     */
    void initData(const std::string& identifier);

    /**
     * Updates snapshot
     */
    void updateDataSnapshot(const std::string& identifier, double value);
};
