#include "Statistics.h"
#include "tarch/Assertions.h"
#include "tarch/multicore/Lock.h"
#include "tarch/mpi/Rank.h"

#include "tarch/la/ScalarOperations.h"

#include <fstream>


tarch::logging::Statistics          tarch::logging::Statistics::_singleton;
tarch::logging::Log                 tarch::logging::Statistics::_log( "tarch::logging::Statistics" );
tarch::multicore::BooleanSemaphore  tarch::logging::Statistics::_semaphore;


tarch::logging::Statistics& tarch::logging::Statistics::getInstance() {
  static tarch::logging::Statistics singleton;
  return singleton;
}



tarch::logging::Statistics::Statistics():
  _maxCountInBetweenTwoMeasurements(1000),
  _maxTimeInBetweenTwoMeasurements(1.0),
  _globalWatch( "tarch::logging::Statistics", "Statistics", false) {
}



tarch::logging::Statistics::DataSet::DataSet():
  _watch("tarch::logging::Statistics", "Statistics", false),
  _counter(0) {
}


tarch::logging::Statistics::DataSet::DataSet(double time):
  _watch("tarch::logging::Statistics", "Statistics", false),
  _counter(0) {
  _data.push_back( std::make_tuple(time,0.0,0.0,0.0,0) );
}


std::string tarch::logging::Statistics::DataSet::toString() const {
  std::ostringstream msg;
  msg << "(#" << _data.size() << ", ("
      << std::get<0>( _data.back() ) << ", "
      << std::get<1>( _data.back() ) << ", "
      << std::get<2>( _data.back() ) << ", "
      << std::get<3>( _data.back() ) << ", "
      << std::get<4>( _data.back() ) << "))";
  return msg.str();
}


void tarch::logging::Statistics::DataSet::createNewSnapshot(double time, bool clearAfterDatabaseDump) {
  double lastValueFromPreviousSnapshot = clearAfterDatabaseDump ? 0.0 : std::get<1>( _data.back() );
  _data.push_back( std::make_tuple(
    time,
    lastValueFromPreviousSnapshot,
    lastValueFromPreviousSnapshot,
    lastValueFromPreviousSnapshot,
    0
  ));
}


void tarch::logging::Statistics::clear() {
  _dataSetMap.clear();
}


bool tarch::logging::Statistics::acceptNewData(const std::string& identifier, bool disableSampling, bool clearAfterDatabaseDump) {
  _dataSetMap[identifier]._counter++;
  _dataSetMap[identifier]._watch.stop();

  bool result = false;

  constexpr double DifferenceThreshold = 0.1;

  if (
    disableSampling
  ) {
   result = true;
  }
  else if (
    _dataSetMap[identifier]._data.size()>1
    and
    std::get<1>( _dataSetMap[identifier]._data[_dataSetMap[identifier]._data.size()-1] ) == std::get<1>( _dataSetMap[identifier]._data[_dataSetMap[identifier]._data.size()-2] )
  ) {
    result = false;
  }
  else if (
    _dataSetMap[identifier]._counter>=_maxCountInBetweenTwoMeasurements
    or
    _dataSetMap[identifier]._watch.getCalendarTime() > _maxTimeInBetweenTwoMeasurements
  ) {
    result = true;
  }
  else if (
    _dataSetMap[identifier]._data.size()>1
    and
    std::get<1>( _dataSetMap[identifier]._data[_dataSetMap[identifier]._data.size()-1] ) > std::get<1>( _dataSetMap[identifier]._data[_dataSetMap[identifier]._data.size()-2] ) * (1.0 + DifferenceThreshold)
    and
    not clearAfterDatabaseDump
  ) {
    result = true;
  }
  else if (
    _dataSetMap[identifier]._data.size()>1
    and
    std::get<1>( _dataSetMap[identifier]._data[_dataSetMap[identifier]._data.size()-1] ) < std::get<1>( _dataSetMap[identifier]._data[_dataSetMap[identifier]._data.size()-2] ) * (1.0 - DifferenceThreshold)
    and
    not clearAfterDatabaseDump
  ) {
    result = true;
  }


  if (result) {
    _dataSetMap[identifier]._counter = 0;
    _dataSetMap[identifier]._watch.start();
    return true;
  }
  else return false;
}


void tarch::logging::Statistics::initData(const std::string& identifier) {
  if ( _dataSetMap.count( identifier )==0 ) {
     _globalWatch.stop();
     double t = _globalWatch.getCalendarTime();

    _dataSetMap.insert( std::pair<std::string,DataSet>( identifier, DataSet(t) ));
  }
}


void tarch::logging::Statistics::updateDataSnapshot(const std::string& identifier, double value) {
  assertion( _dataSetMap.count( identifier )==1 );
  std::get<1>( _dataSetMap[identifier]._data.back() ) = value;
  std::get<2>( _dataSetMap[identifier]._data.back() ) = std::min(std::get<2>( _dataSetMap[identifier]._data.back() ), value);
  std::get<3>( _dataSetMap[identifier]._data.back() ) = std::max(std::get<3>( _dataSetMap[identifier]._data.back() ), value);
  std::get<4>( _dataSetMap[identifier]._data.back() ) = 1 + std::get<4>( _dataSetMap[identifier]._data.back() );
}


#ifdef TrackStatistics
void tarch::logging::Statistics::log( const std::string& identifier, const std::string& value ) {
  tarch::multicore::Lock lock(_semaphore);
  if ( _logMessageMap.count( identifier )==0 ) {
    _logMessageMap.insert( std::pair< std::string, LogMessage >( identifier, LogMessage() ) );
  }

  _globalWatch.stop();
  double t = _globalWatch.getCalendarTime();

  _logMessageMap[identifier]._data.push_back( std::tuple<double,std::string>(t,value) );
}


void tarch::logging::Statistics::log( const std::string& identifier, double value, bool disableSampling ) {
  tarch::multicore::Lock lock(_semaphore);
  initData(identifier);
  updateDataSnapshot(identifier,value);
  if ( acceptNewData(identifier, disableSampling, false /*clearAfterDatabaseDump*/) ) {
    _globalWatch.stop();
    double t = _globalWatch.getCalendarTime();
    logDebug( "log(string,double)", identifier << "=" << value );
    _dataSetMap[identifier].createNewSnapshot(t, false);
  }
}


void tarch::logging::Statistics::inc( const std::string& identifier, double value, bool disableSampling, bool clearAfterDatabaseDump ) {
  tarch::multicore::Lock lock(_semaphore);
  initData( identifier );

  double newValue = std::get<1>( _dataSetMap[identifier]._data.back() ) + value;
  updateDataSnapshot(identifier,newValue);

  if ( acceptNewData(identifier, disableSampling, clearAfterDatabaseDump) ) {
    _globalWatch.stop();
    double t = _globalWatch.getCalendarTime();
    _dataSetMap[identifier].createNewSnapshot(t, clearAfterDatabaseDump);
  }
}
#endif


void tarch::logging::Statistics::writeToCSV( [[maybe_unused]] std::string filename ) {
#ifdef TrackStatistics
  tarch::multicore::Lock lock(_semaphore);

  logDebug( "writeToCSV(string)", "start to dump statistics into file " << filename );

  if (tarch::mpi::Rank::getInstance().getNumberOfRanks()>0 ) {
    filename += "-rank-" + std::to_string( tarch::mpi::Rank::getInstance().getRank() );
  }

  filename += ".csv";

  // only for stats
  int totalNumberOfEntries = 0;
  for (auto& p: _dataSetMap) {
    totalNumberOfEntries += p.second._data.size();
  }
  for (auto& p: _logMessageMap) {
    totalNumberOfEntries += p.second._data.size();
  }
  logInfo( "writeToCSV(string)", "write statistics to file " << filename << " (total no of entries=" << totalNumberOfEntries << ")" );

  std::ofstream file( filename );
  file << "t";
  for (auto& p: _dataSetMap) {
    file << ", " << p.first;
  }
  for (auto& p: _logMessageMap) {
    file << ", " << p.first;
  }
  file << std::endl;

  double t = 0.0;
  while (t<std::numeric_limits<double>::max()) {
    // find minimum time stamp over both types of logs
    t   = std::numeric_limits<double>::max();

    auto computeT = [&](double currentT) {
      t  =  std::min( t,  currentT );
    };

    for (auto& p: _dataSetMap) {
      if (not p.second._data.empty()) {
        computeT( std::get<0>(p.second._data.front()) );
      }
    } 
    for (auto& p: _logMessageMap) {
      if (not p.second._data.empty()) {
        computeT( std::get<0>(p.second._data.front()) );
      }
    }

    // plot data. If an entry is a fit, remove it from sequence
    if (t<std::numeric_limits<double>::max()) {
      file << t; 
      for (auto& p: _dataSetMap) {
        if (
          not p.second._data.empty()
          and
          tarch::la::smallerEquals( std::get<0>(p.second._data.front()), t )
        ) {
          file << ", (" << std::get<1>(p.second._data.front())
              << "/" << std::get<2>(p.second._data.front())
              << "/" << std::get<3>(p.second._data.front())
              << "/#" << std::get<4>(p.second._data.front())
              << ")";
          p.second._data.erase(p.second._data.begin());
        }
        else {
          file << ", ";
        }
      }
      for (auto& pp: _logMessageMap) {
        if (
          not pp.second._data.empty()
          and
          tarch::la::smallerEquals( std::get<0>(pp.second._data.front()), t )
        ) {
          file << ", \"" << std::get<1>(pp.second._data.front()) << "\" ";
          pp.second._data.erase(pp.second._data.begin());
	  file << ", ";
        }
        else {
          file << ", ";
        }
      }
    }
    file << std::endl;
  }

  // clear maps. Within this clear, I get seg faults on some computers when the program terminates
  _dataSetMap.clear();
  _logMessageMap.clear();

  // close output file
  file.close();
  #else
  logWarning( "writeToCSV(string)", "no statistics available. Recompile with -DTrackStatistics for runtime sampling" );
  #endif
}
