#include "tarch/accelerator/GPUMemoryManager.h"

#if defined(GPUOffloadingOMP)

#include "tarch/Assertions.h"
#include "tarch/multicore/Lock.h"
#include "tarch/accelerator/accelerator.h"

#include <omp.h>

#if defined(OMPUnifiedSharedMemory)
#pragma omp requires unified_shared_memory
#endif

tarch::logging::Log  tarch::accelerator::GPUMemoryManager::_log( "tarch::accelerator::GPUMemoryManager" );
int tarch::accelerator::GPUMemoryManager::_allocatedMemory;

tarch::accelerator::GPUMemoryManager& tarch::accelerator::GPUMemoryManager::getInstance() {
  static tarch::accelerator::GPUMemoryManager singleton;
  return singleton;
}

void* tarch::accelerator::GPUMemoryManager::allocateRawData( int size, int device ) {
  #if defined(OMPUnifiedSharedMemory)
  return malloc(size);
  #else
  tarch::multicore::Lock lock(_semaphore);

  // Is device initialised?
  if ( _deviceMemory.count(device)==0 ) {
    _deviceMemory.insert( std::pair<int, DeviceMemory>(device,DeviceMemory()) );
  }

  // Is a free page left? If not, insert one
  if ( _deviceMemory.at(device).freePages[size].empty() ) {
    _deviceMemory.at(device).freePages[size].push_back(
      omp_target_alloc(size, device)
    );
    assertion(_deviceMemory.at(device).freePages[size][0] != nullptr);
    _allocatedMemory += size;
  }

  // Grab fitting page
  void* result = _deviceMemory.at(device).freePages.at(size).back();
  _deviceMemory.at(device).freePages.at(size).pop_back();

  // Memorize page meta data, i.e. size and device
  assertion( _deviceMemory.at(device).usedPages.count(result)==0 );
  _deviceMemory.at(device).usedPages.insert( std::pair<void*,int>(result,size) );

  // Return result
  return result;
  #endif
}

void tarch::accelerator::GPUMemoryManager::free(void* devicePointer, int device) {
  #if defined(OMPUnifiedSharedMemory)
  free(devicePointer);
  #else
  tarch::multicore::Lock lock(_semaphore);

  assertion( _deviceMemory.count(device)==1 );
  assertion( _deviceMemory.at(device).usedPages.count(devicePointer)==1 );

  int size = _deviceMemory.at(device).usedPages.at(devicePointer);
  _deviceMemory.at(device).usedPages.erase(devicePointer);

  assertion( _deviceMemory.at(device).freePages.count(size)==1 );
  _deviceMemory.at(device).freePages.at(size).push_back( devicePointer );
  #endif
}

tarch::accelerator::GPUMemoryManager::~GPUMemoryManager() {
  clear();
}

void tarch::accelerator::GPUMemoryManager::clear() {
  tarch::multicore::Lock lock(_semaphore);

  for (auto& deviceMemory: _deviceMemory) {
    int device = deviceMemory.first;
    logDebug( "clear()", "free data of device " << device );

    if ( not deviceMemory.second.usedPages.empty() ) {
      logError( "clear()", "used pages on device " << device << " are not empty. " << deviceMemory.second.usedPages.size() << " memory blocks still in use" );
    }

    for (auto& p: deviceMemory.second.freePages)
    for (auto& pp: p.second) {
      omp_target_free(pp, device);
    }
  }

  _deviceMemory.clear();
  logInfo("clear()", "Total device memory allocated: " << _allocatedMemory);
}

void tarch::accelerator::GPUMemoryManager::copyData(void* destination, void* source, size_t length, tarch::accelerator::DeviceType destinationDevice, tarch::accelerator::DeviceType sourceDevice){
  #if defined(OMPUnifiedSharedMemory)
  memcpy(destination, source, length);
  #else
  omp_target_memcpy(destination, source, length, 0, 0, destinationDevice, sourceDevice);
  #endif
}
#endif
