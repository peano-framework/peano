#include <iostream>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <cstdlib>


// *** NOTE: These are the CUDA-based functions that the generated MLIR code call
//           to setup the streams, load and launch the kernels etc.
//           A similar MLIRbridgeROCM.cpp will be required for AMD ROCm support

extern "C" {
  typedef void* GPUModule;
  typedef void* GPUFunction;
  typedef void* GPUStream;
  
  GPUModule mgpuModuleLoad(const void* binary, size_t size);

  void mgpuModuleUnload(GPUModule module);

  GPUFunction mgpuModuleGetFunction(GPUModule module, const char* name);

  // Stream management
  GPUStream mgpuStreamCreate();

  void mgpuStreamDestroy(GPUStream stream);

  void mgpuStreamSynchronize(GPUStream stream);

  void mgpuLaunchKernel(GPUFunction func, 
                        int64_t gridDimX, int64_t gridDimY, int64_t gridDimZ,
                        int64_t blockDimX, int64_t blockDimY, int64_t blockDimZ,
                        int32_t smemSize,
                        GPUStream stream,
                        void** kernelParams, void** extraParams,
                        int64_t paramCount);



  void* mgpuMemAlloc(int64_t size, void* stream, int8_t flags);

  int32_t mgpuMemFree(void* ptr, void* stream);


void setupMLIRContextGPU(CUdevice targetDevice);

void releaseMLIRContextGPU(CUdevice targetDevice);

  }