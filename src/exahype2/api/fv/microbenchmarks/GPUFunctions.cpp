#include <iostream>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include "GPUFunctions.h"
#include "tarch/accelerator/GPUMemoryManager.h"

// *** NOTE: These are the CUDA-based functions that the generated MLIR code call
//           to setup the streams, load and launch the kernels etc.
//           A similar MLIRbridgeROCM.cpp will be required for AMD ROCm support

extern "C" {
  typedef void* GPUModule;
  typedef void* GPUFunction;
  typedef void* GPUStream;
  
  GPUModule mgpuModuleLoad(const void* binary, size_t size) {
    if (!binary || size == 0) {
      std::cerr << "[CUDA ERROR] Invalid binary data!" << std::endl;
      return nullptr;
    }
    
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS && res != 15) {
      std::cerr << "[CUDA ERROR] Failed to initialize CUDA driver: " << (int)res << std::endl;
    }
    
    CUdevice device = 0;

    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
      std::cerr << "[CUDA ERROR] Failed to get device: " << (int)res << std::endl;
      return nullptr;
    }
 
    // Load PTX as CUDA module
    CUmodule cu_module = nullptr;
    res = cuModuleLoadData(&cu_module, binary);
    if (res != CUDA_SUCCESS) {
      std::cerr << "[CUDA ERROR] Failed to load module: " << (int)res << std::endl;
      return nullptr;
    }
    return (GPUModule)cu_module;
  }

  void mgpuModuleUnload(GPUModule module) {
    if (module) {
      cuModuleUnload((CUmodule)module);
    }
  }

  GPUFunction mgpuModuleGetFunction(GPUModule module, const char* name) {
    if (!module) {
      std::cerr << "[ERROR] Invalid module handle" << std::endl;
      return nullptr;
    }
    
    CUfunction cu_kernel = nullptr;
    CUresult res = cuModuleGetFunction(&cu_kernel, (CUmodule)module, name);
    if (res != CUDA_SUCCESS) {
      std::cerr << "[CUDA ERROR] Failed to get kernel: " << (int)res << std::endl;
      return nullptr;
    }
    return (GPUFunction)cu_kernel;
  }

  // Stream management
  GPUStream mgpuStreamCreate() {
    cudaStream_t stream = nullptr;

    CUresult res = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    if (res != CUDA_SUCCESS) {
      std::cerr << "[CUDA ERROR] Failed to create stream: " << (int)res << std::endl;
      return nullptr;
    }
    return (GPUStream)stream;
  }

  void mgpuStreamDestroy(GPUStream stream) {
    if (stream) {
      CUresult res = cuStreamDestroy((cudaStream_t)stream);
      if (res != CUDA_SUCCESS) {
        std::cerr << "[CUDA ERROR] Failed to destroy stream: " << (int)res << std::endl;
      }
    }
  }

  void mgpuStreamSynchronize(GPUStream stream) {
    if (stream) {
      CUresult res = cuStreamSynchronize((cudaStream_t)stream);
      if (res != CUDA_SUCCESS) {
        std::cerr << "[CUDA ERROR] Failed to synchronize stream: " << (int)res << std::endl;
      }
    }
  }

  void mgpuLaunchKernel(GPUFunction func, 
                        int64_t gridDimX, int64_t gridDimY, int64_t gridDimZ,
                        int64_t blockDimX, int64_t blockDimY, int64_t blockDimZ,
                        int32_t smemSize,
                        GPUStream stream,
                        void** kernelParams, void** extraParams,
                        int64_t paramCount) {
    if (!func) {
      std::cerr << "[ERROR] Kernel function handle is null!" << std::endl;
      return;
    }

    void** gpu_params = (void**)kernelParams;
    
    CUresult res = cuLaunchKernel(
        (CUfunction)func,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        smemSize,
        (cudaStream_t)stream,  
        gpu_params,
        nullptr
    );
    
    if (res != CUDA_SUCCESS) {
      const char* errStr = nullptr;
      cuGetErrorName(res, &errStr);
      std::cerr << "[ERROR] cuLaunchKernel failed: " << errStr << std::endl;
      return;
    }
  }



  void* mgpuMemAlloc(int64_t size, void* stream, int8_t flags) { 
    void* ptr = nullptr;

    cudaError_t res = cudaMallocManaged(&ptr, size);
    if (res != cudaSuccess) {
        std::cerr << "[CUDA ERROR] Failed to allocate GPU memory: " << cudaGetErrorString(res) << std::endl;
        return nullptr;
    }
    return ptr;
    
    return malloc(size);
  }

  int32_t mgpuMemFree(void* ptr, void* stream) {
      if (!ptr) return 0;

      cudaPointerAttributes attrs;
      cudaError_t attrRes = cudaPointerGetAttributes(&attrs, ptr);
      cudaGetLastError(); // clear error state regardless

      if (attrRes == cudaSuccess && attrs.type == cudaMemoryTypeManaged) {
          // Only free memory we actually allocated with cudaMallocManaged
          cudaError_t res = cudaFree(ptr);
          if (res != cudaSuccess) {
              std::cerr << "[CUDA ERROR] Failed to free GPU memory: "
                        << cudaGetErrorString(res) << std::endl;
              return -1;
          }
      } 
      return 0;
    tarch::accelerator::GPUMemoryManager::getInstance().free(ptr, deviceNumber);
    return 0;
  }

}


void setupMLIRContextGPU(CUdevice targetDevice) {
    cuInit(0);
    cuDeviceGet(&targetDevice, 0);

    CUcontext context;
    cuDevicePrimaryCtxRetain(&context, targetDevice);
    cuCtxSetCurrent(context);
}

void releaseMLIRContextGPU(CUdevice targetDevice) {
    cuDevicePrimaryCtxRelease(targetDevice);
}