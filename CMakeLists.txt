# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
cmake_minimum_required(VERSION 3.21)

set(PEANO_PROJECT_NAME "Peano")
set(PEANO_PROJECT_DESCRIPTION "Peano is an open source framework for solvers on dynamically adaptive Cartesian meshes")
set(PEANO_AUTHOR_ORGANIZATION "Durham University and Technische Universitaet Muenchen")
set(PEANO_AUTHOR_DOMAIN "www.peano-framework.org")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GetGitRevisionDescription)
get_git_head_revision(GIT_BRANCH GIT_COMMIT_HASH)
get_git_commit_message(GIT_COMMIT_MESSAGE)
get_git_commit_timestamp(GIT_COMMIT_TIMESTAMP)

set(PEANO_VERSION_REVISION "${GIT_COMMIT_HASH}")
set(PEANO_GIT_BRANCH "${GIT_BRANCH}")
set(PEANO_GIT_HASH "${GIT_COMMIT_HASH}")
set(PEANO_GIT_COMMIT_MESSAGE "${GIT_COMMIT_MESSAGE}")
set(PEANO_GIT_URL "https://github.com/tobiasweinzierl/peano")
message(STATUS "Revision: ${GIT_BRANCH} -m \"${GIT_COMMIT_MESSAGE}\" (${GIT_COMMIT_HASH}) ${GIT_COMMIT_TIMESTAMP}")

project(${PEANO_PROJECT_NAME}
  DESCRIPTION ${PEANO_PROJECT_DESCRIPTION}
  HOMEPAGE_URL ${PEANO_AUTHOR_DOMAIN}
  LANGUAGES
    C
    CXX
    #Fortran
)

set(CMAKE_C_STANDARD 17 CACHE STRING "")
set(CMAKE_C_STANDARD_REQUIRED TRUE)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_INLINES_HIDDEN YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE INTERNAL "")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

file(WRITE "${PROJECT_BINARY_DIR}/.gitignore" "*")

include(GNUInstallDirs)
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(Ccache)
include(CheckIncludeFiles)
include(CheckIncludeFileCXX)

include(TestBigEndian)
test_big_endian(BIG_ENDIAN)

include(ProcessorCount)
ProcessorCount(AVAILABLE_PROCESSOR_COUNT)

include(CTest)
enable_testing()

# https://stackoverflow.com/questions/31355692/how-do-i-enable-link-time-optimization-lto-with-cmake
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED)
  message(STATUS "IPO/LTO enabled")
  set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION 1)
else()
  #message(WARNING "IPO/LTO not supported: ${IPO_ERROR}")
  message(WARNING "IPO/LTO not supported")
endif()

add_library(PeanoOptions INTERFACE)
target_compile_features(PeanoOptions INTERFACE cxx_std_${CMAKE_CXX_STANDARD})
install(TARGETS PeanoOptions EXPORT PeanoTargets)

if(${CMAKE_CXX_COMPILER_ID} MATCHES "NVHPC|PGI")
  target_compile_options(PeanoOptions INTERFACE
    -W
    -Wall
    -Wextra
    --diag_suppress=unrecognized_attribute
  )
else()
  target_compile_options(PeanoOptions INTERFACE
    -w
    -W
    -Wall
    -Wextra
    -Wpedantic
    -Wno-attributes
    -Wnon-virtual-dtor
    -Wreturn-type
    -Wno-system-headers
    $<$<CONFIG:Debug>:-O0;-ggdb;-fno-elide-constructors>
  )
endif()

option(WITH_MPI "Build with Message Passing Interface (MPI) support" OFF)
option(WITH_TARGETDART "Build with targetDART support" OFF)
option(WITH_OTTER "Build with Otter support" OFF)
option(WITH_PETSC "Build with PETSc support" OFF)
option(WITH_LIBXSMM "Build with LIBXSMM support" OFF)
option(WITH_VTK "Build with VTK support" OFF)
option(WITH_HDF5 "Build with HDF5 support" OFF)
option(WITH_GSL "Build with GSL support (e.g. for ExaGRyPE's TwoPunctures)" OFF)
option(WITH_NETCDF "Build with NetCDF support" OFF)
option(WITH_EASI "Build with easi support" OFF)

option(ENABLE_PARTICLES "Enable the particles toolbox" OFF)
option(ENABLE_FINITEELEMENTS "Enable the finite elements toolbox" OFF)
option(ENABLE_BLOCKSTRUCTURED "Enable the blockstructured toolbox" OFF)
option(ENABLE_LOADBALANCING "Enable the loadbalancing toolbox" OFF)
option(ENABLE_CURVI "Enable the Curvi toolbox" OFF)
option(ENABLE_EXAHYPE "Enable ExaHyPE 2" OFF)
option(ENABLE_EXAGRYPE "Enable ExaGRyPE (3d only, requires ExaHyPE 2)" OFF)
option(ENABLE_MGHYPE "Enable MGHyPE" OFF)
option(ENABLE_EXAPARTY "Enable ExaParty" OFF)
option(ENABLE_CCZ4_AHF_POSTPROCESSOR "Build ExaGRyPE's apparent-horizon-finder postprocessing executable (requires ExaGRyPE)" OFF)

list(APPEND AVAILABLE_TOOLCHAINS
  "no"
  "intel"
  "nvidia"
  "scorep"
  "chrome"
  "gnu"
)
set(WITH_TOOLCHAIN "no" CACHE STRING "${AVAILABLE_TOOLCHAINS}")
set_property(CACHE WITH_TOOLCHAIN PROPERTY STRINGS ${AVAILABLE_TOOLCHAINS})

if(WITH_TOOLCHAIN STREQUAL "scorep")
  find_library(PAPI libpapi.a papi)
  find_library(CUBE4W libcube4w.a cube4w)
  target_link_libraries(PeanoOptions INTERFACE ${PAPI} ${CUBE4W} ${SCOREP_LIBRARIES})
  target_link_options(PeanoOptions INTERFACE ${SCOREP_LINK_OPTION})
  target_compile_options(PeanoOptions INTERFACE ${SCOREP_COMPILE_FLAGS})
  target_compile_definitions(PeanoOptions INTERFACE UseScoreP UseLogService=ScorePLogger)
elseif(WITH_TOOLCHAIN STREQUAL "itac")
  target_link_libraries(PeanoOptions INTERFACE VT::VT)
  target_include_directories(PeanoOptions SYSTEM INTERFACE ${VT_INCLUDE_DIRS})
  target_link_directories(PeanoOptions INTERFACE ${VT_LIBRARIES_DIR})
  target_compile_definitions(PeanoOptions INTERFACE UseITAC UseLogService=ITACLogger)
elseif(WITH_TOOLCHAIN STREQUAL "nvidia")
  check_include_files(nvToolsExt.h HAVE_NVTOOLSEXT_H)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(PeanoOptions INTERFACE CUDA::nvToolsExt)
  target_compile_definitions(PeanoOptions INTERFACE UseNVTX UseLogService=NVTXLogger)
endif()

list(APPEND AVAILABLE_MULTITHREADING
  "no"
  "omp"
  "sycl"
  "tbb"
  "cpp"
)
set(WITH_MULTITHREADING "no" CACHE STRING "${AVAILABLE_MULTITHREADING}")
set_property(CACHE WITH_MULTITHREADING PROPERTY STRINGS ${AVAILABLE_MULTITHREADING})

if(WITH_MULTITHREADING STREQUAL "omp")
  find_package(OpenMP REQUIRED)
  check_include_files(omp.h HAVE_OMP_H)
  target_link_libraries(PeanoOptions INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(PeanoOptions INTERFACE SharedOMP)
  set(ONLY_IF_MULTITHREADING_FIND_DEPENDENCY "find_dependency(OpenMP)")
elseif(WITH_MULTITHREADING STREQUAL "tbb")
  find_package(TBB REQUIRED)
  target_include_directories(PeanoOptions INTERFACE ${TBB_INCLUDE_DIRS})
  target_compile_definitions(PeanoOptions INTERFACE ${TBB_DEFINITIONS})
  target_link_libraries(PeanoOptions INTERFACE ${TBB_LIBRARIES})
  target_compile_definitions(PeanoOptions INTERFACE SharedTBB)
elseif(WITH_MULTITHREADING STREQUAL "cpp")
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  target_link_libraries(PeanoOptions INTERFACE Threads::Threads)
  target_compile_definitions(PeanoOptions INTERFACE SharedCPP)
  set(ONLY_IF_MULTITHREADING_FIND_DEPENDENCY "find_dependency(Threads::Threads)")
elseif(WITH_MULTITHREADING STREQUAL "sycl")
  #check_include_file_cxx(sycl/sycl.hpp HAVE_CL_SYCL_HPP) # Needs -fsycl to find the header. TODO: Add globally
  target_compile_options(PeanoOptions INTERFACE -fsycl)
  target_link_options(PeanoOptions INTERFACE -fsycl)
  target_compile_definitions(PeanoOptions INTERFACE SharedSYCL)
else()
  set(ONLY_IF_MULTITHREADING_FIND_DEPENDENCY "# Multithreading not used")
endif()

list(APPEND AVAILABLE_ACCELERATORS
  "no"
  "omp"
  "sycl"
  "cpp"
)
set(WITH_GPU "no" CACHE STRING "${AVAILABLE_ACCELERATORS}")
set_property(CACHE WITH_GPU PROPERTY STRINGS ${AVAILABLE_ACCELERATORS})

list(APPEND AVAILABLE_ACCELERATOR_ARCHITECTURES
  "none"
  "sm_70"
  "sm_72"
  "sm_75"
  "sm_80"
  "sm_86"
  "sm_89"
  "sm_90"
  "gfx906"
  "gfx908"
  "gfx90a"
  "gfx942"
  "spir64"
)
set(WITH_GPU_ARCH "none" CACHE STRING "${AVAILABLE_ACCELERATOR_ARCHITECTURES}")
set_property(CACHE WITH_GPU_ARCH PROPERTY STRINGS ${AVAILABLE_ACCELERATOR_ARCHITECTURES})

if(WITH_GPU_ARCH MATCHES "^sm_")
  set(PEANO_NVIDIA_GPU TRUE)
elseif(WITH_GPU_ARCH MATCHES "^gfx")
  set(PEANO_AMD_GPU TRUE)
elseif(WITH_GPU_ARCH MATCHES "^spir")
  set(PEANO_INTEL_GPU TRUE)
endif()

if(NOT WITH_GPU STREQUAL "no" AND WITH_GPU_ARCH STREQUAL "none")
  message(FATAL_ERROR "No GPU architecture chosen. Make sure to also set WITH_GPU_ARCH")
endif()

if(WITH_GPU STREQUAL "omp")
  include(OpenMPTarget)
elseif(WITH_GPU STREQUAL "cpp")
  include(StdPar)
elseif(WITH_GPU STREQUAL "sycl")
  include(SYCL)
else()
  set(ONLY_IF_GPU_FIND_DEPENDENCY "# GPU not used")
endif()

list(APPEND PEANO_DIMENSIONS
  "2"
  "3"
)

set(CMAKE_BUILD_TYPE "" CACHE INTERNAL "")
list(APPEND PEANO_BUILD_TYPES
  "Release"
  "Debug"
  "Trace"
  "Asserts"
  "Stats"
)

set(PEANO_CXX_FLAGS_RELEASE -O3 CACHE INTERNAL "")
set(PEANO_RELEASE_DEFINITIONS -DPeanoDebug=0 -DNDEBUG CACHE INTERNAL "")

set(PEANO_CXX_FLAGS_DEBUG -g -O0 CACHE INTERNAL "")
set(PEANO_DEBUG_DEFINITIONS -DPeanoDebug=4 CACHE INTERNAL "")

set(PEANO_CXX_FLAGS_STATS -O3 CACHE INTERNAL "")
set(PEANO_STATS_DEFINITIONS -DPeanoDebug=0 -DNDEBUG -DTrackStatistics CACHE INTERNAL "")

set(PEANO_CXX_FLAGS_ASSERTS -g -O3 CACHE INTERNAL "")
set(PEANO_ASSERTS_DEFINITIONS -DPeanoDebug=2 CACHE INTERNAL "")

set(PEANO_CXX_FLAGS_TRACE -g -O2 CACHE INTERNAL "")
set(PEANO_TRACE_DEFINITIONS -DPeanoDebug=1 -DNDEBUG CACHE INTERNAL "")

if(WITH_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(PeanoOptions INTERFACE MPI::MPI_CXX)
  target_compile_definitions(PeanoOptions INTERFACE Parallel)
  set(ONLY_IF_MPI_FIND_DEPENDENCY "find_dependency(MPI)")
else()
  set(ONLY_IF_MPI_FIND_DEPENDENCY "# MPI not used")
endif()

if(WITH_OTTER)
  find_package(Otter CONFIG)
  target_link_libraries(PeanoOptions INTERFACE Otter::Otter)
  target_compile_definitions(PeanoOptions INTERFACE UseOtter)
  set(ONLY_IF_OTTER_FIND_DEPENDENCY "find_dependency(Otter)")
else()
  set(ONLY_IF_OTTER_FIND_DEPENDENCY "# Otter not used")
endif()

if(WITH_PETSC)
  find_package(PETSc REQUIRED)
  target_compile_definitions(PeanoOptions INTERFACE UsePETSc)
  target_link_libraries(PeanoOptions INTERFACE ${PETSc_LIBRARIES})
  target_include_directories(PeanoOptions INTERFACE ${PETSc_INCLUDE_DIRS})
endif()

if(WITH_TARGETDART)
  target_compile_definitions(PeanoOptions INTERFACE UseTargetDART)
endif()

if(WITH_LIBXSMM)
  find_package(LIBXSMM REQUIRED)
  target_link_libraries(PeanoOptions INTERFACE LIBXSMM::LIBXSMM)
  target_compile_definitions(PeanoOptions INTERFACE UseLibXSMM)
endif()

if(WITH_VTK)
  find_package(VTK REQUIRED)
  target_link_libraries(PeanoOptions INTERFACE VTK::CommonCore)
  target_compile_definitions(PeanoOptions INTERFACE UseVTK)
  set(ONLY_IF_VTK_FIND_DEPENDENCY "find_dependency(VTK)")
else()
  set(ONLY_IF_VTK_FIND_DEPENDENCY "# VTK not used")
endif()

if(WITH_GSL)
  find_package(GSL REQUIRED)
  target_link_libraries(PeanoOptions INTERFACE GSL::gsl GSL::gslcblas)
  target_compile_definitions(PeanoOptions INTERFACE UseGSL)
  set(ONLY_IF_GSL_FIND_DEPENDENCY "find_dependency(GSL)")
else()
  set(ONLY_IF_GSL_FIND_DEPENDENCY "# GSL not used")
endif()

if(WITH_HDF5)
  find_package(HDF5 REQUIRED COMPONENTS C CXX)
  target_link_libraries(PeanoOptions INTERFACE HDF5::HDF5)
  target_compile_definitions(PeanoOptions INTERFACE UseHDF5)
  set(ONLY_IF_HDF5_FIND_DEPENDENCY "find_dependency(HDF5 COMPONENTS C CXX)")
else()
  set(ONLY_IF_HDF5_FIND_DEPENDENCY "# HDF5 not used")
endif()

if(WITH_NETCDF)
  find_package(NetCDF REQUIRED)
  target_link_libraries(PeanoOptions INTERFACE ${NETCDF_LIBRARIES})
  target_include_directories(PeanoOptions INTERFACE ${NETCDF_INCLUDES})
  target_compile_definitions(PeanoOptions INTERFACE UseNetCDF)
endif()

if(WITH_EASI)
  find_package(easi REQUIRED)
  target_link_libraries(PeanoOptions INTERFACE easi::easi)
  target_compile_definitions(PeanoOptions INTERFACE UseEasi UseAsagi)
  set(ONLY_IF_EASI_FIND_DEPENDENCY "find_dependency(easi)")
else()
  set(ONLY_IF_EASI_FIND_DEPENDENCY "# easi not used")
endif()

if(ENABLE_LOADBALANCING)
  target_compile_definitions(PeanoOptions INTERFACE UseToolboxLoadbalancing)
endif()

if(ENABLE_BLOCKSTRUCTURED)
  target_compile_definitions(PeanoOptions INTERFACE UseToolboxBlockstructured)
endif()

if(ENABLE_PARTICLES)
  target_compile_definitions(PeanoOptions INTERFACE UseToolboxParticles)
endif()

if(ENABLE_MULTIPRECISION)
  target_compile_definitions(PeanoOptions INTERFACE UseToolboxMultiprecision)
endif()

if(ENABLE_FINITEELEMENTS)
  target_compile_definitions(PeanoOptions INTERFACE UseToolboxFiniteElements)
endif()

if(ENABLE_CURVI)
  target_compile_definitions(PeanoOptions INTERFACE UseToolboxCurvi)
endif()

if(ENABLE_EXAHYPE)
  target_compile_definitions(PeanoOptions INTERFACE UseExaHyPE)
endif()

if(ENABLE_EXAGRYPE)
  if(NOT ENABLE_EXAHYPE)
    message(FATAL_ERROR "ExaGRyPE requires ExaHyPE 2 (set -DENABLE_EXAHYPE=ON)")
  endif()
  target_compile_definitions(PeanoOptions INTERFACE UseExaGRyPE)
endif()

if(ENABLE_EXAPARTY)
  target_compile_definitions(PeanoOptions INTERFACE UseExaParty)
endif()

target_include_directories(PeanoOptions INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_subdirectory(src)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${PEANO_PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PEANO_VERSION_REVISION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake/Config.cmake.in
  ${CMAKE_BINARY_DIR}/${PEANO_PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}
)

install(DIRECTORY
  ${CMAKE_SOURCE_DIR}/python
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

install(FILES
  ${CMAKE_SOURCE_DIR}/LICENSE
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

export(EXPORT ${PEANO_PROJECT_NAME}Targets
  FILE ${CMAKE_BINARY_DIR}/${PEANO_PROJECT_NAME}Targets.cmake
  NAMESPACE ${PEANO_PROJECT_NAME}::
)

install(EXPORT ${PEANO_PROJECT_NAME}Targets
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  NAMESPACE ${PEANO_PROJECT_NAME}::
)

install(FILES
  ${CMAKE_BINARY_DIR}/${PEANO_PROJECT_NAME}Config.cmake
  ${CMAKE_BINARY_DIR}/${PEANO_PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)
