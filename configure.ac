#
# Recreate build environment:
# ---------------------------
#   libtoolize
#   aclocal
#   autoconf
#   autoheader
#   automake --add-missing
#
# Summary:
#   libtoolize; aclocal; autoconf; autoheader; cp src/config.h.in .; automake --add-missing
#
#
# Make and test:
# ---------------------------
#   make -j
#   make check
#
#
# Cleanup
# ---------------------------
#   make clean
#   make distclean
#   git clean -xdf
#   rm -rf submodules
#
# Summary:
#   make clean && make distclean && git clean -xf && rm -rf submodules
#
#
AC_INIT([Peano], [4.0])
AC_CONFIG_MACRO_DIR([m4])

# https://stackoverflow.com/questions/3116645/default-compiler-flags-with-autotools
if test -z "$CXXFLAGS"; then
  CXXFLAGS=' '
fi
if test -z "$CFLAGS"; then
  CFLAGS=' '
fi

AM_INIT_AUTOMAKE(foreign -Wall -Werror)
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_CONFIG_HEADERS([src/config.h])

AC_CANONICAL_BUILD

AC_PROG_INSTALL
AC_PROG_MAKE_SET

#
# Default installation is current path
#
AC_PREFIX_DEFAULT("libs")

#
# This is just a sanity check whether the right files/directories do exist
#
AC_CONFIG_SRCDIR([src/peano4/peano.cpp])

#
# Pipe system-specific findings into a file config.h
#
AC_CONFIG_HEADERS([config.h])

#
# I wanna build libraries
#
AM_PROG_AR
LT_INIT
AC_CONFIG_MACRO_DIRS([m4])

#
# Checks for programs
#
AC_PROG_CXX
#AC_PROG_FC

AC_LANG(C++)

#AX_COMPILER_VENDOR

AR="ar"
# The "D" modifier requests deterministic archives. It is a GNU ar extension;
# BSD ar (the default on macOS) rejects it with "illegal option -- D". Probe for
# support and fall back to plain "rc" so the build works with either archiver.
AC_MSG_CHECKING([whether $AR accepts the D (deterministic) modifier])
: > conftest_ar.$$
if $AR rcD conftest_ar_$$.a conftest_ar.$$ >/dev/null 2>&1; then
  ARFLAGS="rcD"
  AC_MSG_RESULT([yes])
else
  ARFLAGS="rc"
  AC_MSG_RESULT([no])
fi
rm -f conftest_ar.$$ conftest_ar_$$.a
AC_SUBST(AR)
AC_SUBST(ARFLAGS)

srcdir_full_path=`cd $srcdir && pwd`

#
# ===============================================================
# Define all the compiler and linker variables of Peano
# ===============================================================
#
CXXFLAGS_PEANO_2D_RELEASE=" -DDimensions=2 -DPeanoDebug=0 "
CXXFLAGS_PEANO_2D_STATS="   -DDimensions=2 -DPeanoDebug=0 -DTrackStatistics "
CXXFLAGS_PEANO_2D_TRACE="   -DDimensions=2 -DPeanoDebug=1 -g "
CXXFLAGS_PEANO_2D_ASSERTS=" -DDimensions=2 -DPeanoDebug=2 -g "
CXXFLAGS_PEANO_2D_DEBUG="   -DDimensions=2 -DPeanoDebug=4 -g -O0 "

CXXFLAGS_PEANO_3D_RELEASE=" -DDimensions=3 -DPeanoDebug=0 "
CXXFLAGS_PEANO_3D_STATS="   -DDimensions=3 -DPeanoDebug=0 -DTrackStatistics "
CXXFLAGS_PEANO_3D_TRACE="   -DDimensions=3 -DPeanoDebug=1 -g "
CXXFLAGS_PEANO_3D_ASSERTS=" -DDimensions=3 -DPeanoDebug=2 -g "
CXXFLAGS_PEANO_3D_DEBUG="   -DDimensions=3 -DPeanoDebug=4 -g -O0 "

# Note: the flag "-g" is essentially a flag for the compiler driver.
# Given as a linker flag, the linker does not complain but it will be ignored.
# Taken from GNU ld documentation: https://ftp.gnu.org/old-gnu/Manuals/ld-2.9.1/html_node/ld_3.html
LDFLAGS_PEANO_RELEASE=" -L$srcdir_full_path/src "
LDFLAGS_PEANO_STATS="   -L$srcdir_full_path/src "
LDFLAGS_PEANO_ASSERTS=" -L$srcdir_full_path/src "
LDFLAGS_PEANO_TRACE="   -L$srcdir_full_path/src "
LDFLAGS_PEANO_DEBUG="   -L$srcdir_full_path/src "

LDADD_PEANO_2D_RELEASE=" -lPeano4Core2d         -lTarch"
LDADD_PEANO_2D_STATS="   -lPeano4Core2d_stats   -lTarch_stats"
LDADD_PEANO_2D_ASSERTS=" -lPeano4Core2d_asserts -lTarch_asserts"
LDADD_PEANO_2D_TRACE="   -lPeano4Core2d_trace   -lTarch_trace"
LDADD_PEANO_2D_DEBUG="   -lPeano4Core2d_debug   -lTarch_debug"

LDADD_PEANO_3D_RELEASE=" -lPeano4Core3d         -lTarch"
LDADD_PEANO_3D_STATS="   -lPeano4Core3d_stats   -lTarch_stats"
LDADD_PEANO_3D_ASSERTS=" -lPeano4Core3d_asserts -lTarch_asserts"
LDADD_PEANO_3D_TRACE="   -lPeano4Core3d_trace   -lTarch_trace"
LDADD_PEANO_3D_DEBUG="   -lPeano4Core3d_debug   -lTarch_debug"


#
# ===============================================================
# Parallelisation backends
#
# These configurations establish the multithreading, MPI and GPU
# environment.
# ===============================================================
#

#
# Multithreading
#
AC_ARG_WITH(
  multithreading,
  AS_HELP_STRING([--with-multithreading],[switch on multithreading [default=no,omp,tbb,tbb_extension,sycl,cpp]]),
  [
    AC_MSG_CHECKING(for multithreading)
    case "${withval}" in
      no)     AC_MSG_RESULT([no multithreading]);                    multithreading=false ;;
      omp)    AC_MSG_RESULT([build with OpenMP]);                    multithreading=omp ;;
      tbb)    AC_MSG_RESULT([build with TBB]);                       multithreading=tbb ;;
      tbb_extension)    AC_MSG_RESULT([build with TBB]);             multithreading=tbb_extension ;;
      sycl)   AC_MSG_RESULT([build with SYCL tasking on host]);      multithreading=sycl ;;
      cpp)    AC_MSG_RESULT([build with C++11 threading support]);   multithreading=cpp ;;
      *)      AC_MSG_FAILURE([default=no (no multithreading). Use one of the available targets]); multithreading=false ;;
    esac
  ],
  [
    AC_MSG_CHECKING(for multithreading);
    AC_MSG_RESULT(no); multithreading=false
  ]
)

if test "$multithreading" == cpp; then
  AC_DEFINE([SharedCPP], [1], [SharedCPP])
  AC_SEARCH_LIBS(pthread_create,pthread)
fi

if test "$multithreading" == tbb; then
  AC_DEFINE([SharedTBB], [1], [SharedTBB])
  AC_DEFINE([SharedTBB], [1], [SharedTBB])
  AC_DEFINE([SharedTBB], [1], [SharedTBB])
fi

if test "$multithreading" == tbb_extension; then
  AC_DEFINE([SharedTBB], [1], [SharedTBB])
  AC_DEFINE([SharedTBBExtension], [1], [SharedTBBExtension])

  CXXFLAGS_PEANO_2D_STATS="   $CXXFLAGS_PEANO_2D_STATS   -Isrc/tbb"
  CXXFLAGS_PEANO_2D_RELEASE=" $CXXFLAGS_PEANO_2D_RELEASE -Isrc/tbb"
  CXXFLAGS_PEANO_2D_ASSERTS=" $CXXFLAGS_PEANO_2D_ASSERTS -DTBB_USE_ASSERT -Isrc/tbb"
  CXXFLAGS_PEANO_2D_TRACE="   $CXXFLAGS_PEANO_2D_TRACE   -DTBB_USE_ASSERT -Isrc/tbb"
  CXXFLAGS_PEANO_2D_DEBUG="   $CXXFLAGS_PEANO_2D_DEBUG   -DTBB_USE_ASSERT -Isrc/tbb"

  CXXFLAGS_PEANO_3D_STATS="   $CXXFLAGS_PEANO_3D_STATS   -Isrc/tbb"
  CXXFLAGS_PEANO_3D_RELEASE=" $CXXFLAGS_PEANO_3D_RELEASE -Isrc/tbb"
  CXXFLAGS_PEANO_3D_ASSERTS=" $CXXFLAGS_PEANO_3D_ASSERTS -DTBB_USE_ASSERT -Isrc/tbb"
  CXXFLAGS_PEANO_3D_TRACE="   $CXXFLAGS_PEANO_3D_TRACE   -DTBB_USE_ASSERT -Isrc/tbb"
  CXXFLAGS_PEANO_3D_DEBUG="   $CXXFLAGS_PEANO_3D_DEBUG   -DTBB_USE_ASSERT -Isrc/tbb"

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/tbb "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/tbb "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/tbb "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/tbb "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/tbb "

  LDADD_PEANO_2D_RELEASE+=" -ltbb_extension"
  LDADD_PEANO_2D_STATS+="   -ltbb_extension"
  LDADD_PEANO_2D_ASSERTS+=" -ltbb_extension_debug"
  LDADD_PEANO_2D_TRACE+="   -ltbb_extension_trace"
  LDADD_PEANO_2D_DEBUG+="   -ltbb_extension_debug"

  LDADD_PEANO_3D_RELEASE+=" -ltbb_extension"
  LDADD_PEANO_3D_STATS+="   -ltbb_extension"
  LDADD_PEANO_3D_ASSERTS+=" -ltbb_extension_debug"
  LDADD_PEANO_3D_TRACE+="   -ltbb_extension_trace"
  LDADD_PEANO_3D_DEBUG+="   -ltbb_extension_debug"
fi
AM_CONDITIONAL([UseTBBExtension], [test "x$multithreading" = xtbb_extension])

if test "$multithreading" == omp; then
  AC_CHECK_HEADERS([omp.h],[],[AC_MSG_WARN([header <omp.h> not found])])
  AC_DEFINE([SharedOMP], [1], [SharedOMP])
fi

if test "$multithreading" == sycl; then
  AC_CHECK_HEADERS([CL/sycl.hpp],[],[AC_MSG_WARN([header <CL/sycl.hpp> not found])])
  AC_DEFINE([SharedSYCL], [1], [SharedSYCL])
fi


#
# MPI Options
#
# Logic: user provides mpi compiler command
#        check if that command exists
#        overwrite CXX to be the user supplied value
#
AC_ARG_WITH(mpi,
  AS_HELP_STRING([--with-mpi],[switch on distributed memory parallelisation [default=no, mpiCC command]]),
  [ AC_MSG_CHECKING(for mpi)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);                              mpi=false ;;
      *)     AC_MSG_RESULT(use mpi c++ command ${withval});  mpi=${withval}  ;;
    esac ],
  [ AC_MSG_CHECKING(for mpi); AC_MSG_RESULT(no); mpi=false ])

#AC_DEFUN([AC_PROG_USERMPI], [AC_CHECK_PROG(USERMPI,"${mpi}",yes)])

if test "$mpi" != false; then
  CXX=$mpi
  AC_DEFINE([Parallel], [1], [Parallel])

  AC_CHECK_PROG( USERMPI, $mpi, "yes" )
  if test x"$USERMPI" != x"yes" ; then
    AC_MSG_WARN([The specified MPI compiler (${mpi}) was not found in the path. If it is a global path, please ignore this message.])
  fi
fi


#
# GPU support
#
AC_ARG_WITH(
  gpu,
  AS_HELP_STRING([--with-gpu],[build with the GPU support. Not all targets are supported for all programming models, but you will need some threading model enabled [default=no,omp,hip,sycl,cpp]]),
  [
    AC_MSG_CHECKING(for gpu)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);                                             gpu=false ;;
      omp)   AC_MSG_RESULT(build with OpenMP offloading to GPUs);           gpu=omp ;;
      omp_unified_memory)   AC_MSG_RESULT(build with OpenMP offloading to GPUs, using unified memory);           gpu=omp ;;
      sycl)  AC_MSG_RESULT(build with SYCL GPU support);                    gpu=sycl ;;
      cpp)   AC_MSG_RESULT(build with C++ GPU support);                     gpu=cpp ;;
      hip)   AC_MSG_RESULT(build with HIP GPU support);                     gpu=hip ;;
      *)     AC_MSG_FAILURE([use omp, sycl, cpp or hip to define target]);  gpu=false ;;
    esac
  ],
  [
    AC_MSG_CHECKING(for gpu);
    AC_MSG_RESULT(no); gpu=false
  ]
)
AC_ARG_WITH(
  gpu_vendor,
  AS_HELP_STRING([--with-gpu-vendor],[select which GPU vendor is being used [default=unset,amd,intel,nvidia]]),
  [
    AC_MSG_CHECKING(for gpu_vendor)
    case "${withval}" in
      unset)    AC_MSG_RESULT(Do not set GPU vendor);                                             gpu_vendor=unset ;;
      amd)    AC_MSG_RESULT(build with AMD GPU support);                                             gpu_vendor=amd ;;
      intel)   AC_MSG_RESULT(build with Intel GPU support);           gpu_vendor=intel ;;
      nvidia)  AC_MSG_RESULT(build with NVIDIA GPU support);                    gpu_vendor=nvidia ;;
      *)     AC_MSG_FAILURE([use amd, intel or nvidia to define target]);  gpu_vendor=unset ;;
    esac
  ],
  [
    AC_MSG_CHECKING(for gpu_vendor);
    AC_MSG_RESULT(unset); gpu_vendor=unset
  ]
)

GPU_VENDOR=""
if test "$gpu_vendor" == amd; then
  GPU_VENDOR="AMD"
  
  if test "$gpu" == omp; then
    CXXFLAGS="$CXXFLAGS -fopenmp-targets=amdgcn-amd-amdhsa"
    LDFLAGS="$LDFLAGS -fopenmp-targets=amdgcn-amd-amdhsa"
  fi
  if test "$gpu" == sycl; then
    CXXFLAGS="$CXXFLAGS -fsycl-targets=amdgcn-amd-amdhsa"
    LDFLAGS="$LDFLAGS -fsycl-targets=amdgcn-amd-amdhsa"
  fi
fi

if test "$gpu_vendor" == intel; then
  GPU_VENDOR="INTEL"
  if test "$gpu" == omp; then
    CXXFLAGS="$CXXFLAGS -fopenmp-targets=spir64"
    LDFLAGS="$LDFLAGS -fopenmp-targets=spir64"
  fi
  if test "$gpu" == sycl; then
    CXXFLAGS="$CXXFLAGS -fsycl-targets=spir64"
    LDFLAGS="$LDFLAGS -fsycl-targets=spir64"
  fi
fi

if test "$gpu_vendor" == nvidia; then
  GPU_VENDOR="NVIDIA"
  if test "$gpu" == omp; then
    CXXFLAGS="$CXXFLAGS -fopenmp-targets=nvptx64-nvidia-cuda"
    LDFLAGS="$LDFLAGS -fopenmp-targets=nvptx64-nvidia-cuda"
  fi
  if test "$gpu" == sycl; then
    CXXFLAGS="$CXXFLAGS -fsycl-targets=nvptx64-nvidia-cuda"
    LDFLAGS="$LDFLAGS -fsycl-targets=nvptx64-nvidia-cuda"
  fi
fi

AC_ARG_WITH(
  gpu_arch,
  AS_HELP_STRING([--with-gpu-arch],[select which GPU architecture is being used, default=unset]),
  [
    AC_MSG_CHECKING(for gpu_arch)
    case "${withval}" in
      unset)    AC_MSG_RESULT(Do not set GPU architecture);                                             gpu_arch=unset ;;
      *)     AC_MSG_RESULT(GPU architecture set to ${withval});  gpu_arch=${withval} ;;
    esac
  ],
  [
    AC_MSG_CHECKING(for gpu_arch);
    AC_MSG_RESULT(unset); gpu_arch=unset
  ]
)

GPU_ARCH=""
if test "$gpu_arch" != unset; then
  CXXFLAGS="$CXXFLAGS --offload-arch=${gpu_arch}"
  LDFLAGS="$LDFLAGS --offload-arch=${gpu_arch}"
  GPU_ARCH=${gpu_arch}
fi

AC_SUBST(GPU_VENDOR)
AC_SUBST(GPU_ARCH)

if test "$gpu" == omp; then
  AC_CHECK_HEADERS([omp.h],[],[AC_MSG_WARN([header <omp.h> not found])])
  AC_DEFINE([GPUOffloadingOMP], [1], [GPUOffloadingOMP])
fi

if test "$gpu" == omp_unified_memory; then
  AC_CHECK_HEADERS([omp.h],[],[AC_MSG_WARN([header <omp.h> not found])])
  AC_DEFINE([GPUOffloadingOMP], [1], [GPUOffloadingOMP])
  AC_DEFINE([OMPUnifiedSharedMemory], [1], [OMPUnifiedSharedMemory])
fi

if test "$gpu" == sycl; then
  AC_DEFINE([GPUOffloadingSYCL], [1], [GPUOffloadingSYCL])
fi

if test "$gpu" == hip; then
  AC_DEFINE([GPUOffloadingHIP], [1], [GPUOffloadingHIP])
fi

if test "$gpu" == cpp; then
  AC_DEFINE([GPUOffloadingCPP], [1], [GPUOffloadingCPP])
fi

#
# PETSc
#
AC_ARG_WITH(petsc,
  AS_HELP_STRING([--with-petsc],[switch on PETSc support]),
  [ AC_MSG_CHECKING(for petsc)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);                              petsc=false ;;
      *)     AC_MSG_RESULT(yes);                             petsc=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for petsc); AC_MSG_RESULT(no); petsc=false ])
AM_CONDITIONAL([UsePETSc], [test "x$petsc" = xtrue])

#
# ===============================================================
# Postprocessing and I/O
# ===============================================================
#

#
# VTK
#
AC_ARG_WITH(vtk,
  AS_HELP_STRING([--with-vtk],[enable VTK and thus build Peano's conversion tools [default=no, prefix]. prefix is where VTK include files are to be found, such as /usr/local/include/vtk-8.90]),
  [ AC_MSG_CHECKING(for vtk)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);     vtk_include=false ;;
      yes)   AC_MSG_RESULT(build against VTK installation in default location (/usr/include));    vtk_include="/usr/include"  ;;
      *)     AC_MSG_RESULT(build against VTK installation);    vtk_include=${withval}  ;;
    esac ],
  [ AC_MSG_CHECKING(for vtk); AC_MSG_RESULT(no); vtk_include=false ])

AM_CONDITIONAL([UseVTK],[test "$vtk_include" != false])

if test "$vtk_include" != false; then
  AC_DEFINE([UseVTK], [1], [UseVTK])
  AC_CHECK_FILE([$vtk_include/vtkVersion.h], [vtkFound="OK"])
  VTK_INCLUDE=-I$vtk_include
  AC_MSG_RESULT(use VTK include argument $VTK_INCLUDE)

  AC_SUBST([VTK_INCLUDE])

  #
  # If you specify a version, then I use this one. Otherwise, I search $vtk_include
  #
  AC_ARG_WITH(vtk_version,
    AS_HELP_STRING([--with-vtk-version],[specify vtk version (major) [default=autodetect]]),
    [ AC_MSG_CHECKING(for vtk-version)
      case "${withval}" in
      *)     AC_MSG_RESULT(build against VTK version ${withval});    vtk_version=${withval}  ;;
      esac ],
    [ AC_MSG_CHECKING(for vtk-version); AC_MSG_RESULT(not set - try to autodetect); vtk_version="autodetect" ])

  if test x$vtk_version == x"autodetect"; then
    vtk_version_line=$(grep "define VTK_MAJOR_VERSION " $vtk_include/*.h)
    vtk_version=$(echo $vtk_version_line  | rev | cut -d' ' -f 1 | rev)

    if test x$vtk_version != "x"; then
      AC_MSG_NOTICE([detected VTK version $vtk_version])
    else
      AC_MSG_ERROR([no VTK manually specified and no VTK version could be detected at VTK prefix '$vtk_include'!])
    fi
  fi

  AC_ARG_WITH(vtk-suffix,
    AS_HELP_STRING([--with-vtk-suffix],[specify which suffix the vtk libraries carry [default=autodetect, use for no prefix]]),
    [ AC_MSG_CHECKING(for vtk-suffix)
      case "${withval}" in
        yes)   AC_MSG_RESULT(assume VTK libraries have no suffix);    vtk_suffix=""  ;;
        no)    AC_MSG_RESULT(not set - try to autodetect); vtk_suffix="autodetect";;
        *)     AC_MSG_RESULT(use suffix ${withval});    vtk_suffix=${withval}  ;;
      esac
    ],
    [ AC_MSG_CHECKING(for vtk-suffix); AC_MSG_RESULT(not set - try to autodetect); vtk_suffix="autodetect" ])

  #
  # So far, every VTK version we had tested had vtkIOCore, so we test this one. On the
  # long term, there might be a need for ifs here checking the version.
  #
  # You have to very careful with the spaces here. A missing space before/after the ==
  # destroys the semantics. The x that I add before rhs and lhs of the comparison is
  # a cheap trick. It ensures that the expression is valid even if vtk_suffix is
  # empty
  #

  if test x$vtk_suffix == x"autodetect"; then
    vtk_version_line=$(grep "define VTK_MAJOR_VERSION " $vtk_include/*.h)
    vtk_major_version=$(echo $vtk_version_line  | rev | cut -d' ' -f 1 | rev)
    vtk_version_line=$(grep "define VTK_MINOR_VERSION " $vtk_include/*.h)
    vtk_minor_version=$(echo $vtk_version_line  | rev | cut -d' ' -f 1 | rev)
    vtk_suffix=-$vtk_major_version.$vtk_minor_version
    AC_MSG_RESULT(assume library suffix is $vtk_suffix)
  fi

  AC_CHECK_LIB(vtkIOCore$vtk_suffix,main,,[AC_MSG_ERROR([vtk library not found])])
  VTK_LIBRARY_SUFFIX=$vtk_suffix
  AC_SUBST([VTK_LIBRARY_SUFFIX])
fi

AM_CONDITIONAL([VTK_LIBRARY_VERSION_7],[test "$vtk_version" = "7"])
AM_CONDITIONAL([VTK_LIBRARY_VERSION_8],[test "$vtk_version" = "8"])
AM_CONDITIONAL([VTK_LIBRARY_VERSION_9],[test "$vtk_version" = "9"])


#
# HDF5
#
AC_ARG_WITH(hdf5,
  AS_HELP_STRING([--with-hdf5],[enable HDF5 support [default=no]]),
  [ AC_MSG_CHECKING(for hdf5)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);   use_hdf5=false ;;
      *)     AC_MSG_RESULT(yes);  use_hdf5=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for hdf5); AC_MSG_RESULT(no); use_hdf5=false ])
AM_CONDITIONAL([UseHDF5],[test "x$use_hdf5" = xtrue])

if test "$use_hdf5" == true; then
  AC_DEFINE([UseHDF5], [UseHDF5], [UseHDF5])
fi


#
# GSL (GNU Scientific Library), e.g. for ExaGRyPE's TwoPunctures initial data
#
AC_ARG_WITH(gsl,
  AS_HELP_STRING([--with-gsl],[enable GSL support [default=no]]),
  [ AC_MSG_CHECKING(for gsl)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);   use_gsl=false ;;
      *)     AC_MSG_RESULT(yes);  use_gsl=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for gsl); AC_MSG_RESULT(no); use_gsl=false ])
AM_CONDITIONAL([UseGSL],[test "x$use_gsl" = xtrue])

if test "$use_gsl" == true; then
  AC_DEFINE([UseGSL], [UseGSL], [UseGSL])
fi


#
# NetCDF
#
AC_ARG_WITH(netcdf,
  AS_HELP_STRING([--with-netcdf],[enable NetCDF [default=false]]),
  [ AC_MSG_CHECKING(for netcdf)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    use_netcdf=false ;;
      *)     AC_MSG_RESULT(yes);   use_netcdf=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for netcdf); AC_MSG_RESULT(no); use_netcdf=false ])
AM_CONDITIONAL([UseNetCDF],[test "x$use_netcdf" = xtrue])

if test "$use_netcdf" == true; then
  AC_DEFINE([UseNetCDF], [UseNetCDF], [UseNetCDF])
fi


#
# Otter
#
AC_ARG_WITH(otter,
  AS_HELP_STRING([--with-otter],[enable Otter [default=false]]),
  [ AC_MSG_CHECKING(for otter)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    use_otter=false ;;
      *)     AC_MSG_RESULT(yes);   use_otter=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for otter); AC_MSG_RESULT(no); use_otter=false ])
AM_CONDITIONAL([UseOtter],[test "x$use_otter" = xtrue])

# internal toolboxes don't need ifdefs, but here we need them
if test "$use_otter" == true; then
  AC_DEFINE([UseOtter], [UseOtter], [UseOtter])
fi


#
# OPENBLAS
# This is needed by LAPACK and libxsmmjit, so we enable/disable
# here, make sure we add "-lopenblas" to LIBS, and do not
# allow configuring --with-lapack or --with-libxsmm without it.
#
AC_ARG_WITH(openblas,
  AS_HELP_STRING([--with-openblas],[enable OpenBLAS [default=false]]),
  [ AC_MSG_CHECKING(for openblas)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    use_openblas=false ;;
      *)     AC_MSG_RESULT(yes);   use_openblas=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for openblas); AC_MSG_RESULT(no); use_openblas=false ])
AM_CONDITIONAL([UseOpenblas],[test "x$use_openblas" = xtrue])

if test "$use_openblas" == true; then
  AC_DEFINE([UseOpenblas], [UseOpenblas], [UseOpenblas])
fi


#
# GSL
#
# We use pkg-config if it knows GSL and fall back to the libraries' default
# names otherwise. GSL is appended to the link line, i.e. it comes after all
# Peano libraries (which are prepended) that might use it.
#
if test "$use_gsl" != false; then
  AC_PATH_PROG([PKG_CONFIG], [pkg-config])
  if test -n "$PKG_CONFIG" && $PKG_CONFIG --exists gsl; then
    GSL_INCLUDE_DIR=`$PKG_CONFIG --cflags gsl`
    GSL_LIBS=`$PKG_CONFIG --libs gsl`
  else
    GSL_INCLUDE_DIR=""
    GSL_LIBS="-lgsl -lgslcblas -lm"
  fi
  AC_MSG_NOTICE([GSL_INCLUDE_DIR: ${GSL_INCLUDE_DIR}])
  AC_MSG_NOTICE([GSL_LIBS: ${GSL_LIBS}])

  CXXFLAGS_PEANO_2D_RELEASE=" $CXXFLAGS_PEANO_2D_RELEASE ${GSL_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_STATS="   $CXXFLAGS_PEANO_2D_STATS ${GSL_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_ASSERTS=" $CXXFLAGS_PEANO_2D_ASSERTS ${GSL_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_TRACE="   $CXXFLAGS_PEANO_2D_TRACE ${GSL_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_DEBUG="   $CXXFLAGS_PEANO_2D_DEBUG ${GSL_INCLUDE_DIR} "

  CXXFLAGS_PEANO_3D_RELEASE=" $CXXFLAGS_PEANO_3D_RELEASE ${GSL_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_STATS="   $CXXFLAGS_PEANO_3D_STATS ${GSL_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_ASSERTS=" $CXXFLAGS_PEANO_3D_ASSERTS ${GSL_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_TRACE="   $CXXFLAGS_PEANO_3D_TRACE ${GSL_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_DEBUG="   $CXXFLAGS_PEANO_3D_DEBUG ${GSL_INCLUDE_DIR} "

  LDADD_PEANO_2D_RELEASE=" $LDADD_PEANO_2D_RELEASE ${GSL_LIBS} "
  LDADD_PEANO_2D_STATS="   $LDADD_PEANO_2D_STATS ${GSL_LIBS} "
  LDADD_PEANO_2D_ASSERTS=" $LDADD_PEANO_2D_ASSERTS ${GSL_LIBS} "
  LDADD_PEANO_2D_TRACE="   $LDADD_PEANO_2D_TRACE ${GSL_LIBS} "
  LDADD_PEANO_2D_DEBUG="   $LDADD_PEANO_2D_DEBUG ${GSL_LIBS} "

  LDADD_PEANO_3D_RELEASE=" $LDADD_PEANO_3D_RELEASE ${GSL_LIBS} "
  LDADD_PEANO_3D_STATS="   $LDADD_PEANO_3D_STATS ${GSL_LIBS} "
  LDADD_PEANO_3D_ASSERTS=" $LDADD_PEANO_3D_ASSERTS ${GSL_LIBS} "
  LDADD_PEANO_3D_TRACE="   $LDADD_PEANO_3D_TRACE ${GSL_LIBS} "
  LDADD_PEANO_3D_DEBUG="   $LDADD_PEANO_3D_DEBUG ${GSL_LIBS} "
fi


#
# HDF5 (parallel version requires --with-mpi)
#
if test "$use_hdf5" != false; then
  AC_PATH_PROG([PKG_CONFIG], [pkg-config])

  HDF5_INCLUDE_DIR=`$PKG_CONFIG --cflags hdf5_cpp`
  HDF5_LIBS_DIR=`$PKG_CONFIG --libs-only-L hdf5_cpp`
  HDF5_LIBS=`$PKG_CONFIG --libs-only-l hdf5_cpp`

  CXXFLAGS_PEANO_2D_RELEASE=" $CXXFLAGS_PEANO_2D_RELEASE ${HDF5_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_STATS="   $CXXFLAGS_PEANO_2D_STATS ${HDF5_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_ASSERTS=" $CXXFLAGS_PEANO_2D_ASSERTS ${HDF5_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_TRACE="   $CXXFLAGS_PEANO_2D_TRACE ${HDF5_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_DEBUG="   $CXXFLAGS_PEANO_2D_DEBUG ${HDF5_INCLUDE_DIR} "

  CXXFLAGS_PEANO_3D_RELEASE=" $CXXFLAGS_PEANO_3D_RELEASE ${HDF5_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_STATS="   $CXXFLAGS_PEANO_3D_STATS ${HDF5_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_ASSERTS=" $CXXFLAGS_PEANO_3D_ASSERTS ${HDF5_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_TRACE="   $CXXFLAGS_PEANO_3D_TRACE ${HDF5_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_DEBUG="   $CXXFLAGS_PEANO_3D_DEBUG ${HDF5_INCLUDE_DIR} "

  LDADD_PEANO_2D_RELEASE=" $LDADD_PEANO_2D_RELEASE ${HDF5_LIBS} "
  LDADD_PEANO_2D_STATS="   $LDADD_PEANO_2D_STATS ${HDF5_LIBS} "
  LDADD_PEANO_2D_ASSERTS=" $LDADD_PEANO_2D_ASSERTS ${HDF5_LIBS} "
  LDADD_PEANO_2D_TRACE="   $LDADD_PEANO_2D_TRACE ${HDF5_LIBS} "
  LDADD_PEANO_2D_DEBUG="   $LDADD_PEANO_2D_DEBUG ${HDF5_LIBS} "

  LDADD_PEANO_3D_RELEASE=" $LDADD_PEANO_3D_RELEASE ${HDF5_LIBS} "
  LDADD_PEANO_3D_STATS="   $LDADD_PEANO_3D_STATS ${HDF5_LIBS} "
  LDADD_PEANO_3D_ASSERTS=" $LDADD_PEANO_3D_ASSERTS ${HDF5_LIBS} "
  LDADD_PEANO_3D_TRACE="   $LDADD_PEANO_3D_TRACE ${HDF5_LIBS} "
  LDADD_PEANO_3D_DEBUG="   $LDADD_PEANO_3D_DEBUG ${HDF5_LIBS} "

  LDFLAGS_PEANO_RELEASE+=" ${HDF5_LIBS_DIR} "
  LDFLAGS_PEANO_STATS+="   ${HDF5_LIBS_DIR} "
  LDFLAGS_PEANO_ASSERTS+=" ${HDF5_LIBS_DIR} "
  LDFLAGS_PEANO_TRACE+="   ${HDF5_LIBS_DIR} "
  LDFLAGS_PEANO_DEBUG+="   ${HDF5_LIBS_DIR} "

  AC_MSG_NOTICE([HDF5_INCLUDE_DIR: ${HDF5_INCLUDE_DIR}])
  AC_MSG_NOTICE([HDF5_LIBS_DIR: ${HDF5_LIBS_DIR}])
  AC_MSG_NOTICE([HDF5_LIBS: ${HDF5_LIBS}])

  AC_SUBST([HDF5_INCLUDE_DIR])
  AC_SUBST([HDF5_LIBS_DIR])
  AC_SUBST([HDF5_LIBS])
fi


#
# NetCDF
#
if test "$use_netcdf" != false; then
  AC_PATH_PROG([PKG_CONFIG], [pkg-config])

  NETCDF_INCLUDE_DIR=`$PKG_CONFIG --cflags netcdf`
  NETCDF_LIBS_DIR=`$PKG_CONFIG --libs-only-L netcdf`
  NETCDF_LIBS=`$PKG_CONFIG --libs-only-l netcdf`

  CXXFLAGS_PEANO_2D_RELEASE=" $CXXFLAGS_PEANO_2D_RELEASE ${NETCDF_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_STATS="   $CXXFLAGS_PEANO_2D_STATS ${NETCDF_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_ASSERTS=" $CXXFLAGS_PEANO_2D_ASSERTS ${NETCDF_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_TRACE="   $CXXFLAGS_PEANO_2D_TRACE ${NETCDF_INCLUDE_DIR} "
  CXXFLAGS_PEANO_2D_DEBUG="   $CXXFLAGS_PEANO_2D_DEBUG ${NETCDF_INCLUDE_DIR} "

  CXXFLAGS_PEANO_3D_RELEASE=" $CXXFLAGS_PEANO_3D_RELEASE ${NETCDF_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_STATS="   $CXXFLAGS_PEANO_3D_STATS ${NETCDF_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_ASSERTS=" $CXXFLAGS_PEANO_3D_ASSERTS ${NETCDF_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_TRACE="   $CXXFLAGS_PEANO_3D_TRACE ${NETCDF_INCLUDE_DIR} "
  CXXFLAGS_PEANO_3D_DEBUG="   $CXXFLAGS_PEANO_3D_DEBUG ${NETCDF_INCLUDE_DIR} "

  LDADD_PEANO_2D_RELEASE=" $LDADD_PEANO_2D_RELEASE ${NETCDF_LIBS} "
  LDADD_PEANO_2D_STATS="   $LDADD_PEANO_2D_STATS ${NETCDF_LIBS} "
  LDADD_PEANO_2D_ASSERTS=" $LDADD_PEANO_2D_ASSERTS ${NETCDF_LIBS} "
  LDADD_PEANO_2D_TRACE="   $LDADD_PEANO_2D_TRACE ${NETCDF_LIBS} "
  LDADD_PEANO_2D_DEBUG="   $LDADD_PEANO_2D_DEBUG ${NETCDF_LIBS} "

  LDADD_PEANO_3D_RELEASE=" $LDADD_PEANO_3D_RELEASE ${NETCDF_LIBS} "
  LDADD_PEANO_3D_STATS="   $LDADD_PEANO_3D_STATS ${NETCDF_LIBS} "
  LDADD_PEANO_3D_ASSERTS=" $LDADD_PEANO_3D_ASSERTS ${NETCDF_LIBS} "
  LDADD_PEANO_3D_TRACE="   $LDADD_PEANO_3D_TRACE ${NETCDF_LIBS} "
  LDADD_PEANO_3D_DEBUG="   $LDADD_PEANO_3D_DEBUG ${NETCDF_LIBS} "

  LDFLAGS_PEANO_RELEASE+=" ${NETCDF_LIBS_DIR} "
  LDFLAGS_PEANO_STATS+="   ${NETCDF_LIBS_DIR} "
  LDFLAGS_PEANO_ASSERTS+=" ${NETCDF_LIBS_DIR} "
  LDFLAGS_PEANO_TRACE+="   ${NETCDF_LIBS_DIR} "
  LDFLAGS_PEANO_DEBUG+="   ${NETCDF_LIBS_DIR} "

  AC_MSG_NOTICE([NETCDF_INCLUDE_DIR: ${NETCDF_INCLUDE_DIR}])
  AC_MSG_NOTICE([NETCDF_LIBS_DIR: ${NETCDF_LIBS_DIR}])
  AC_MSG_NOTICE([NETCDF_LIBS: ${NETCDF_LIBS}])

  AC_SUBST([NETCDF_INCLUDE_DIR])
  AC_SUBST([NETCDF_LIBS_DIR])
  AC_SUBST([NETCDF_LIBS])
fi


#
# LIBXSMM
#
AC_ARG_WITH(libxsmm,
  AS_HELP_STRING([--with-libxsmm],[enable LIBXSMM [default=false]]),
  [ AC_MSG_CHECKING(for libxsmm)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    use_libxsmm=false ;;
      *)     AC_MSG_RESULT(yes);   use_libxsmm=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for libxsmm); AC_MSG_RESULT(no); use_libxsmm=false ])
AM_CONDITIONAL([UseLibXSMM],[test "x$use_libxsmm" = xtrue])

if test "$use_libxsmm" == true; then
  AC_DEFINE([UseLibXSMM], [UseLibXSMM], [UseLibXSMM])
fi


#
# LIBXSMM JIT
#
# Check to see if user enabled OpenBLAS
#
AC_ARG_WITH(libxsmmjit,
  AS_HELP_STRING([--with-libxsmmjit],[enable LIBXSMM JIT [default=false]]),
  [ AC_MSG_CHECKING(for libxsmmjit)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    use_libxsmmJIT=false ;;
      *)     AC_MSG_RESULT(yes);   use_libxsmmJIT=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for libxsmmjit); AC_MSG_RESULT(no); use_libxsmmJIT=false ])
AM_CONDITIONAL([UselibxsmmJIT],[test "x$use_libxsmmJIT" = xtrue])

if test "$use_libxsmmJIT" == true; then
  if test "$use_openblas" == false; then
  AC_MSG_ERROR([LIBXSMM JIT requires OpenBLAS. Re-reun with --with-openblas])
  else
  AC_DEFINE([UselibxsmmJIT], [UselibxsmmJIT], [UselibxsmmJIT])
  fi
fi


#
# BLIS
#
AC_ARG_WITH(blis,
  AS_HELP_STRING([--with-blis],[enable BLIS [default=false]]),
  [ AC_MSG_CHECKING(for blis)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    use_BLIS=false ;;
      *)     AC_MSG_RESULT(yes);   use_BLIS=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for blis); AC_MSG_RESULT(no); use_BLIS=false ])
AM_CONDITIONAL([UseBLIS],[test "x$use_BLIS" = xtrue])

if test "$use_BLIS" == true; then
  AC_DEFINE([UseBLIS], [UseBLIS], [UseBLIS])
fi


#
# Eigen
#
AC_ARG_WITH(Eigen,
  AS_HELP_STRING([--with-Eigen],[enable Eigen [default=false]]),
  [ AC_MSG_CHECKING(for Eigen)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    use_Eigen=false ;;
      *)     AC_MSG_RESULT(yes);   use_Eigen=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for Eigen); AC_MSG_RESULT(no); use_Eigen=false ])
AM_CONDITIONAL([UseEigen],[test "x$use_Eigen" = xtrue])

if test "$use_Eigen" == true; then
  AC_DEFINE([UseEigen], [UseEigen], [UseEigen])
fi


#
# LAPACK
#
AC_ARG_WITH(lapack,
  AS_HELP_STRING([--with-lapack],[enable LAPACK [default=false]]),
  [ AC_MSG_CHECKING(for lapack)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    use_lapack=false ;;
      *)     AC_MSG_RESULT(yes);   use_lapack=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for lapack); AC_MSG_RESULT(no); use_lapack=false ])
AM_CONDITIONAL([UseLapack],[test "x$use_lapack" = xtrue])

if test "$use_lapack" == true; then
  if test "$use_openblas" == false; then
  AC_MSG_ERROR([Lapack requires OpenBLAS. Re-reun with --with-openblas])
  else
  AC_DEFINE([UseLapack], [UseLapack], [UseLapack])
  fi
fi


#
# ===============================================================
# Tailored toolchains
#
# A toolchain in Peano in particular, bespoke compile and build
# settings that work well with particular (performance) analysis
# tools or a suite of dev tools by a particular vendor, or some
# bespoke postprocessing tools.
# ===============================================================
#

#
# Vendor-specific toolchain (logging devices, memory allocation libraries, ...)
#
AC_ARG_WITH(toolchain,
  AS_HELP_STRING([--with-toolchain],[build with vendor-specific toolchain [default=no,itac,itt,nvptx,scorep,chrome,gnu]]),
  [ AC_MSG_CHECKING(for toolchain)
    case "${withval}" in
      no)        AC_MSG_RESULT(no);                                           toolchain=false ;;
      itac)      AC_MSG_RESULT(build with support for the Intel's ITAC);      toolchain=itac  ;;
      itt)       AC_MSG_RESULT(build with support for the Intel's Instrumentation and Tracing Technology (ITT));      toolchain=itt  ;;
      nvidia)    AC_MSG_RESULT(build with support for the NVIDIA SDK);        toolchain=nvidia ;;
      scorep)    AC_MSG_RESULT(build with support for the Score-P);           toolchain=scorep ;;
      chrome)    AC_MSG_RESULT(build with logging support for Google Chrome); toolchain=chrome ;;
      gnu)       AC_MSG_RESULT(build with specialised GNU flags); toolchain=gnu ;;
      *)         AC_MSG_FAILURE([default=no (no bespoke toolchain)]); toolchain=false ;;
    esac
  ],
  [
    AC_MSG_CHECKING(for toolchain);
    AC_MSG_RESULT(no); toolchain=false
  ]
)

if test "$toolchain" == itac; then
  AC_DEFINE([UseITAC], [1], [UseITAC])
  AC_DEFINE([UseLogService], [ITACLogger], [ITACLogger])

  CXXFLAGS_PEANO_2D_RELEASE+=" -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_2D_STATS+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_2D_ASSERTS+=" -DTBB_USE_THREADING_TOOLS -parallel-source-info=2 -DTBB_USE_ASSERT"
  CXXFLAGS_PEANO_2D_TRACE+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_2D_DEBUG+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2 -DTBB_USE_ASSERT"

  CXXFLAGS_PEANO_3D_RELEASE+=" -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_3D_STATS+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_3D_ASSERTS+=" -DTBB_USE_THREADING_TOOLS -parallel-source-info=2 -DTBB_USE_ASSERT"
  CXXFLAGS_PEANO_3D_TRACE+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_3D_DEBUG+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2 -DTBB_USE_ASSERT"
fi

if test "$toolchain" == itt; then
  AC_DEFINE([UseITT], [1], [UseITT])
  AC_DEFINE([UseLogService], [ITTLogger], [ITTLogger])

  CXXFLAGS_PEANO_2D_RELEASE+=" -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_2D_STATS+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_2D_ASSERTS+=" -DTBB_USE_THREADING_TOOLS -parallel-source-info=2 -DTBB_USE_ASSERT"
  CXXFLAGS_PEANO_2D_TRACE+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_2D_DEBUG+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2 -DTBB_USE_ASSERT"

  CXXFLAGS_PEANO_3D_RELEASE+=" -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_3D_STATS+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_3D_ASSERTS+=" -DTBB_USE_THREADING_TOOLS -parallel-source-info=2 -DTBB_USE_ASSERT"
  CXXFLAGS_PEANO_3D_TRACE+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2"
  CXXFLAGS_PEANO_3D_DEBUG+="   -DTBB_USE_THREADING_TOOLS -parallel-source-info=2 -DTBB_USE_ASSERT"
fi

if test "$toolchain" == scorep; then
  AC_DEFINE([UseLogService], [ScorePLogger], [ScorePLogger])
  AC_DEFINE([UseScoreP],     [UseScoreP], [UseScoreP])
fi

if test "$toolchain" == nvptx; then
  AC_DEFINE([UseNVIDIA], [1], [UseNVIDIA])
  AC_DEFINE([UseLogService], [NVTXLogger], [NVTXLogger])
  AC_CHECK_HEADERS([nvtx3/nvToolsExt.h],[],[AC_MSG_ERROR([header <nvtx3/nvToolsExt.h> not found])])
fi

if test "$toolchain" == chrome; then
  AC_DEFINE([UseLogService], [ChromeTraceFileLogger], [ChromeTraceFileLogger])
fi

if test "$toolchain" == gnu; then
  LDFLAGS_PEANO_RELEASE+=" -flto "
  LDFLAGS_PEANO_STATS+="   -flto "
  LDFLAGS_PEANO_ASSERTS+=" -flto "
  LDFLAGS_PEANO_TRACE+="   -flto "
  LDFLAGS_PEANO_DEBUG+="   -flto "
fi


#
# ===============================================================
# Toolboxes (actives with enable)
#
# A toolbox is an optional extension of Peano (package). As there
# are dependencies between the toolboxes, this part only analyses
# whether a toolbox should be used or not. It does not react to
# this choice.
# ===============================================================
#

#
# FEM
#
AC_ARG_ENABLE(finiteelements,
  AS_HELP_STRING([--enable-finiteelements],[enable finite elements toolbox  [default=false]]),
  [ AC_MSG_CHECKING(for finiteelements)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    toolbox_finiteelements=false ;;
      *)     AC_MSG_RESULT(yes);   toolbox_finiteelements=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for finiteelements); AC_MSG_RESULT(no); toolbox_finiteelements=false ])
AM_CONDITIONAL([UseToolboxFiniteElements],[test "x$toolbox_finiteelements" = xtrue])


#
# Multiprecision
#
AC_ARG_ENABLE(multiprecision,
  AS_HELP_STRING([--enable-multiprecision],[enable multiprecision  [default=false]]),
  [ AC_MSG_CHECKING(for multiprecision)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    toolbox_multiprecision=false ;;
      *)     AC_MSG_RESULT(yes);   toolbox_multiprecision=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for multiprecision); AC_MSG_RESULT(no); toolbox_multiprecision=false ])
AM_CONDITIONAL([UseToolboxMultiprecision],[test "x$toolbox_multiprecision" = xtrue])


#
# Load balancing
#
AC_ARG_ENABLE(loadbalancing,
  AS_HELP_STRING([--enable-loadbalancing],[enable loadbalancing  [default=false]]),
  [ AC_MSG_CHECKING(for loadbalancing)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    toolbox_loadbalancing=false ;;
      *)     AC_MSG_RESULT(yes);   toolbox_loadbalancing=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for loadbalancing); AC_MSG_RESULT(no); toolbox_loadbalancing=false ])
AM_CONDITIONAL([UseToolboxLoadBalancing],[test "x$toolbox_loadbalancing" = xtrue])


#
# Particles
#
AC_ARG_ENABLE(particles,
  AS_HELP_STRING([--enable-particles],[enable particles  [default=false]]),
  [ AC_MSG_CHECKING(for particles)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    toolbox_particles=false ;;
      *)     AC_MSG_RESULT(yes);   toolbox_particles=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for particles); AC_MSG_RESULT(no); toolbox_particles=false ])
AM_CONDITIONAL([UseToolboxParticles],[test "x$toolbox_particles" = xtrue])


#
# ExaHyPE 2
#
AC_ARG_ENABLE(exahype,
  AS_HELP_STRING([--enable-exahype],[enable ExaHyPE2 [default=false]]),
  [ AC_MSG_CHECKING(for exahype)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    toolbox_exahype=false ;;
      *)     AC_MSG_RESULT(yes);   toolbox_exahype=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for exahype); AC_MSG_RESULT(no); toolbox_exahype=false ])
AM_CONDITIONAL([UseExaHyPE],[test "x$toolbox_exahype" = xtrue])


#
# ExaGRyPE
#
AC_ARG_ENABLE(exagrype,
  AS_HELP_STRING([--enable-exagrype],[enable ExaGRyPE (3d only, requires ExaHyPE 2) [default=false]]),
  [ AC_MSG_CHECKING(for exagrype)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    toolbox_exagrype=false ;;
      *)     AC_MSG_RESULT(yes);   toolbox_exagrype=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for exagrype); AC_MSG_RESULT(no); toolbox_exagrype=false ])
AM_CONDITIONAL([UseExaGRyPE],[test "x$toolbox_exagrype" = xtrue])


#
# Curvi
#
AC_ARG_ENABLE(curvi,
  AS_HELP_STRING([--enable-curvi],[enable Curvi [default=false]]),
  [ AC_MSG_CHECKING(for curvi)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    toolbox_curvi=false ;;
      *)     AC_MSG_RESULT(yes);   toolbox_curvi=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for curvi); AC_MSG_RESULT(no); toolbox_curvi=false ])
AM_CONDITIONAL([UseToolboxCurvi],[test "x$toolbox_curvi" = xtrue])


#
# ExaParty
#
AC_ARG_ENABLE(exaparty,
  AS_HELP_STRING([--enable-exaparty],[enable ExaParty [default=false]]),
  [ AC_MSG_CHECKING(for exaparty)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    toolbox_exaparty=false ;;
      *)     AC_MSG_RESULT(yes);   toolbox_exaparty=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for exaparty); AC_MSG_RESULT(no); toolbox_exaparty=false ])
AM_CONDITIONAL([UseExaParty],[test "x$toolbox_exaparty" = xtrue])


#
# Multigrid
#
AC_ARG_ENABLE(mghype,
  AS_HELP_STRING([--enable-mghype],[enable MGHyPE [default=false]]),
  [ AC_MSG_CHECKING(for mghype)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    toolbox_mghype=false ;;
      *)     AC_MSG_RESULT(yes);   toolbox_mghype=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for mghype); AC_MSG_RESULT(no); toolbox_mghype=false ])
AM_CONDITIONAL([UseMGHyPE],[test "x$toolbox_mghype" = xtrue])


#
# Blockstructured
#
AC_ARG_ENABLE(blockstructured,
  AS_HELP_STRING([--enable-blockstructured],[enable blockstructured (SAMR) support [default=false]]),
  [ AC_MSG_CHECKING(for blockstructured)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    toolbox_blockstructured=false ;;
      *)     AC_MSG_RESULT(yes);   toolbox_blockstructured=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for blockstructured); AC_MSG_RESULT(no); toolbox_blockstructured=false ])
AM_CONDITIONAL([UseToolboxBlockstructured],[test "x$toolbox_blockstructured" = xtrue])


#
# Delta
#
AC_ARG_WITH(delta,
  AS_HELP_STRING([--with-delta],[enable Delta [default=no]]),
  [ AC_MSG_CHECKING(for delta)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);   toolbox_delta=false ;;
      *)     AC_MSG_RESULT(yes);  toolbox_delta=${withval}  ;;
    esac ],
  [ AC_MSG_CHECKING(for delta); AC_MSG_RESULT(no); toolbox_delta=false ])
# This whole autotools thing is not very nice: If I wanna use an if ... endif (
# note the syntax which is different to the present file, then I have to declare
# the variable as I do in the line below. Otherwise, the if thing is not
# "exported". Seems the autoconf and the makefile generator are completely
# separate with inconsistent syntax.
AM_CONDITIONAL([UseDelta],[test "$toolbox_delta" != false])


#
# ===============================================================
# Setup toolboxes and extensions
#
# Run through the set of toolboxes and extensions that are enabled,
# configure them properly, i.e., augment the Peano compile and
# linker flags, and check for consistency, i.e., whether all
# dependencies are ok.
# ===============================================================
#

#
# FEM
#
if test "$toolbox_finiteelements" != false; then
  AC_DEFINE([UseToolboxFiniteElements], [1], [UseToolboxFiniteElements])

  LDADD_PEANO_2D_RELEASE=" -lToolboxFiniteElements2d         $LDADD_PEANO_2D_RELEASE"
  LDADD_PEANO_2D_STATS="   -lToolboxFiniteElements2d_stats   $LDADD_PEANO_2D_STATS "
  LDADD_PEANO_2D_ASSERTS=" -lToolboxFiniteElements2d_asserts $LDADD_PEANO_2D_ASSERTS "
  LDADD_PEANO_2D_TRACE="   -lToolboxFiniteElements2d_trace   $LDADD_PEANO_2D_TRACE "
  LDADD_PEANO_2D_DEBUG="   -lToolboxFiniteElements2d_debug   $LDADD_PEANO_2D_DEBUG "

  LDADD_PEANO_3D_RELEASE=" -lToolboxFiniteElements3d         $LDADD_PEANO_3D_RELEASE"
  LDADD_PEANO_3D_STATS="   -lToolboxFiniteElements3d_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS=" -lToolboxFiniteElements3d_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE="   -lToolboxFiniteElements3d_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG="   -lToolboxFiniteElements3d_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/toolbox/finiteelements "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/toolbox/finiteelements "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/toolbox/finiteelements "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/toolbox/finiteelements "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/toolbox/finiteelements "
fi


#
# Multiprecision
#
if test "$toolbox_multiprecision" != false; then
  AC_DEFINE([UseToolboxMultiprecision], [1], [UseToolboxMultiprecision])

  LDADD_PEANO_2D_RELEASE=" -lToolboxMultiprecision         $LDADD_PEANO_2D_RELEASE "
  LDADD_PEANO_2D_STATS="   -lToolboxMultiprecision_stats   $LDADD_PEANO_2D_STATS "
  LDADD_PEANO_2D_ASSERTS=" -lToolboxMultiprecision_asserts $LDADD_PEANO_2D_ASSERTS "
  LDADD_PEANO_2D_TRACE="   -lToolboxMultiprecision_trace   $LDADD_PEANO_2D_TRACE "
  LDADD_PEANO_2D_DEBUG="   -lToolboxMultiprecision_debug   $LDADD_PEANO_2D_DEBUG "

  LDADD_PEANO_3D_RELEASE=" -lToolboxMultiprecision         $LDADD_PEANO_3D_RELEASE "
  LDADD_PEANO_3D_STATS="   -lToolboxMultiprecision_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS=" -lToolboxMultiprecision_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE="   -lToolboxMultiprecision_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG="   -lToolboxMultiprecision_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/toolbox/multiprecision "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/toolbox/multiprecision "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/toolbox/multiprecision "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/toolbox/multiprecision "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/toolbox/multiprecision "
fi


#
# Load balancing
#
if test "$toolbox_loadbalancing" != false; then
  AC_DEFINE([UseToolboxLoadBalancing], [1], [UseToolboxLoadBalancing])

  LDADD_PEANO_2D_RELEASE=" -lToolboxLoadBalancing2d         $LDADD_PEANO_2D_RELEASE "
  LDADD_PEANO_2D_STATS="   -lToolboxLoadBalancing2d_stats   $LDADD_PEANO_2D_STATS "
  LDADD_PEANO_2D_ASSERTS=" -lToolboxLoadBalancing2d_asserts $LDADD_PEANO_2D_ASSERTS "
  LDADD_PEANO_2D_TRACE="   -lToolboxLoadBalancing2d_trace   $LDADD_PEANO_2D_TRACE "
  LDADD_PEANO_2D_DEBUG="   -lToolboxLoadBalancing2d_debug   $LDADD_PEANO_2D_DEBUG "

  LDADD_PEANO_3D_RELEASE=" -lToolboxLoadBalancing3d         $LDADD_PEANO_3D_RELEASE "
  LDADD_PEANO_3D_STATS="   -lToolboxLoadBalancing3d_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS=" -lToolboxLoadBalancing3d_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE="   -lToolboxLoadBalancing3d_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG="   -lToolboxLoadBalancing3d_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/toolbox/loadbalancing "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/toolbox/loadbalancing "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/toolbox/loadbalancing "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/toolbox/loadbalancing "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/toolbox/loadbalancing "
fi


#
# Blockstructured
#
if test "$toolbox_blockstructured" != false; then
  AC_DEFINE([UseToolboxBlockstructured], [1], [UseToolboxBlockstructured])

  LDADD_PEANO_2D_RELEASE=" -lToolboxBlockstructured2d         $LDADD_PEANO_2D_RELEASE "
  LDADD_PEANO_2D_STATS="   -lToolboxBlockstructured2d_stats   $LDADD_PEANO_2D_STATS "
  LDADD_PEANO_2D_ASSERTS=" -lToolboxBlockstructured2d_asserts $LDADD_PEANO_2D_ASSERTS "
  LDADD_PEANO_2D_TRACE="   -lToolboxBlockstructured2d_trace   $LDADD_PEANO_2D_TRACE "
  LDADD_PEANO_2D_DEBUG="   -lToolboxBlockstructured2d_debug   $LDADD_PEANO_2D_DEBUG "

  LDADD_PEANO_3D_RELEASE=" -lToolboxBlockstructured3d         $LDADD_PEANO_3D_RELEASE "
  LDADD_PEANO_3D_STATS="   -lToolboxBlockstructured3d_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS=" -lToolboxBlockstructured3d_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE="   -lToolboxBlockstructured3d_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG="   -lToolboxBlockstructured3d_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/toolbox/blockstructured "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/toolbox/blockstructured "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/toolbox/blockstructured "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/toolbox/blockstructured "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/toolbox/blockstructured "
fi


#
# Particles
#
if test "$toolbox_particles" != false; then
  AC_DEFINE([UseToolboxParticles], [1], [UseToolboxParticles])

  LDADD_PEANO_2D_RELEASE+=" -lToolboxParticles2d         $LDADD_PEANO_2D_RELEASE "
  LDADD_PEANO_2D_STATS+="   -lToolboxParticles2d_stats   $LDADD_PEANO_2D_STATS "
  LDADD_PEANO_2D_ASSERTS+=" -lToolboxParticles2d_asserts $LDADD_PEANO_2D_ASSERTS "
  LDADD_PEANO_2D_TRACE+="   -lToolboxParticles2d_trace   $LDADD_PEANO_2D_TRACE "
  LDADD_PEANO_2D_DEBUG+="   -lToolboxParticles2d_debug   $LDADD_PEANO_2D_DEBUG "

  LDADD_PEANO_3D_RELEASE+=" -lToolboxParticles3d         $LDADD_PEANO_3D_RELEASE "
  LDADD_PEANO_3D_STATS+="   -lToolboxParticles3d_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS+=" -lToolboxParticles3d_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE+="   -lToolboxParticles3d_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG+="   -lToolboxParticles3d_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/toolbox/particles "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/toolbox/particles "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/toolbox/particles "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/toolbox/particles "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/toolbox/particles "
fi


#
# PETSC
#
if test "$petsc" != false; then
  AC_DEFINE([UsePETSc], [1], [UsePETSc])
  if test "$toolbox_finiteelements" == false; then
    AC_MSG_ERROR([PETSc requires the finiteelements toolbox (run with --help to see the correct enable statement)])
  fi
  AC_CHECK_HEADER([petsc.h], [
  ], [
    AC_MSG_ERROR([PETSc headers not found])
  ])
  # Check for libraries
  AC_CHECK_LIB([petsc], [MatCreate], [
    PETSC_LIBS="ac_cv_search_MatCreate"
  ], [
    AC_MSG_ERROR([PETSc library not found. Did you forget to include -lpetsc to LDFLAGS?])
  ])
  LDADD_PEANO_2D_RELEASE=" -lPETSc         $LDADD_PEANO_2D_RELEASE "
  LDADD_PEANO_2D_STATS="   -lPETSc_stats   $LDADD_PEANO_2D_STATS "
  LDADD_PEANO_2D_ASSERTS=" -lPETSc_asserts $LDADD_PEANO_2D_ASSERTS "
  LDADD_PEANO_2D_TRACE="   -lPETSc_trace   $LDADD_PEANO_2D_TRACE "
  LDADD_PEANO_2D_DEBUG="   -lPETSc_debug   $LDADD_PEANO_2D_DEBUG "

  LDADD_PEANO_3D_RELEASE=" -lPETSc         $LDADD_PEANO_3D_RELEASE "
  LDADD_PEANO_3D_STATS="   -lPETSc_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS=" -lPETSc_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE="   -lPETSc_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG="   -lPETSc_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/petsc "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/petsc "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/petsc "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/petsc "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/petsc "
fi


#
# ExaHyPE 2
#
if test "$toolbox_exahype" != false; then
  AC_DEFINE([UseExaHyPE], [1], [UseExaHyPE])
  if test "$toolbox_loadbalancing" == false; then
    AC_MSG_ERROR([ExaHyPE 2 requires the load balancing toolbox (run with --help to see the correct enable statement)])
  fi
  if test "$toolbox_blockstructured" == false; then
    AC_MSG_ERROR([ExaHyPE 2 requires the blockstructured toolbox (run with --help to see the correct enable statement)])
  fi
  LDADD_PEANO_2D_RELEASE=" -lExaHyPE2Core2d         $LDADD_PEANO_2D_RELEASE "
  LDADD_PEANO_2D_STATS="   -lExaHyPE2Core2d_stats   $LDADD_PEANO_2D_STATS "
  LDADD_PEANO_2D_ASSERTS=" -lExaHyPE2Core2d_asserts $LDADD_PEANO_2D_ASSERTS "
  LDADD_PEANO_2D_TRACE="   -lExaHyPE2Core2d_trace   $LDADD_PEANO_2D_TRACE "
  LDADD_PEANO_2D_DEBUG="   -lExaHyPE2Core2d_debug   $LDADD_PEANO_2D_DEBUG "

  LDADD_PEANO_3D_RELEASE=" -lExaHyPE2Core3d         $LDADD_PEANO_3D_RELEASE "
  LDADD_PEANO_3D_STATS="   -lExaHyPE2Core3d_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS=" -lExaHyPE2Core3d_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE="   -lExaHyPE2Core3d_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG="   -lExaHyPE2Core3d_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/exahype2 "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/exahype2 "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/exahype2 "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/exahype2 "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/exahype2 "
fi


#
# ExaGRyPE
#
# ExaGRyPE is built for 3d only, so we only extend the 3d link lines. As
# ExaGRyPE builds on top of ExaHyPE 2, this block has to come after the
# ExaHyPE 2 block: we prepend the libraries, so ExaGRyPE ends up in front of
# ExaHyPE 2 on the link line.
#
if test "$toolbox_exagrype" != false; then
  AC_DEFINE([UseExaGRyPE], [1], [UseExaGRyPE])
  if test "$toolbox_exahype" == false; then
    AC_MSG_ERROR([ExaGRyPE requires ExaHyPE 2 (run with --help to see the correct enable statement)])
  fi
  LDADD_PEANO_3D_RELEASE=" -lExaGRyPECore3d         $LDADD_PEANO_3D_RELEASE "
  LDADD_PEANO_3D_STATS="   -lExaGRyPECore3d_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS=" -lExaGRyPECore3d_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE="   -lExaGRyPECore3d_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG="   -lExaGRyPECore3d_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/exagrype "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/exagrype "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/exagrype "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/exagrype "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/exagrype "
fi


#
# Curvi
#
if test "$toolbox_curvi" != false; then
  AC_DEFINE([UseToolboxCurvi], [1], [UseToolboxCurvi])

  LDADD_PEANO_2D_RELEASE=" -lToolboxCurvi         $LDADD_PEANO_2D_RELEASE "
  LDADD_PEANO_2D_STATS="   -lToolboxCurvi_stats   $LDADD_PEANO_2D_STATS "
  LDADD_PEANO_2D_ASSERTS=" -lToolboxCurvi_asserts $LDADD_PEANO_2D_ASSERTS "
  LDADD_PEANO_2D_TRACE="   -lToolboxCurvi_trace   $LDADD_PEANO_2D_TRACE "
  LDADD_PEANO_2D_DEBUG="   -lToolboxCurvi_debug   $LDADD_PEANO_2D_DEBUG "

  LDADD_PEANO_3D_RELEASE=" -lToolboxCurvi         $LDADD_PEANO_3D_RELEASE "
  LDADD_PEANO_3D_STATS="   -lToolboxCurvi_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS=" -lToolboxCurvi_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE="   -lToolboxCurvi_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG="   -lToolboxCurvi_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/toolbox/curvi "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/toolbox/curvi "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/toolbox/curvi "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/toolbox/curvi "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/toolbox/curvi "
fi


#
# Multigrid
#
if test "$toolbox_mghype" != false; then
  AC_DEFINE([UseMGHyPE], [1], [UseMGHyPE])
  if test "$toolbox_finiteelements" == false; then
    AC_MSG_ERROR([MGHyPE (mghype) requires the finiteelements toolbox (run with --help to see the correct enable statement)])
  fi

  LDADD_PEANO_2D_RELEASE=" -lMultigrid2d         $LDADD_PEANO_2D_RELEASE "
  LDADD_PEANO_2D_STATS="   -lMultigrid2d_stats   $LDADD_PEANO_2D_STATS "
  LDADD_PEANO_2D_ASSERTS=" -lMultigrid2d_asserts $LDADD_PEANO_2D_ASSERTS "
  LDADD_PEANO_2D_TRACE="   -lMultigrid2d_trace   $LDADD_PEANO_2D_TRACE "
  LDADD_PEANO_2D_DEBUG="   -lMultigrid2d_debug   $LDADD_PEANO_2D_DEBUG "

  LDADD_PEANO_3D_RELEASE=" -lMultigrid3d         $LDADD_PEANO_3D_RELEASE "
  LDADD_PEANO_3D_STATS="   -lMultigrid3d_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS=" -lMultigrid3d_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE="   -lMultigrid3d_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG="   -lMultigrid3d_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/mghype "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/mghype "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/mghype "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/mghype "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/mghype "
fi


#
# OPENBLAS
#
if test "$use_openblas" != false; then
  # Find the lapack header
  AC_CHECK_HEADER([cblas.h], [
  ], [
    AC_MSG_ERROR([OpenBLAS headers not found])
  ])
  # Check for libraries
  AC_CHECK_LIB([openblas], [cblas_daxpy], [
    OPENBLAS_LIBS="ac_cv_search_cblas_daxpy"
  ], [
    AC_MSG_ERROR([OpenBLAS library not found. Did you forget to include -lopenblas to LDFLAGS?])
  ])
  AC_SUBST([OPENBLAS_LIBS])
  OPENBLAS_INCLUDE_DIR=$(dirname $OPENBLAS_LIBS)
  AC_SUBST([OPENBLAS_INCLUDE_DIR])

  LDFLAGS_PEANO_RELEASE+=" -lopenblas "
  LDFLAGS_PEANO_STATS+="   -lopenblas "
  LDFLAGS_PEANO_ASSERTS+=" -lopenblas "
  LDFLAGS_PEANO_TRACE+="   -lopenblas "
  LDFLAGS_PEANO_DEBUG+="   -lopenblas "
fi


#
# NetCDF
#
if test "$use_netcdf" != false; then
  # Find the NetCDF header
  AC_CHECK_HEADERS([netcdf.h], [
  ], [
    AC_MSG_ERROR([NetCDF headers not found])
  ])
  # Check for libraries
  AC_CHECK_LIB([netcdf], [nc_open], [
    NETCDF_LIBS="netcdf"
  ], [
    AC_MSG_ERROR([NetCDF library not found. Did you forget to include -lnetcdf to LDFLAGS?])
  ])
  AC_SUBST([NETCDF_LIBS])
  NETCDF_INCLUDE_DIR=$(dirname $NETCDF_LIBS)
  AC_SUBST([NETCDF_INCLUDE_DIR])
fi


#
# LIBXSMM
#
if test "$use_libxsmm" != false; then
  echo "Checking whether LIBXSMM needs to be installed or built"
  LIBXSMM_PATH=$(command -v libxsmm_gemm_generator)
  if @<:@ "$LIBXSMM_PATH" @:>@; then
    echo "Found preinstalled LIBXSMM at $LIBXSMM_PATH"
  else
    if @<:@ ! -d submodules @:>@; then
      mkdir submodules
      echo "Created libxsmm directory in submodules"
    fi
    if @<:@ ! -e submodules/libxsmm/.git @:>@; then
      echo "Initializing LIBXSMM in submodules/libxsmm, this may take a while."
      cd submodules
      git clone https://github.com/libxsmm/libxsmm.git
      cd libxsmm
      # Cleanup documentation to save space
      rm -rf samples/ # Over 100M of not needed stuff
      rm -rf documentation/ # Around 10M of documentation
      cd ../..
    fi

    # Build LIBXSMM
    if @<:@ ! -d submodules/libxsmm/bin @:>@ || @<:@ ! -e submodules/libxsmm/bin/libxsmm_gemm_generator @:>@; then
      REBUILD_LIBXSMM=true
    fi
    if @<:@ "$REBUILD_LIBXSMM" = true @:>@; then
      echo "Building LIBXSMM GEMM generator"
      cd submodules/libxsmm
      git restore samples
      make realclean --silent
      make generator --silent
      rm -rf samples
      cd ../..
    fi
    echo "Finished building LIBXSMM GEMM generator"
  fi
fi


#
# LIBXSMM JIT
#
if test "$use_libxsmmJIT" != false; then
  echo "Checking whether LIBXSMM JIT needs to be installed or built"
  if @<:@ ! -d submodules @:>@; then
    mkdir submodules
    echo "Created libxsmmJIT directory in submodules"
  fi
  if @<:@ ! -e submodules/libxsmmJIT/.git @:>@; then
    echo "Initializing LIBXSMM JIT in submodules/libxsmmJIT, this may take a while."
    cd submodules
    git clone https://github.com/libxsmm/libxsmm.git libxsmmJIT
    cd libxsmmJIT
    cd ../..
  fi

  # Build LIBXSMM JIT
  if @<:@ ! -d submodules/libxsmmJIT/lib @:>@ || @<:@ ! -e submodules/libxsmmJIT/include @:>@; then
    REBUILD_libxsmmJIT=true
  fi
  if @<:@ "$REBUILD_libxsmmJIT" = true @:>@; then
    echo "Building LIBXSMM JIT"
    cd submodules/libxsmmJIT
    make --silent
    cd ../..
  fi
  echo "Finished building LIBXSMM JIT"
fi


#
# BLIS
#
if test "$use_BLIS" != false; then
  echo "Checking whether BLIS needs to be installed or built"
  if @<:@ ! -d submodules @:>@; then
    mkdir submodules
    echo "Created BLIS directory in submodules"
  fi
  if @<:@ ! -e submodules/blis/.git @:>@; then
    echo "Initializing BLIS in submodules/blis, this may take a while."
    cd submodules
    git clone https://github.com/flame/blis.git
    cd blis
    cd ../..
  fi

  # Build BLIS
  if @<:@ ! -d submodules/blis/lib @:>@ || @<:@ ! -e submodules/blis/include @:>@; then
    REBUILD_BLIS=true
  fi
  if @<:@ "$REBUILD_BLIS" = true @:>@; then
    echo "Building BLIS"
    cd submodules/blis
    ./configure auto
    make --silent
    make check
    cd ../..
  fi
  echo "Finished building BLIS"
fi


#
# Eigen
#
if test "$use_Eigen" != false; then
  echo "Checking whether Eigen needs to be installed or built"
  if @<:@ ! -d submodules @:>@; then
    mkdir submodules
    echo "Created Eigen directory in submodules"
  fi
  if @<:@ ! -e submodules/Eigen/.git @:>@; then
    echo "Initializing Eigen in submodules/Eigen, this may take a while."
    cd submodules
    git clone https://gitlab.com/libeigen/eigen.git
    cd eigen
    cd ../..
  fi
fi


#
# LAPACK
#
if test "$use_lapack" != false; then
  # Find the lapack header
  AC_CHECK_HEADER([lapacke.h], [
  ], [
    AC_MSG_ERROR([LAPACK headers not found])
  ])
  # Check for libraries. The lapack libraries are
  # libopenblas, but if this doesn't work, try
  # configuring with -llapack.
  AC_CHECK_LIB([openblas], [LAPACKE_dgetrf], [
    LAPACK_LIBS="ac_cv_search_LAPACKE_dgetrf"
  ], [
    AC_MSG_ERROR([LAPACK library not found.])
  ])
  AC_SUBST([LAPACK_LIBS])
  LAPACK_INCLUDE_DIR=$(dirname $LAPACK_LIBS)
  AC_SUBST([LAPACK_INCLUDE_DIR])
fi


#
# ExaParty
#
if test "$toolbox_exaparty" != false; then
  AC_DEFINE([UseExaParty], [1], [UseExaParty])
  if test "$toolbox_loadbalancing" == false; then
    AC_MSG_ERROR([ExaParty requires the load balancing toolbox (run with --help to see the correct enable statement)])
  fi
  if test "$toolbox_particles" == false; then
    AC_MSG_ERROR([ExaParty requires the particles toolbox (run with --help to see the correct enable statement)])
  fi
  LDADD_PEANO_2D_RELEASE=" -lExaPartyCore2d         $LDADD_PEANO_2D_RELEASE "
  LDADD_PEANO_2D_STATS="   -lExaPartyCore2d_stats   $LDADD_PEANO_2D_STATS "
  LDADD_PEANO_2D_ASSERTS=" -lExaPartyCore2d_asserts $LDADD_PEANO_2D_ASSERTS "
  LDADD_PEANO_2D_TRACE="   -lExaPartyCore2d_trace   $LDADD_PEANO_2D_TRACE "
  LDADD_PEANO_2D_DEBUG="   -lExaPartyCore2d_debug   $LDADD_PEANO_2D_DEBUG "

  LDADD_PEANO_3D_RELEASE=" -lExaPartyCore3d         $LDADD_PEANO_3D_RELEASE "
  LDADD_PEANO_3D_STATS="   -lExaPartyCore3d_stats   $LDADD_PEANO_3D_STATS "
  LDADD_PEANO_3D_ASSERTS=" -lExaPartyCore3d_asserts $LDADD_PEANO_3D_ASSERTS "
  LDADD_PEANO_3D_TRACE="   -lExaPartyCore3d_trace   $LDADD_PEANO_3D_TRACE "
  LDADD_PEANO_3D_DEBUG="   -lExaPartyCore3d_debug   $LDADD_PEANO_3D_DEBUG "

  LDFLAGS_PEANO_RELEASE+=" -L$srcdir_full_path/src/exaparty "
  LDFLAGS_PEANO_STATS+="   -L$srcdir_full_path/src/exaparty "
  LDFLAGS_PEANO_ASSERTS+=" -L$srcdir_full_path/src/exaparty "
  LDFLAGS_PEANO_TRACE+="   -L$srcdir_full_path/src/exaparty "
  LDFLAGS_PEANO_DEBUG+="   -L$srcdir_full_path/src/exaparty "
fi


#
# SmartMPI
#
AC_ARG_WITH(smartmpi,
  AS_HELP_STRING([--with-smartmpi],[build with SmartMPI [default=no,Merged,Alternating]]),
  [ AC_MSG_CHECKING(for smartmpi)
    case "${withval}" in
      no)               AC_MSG_RESULT(no);    smartmpi=false ;;
      Merged)           AC_MSG_RESULT(yes; use topology Merged);            smartmpi=Merged ;;
      Alternating)      AC_MSG_RESULT(yes; use topology Alternating);       smartmpi=Alternating ;;
      *)         AC_MSG_RESULT(failed);   AC_MSG_ERROR([invalid option for SmartMPI]) ;;
    esac ],
  [ AC_MSG_CHECKING(for smartmpi); AC_MSG_RESULT(no); smartmpi=false ])
AM_CONDITIONAL([UseSmartMPI],[test "x$smartmpi" = xtrue])
if test "$smartmpi" == Merged; then
  AC_DEFINE([UseSmartMPI], [Merged], [UseSmartMPI])
fi
if test "$smartmpi" == Alternating; then
  AC_DEFINE([UseSmartMPI], [Alternating], [UseSmartMPI])
fi


#
# targetDART
#
AC_ARG_WITH(targetdart,
  AS_HELP_STRING([--with-targetdart],[build with targetDART [default=no]]),
  [ AC_MSG_CHECKING(for targetdart)
    case "${withval}" in
      no)    AC_MSG_RESULT(no);    targetdart=false ;;
      *)     AC_MSG_RESULT(yes);   targetdart=true  ;;
    esac ],
  [ AC_MSG_CHECKING(for targetdart); AC_MSG_RESULT(no); targetdart=false ])
AM_CONDITIONAL([UseTargetDART],[test "x$targetdart" = xtrue])
if test "$targetdart" == true; then
  AC_DEFINE([UseTargetDART], [UseTargetDART], [UseTargetDART])
fi


#
# ===============================================================
#   Export variables
# ===============================================================
#
AC_SUBST(CXXFLAGS_PEANO_2D_RELEASE)
AC_SUBST(CXXFLAGS_PEANO_2D_STATS)
AC_SUBST(CXXFLAGS_PEANO_2D_ASSERTS)
AC_SUBST(CXXFLAGS_PEANO_2D_TRACE)
AC_SUBST(CXXFLAGS_PEANO_2D_DEBUG)

AC_SUBST(CXXFLAGS_PEANO_3D_RELEASE)
AC_SUBST(CXXFLAGS_PEANO_3D_STATS)
AC_SUBST(CXXFLAGS_PEANO_3D_ASSERTS)
AC_SUBST(CXXFLAGS_PEANO_3D_TRACE)
AC_SUBST(CXXFLAGS_PEANO_3D_DEBUG)

AC_SUBST(LDFLAGS_PEANO_RELEASE)
AC_SUBST(LDFLAGS_PEANO_STATS)
AC_SUBST(LDFLAGS_PEANO_ASSERTS)
AC_SUBST(LDFLAGS_PEANO_TRACE)
AC_SUBST(LDFLAGS_PEANO_DEBUG)

AC_SUBST(LDADD_PEANO_2D_RELEASE)
AC_SUBST(LDADD_PEANO_2D_STATS)
AC_SUBST(LDADD_PEANO_2D_ASSERTS)
AC_SUBST(LDADD_PEANO_2D_TRACE)
AC_SUBST(LDADD_PEANO_2D_DEBUG)

AC_SUBST(LDADD_PEANO_3D_RELEASE)
AC_SUBST(LDADD_PEANO_3D_STATS)
AC_SUBST(LDADD_PEANO_3D_ASSERTS)
AC_SUBST(LDADD_PEANO_3D_TRACE)
AC_SUBST(LDADD_PEANO_3D_DEBUG)

AC_CONFIG_FILES([Makefile
  src/Makefile
  src/toolbox/blockstructured/Makefile
  src/toolbox/multiprecision/Makefile
  src/toolbox/finiteelements/Makefile
  src/toolbox/loadbalancing/Makefile
  src/toolbox/particles/Makefile
  src/toolbox/curvi/Makefile
  src/tbb/Makefile
  src/convert/Makefile
  src/exahype2/Makefile
  src/exagrype/Makefile
  src/mghype/Makefile
  src/petsc/Makefile
  src/exaparty/Makefile
  src/unittests/Makefile
])
AC_OUTPUT
