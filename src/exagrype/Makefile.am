AUTOMAKE_OPTIONS = subdir-objects

#
# ExaGRyPE solves the Einstein equations and hence is built for 3d only.
#
if UseExaGRyPE
  lib_LIBRARIES = libExaGRyPECore3d.a  libExaGRyPECore3d_debug.a  libExaGRyPECore3d_stats.a  libExaGRyPECore3d_asserts.a  libExaGRyPECore3d_trace.a
endif


ExaGRyPESources = \
  UnitTests.cpp \
  ccz4/Kernels.cpp \
  ccz4/SecondOrderAuxiliaryVariablesReconstruction.cpp \
  ccz4/tests/FlatSpacetimeTest.cpp \
  euler/EulerKernels.cpp \
  euler/initialvalues/InitialValues.cpp \
  initialvalues/ApplyTwoPunctures.cpp \
  initialvalues/InitialValues.cpp

#
# TwoPunctures (initial data for black holes) needs GSL
#
if UseGSL
ExaGRyPESources += \
  twopunctures/CoordTransf.cpp \
  twopunctures/Equations.cpp \
  twopunctures/FuncAndJacobian.cpp \
  twopunctures/Newton.cpp \
  twopunctures/TP_Logging.cpp \
  twopunctures/TP_Parameters.cpp \
  twopunctures/TP_Utilities.cpp \
  twopunctures/TwoPunctures.cpp
endif

libExaGRyPECore3d_a_CXXFLAGS          = $(CXXFLAGS_PEANO_3D_RELEASE)
libExaGRyPECore3d_trace_a_CXXFLAGS    = $(CXXFLAGS_PEANO_3D_TRACE)
libExaGRyPECore3d_asserts_a_CXXFLAGS  = $(CXXFLAGS_PEANO_3D_ASSERTS)
libExaGRyPECore3d_debug_a_CXXFLAGS    = $(CXXFLAGS_PEANO_3D_DEBUG)
libExaGRyPECore3d_stats_a_CXXFLAGS    = $(CXXFLAGS_PEANO_3D_STATS)

libExaGRyPECore3d_a_SOURCES         = $(ExaGRyPESources)
libExaGRyPECore3d_trace_a_SOURCES   = $(ExaGRyPESources)
libExaGRyPECore3d_asserts_a_SOURCES = $(ExaGRyPESources)
libExaGRyPECore3d_debug_a_SOURCES   = $(ExaGRyPESources)
libExaGRyPECore3d_stats_a_SOURCES   = $(ExaGRyPESources)
