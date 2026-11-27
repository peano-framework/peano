AUTOMAKE_OPTIONS = subdir-objects

#
# By default, automake does dfs. We need the libs to be built first however.
# So we add the . explicitly. Please note that it is not enough to add your
# directory here. You also have to add any new directory to the root's
# configure.ac script and rerun the whole autotool pipeline (see comment
# within configure.ac as well as its very last line).
#
SUBDIRS = . \
	toolbox/blockstructured \
	toolbox/finiteelements \
	toolbox/loadbalancing \
	toolbox/multiprecision \
	toolbox/particles \
	toolbox/curvi \
	convert \
	exahype2 \
	exagrype \
	mghype \
	petsc \
	exaparty \
	tbb \
	unittests

#
# I want Peano to be as fast as possible and thus build it as static library
# such that a linker/compiler can ipo. One of the other reasons for this is
# that I have a couple of singletons and I did struggle with singletons and
# shared libraries.
#
lib_LIBRARIES = libTarch.a        libTarch_debug.a        libTarch_asserts.a        libTarch_trace.a        libTarch_stats.a \
								libPeano4Core2d.a libPeano4Core2d_debug.a libPeano4Core2d_asserts.a libPeano4Core2d_trace.a libPeano4Core2d_stats.a \
								libPeano4Core3d.a libPeano4Core3d_debug.a libPeano4Core3d_asserts.a libPeano4Core3d_trace.a libPeano4Core3d_stats.a

Tarch_SOURCES = tarch/tarch.cpp \
	tarch/Enumerator.cpp \
	tarch/NonCriticalAssertions.cpp \
	tarch/UnitTests.cpp \
	tarch/la/tests/DynamicMatrixTest.cpp \
	tarch/la/tests/MatrixVectorTest.cpp \
	tarch/la/tests/MatrixTest.cpp \
	tarch/la/tests/ScalarTest.cpp \
	tarch/la/tests/GramSchmidtTest.cpp \
	tarch/la/tests/LUDecompositionTest.cpp \
	tarch/la/tests/VectorTest.cpp \
	tarch/la/ScalarOperations.cpp \
	tarch/la/Scalar.cpp \
	tarch/la/DynamicMatrix.cpp \
	tarch/la/DynamicMatrixOperations.cpp \
	tarch/la/GramSchmidt.cpp \
	tarch/la/LUDecompositionDynamic.cpp \
	tarch/accelerator/accelerator.cpp \
	tarch/accelerator/Device.cpp \
	tarch/accelerator/cpp/accelerator.cpp \
	tarch/accelerator/cpp/Device.cpp \
	tarch/accelerator/omp/accelerator.cpp \
	tarch/accelerator/omp/Device.cpp \
	tarch/accelerator/omp/GPUMemoryManager.cpp \
	tarch/accelerator/sycl/accelerator.cpp \
	tarch/accelerator/sycl/Device.cpp \
	tarch/accelerator/sycl/GPUMemoryManager.cpp \
	tarch/accelerator/hip/accelerator.cpp \
	tarch/multicore/BooleanSemaphore.cpp \
	tarch/multicore/multicore.cpp \
	tarch/multicore/Core.cpp \
	tarch/multicore/Lock.cpp \
	tarch/multicore/TaskEnumerator.cpp \
	tarch/multicore/MultiReadSingleWriteSemaphore.cpp \
	tarch/multicore/MultiReadSingleWriteLock.cpp \
	tarch/multicore/RecursiveLock.cpp \
	tarch/multicore/RecursiveSemaphore.cpp \
	tarch/multicore/smartScheduling.cpp \
	tarch/multicore/Task.cpp \
	tarch/multicore/cpp/multicore.cpp \
	tarch/multicore/cpp/BooleanSemaphore.cpp \
	tarch/multicore/cpp/Core.cpp \
	tarch/multicore/cpp/RecursiveSemaphore.cpp \
	tarch/multicore/cpp/Tasks.cpp \
	tarch/multicore/omp/multicore.cpp \
	tarch/multicore/omp/BooleanSemaphore.cpp \
	tarch/multicore/omp/Core.cpp \
	tarch/multicore/omp/RecursiveSemaphore.cpp \
	tarch/multicore/omp/Tasks.cpp \
	tarch/multicore/omp/TasksWithOpenMP5.cpp \
	tarch/multicore/omp/TasksWithOpenMP6.cpp \
	tarch/multicore/taskfusion/taskfusion.cpp \
	tarch/multicore/taskfusion/FusableTasksQueue.cpp \
	tarch/multicore/taskfusion/LogReadyTask.cpp \
	tarch/multicore/taskfusion/ProcessOneReadyTask.cpp \
	tarch/multicore/taskfusion/ProcessSetOfReadyTasks.cpp \
	tarch/multicore/tbb/multicore.cpp \
	tarch/multicore/tbb/BooleanSemaphore.cpp \
	tarch/multicore/tbb/Core.cpp \
	tarch/multicore/tbb/MultiReadSingleWriteSemaphore.cpp \
	tarch/multicore/tbb/RecursiveSemaphore.cpp \
	tarch/multicore/tbb/Tasks.cpp \
	tarch/multicore/tbb/TaskHandleRepository.cpp \
	tarch/multicore/sycl/multicore.cpp \
	tarch/multicore/sycl/BooleanSemaphore.cpp \
	tarch/multicore/sycl/Core.cpp \
	tarch/multicore/sycl/RecursiveSemaphore.cpp \
	tarch/multicore/sycl/Tasks.cpp \
	tarch/multicore/orchestration/AllOnGPU.cpp \
	tarch/multicore/orchestration/Hardcoded.cpp \
	tarch/multicore/orchestration/Strategy.cpp \
	tarch/tests/TestCaseCollection.cpp \
	tarch/tests/TreeTestCaseCollection.cpp \
	tarch/tests/TestCase.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTUTextFileWriter_CellWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTKTextFileWriter_VertexDataWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTKTextFileWriter_VertexWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/tests/VTKBinaryFileTestCase.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTKBinaryFileWriter_CellWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTUTextFileWriter_CellDataWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTKTextFileWriter_CellWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTUTextFileWriter_VertexDataWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTKBinaryFileWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTUTextFileWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTKBinaryFileWriter_VertexWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTKBinaryFileWriter_CellDataWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTKTextFileWriter_CellDataWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTKBinaryFileWriter_VertexDataWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTUTextFileWriter_VertexWriter.cpp \
	tarch/plotter/griddata/unstructured/vtk/VTKTextFileWriter.cpp \
	tarch/plotter/PVDTimeSeriesWriter.cpp \
	tarch/plotter/griddata/blockstructured/PeanoTextPatchFileWriter.cpp \
	tarch/plotter/griddata/blockstructured/PeanoTextPatchFileWriter_VertexDataWriter.cpp \
	tarch/plotter/griddata/blockstructured/PeanoHDF5PatchFileWriter.cpp \
	tarch/plotter/griddata/blockstructured/PeanoHDF5PatchFileWriter_CellDataWriter.cpp \
	tarch/plotter/griddata/blockstructured/PeanoHDF5PatchFileWriter_VertexDataWriter.cpp \
	tarch/plotter/griddata/blockstructured/PeanoTextPatchFileWriter_CellDataWriter.cpp \
	tarch/plotter/griddata/blockstructured/PeanoBinaryPatchFileWriter.cpp \
	tarch/plotter/griddata/blockstructured/PeanoBinaryPatchFileWriter_VertexDataWriter.cpp \
	tarch/plotter/griddata/blockstructured/PeanoBinaryPatchFileWriter_CellDataWriter.cpp \
	tarch/plotter/pointdata/vtk/VTKWriter.cpp \
	tarch/reader/NetCDFFileReader.cpp \
	tarch/reader/NetCDFFieldParser.cpp \
	tarch/reader/PeanoTextPatchFileReader.cpp \
	tarch/reader/PeanoBinaryPatchFileReader.cpp \
	tarch/mpi/BooleanSemaphore.cpp \
	tarch/mpi/Lock.cpp \
	tarch/mpi/Rank.cpp \
	tarch/mpi/mpi.cpp \
	tarch/mpi/DoubleMessage.cpp \
	tarch/mpi/IntegerMessage.cpp \
	tarch/mpi/StringMessage.cpp \
	tarch/mpi/tests/StringTest.cpp \
	tarch/multicore/tests/TaskTest.cpp \
	tarch/logging/CommandLineLogger.cpp \
	tarch/logging/ChromeTraceFileLogger.cpp \
	tarch/logging/ITACLogger.cpp \
	tarch/logging/ITTLogger.cpp \
	tarch/logging/Log.cpp \
	tarch/logging/LogFilterFileReader.cpp \
	tarch/logging/NoLogger.cpp \
	tarch/logging/NVTXLogger.cpp \
	tarch/logging/ScorePLogger.cpp \
	tarch/logging/LogFilter.cpp \
	tarch/logging/Statistics.cpp \
	tarch/timing/Watch.cpp \
	tarch/timing/Measurement.cpp \
	tarch/timing/GlidingAverageMeasurement.cpp \
	tarch/services/Service.cpp \
	tarch/services/ServiceRepository.cpp

libTarch_a_SOURCES         = $(Tarch_SOURCES)
libTarch_stats_a_SOURCES   = $(Tarch_SOURCES)
libTarch_trace_a_SOURCES   = $(Tarch_SOURCES)
libTarch_asserts_a_SOURCES = $(Tarch_SOURCES)
libTarch_debug_a_SOURCES   = $(Tarch_SOURCES)

libTarch_a_CXXFLAGS         = $(CXXFLAGS_PEANO_2D_RELEASE)
libTarch_trace_a_CXXFLAGS   = $(CXXFLAGS_PEANO_2D_TRACE)
libTarch_asserts_a_CXXFLAGS = $(CXXFLAGS_PEANO_2D_ASSERTS)
libTarch_debug_a_CXXFLAGS   = $(CXXFLAGS_PEANO_2D_DEBUG)
libTarch_stats_a_CXXFLAGS   = $(CXXFLAGS_PEANO_2D_STATS)

Peano4Core_SOURCES = \
	peano4/datamanagement/CellMarker.cpp \
	peano4/datamanagement/FaceMarker.cpp \
	peano4/datamanagement/VertexMarker.cpp \
	peano4/datamanagement/tests/DataMovementTest.cpp \
	peano4/peano.cpp \
	peano4/RunMultipleSteps.cpp \
	peano4/DefaultStepReport.cpp \
	peano4/UnitTests.cpp \
	peano4/grid/AutomatonState.cpp \
	peano4/grid/EmptyTraversalObserver.cpp \
	peano4/grid/grid.cpp \
	peano4/grid/GridControlEvent.cpp \
	peano4/grid/GridVertex.cpp \
	peano4/grid/GridStatistics.cpp \
	peano4/grid/GridTraversalEvent.cpp \
	peano4/grid/GridTraversalEventGenerator.cpp \
	peano4/grid/LoadStoreComputeFlag.cpp \
	peano4/grid/PeanoCurve.cpp \
	peano4/grid/Spacetree.cpp \
	peano4/grid/TraversalObserver.cpp \
	peano4/grid/TraversalVTKPlotter.cpp \
	peano4/grid/tests/GridControlEventTest.cpp \
	peano4/grid/tests/SpacetreeTest.cpp \
	peano4/grid/tests/GridTraversalEventGeneratorTest.cpp \
	peano4/parallel/Node.cpp \
	peano4/parallel/parallel.cpp \
	peano4/parallel/SpacetreeSet.cpp \
	peano4/parallel/StartTraversalMessage.cpp \
	peano4/parallel/TreeEntry.cpp \
	peano4/parallel/TreeManagementMessage.cpp \
	peano4/parallel/tests/PingPongTest.cpp \
	peano4/parallel/tests/NodeTest.cpp \
	peano4/stacks/stacks.cpp \
	peano4/stacks/STDVectorStack.cpp \
	peano4/stacks/tests/GeneratedTestObject.cpp \
	peano4/stacks/tests/GeneratedTestObject_Flattened.cpp \
	peano4/stacks/tests/STDVectorStackOverObjectsWithSmartPointersTest.cpp \
	peano4/utils/Loop.cpp \
	peano4/utils/tests/ParallelDForTest.cpp
	

libPeano4Core2d_a_SOURCES         = $(Peano4Core_SOURCES)
libPeano4Core2d_trace_a_SOURCES   = $(Peano4Core_SOURCES)
libPeano4Core2d_asserts_a_SOURCES = $(Peano4Core_SOURCES)
libPeano4Core2d_debug_a_SOURCES   = $(Peano4Core_SOURCES)
libPeano4Core2d_stats_a_SOURCES   = $(Peano4Core_SOURCES)

libPeano4Core3d_a_SOURCES         = $(Peano4Core_SOURCES)
libPeano4Core3d_trace_a_SOURCES   = $(Peano4Core_SOURCES)
libPeano4Core3d_asserts_a_SOURCES = $(Peano4Core_SOURCES)
libPeano4Core3d_debug_a_SOURCES   = $(Peano4Core_SOURCES)
libPeano4Core3d_stats_a_SOURCES   = $(Peano4Core_SOURCES)

libPeano4Core2d_a_CXXFLAGS          = $(CXXFLAGS_PEANO_2D_RELEASE)
libPeano4Core2d_stats_a_CXXFLAGS    = $(CXXFLAGS_PEANO_2D_STATS)
libPeano4Core2d_trace_a_CXXFLAGS    = $(CXXFLAGS_PEANO_2D_TRACE)
libPeano4Core2d_asserts_a_CXXFLAGS  = $(CXXFLAGS_PEANO_2D_ASSERTS)
libPeano4Core2d_debug_a_CXXFLAGS    = $(CXXFLAGS_PEANO_2D_DEBUG)

libPeano4Core3d_a_CXXFLAGS          = $(CXXFLAGS_PEANO_3D_RELEASE)
libPeano4Core3d_stats_a_CXXFLAGS    = $(CXXFLAGS_PEANO_3D_STATS)
libPeano4Core3d_trace_a_CXXFLAGS    = $(CXXFLAGS_PEANO_3D_TRACE)
libPeano4Core3d_asserts_a_CXXFLAGS  = $(CXXFLAGS_PEANO_3D_ASSERTS)
libPeano4Core3d_debug_a_CXXFLAGS    = $(CXXFLAGS_PEANO_3D_DEBUG)
