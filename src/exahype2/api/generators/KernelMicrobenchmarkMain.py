# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api

from ..Project import Project


class KernelMicrobenchmarkMain(Project.Generator):
    """!

    Optional realisation of exahype2.api.Project.mainfile_generator

    It constructs a main function that times the performance of different functions
    and assesses their accuracy.

    """

    def __init__(self, project: Project, function_calls: str, additional_includes: str):
        super().__init__(project)
        self.function_calls = function_calls
        self.additional_includes = additional_includes

    def construct_output(self, peano_project):
        peano4.api.Logger().log(
            "Create main file for {}".format(self.project._project_name)
        )
        peano_project.main = _ExaHyPEKernelBenchmarkMainFileSequence(peano_project, self.project, self.function_calls, self.additional_includes)


class _ExaHyPEKernelBenchmarkMainFileSequence:
    """!

    Adapter that speaks the construct_output(output, main_name) protocol
    peano4.api.Project.generate() expects from its main attribute (see
    peano4.api.runner.DefaultSequence, whose job this class takes over for
    an ExaHyPE 2 project), but fills the header/implementation pair with
    ExaHyPE-flavoured content instead of the mesh-agnostic default.

    project is the peano4-level project (peano4.api.Project) this sequence
    is attached to as .main - it is where SUBDIRECTORIES/SUBNAMESPACE/
    rootnamespace come from, mirroring peano4.api.runner.DefaultSequence.

    exahype_project is the exahype2.api.Project this main file is generated
    for - it is where the runtime settings (TIMEOUT, NUMBER_OF_THREADS,
    GPUS, MULTICORE_ORCHESTRATION) come from, as those live on the ExaHyPE
    2 project rather than on the plain peano4 one.

    """

    HEADER_TEMPLATE = """
// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

int main(int argc, char** argv);
"""

    CPP_TEMPLATE = """
// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "{{CLASSNAME}}.h"

#include "peano4/peano.h"
#include "exahype2/Simulation.h"
#include "exahype2/CellData.h"
#include "tarch/NonCriticalAssertions.h"
#include "tarch/accelerator/accelerator.h"
#include "tarch/accelerator/Device.h"
#include "tarch/multicore/multicore.h"
#include "tarch/multicore/Core.h"
#include "tarch/multicore/BooleanSemaphore.h"
#include "tarch/multicore/Lock.h"
#include "tarch/logging/Log.h"
#include "tarch/logging/LogFilterFileReader.h"
#include "tarch/timing/Measurement.h"
#include "tarch/timing/Watch.h"
{{ADDITIONAL_INCLUDES}}

{% if SUBDIRECTORIES -%}
{% for subdirectory in SUBDIRECTORIES %}
#include "{{subdirectory}}Constants.h"
#include "{{subdirectory}}repositories/DataRepository.h"
#include "{{subdirectory}}repositories/StepRepository.h"
#include "{{subdirectory}}repositories/SolverRepository.h"
#include "{{subdirectory}}runners/SetupSimulation.h"
#include "{{subdirectory}}runners/TimeStep.h"
#include "{{subdirectory}}runners/Plot.h"
#include "{{subdirectory}}runners/MeshAdaptation.h"
{% endfor -%}
{% else %}
#include "Constants.h"
#include "repositories/DataRepository.h"
#include "repositories/StepRepository.h"
#include "repositories/SolverRepository.h"
#include "runners/SetupSimulation.h"
#include "runners/TimeStep.h"
#include "runners/Plot.h"
#include "runners/MeshAdaptation.h"
{%- endif %}

using namespace {{NAMESPACE | join("::")}};
tarch::logging::Log _log( "{{NAMESPACE | join("::")}}" );
::tarch::timing::Measurement         timingComputeKernel;
::tarch::multicore::BooleanSemaphore validateOutcomeSemaphore;

double   validMaxEigenvalue = 0.0;
double*  validOutcome       = nullptr;

constexpr double TimeStamp    = 0.5;
constexpr double TimeStepSize = 1e-6;
constexpr double CellSize     = 0.1;
constexpr double CellOffset   = 4.0;

constexpr int NumberOfInputEntriesPerPatch
  = (repositories::{{SOLVER_INSTANCE}}.PatchSize + 2 * repositories::{{SOLVER_INSTANCE}}.HaloSize)
  * (repositories::{{SOLVER_INSTANCE}}.PatchSize + 2 * repositories::{{SOLVER_INSTANCE}}.HaloSize)
#if Dimensions == 3
  * (repositories::{{SOLVER_INSTANCE}}.PatchSize + 2 * repositories::{{SOLVER_INSTANCE}}.HaloSize)
#endif
  * (repositories::{{SOLVER_INSTANCE}}.NumberOfUnknowns + repositories::{{SOLVER_INSTANCE}}.NumberOfAuxiliaryVariables);

constexpr int NumberOfOutputEntriesPerPatch
  = (repositories::{{SOLVER_INSTANCE}}.PatchSize + 0)
  * (repositories::{{SOLVER_INSTANCE}}.PatchSize + 0)
#if Dimensions == 3
  * (repositories::{{SOLVER_INSTANCE}}.PatchSize + 0)
#endif
  * (repositories::{{SOLVER_INSTANCE}}.NumberOfUnknowns + repositories::{{SOLVER_INSTANCE}}.NumberOfAuxiliaryVariables);

constexpr int NumberOfFiniteVolumesPerPatch
  = repositories::{{SOLVER_INSTANCE}}.PatchSize
  * repositories::{{SOLVER_INSTANCE}}.PatchSize
#if Dimensions == 3
  * repositories::{{SOLVER_INSTANCE}}.PatchSize
#endif
  ;
  
/**
 * Set input data according to the initial conditions provided for the solver
 */
void initInputData(double* Q) {
  parallelDfor( volume, repositories::{{SOLVER_INSTANCE}}.PatchSize + 2 * repositories::{{SOLVER_INSTANCE}}.HaloSize) {
    int index = peano4::utils::dLinearised(volume,repositories::{{SOLVER_INSTANCE}}.PatchSize + 2 * repositories::{{SOLVER_INSTANCE}}.HaloSize) * (repositories::{{SOLVER_INSTANCE}}.NumberOfUnknowns + repositories::{{SOLVER_INSTANCE}}.NumberOfAuxiliaryVariables);
    repositories::{{SOLVER_INSTANCE}}.initPointWisely(
      ::exahype2::fv::getVolumeCentre( CellOffset + 0.5 * CellSize, CellSize, repositories::{{SOLVER_INSTANCE}}.PatchSize, volume),
      Q + index
    );
  } endParallelDfor
}

/**
 * Store outcome of one compute kernel
 *
 * Make a persistent snapshot of a solution and assume, from hereon, that
 * this snapshot is the valid data. You can call this routine as often as
 * you want. Only the very first call will trigger a snapshot.
 */
void storeOutcome(const double* Q, const double maxEigenvalue) {
  ::tarch::multicore::Lock lock(validateOutcomeSemaphore);
  if (validOutcome == nullptr) {
    validOutcome = new double[NumberOfOutputEntriesPerPatch]{0.0};
    std::memcpy(validOutcome, Q, sizeof(double) * NumberOfOutputEntriesPerPatch);
    validMaxEigenvalue = maxEigenvalue;
    logInfo("storeOutcome()", "Bookmarked reference solution");
  }
}


/**
 * Validate data against pre-stored simulation outcome
 *
 * Works if and only if storeOutcome has been performed before.
 *
 * @return Tuple of maximum difference and total number of errors.
 */
std::tuple<double, int> validateOutcome(
  double*      Q,
  int          patchIndex,
  const double maxEigenvalue
) {
  ::tarch::multicore::Lock lock(validateOutcomeSemaphore);
  int                      index         = 0;
  int                      errors        = 0;
  double                   maxDifference = 0.0;

  for (int i = 0; i < NumberOfOutputEntriesPerPatch; i++) {
    if (not ::tarch::la::equals(Q[i], validOutcome[i], Accuracy)) {
      errors++;
      std::cerr.precision(16);
      logError(
        "validateOutcome()",
        std::fixed
          << "Q[" << i << "]!=validOutcome[" << i << "]: " << Q[i]
          << "!=" << validOutcome[i]
      );
    }
    maxDifference = std::max(maxDifference, std::abs(Q[i] - validOutcome[i]));
    index++;
  }

  if (not tarch::la::equals(maxEigenvalue, validMaxEigenvalue, Accuracy)) {
    std::cerr.precision(16);
    logError(
      "validateOutcome()",
      std::fixed
        << " maxEigenvalue[" << patchIndex << "]!=validMaxEigenvalue["
        << patchIndex << "]: " << maxEigenvalue << "!=" << validMaxEigenvalue
    );
    errors++;
  }

  return {maxDifference, errors};
}


void reportRuntime(
  const std::string&                  kernelIdentificator,
  const ::tarch::timing::Measurement& timingKernelLaunch,
  int                                 patches
) {
  std::stringstream ss;
  ss << "\\n";
  ss << kernelIdentificator << ":\\n\\t";

  ss << timingComputeKernel.getValue() << " |\\n\\t";
  ss
    << (timingComputeKernel.getValue() / patches / NumberOfFiniteVolumesPerPatch
       )
    << " |\\n\\t";
  ss << timingComputeKernel.toString() << " |\\n\\t";

  ss << timingKernelLaunch.getValue() << " |\\n\\t";
  ss
    << (timingKernelLaunch.getValue() / patches / NumberOfFiniteVolumesPerPatch
       );
  ss << " |\\n\\t" << timingKernelLaunch.toString();

  logInfo("reportRuntime()", ss.str());
  std::cout.flush();
}

/**
 * We want to use all kernels exactly the same way. However, the various
 * kernels all have slightly different signatures. So we write small helper
 * functions (wrappers) which map the generic test signature onto the
 * specific kernel.
 *
 * To make this possible, all parameters which are not part of the generic
 * interface, i.e., which are not patch data or the boolean, have to be
 * mapped onto template arguments.
 */
//template <void (*Kernel)(
//  exahype2::CellData<double, double>&,
//  const ::exahype2::fv::rusanov::FluxFunctor&,
//  const ::exahype2::fv::rusanov::MaxEigenvalueFunctor&,
//  tarch::timing::Measurement&
//)>
//void wrapHostKernel(int device, exahype2::CellData<double, double>& patchData) {
//  assertionEquals(device, tarch::accelerator::Device::HostDevice);
//  Kernel(
//    patchData,
//    [&](
//      const double* __restrict__ Q,
//      const tarch::la::Vector<Dimensions, double>& x,
//      int                                          normal,
//      double* __restrict__ F
//    )->void {
//      repositories::instanceOfEulerFVSmooth.flux(Q, x, normal, F);
//    },
//    [&](
//      const double* __restrict__ Q,
//      const tarch::la::Vector<Dimensions, double>& x,
//      int                                          normal,
//      double* __restrict__ lambdaMax
//    )->void {
//      repositories::instanceOfEulerFVSmooth.maxEigenvalue(Q, x, normal, lambdaMax);
//    },
//    timingComputeKernel
//  );
//}

template <void (*Kernel)(exahype2::CellData<double, double>&, tarch::timing::Measurement&)>
void wrapHostKernel(int device, exahype2::CellData<double, double>& patchData) {
  assertionEquals(device, tarch::accelerator::Device::HostDevice);
  Kernel(patchData, timingComputeKernel);
}

 template <void (*Kernel)(exahype2::CellData<double, double>&, tarch::timing::Measurement&, peano4::utils::LoopPlacement),
          peano4::utils::LoopPlacement loopPlacement>
void wrapHostKernel(int device, exahype2::CellData<double, double>& patchData) {
  assertionEquals(device, tarch::accelerator::Device::HostDevice);
  Kernel(patchData, timingComputeKernel, loopPlacement);
}

template <void (*Kernel)(int, exahype2::CopyCellDataGPU<double, double>&, tarch::timing::Measurement&)>
void wrapDeviceKernel(int device, ::exahype2::CellData<double, double>& patchData) {
  assertion(device != tarch::accelerator::Device::HostDevice);
  ::exahype2::CopyCellDataGPU<double, double> gpuCellData(patchData, NumberOfInputEntriesPerPatch, NumberOfOutputEntriesPerPatch, device);
  Kernel(device, gpuCellData, timingComputeKernel);
  gpuCellData.copyBackAndDestroy();
}

template <void (*Kernel)(exahype2::CellDataFlattened<double, double>&, tarch::timing::Measurement&)>
void wrapFlattenedHostKernel(int device, ::exahype2::CellData<double, double>& patchData) {
  assertion(device == tarch::accelerator::Device::HostDevice);
  exahype2::CellDataFlattened<double, double> cellData(patchData, NumberOfInputEntriesPerPatch, NumberOfOutputEntriesPerPatch);
  Kernel(cellData, timingComputeKernel);
  cellData.copyBackAndDestroy();
}

template <void (*Kernel)(exahype2::MLIRCellDataFlattened&, tarch::timing::Measurement&)>
void wrapMLIRFlattenedHostKernel(int device, ::exahype2::CellData<double, double>& patchData) {
  assertion(device == tarch::accelerator::Device::HostDevice);
  exahype2::MLIRCellDataFlattened mlirCellData(patchData, NumberOfInputEntriesPerPatch, NumberOfOutputEntriesPerPatch);
  Kernel(mlirCellData, timingComputeKernel);
  mlirCellData.copyBackAndDestroy();
}

template <void (*Kernel)(exahype2::MLIRCellDataNested&, tarch::timing::Measurement&)>
void wrapMLIRNestedHostKernel(int device, ::exahype2::CellData<double, double>& patchData) {
  assertion(device == tarch::accelerator::Device::HostDevice);
  exahype2::MLIRCellDataNested mlirCellData(patchData, NumberOfInputEntriesPerPatch, NumberOfOutputEntriesPerPatch);
  Kernel(mlirCellData, timingComputeKernel);
  mlirCellData.copyBackAndDestroy();
}

/**
 * Run the benchmark for one particular number of patches
 *
 * This operation benchmarks exclusively the host performance. This happens in
 * two steps: We first assess the baseline performance, i.e., the serial kernel
 * call, and then we look if the multithreaded stateless optimisation pays off
 * and what speedup we obtain.
 *
 * These two steps may run on multiple threads to mimic the spacetree tasking
 * and domain decomposition. In OpenMP for example, there is
 * one parallel loop around the two steps which invokes the same benchmark per
 * loop entry. However, only the first thread always will report its runtime.
 * If you are not bandwidth-bound, the data you obtain should thus be
 * independent of the number of threads used.
 *
 * @param numberOfPatches Number of patches to study
 */
void runBenchmarks(int numberOfPatches, int launchingThreads) {
  ::peano4::grid::GridTraversalEvent event;
  event.setX(CellOffset);
  event.setH(CellSize);
  ::peano4::datamanagement::CellMarker marker(event);

  auto assessKernel =
    [&](
      std::function<void(int device, ::exahype2::CellData<double, double>& patchData)> kernel,
      const std::string& markerName,
      int                launchingThreads,
      int                device,
      int                patches
    ) -> void {
    timingComputeKernel.erase();
    ::tarch::timing::Measurement timingKernelLaunch;

    // TODO: Does the number of samples change the outcome of the solution?
    for (int j = 0; j < NumberOfSamples; j++) {
      //parallelFor(launchingThread, launchingThreads) {
      //for (int launchingThread=0; launchingThread < launchingThreads; launchingThread++){
        ::exahype2::CellData<double, double> patchData(patches);
        for (int i = 0; i < patches; i++) {
          patchData.QIn[i] = ::tarch::allocateMemory<double>(
            NumberOfInputEntriesPerPatch,
            ::tarch::MemoryLocation::ManagedSharedAcceleratorDeviceMemory
          );
          patchData.t[i]    = TimeStamp;
          patchData.dt[i]   = TimeStepSize;
          patchData.QOut[i] = ::tarch::allocateMemory<double>(
            NumberOfOutputEntriesPerPatch,
            ::tarch::MemoryLocation::ManagedSharedAcceleratorDeviceMemory
          );
          patchData.cellCentre[i] = ::tarch::la::Vector<Dimensions, double>(
            CellOffset + 0.5 * CellSize
          );
          patchData.cellSize[i] = ::tarch::la::Vector<Dimensions, double>(
            CellSize
          );
          patchData.maxEigenvalue[i] = 0.0;
          initInputData(patchData.QIn[i]);
          std::memset(
            patchData.QOut[i],
            0.0,
            NumberOfOutputEntriesPerPatch * sizeof(double)
          );
        }

        ::tarch::timing::Watch
          watchKernelLaunch("::runBenchmarks", "assessKernel(...)", false);
        kernel(device, patchData);
        watchKernelLaunch.stop();
        timingKernelLaunch.setValue(watchKernelLaunch.getCalendarTime());

        if constexpr (Accuracy > 0.0) {
          int    errors        = 0;
          double maxDifference = 0.0;
          for (int i = 0; i < patches; i++) {
            storeOutcome(patchData.QOut[i], patchData.maxEigenvalue[i]);
            auto [maxDifferencePerPatch, errorsPerPatch] = validateOutcome(
              patchData.QOut[i],
              i,
              patchData.maxEigenvalue[i]
            );
            errors += errorsPerPatch;
            maxDifference = std::max(maxDifference, maxDifferencePerPatch);
          }

          if (errors > 0) {
            logError(
              "runBenchmarks()",
              "Max difference of outcome from \\""
                << markerName << "\\" for all patches is " << maxDifference
                << " (admissible accuracy=" << Accuracy << ")"
                << " for " << errors << " entries"
            );
            std::abort();
          }
        }

        for (int i = 0; i < patches; i++) {
          ::tarch::freeMemory(
            patchData.QIn[i],
            ::tarch::MemoryLocation::ManagedSharedAcceleratorDeviceMemory
          );
          ::tarch::freeMemory(
            patchData.QOut[i],
            ::tarch::MemoryLocation::ManagedSharedAcceleratorDeviceMemory
          );
        }
      //}
      //endParallelFor
    }

    if (::tarch::mpi::Rank::getInstance().getRank() == ::tarch::mpi::Rank::getGlobalMasterRank()) {
      reportRuntime(markerName, timingKernelLaunch, patches);
    }
  };

  // Kernel launches
  const int device = 0;
  const int rank          = ::tarch::mpi::Rank::getInstance().getRank();
  const int numberOfRanks = ::tarch::mpi::Rank::getInstance().getNumberOfRanks(
  );
  const int patchesPerProcess = numberOfPatches / numberOfRanks;
  const int remainder         = numberOfPatches % numberOfRanks;
  const int startPatch        = rank * patchesPerProcess;
  const int endPatch          = startPatch + patchesPerProcess
                       + (rank == numberOfRanks - 1 ? remainder : 0);
  const int localPatches = endPatch - startPatch;

  // Headers
  std::stringstream ss;
  ss << std::left;
  ss << "\\n";
  ss << "Kernel ID:\\n\\t";
  ss << "Compute Kernel Time |\\n\\t";
  ss << "Compute Kernel Time (Normalised) |\\n\\t";
  ss << "Compute Kernel String |\\n\\t";
  ss << "Kernel Launch Time |\\n\\t";
  ss << "Kernel Launch Time (Normalised) |\\n\\t";
  ss << "Kernel Launch String";

  if (::tarch::mpi::Rank::getInstance().getRank() == ::tarch::mpi::Rank::getGlobalMasterRank()) {
    logInfo("runBenchmarks()", "Number of patches per rank: " << localPatches);
    logInfo("runBenchmarks()", ss.str());
  }

  std::string deviceString;
#ifdef Parallel
  std::vector<int> devices(numberOfRanks);
  MPI_Gather(&device, 1, MPI_INT, devices.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  for (int i = 0; i < numberOfRanks; i++) {
    deviceString += std::to_string(devices[i]);
    if (i < numberOfRanks - 1) {
      deviceString += ",";
    }
  }
#else
  deviceString = std::to_string(device);
#endif

{{FUNCTION_CALLS}}
}

int main(int argc, char** argv) {
  constexpr int ExitCodeSuccess             = 0;
  constexpr int ExitCodeInconsistentResults = 1;
  constexpr int ExitCodeInvalidArguments    = 2;
  constexpr int ExitCodeInvalidBuild        = 3;

  peano4::initParallelEnvironment(&argc, &argv);

  // Do this early, so people can use logInfo/logWarning/... properly from
  // here on.
  {% if LOG_FILTER_FILE!="" %}
  if ( not tarch::logging::LogFilterFileReader::parsePlainTextFile( "{{LOG_FILTER_FILE}}" ) ) {
    if (tarch::mpi::Rank::getInstance().isGlobalMaster()) {
      logWarning( "main()", "filter file {{LOG_FILTER_FILE}} not found or file has been corrupted. Use default logging configuration" );
    }
  }
  {% endif %}

  peano4::fillLookupTables();
  {{NAMESPACE | join("::")}}::{{SUBNAMESPACE}}repositories::DataRepository::initDatatypes();
  peano4::initSingletons(
    {{NAMESPACE | join("::")}}::DomainOffset,
    {{NAMESPACE | join("::")}}::DomainSize,
    {{NAMESPACE | join("::")}}::PeriodicBC
  );

  if ( not ::exahype2::configureRuntime(
    {{TIMEOUT}},
    {{NUMBER_OF_THREADS}},
    { {{GPUS|join(",")}} },
    {{MULTICORE_ORCHESTRATION}}
  ) ) {
    return ExitCodeInvalidBuild;
  }

  {% if FENV_ARGS is defined and FENV_ARGS != "" -%}
  feenableexcept({{FENV_ARGS}} );
  {% endif -%}


  for (int n = 0; n < NumberOfPatchesToStudy.size(); n++) {
    runBenchmarks(NumberOfPatchesToStudy[n], NumberOfLaunchingThreads);
  }

  if constexpr (Accuracy > 0.0) {
    logInfo(
      "main()",
      "All kernels yield the same outcome up to a accuracy of "
        << Accuracy << " unless reported otherwise"
    );
  } else {
    logInfo("main()", "No accuracy checks were performed");
  }
  
  ::exahype2::shutdownRuntime(
    {{NAMESPACE | join("::")}}::{{SUBNAMESPACE}}repositories::DataRepository::shutdownDatatypes
  );

  return ExitCodeSuccess;
}
"""

    def __init__(self, project, exahype_project, function_calls, additional_includes):
        self.project = project
        self.exahype_project = exahype_project
        self.function_calls = function_calls
        self.additional_includes = additional_includes

    def construct_output(self, output, main_name):
        d = {
            "SUBDIRECTORIES": list(self.project.subdirectories),
            "SUBNAMESPACE": self.project._subdirectory + "::" if self.project._subdirectory else "",
            "TIMEOUT": self.exahype_project._timeout,
            "NUMBER_OF_THREADS": self.exahype_project._number_of_threads,
            "MULTICORE_ORCHESTRATION": self.exahype_project._multicore_orchestration,
            "GPUS": self.exahype_project._gpus,
            "LOG_FILTER_FILE": self.exahype_project._log_filter_file,
            "SOLVER_NAME": self.exahype_project._solvers[0].name,
            "SOLVER_INSTANCE": self.exahype_project._solvers[0].get_name_of_global_instance(),
            "FUNCTION_CALLS": self.function_calls,
            "ADDITIONAL_INCLUDES": self.additional_includes
        }

        generated_files = peano4.api.output.Jinja2TemplateStringHeaderImplementationFilePair(
            headerfile_template = self.HEADER_TEMPLATE,
            cppfile_template    = self.CPP_TEMPLATE,
            classname           = main_name,
            namespace           = self.project.rootnamespace,
            subdirectory        = "./",
            dictionary          = d,
            default_overwrite   = True,
        )

        output.makefile.add_h_file(main_name + ".h", generated=True)
        output.makefile.add_cpp_file(main_name + ".cpp", generated=True)
        output.add(generated_files)
