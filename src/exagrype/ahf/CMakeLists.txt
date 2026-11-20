cmake_minimum_required(VERSION 3.21)

project(CCZ4ApparentHorizonFinder LANGUAGES CXX)

option(ENABLE_CCZ4_AHF_TESTS "Build CCZ4 apparent-horizon-finder test executables" OFF)

# Debug test builds check every std::array and std::vector index, in the core
# as well as the tests: _GLIBCXX_ASSERTIONS for libstdc++, the extensive
# hardening mode for libc++. Each library ignores the other's macro, and
# neither changes the ABI. Set before any target so that all of them get it.
if(ENABLE_CCZ4_AHF_TESTS)
  add_compile_definitions(
    $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
    $<$<CONFIG:Debug>:_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE>
  )
endif()

if(NOT TARGET PeanoOptions)
  add_library(PeanoOptions INTERFACE)
  target_compile_features(PeanoOptions INTERFACE cxx_std_20)
endif()

set(AHF_CORE_SOURCES
  ah-finder-options.cpp
  ah-finder-result.cpp
  ah-search.cpp
  ahf-postprocess-options.cpp
  merger-orchestration.cpp
  common-horizon-candidates.cpp
  common-horizon-search.cpp
  output.cpp
  interpolation.cpp
  surface-geometry.cpp
  harmonics.cpp
  initial-guess.cpp
  expansion.cpp
  assemble-matrix.cpp
  spectral-solver.cpp
  spacetimes.cpp
  surface-residual.cpp
  interpolated-surface-residual.cpp
  diagnostics.cpp
  iteration.cpp
)

set(AHF_CORE_HEADERS
  ah-finder-options.h
  ah-finder-result.h
  ah-search.h
  ahf-postprocess-options.h
  merger-orchestration.h
  common-horizon-candidates.h
  common-horizon-search.h
  ahf-data.h
  apparent-horizon-finder.h
  peano-input.h
  output.h
  interpolation.h
  surface-geometry.h
  tensor-algebra.h
  harmonics.h
  initial-guess.h
  expansion.h
  assemble-matrix.h
  dense-linear-algebra.h
  spectral-solver.h
  spacetimes.h
  surface-residual.h
  interpolated-surface-residual.h
  analytic-residual.h
  diagnostics.h
  iteration.h
)

add_library(ApparentHorizonFinderCore STATIC
  ${AHF_CORE_SOURCES}
  ${AHF_CORE_HEADERS}
)

target_include_directories(ApparentHorizonFinderCore
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(ApparentHorizonFinderCore
  PUBLIC
    PeanoOptions
)

target_compile_definitions(ApparentHorizonFinderCore PUBLIC IncludeApparentHorizonFinder)

# Only the snapshot reader (file-input), the postprocessing executable and the
# tests that read .vtu files need VTK; the core and its other tests do not.
find_package(VTK QUIET COMPONENTS
  CommonCore
  CommonDataModel
  FiltersCore
  IOXML
)

if(VTK_FOUND)
  add_library(ApparentHorizonFinderPostprocess STATIC
    file-input.cpp
    file-input.h
  )

  target_include_directories(ApparentHorizonFinderPostprocess
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

  target_link_libraries(ApparentHorizonFinderPostprocess
    PUBLIC
      ApparentHorizonFinderCore
    PRIVATE
      VTK::CommonCore
      VTK::CommonDataModel
      VTK::FiltersCore
      VTK::IOXML
  )

  vtk_module_autoinit(
    TARGETS ApparentHorizonFinderPostprocess
    MODULES VTK::CommonCore VTK::CommonDataModel VTK::FiltersCore VTK::IOXML
  )

  add_executable(ahf-postprocess ahf-postprocess.cpp)
  target_link_libraries(ahf-postprocess PRIVATE ApparentHorizonFinderPostprocess)
else()
  message(STATUS "VTK not found: skipping ahf-postprocess and the AHF tests that read .vtu files")
endif()

if(ENABLE_CCZ4_AHF_TESTS)
  enable_testing()

  if(VTK_FOUND)
    add_executable(ahf-io-tests test/io/IoTests.cpp)
    target_link_libraries(ahf-io-tests PRIVATE ApparentHorizonFinderPostprocess)
    add_test(NAME ahf-io-tests COMMAND ahf-io-tests)
  endif()

  add_executable(ahf-data-tests test/data/AhfDataTests.cpp)
  target_link_libraries(ahf-data-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-data-tests COMMAND ahf-data-tests)

  add_executable(ahf-options-tests test/options/AHFinderOptionsTests.cpp)
  target_link_libraries(ahf-options-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-options-tests COMMAND ahf-options-tests)

  add_executable(ahf-postprocess-options-tests
    test/postprocess-options/PostprocessOptionsTests.cpp)
  target_link_libraries(ahf-postprocess-options-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-postprocess-options-tests COMMAND ahf-postprocess-options-tests)

  add_executable(ahf-result-tests test/result/AHFinderResultTests.cpp)
  target_link_libraries(ahf-result-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-result-tests COMMAND ahf-result-tests)

  add_executable(ahf-output-tests test/output/AHOutputTests.cpp)
  target_link_libraries(ahf-output-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-output-tests COMMAND ahf-output-tests)

  add_executable(ahf-search-tests test/search/AHSearchTests.cpp)
  target_link_libraries(ahf-search-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-search-tests COMMAND ahf-search-tests)

  add_executable(ahf-merger-orchestration-tests
    test/merger-orchestration/MergerOrchestrationTests.cpp)
  target_link_libraries(ahf-merger-orchestration-tests
    PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-merger-orchestration-tests
    COMMAND ahf-merger-orchestration-tests)

  add_executable(ahf-common-horizon-candidate-tests
    test/common-horizon-candidates/CommonHorizonCandidateTests.cpp)
  target_link_libraries(ahf-common-horizon-candidate-tests
    PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-common-horizon-candidate-tests
    COMMAND ahf-common-horizon-candidate-tests)

  add_executable(ahf-common-horizon-search-tests
    test/common-horizon-search/CommonHorizonSearchTests.cpp)
  target_link_libraries(ahf-common-horizon-search-tests
    PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-common-horizon-search-tests
    COMMAND ahf-common-horizon-search-tests)

  if(VTK_FOUND)
    add_executable(ahf-multi-horizon-dataset-tests
      test/integration/MultiHorizonDatasetTests.cpp)
    target_link_libraries(ahf-multi-horizon-dataset-tests
      PRIVATE ApparentHorizonFinderPostprocess)
    add_test(NAME ahf-multi-horizon-dataset-tests
      COMMAND ahf-multi-horizon-dataset-tests)
    set_tests_properties(ahf-multi-horizon-dataset-tests
      PROPERTIES LABELS "integration;dataset")
  endif()

  add_executable(ahf-interpolation-tests test/interpolation/InterpolationTests.cpp)
  target_link_libraries(ahf-interpolation-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-interpolation-tests COMMAND ahf-interpolation-tests)

  add_executable(ahf-surface-geometry-tests test/surface-geometry/SurfaceGeometryTests.cpp)
  target_link_libraries(ahf-surface-geometry-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-surface-geometry-tests COMMAND ahf-surface-geometry-tests)

  if(VTK_FOUND)
    add_executable(ahf-sample-t0-user-workflow-tests
      test/integration/SampleT0UserWorkflowTests.cpp)
    target_link_libraries(ahf-sample-t0-user-workflow-tests
      PRIVATE ApparentHorizonFinderPostprocess)
    add_test(NAME ahf-sample-t0-user-workflow-tests
      COMMAND ahf-sample-t0-user-workflow-tests)
    set_tests_properties(ahf-sample-t0-user-workflow-tests
      PROPERTIES LABELS integration)
  endif()

  add_executable(ahf-assemble-matrix-tests test/assemble-matrix/AssembleMatrixTests.cpp)
  target_link_libraries(ahf-assemble-matrix-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-assemble-matrix-tests COMMAND ahf-assemble-matrix-tests)

  add_executable(ahf-harmonics-tests test/harmonics/HarmonicsTests.cpp)
  target_link_libraries(ahf-harmonics-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-harmonics-tests COMMAND ahf-harmonics-tests)

  add_executable(ahf-initial-guess-tests test/initial-guess/InitialGuessTests.cpp)
  target_link_libraries(ahf-initial-guess-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-initial-guess-tests COMMAND ahf-initial-guess-tests)

  add_executable(ahf-spectral-solver-tests test/spectral-solver/SpectralSolverTests.cpp)
  target_link_libraries(ahf-spectral-solver-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-spectral-solver-tests COMMAND ahf-spectral-solver-tests)

  add_executable(ahf-interpolated-surface-residual-tests
    test/interpolated-surface-residual/InterpolatedSurfaceResidualTests.cpp)
  target_link_libraries(ahf-interpolated-surface-residual-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-interpolated-surface-residual-tests
    COMMAND ahf-interpolated-surface-residual-tests)

  add_executable(ahf-solve-pde-for-ah-tests
    test/interpolated-horizon-solver/InterpolatedHorizonSolverTests.cpp)
  target_link_libraries(ahf-solve-pde-for-ah-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-solve-pde-for-ah-tests
    COMMAND ahf-solve-pde-for-ah-tests)

  add_executable(ahf-spacetimes-tests test/spacetimes/SpacetimeTests.cpp)
  target_link_libraries(ahf-spacetimes-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-spacetimes-tests COMMAND ahf-spacetimes-tests)

  add_executable(ahf-analytic-horizon-tests test/analytic-horizon/AnalyticHorizonTests.cpp)
  target_link_libraries(ahf-analytic-horizon-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-analytic-horizon-tests COMMAND ahf-analytic-horizon-tests)

  add_executable(ahf-diagnostics-tests test/diagnostics/DiagnosticsTests.cpp)
  target_link_libraries(ahf-diagnostics-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-diagnostics-tests COMMAND ahf-diagnostics-tests)

  add_executable(ahf-brill-lindquist-tests test/brill-lindquist/BrillLindquistTests.cpp)
  target_link_libraries(ahf-brill-lindquist-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-brill-lindquist-tests COMMAND ahf-brill-lindquist-tests)

  add_executable(ahf-iteration-tests test/iteration/IterationTests.cpp)
  target_link_libraries(ahf-iteration-tests PRIVATE ApparentHorizonFinderCore)
  add_test(NAME ahf-iteration-tests COMMAND ahf-iteration-tests)

  # Figure-data generator for figures/plot-triangle-horizon.py. Deliberately not
  # an add_test: it asserts nothing, it writes a JSON history of the Newton
  # solver. figures/ is not in the repository, so only build it where it exists.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/figures/dump-triangle-horizon.cpp)
    add_executable(ahf-dump-triangle-horizon figures/dump-triangle-horizon.cpp)
    target_link_libraries(ahf-dump-triangle-horizon PRIVATE ApparentHorizonFinderCore)
  endif()
endif()
