# This file is part of the ExaGRyPE project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
file(GLOB_RECURSE SOURCES ${SOURCE_EXTENSIONS_TO_GLOB})
file(GLOB_RECURSE HEADERS ${HEADER_EXTENSIONS_TO_GLOB})
list(FILTER SOURCES EXCLUDE REGEX "\\.template\\.")
list(FILTER HEADERS EXCLUDE REGEX "\\.template\\.")
# TwoPunctures (initial data for black holes) needs GSL
if(NOT WITH_GSL)
  list(FILTER SOURCES EXCLUDE REGEX "/twopunctures/")
  list(FILTER HEADERS EXCLUDE REGEX "/twopunctures/")
endif()
# The apparent-horizon finder is a standalone postprocessing tool with its own
# build (ahf/CMakeLists.txt), not part of the ExaGRyPE library
list(FILTER SOURCES EXCLUDE REGEX "/ahf/")
list(FILTER HEADERS EXCLUDE REGEX "/ahf/")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADERS})

foreach(mode IN LISTS PEANO_BUILD_TYPES)
  string(TOLOWER ${mode} mode_lower)
  string(TOUPPER ${mode} mode_upper)

  # ExaGRyPE solves the Einstein equations and hence is built for 3d only.
  foreach(dimension IN ITEMS 3)
    add_library(ExaGRyPECore${dimension}d_${mode_lower} STATIC)
    target_sources(ExaGRyPECore${dimension}d_${mode_lower} PRIVATE ${SOURCES} ${HEADERS})
    target_link_libraries(ExaGRyPECore${dimension}d_${mode_lower} PRIVATE PeanoOptions)
    target_compile_definitions(ExaGRyPECore${dimension}d_${mode_lower} PUBLIC ${PEANO_${mode_upper}_DEFINITIONS} Dimensions=${dimension})
    target_compile_options(ExaGRyPECore${dimension}d_${mode_lower} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PEANO_CXX_FLAGS_${mode_upper}}>)

    install(TARGETS ExaGRyPECore${dimension}d_${mode_lower}
      EXPORT                      ${PROJECT_NAME}Targets
      RUNTIME DESTINATION         ${CMAKE_INSTALL_BINDIR}
              COMPONENT           ${PROJECT_NAME}_Runtime
      LIBRARY DESTINATION         ${CMAKE_INSTALL_LIBDIR}
              COMPONENT           ${PROJECT_NAME}_Runtime
              NAMELINK_COMPONENT  ${PROJECT_NAME}_Devel
      ARCHIVE DESTINATION         ${CMAKE_INSTALL_LIBDIR}
              COMPONENT           ${PROJECT_NAME}_Devel
      PUBLIC_HEADER DESTINATION   ${CMAKE_INSTALL_INCLUDEDIR}
      PRIVATE_HEADER DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR}
    )
  endforeach()
endforeach()

foreach(pattern IN LISTS HEADER_EXTENSIONS_TO_GLOB)
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN ${pattern}
  )
endforeach()

if(ENABLE_CCZ4_AHF_POSTPROCESSOR)
  add_subdirectory(ahf)
endif()
