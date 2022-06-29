cmake_minimum_required(VERSION 3.0)

project(PLATOANALYZE VERSION 0.0.0 LANGUAGES CXX)

# hdf5 dependency (introduced by exodus) needs C
enable_language(C)

option(PLATOANALYZE_ENABLE_CUDA "Build for Nvidia GPUs using CUDA" OFF)

option(PLATOANALYZE_ENABLE_MPMD "Enable the Plato interface MPMD executable" OFF)

if(PLATOANALYZE_ENABLE_MPMD)
  option(PLATOANALYZE_ENABLE_GEOMETRY "Enable Plato Geometry" OFF)
  option(PLATOANALYZE_ENABLE_ESP      "Enable ESP"            OFF)
endif()

option(PLATOANALYZE_ENABLE_MESHMAP  "Enable Plato MeshMap"  OFF)

option(PLATOANALYZE_ENABLE_ENGINEMESH "Enable EngineMesh"           OFF)
option(PLATOANALYZE_ENABLE_TPETRA     "Enable Tpetra"               OFF)
option(PLATOANALYZE_ENABLE_EPETRA     "Enable Tpetra"               OFF)
option(PLATOANALYZE_ENABLE_PYTHON     "Enable the Python interface" OFF)

option(PLATOANALYZE_1D "Compile with 1D physics"             OFF) # TODO delete
option(PLATOANALYZE_2D "Compile with 2D physics"             OFF) # TODO delete
option(PLATOANALYZE_3D "Compile with 3D physics"             ON ) # TODO delete

option(ELLIPTIC        "Compile with Elliptic physics"       ON )
option(PARABOLIC       "Compile with Parabolic physics"      OFF) # TODO
option(HYPERBOLIC      "Compile with Hyperbolic physics"     OFF) # TODO
option(STABILIZED      "Compile with Stabilized VMS physics" OFF) # TODO
option(PLASTICITY      "Compile with Plasticity physics"     OFF) # TODO
option(HELMHOLTZ       "Compile with Helmholtz filter"       OFF) # TODO
option(EXPRESSION      "Compile with Expression evaluator"   OFF)

option(PLATOANALYZE_UNIT_TEST   "Flag to enable unit tests"  ON)
option(PLATOANALYZE_SMOKE_TESTS "Flag to enable smoke tests" ON)

option(PLATOANALYZE_LONG_LONG_ORDINALTYPE "Flag to change Plato::OrdinalType to 'long long'" OFF)

option( PLATOANALYZE_CHECK_FPE "Flag to enable Plato Analyze checks for floating point exceptions" OFF)


MESSAGE("\n\nConfiguring Plato Analyze Version ${PLATOANALYZE_VERSION_MAJOR}.${PLATOANALYZE_VERSION_MINOR}.${PLATOANALYZE_VERSION_PATCH}")

IF (DEFINED Trilinos_PREFIX)
  SET(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})
ENDIF()

#IF (DEFINED SuperLU_PREFIX)
#  SET(CMAKE_PREFIX_PATH ${SuperLU_PREFIX} ${CMAKE_PREFIX_PATH})
#ENDIF()

#silences warnings about the system not supporting shared libraries
#http://stackoverflow.com/questions/12264299/cmake-on-linux-target-platform-does-not-support-dynamic-linking
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Get Trilinos as one entity
FIND_PACKAGE(Trilinos REQUIRED)
list(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)

IF(NOT Trilinos_FOUND)
  MESSAGE(FATAL_ERROR "Cannot find Trilinos - build exiting!!!")
ENDIF()

# Echo trilinos build info just for fun
MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_PREFIX = ${Trilinos_PREFIX}")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")

set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
set(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
set(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})

set(THREAD_COUNT "" CACHE STRING "Default number of threads to use")
if(THREAD_COUNT)
  set(ALL_THREAD_ARGS "--kokkos-threads=${THREAD_COUNT}")
else()
  set(ALL_THREAD_ARGS "")
endif()

MESSAGE("-- BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")

SET(CMAKE_C_FLAGS ${Trilinos_C_COMPILER_FLAGS} )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Trilinos_CXX_COMPILER_FLAGS} ${CUSTOM_CXX_FLAGS}")

#FIND_PACKAGE(MPI REQUIRED)

## this has to be conditional, or building without cuda fails
if (PLATOANALYZE_ENABLE_CUDA)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
ENDIF ()

MESSAGE("-- CXX Flags being used in this build are : ${CMAKE_CXX_FLAGS}")

# disallow in-source builds

get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)

get_filename_component(bindir "${CMAKE_BINARY_DIR}" REALPATH)

if(${srcdir} STREQUAL ${bindir})

    message("######################################################")

    message("You are attempting to build in your Source Directory.")

    message("You must run cmake from a build directory.")

    message("######################################################")

    message(FATAL_ERROR "In-source builds are forbidden!")

endif()

FIND_PACKAGE(KokkosCore REQUIRED)
FIND_PACKAGE(KokkosContainers REQUIRED)
FIND_PACKAGE(KokkosKernels REQUIRED)
FIND_PACKAGE(Teuchos REQUIRED)

SET(OMEGA_H_PREFIX "" CACHE PATH "Omega_h install location")
IF(OMEGA_H_PREFIX)
  SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OMEGA_H_PREFIX})
  FIND_PACKAGE(Omega_h 9.17.0 REQUIRED)
ENDIF()

# DAI: omega_h2 has a vtkdiff which by default compares all timesteps.
# old omega_h's vtkdiff only compared one timestep.
# at the time of transition, we will keep comparing just one time step
# this flag is needed to do that.

# We need to do this better, but for now lets see how well it works to find the right mpirun
# This needs to be the one on the front end for Cuda and MIC machines - it runs the utilities.

IF(PLATOANALYZE_MPIEXEC)
  MESSAGE("-- User defined PLATOANALYZE_MPIEXEC=${PLATOANALYZE_MPIEXEC}")
  SEPARATE_ARGUMENTS(PLATOANALYZE_MPIEXEC)
ELSE()
  SET(PLATOANALYZE_MPIEXEC)

  #mpirun is frequently found alongside mpicxx, try looking for it there.
  get_filename_component(COMPILER_DIR ${CMAKE_CXX_COMPILER} PATH)

  FIND_FILE(
    PLATOANALYZE_MPIEXEC
    NAMES mpirun mpiexec
    PATHS "${COMPILER_DIR}" ENV PATH
    )

  IF(PLATOANALYZE_MPIEXEC)
    MESSAGE("-- Found PLATOANALYZE_MPIEXEC=${PLATOANALYZE_MPIEXEC}")
  ELSE()
    MESSAGE(FATAL_ERROR "Fatal Error: Cannot find mpirun or mpiexec for PLATOANALYZE_MPIEXEC")
  ENDIF()
ENDIF()

add_definitions(-DPLATOANALYZE_USE_EXPLICIT_INSTANTIATION)

if ( PLATOANALYZE_CHECK_FPE)
  add_definitions(-DPLATOANALYZE_CHECK_FPE)
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    add_definitions(-DPLATOANALYZE_USE_GNU_FPE)
    message("-- Building with GNU FPE checking enabled.")
  else()
    message("-- Building with XMM FPE checking enabled.")
  endif()
else()
  message("-- Building without FPE checking.")
endif()

if(PLATOANALYZE_LONG_LONG_ORDINALTYPE)
  add_definitions(-DPLATOANALYZE_LONG_LONG_ORDINALTYPE)
  message("-- Building with 'long long' ordinal type.")
endif()

add_executable(
    analyze
    src/alg/Run.cpp
    src/alg/Analyze.cpp
  )

if( CMAKE_INSTALL_PREFIX )
  install( TARGETS analyze DESTINATION ${CMAKE_INSTALL_PREFIX}/bin )
endif()
target_include_directories(analyze PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
#target_link_libraries(analyze analyzelib)
target_link_libraries( analyze analyzelib ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} )

IF (DEFINED AMGX_PREFIX)
  ADD_DEFINITIONS(-DHAVE_AMGX)
  FIND_PATH(AMGX_INCLUDE_DIR NAMES amgx_c.h PATHS ${AMGX_PREFIX}/include)
  IF (NOT AMGX_INCLUDE_DIR)
    MESSAGE(FATAL_ERROR "Couldn't find AMGX header!")
  ENDIF()
  FIND_LIBRARY(AMGX_LIB NAMES amgxsh amgx PATHS ${AMGX_PREFIX}/lib)
  IF (NOT AMGX_LIB)
    MESSAGE(FATAL_ERROR "Couldn't find AMGX library!")
  ENDIF()
  IF(PLATOANALYZE_LONG_LONG_ORDINALTYPE)
    message(FATAL_ERROR "Can't build with AMGX when building with 'long long' ordinal type. \n
    Please set -DPLATOANALYZE_LONG_LONG_ORDINALTYPE=OFF or unset AMGX_PREFIX")
  ENDIF()
  SET(AMGX_INCLUDE_DIRS "${AMGX_INCLUDE_DIR}")
  SET(AMGX_LIBRARIES "${AMGX_LIB}")
  SET(AMGX_FOUND TRUE) # a bit of a hack; would be better to check that it actually exists!
  MESSAGE("-- Found AmgX!")
  MESSAGE("-- AMGX_INCLUDE_DIRS = ${AMGX_INCLUDE_DIRS}")
  MESSAGE("-- AMGX_LIBRARIES = ${AMGX_LIBRARIES}")
ENDIF()

SET(SOLVER_INTERFACE_LIBRARIES)

IF( AMGX_FOUND )
  SET(SOLVER_INTERFACE_LIBRARIES
      ${SOLVER_INTERFACE_LIBRARIES}
      ${AMGX_LIBRARIES}
     )
ENDIF()

if (PLATOANALYZE_1D)
  add_definitions(-DPLATOANALYZE_1D)
  message("-- Plato: 1d physics added")
endif()
if (PLATOANALYZE_2D)
  add_definitions(-DPLATOANALYZE_2D)
  message("-- Plato: 2d physics added")
endif()
if (PLATOANALYZE_3D)
  add_definitions(-DPLATOANALYZE_3D)
  message("-- Plato: 3d physics added")
endif()

if(ELLIPTIC)
  message("-- Plato: Elliptic physics added")
  ADD_DEFINITIONS(-DPLATO_ELLIPTIC)
endif()
if(PARABOLIC)
  message("-- Plato: Parabolic physics added")
  ADD_DEFINITIONS(-DPLATO_PARABOLIC)
endif()
if(HYPERBOLIC)
  message("-- Plato: Hyperbolic physics added")
  ADD_DEFINITIONS(-DPLATO_HYPERBOLIC)
endif()
if(STABILIZED)
  message("-- Plato: Stabilized physics added")
  ADD_DEFINITIONS(-DPLATO_STABILIZED)
endif()
if(PLASTICITY)
  message("-- Plato: Plasticity physics added")
  ADD_DEFINITIONS(-DPLATO_PLASTICITY)
endif()
if(HELMHOLTZ)
  message("-- Plato: Helmholtz filter added")
  ADD_DEFINITIONS(-DPLATO_HELMHOLTZ)
endif()

if(EXPRESSION)
  message("-- Plato: Expression evaluator added")
  ADD_DEFINITIONS(-DPLATO_EXPRESSION)
endif()

if (PLATOANALYZE_ENABLE_MESHMAP)
  add_definitions(-DPLATO_MESHMAP)
  find_package(ArborX REQUIRED)
endif()


IF(PLATOANALYZE_ENABLE_ENGINEMESH)
  FIND_PROGRAM( EXODIFF NAMES exodiff PATHS ${Trilinos_PREFIX}/bin NO_DEFAULT_PATH)
  IF(EXODIFF)
    MESSAGE("-- EXODIFF=${EXODIFF}")
  ELSE()
    MESSAGE(FATAL_ERROR "Cannot locate the executable for exodiff, exiting")
  ENDIF()
ELSE()
  FIND_PROGRAM( VTKDIFF NAMES vtkdiff PATHS ${OMEGA_H_PREFIX}/bin NO_DEFAULT_PATH)
  IF(VTKDIFF)
    MESSAGE("-- VTKDIFF=${VTKDIFF}")
  ELSE()
    MESSAGE(FATAL_ERROR "Cannot locate the executable for vtkdiff, exiting")
  ENDIF()
ENDIF()

if (PLATOANALYZE_ENABLE_TPETRA)
  find_package(Tpetra REQUIRED)
  find_package(Belos REQUIRED)
  find_package(Ifpack2 REQUIRED)
  message("adding define: -DPLATO_TPETRA")
  add_definitions(-DPLATO_TPETRA)
endif()
target_link_libraries(analyze ${PLATO_LIBS} ${Trilinos_LIBRARIES})

if (PLATOANALYZE_ENABLE_EPETRA)
  find_package(Epetra REQUIRED)
  find_package(AztecOO REQUIRED)
  message("adding define: -DPLATO_EPETRA")
  add_definitions(-DPLATO_EPETRA)
endif()

###### PLATO ANALYZE APP + PLATO ENGINE EXECUTABLE ######
if (PLATOANALYZE_ENABLE_MPMD)
  FIND_PACKAGE(Epetra REQUIRED)
  set(PLATOENGINE_PREFIX "" CACHE STRING "Path to Plato install")
  find_library(PLATO_DATA_LAYER_LIB NAMES PlatoDataLayer PATHS ${PLATOENGINE_PREFIX}/lib)
  find_library(PLATO_INTERFACE_LIB NAMES PlatoInterface PATHS ${PLATOENGINE_PREFIX}/lib)
  find_library(PLATO_PARSER_LIB NAMES PlatoParser PATHS ${PLATOENGINE_PREFIX}/lib)
  find_library(PLATO_EXCEPTIONS_LIB NAMES PlatoExceptions PATHS ${PLATOENGINE_PREFIX}/lib)
  find_library(PLATO_CONSOLE_LIB NAMES PlatoConsole PATHS ${PLATOENGINE_PREFIX}/lib)
  find_package(Epetra REQUIRED)
  find_package(AztecOO REQUIRED)
  find_package(SEACASExodus REQUIRED)
  find_package(SEACASNemesis REQUIRED)
  find_package(Shards REQUIRED)
  find_package(Intrepid REQUIRED)
  find_package(TeuchosCore REQUIRED)

  set(PLATO_LIBS
      ${PLATO_DATA_LAYER_LIB}
      ${PLATO_INTERFACE_LIB}
      ${PLATO_PARSER_LIB}
      ${PLATO_EXCEPTIONS_LIB}
      ${PLATO_CONSOLE_LIB}
      aztecoo
      epetra
      intrepid
      shards
      sacado
      teuchoscore
      nemesis
      exodus
     )

  add_library( Analyze_App src/Analyze_App.cpp )
  target_include_directories(Analyze_App PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_include_directories(Analyze_App PRIVATE "${PLATOENGINE_PREFIX}/include")
  target_include_directories(Analyze_App PRIVATE "${Trilinos_INCLUDE_DIRS}")
  if( CMAKE_INSTALL_PREFIX )
    install( TARGETS Analyze_App DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
  endif()

  find_package(HDF5 REQUIRED)
  target_link_libraries(Analyze_App HDF5::HDF5)

  add_executable( analyze_MPMD src/Analyze_MPMD.cpp)
  if( CMAKE_INSTALL_PREFIX )
    install( TARGETS analyze_MPMD DESTINATION ${CMAKE_INSTALL_PREFIX}/bin )
  endif()
  target_include_directories(analyze_MPMD PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_include_directories(analyze_MPMD PRIVATE "${PLATOENGINE_PREFIX}/include")
  target_include_directories(analyze_MPMD PRIVATE "${Trilinos_INCLUDE_DIRS}")

  target_include_directories(analyze PRIVATE "${PLATOENGINE_PREFIX}/include")

  if (PLATOANALYZE_ENABLE_GEOMETRY)
    add_definitions(-DPLATO_GEOMETRY)
    find_library(PLATO_GEOMETRY_MLS_LIB NAMES PlatoGeometryMLS PATHS ${PLATOENGINE_PREFIX}/lib)
    find_library(PLATO_GEOMETRY_COGENT_LIB NAMES PlatoGeometryCogent PATHS ${PLATOENGINE_PREFIX}/lib)
    find_package(Pamgen REQUIRED)
    set(PLATO_LIBS ${PLATO_LIBS} ${PLATO_GEOMETRY_MLS_LIB} ${PLATO_GEOMETRY_COGENT_LIB} pamgen)
  endif()

  if (PLATOANALYZE_ENABLE_ENGINEMESH)
    message("Default mesh is EngineMesh")
  else()
    add_definitions(-DUSE_OMEGAH_MESH)
    message("Default mesh is Omega-h mesh")
  endif()

  if (PLATOANALYZE_ENABLE_MESHMAP)
    target_link_libraries(Analyze_App ArborX::ArborX)
  endif()

  if (PLATOANALYZE_ENABLE_ESP)
    find_library(PLATO_GEOMETRY_ESP_LIB NAMES PlatoGeometryESP PATHS ${PLATOENGINE_PREFIX}/lib)
    find_library(ESP_LIB NAMES egads PATHS ${ESP_LIB_DIR})
    find_library(OCSM_LIB NAMES ocsm PATHS ${ESP_LIB_DIR})
    message("adding define: -DPLATO_ESP")
    add_definitions(-DPLATO_ESP)
    include_directories( ${ESP_INC_DIR} )
    set(PLATO_LIBS ${PLATO_LIBS} ${PLATO_GEOMETRY_ESP_LIB} ${OCSM_LIB} ${ESP_LIB} ${OCSM_LIB})
  endif()

  target_link_libraries(Analyze_App ${PLATO_LIBS} ${PLATO_LIBS} analyzelib)
  target_link_libraries(analyze_MPMD Analyze_App ${PLATO_LIBS} ${PLATO_LIBS} analyzelib)

  if (PLATOANALYZE_ENABLE_PYTHON)
      find_package(Python3 COMPONENTS Development)
      include_directories( ${Python3_INCLUDE_DIRS} )
      add_library( PlatoPython SHARED src/PlatoPythonModule.cpp )
      target_include_directories( PlatoPython PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_include_directories( PlatoPython PRIVATE "${PLATOENGINE_PREFIX}/include")
      target_link_libraries( PlatoPython Analyze_App analyzelib ${PLATO_LIBS} ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ${Python3_LIBRARIES} )
      set_target_properties( PlatoPython PROPERTIES PREFIX "" )
      target_compile_options( PlatoPython PRIVATE "-lmpi" )
      if( CMAKE_INSTALL_PREFIX )
        install( TARGETS PlatoPython DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
      endif()
  endif()

endif()

list(REVERSE Trilinos_TPL_LIBRARIES)
list(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)
list(REVERSE Trilinos_TPL_LIBRARIES)
message(STATUS "Trilinos_EXTRA_LD_FLAGS: ${Trilinos_EXTRA_LD_FLAGS}")

IF(OMEGA_H_PREFIX)
 # omega-h adds -Werror which currently breaks build
 set_target_properties(Omega_h::omega_h PROPERTIES INTERFACE_COMPILE_OPTIONS "")
ENDIF()

ENABLE_TESTING()
INCLUDE(CTest)

SET(PLATOANALYZE_RANKS_PER_NODE_FLAG "-npernode" CACHE STRING "Processes per node flag for PLATOANALYZE_MPIEXEC")
SET(PLATOANALYZE_RANKS_FLAG "-n" CACHE STRING "Number of processes flag for PLATOANALYZE_MPIEXEC")
SET(PLATOANALYZE_RANKS_PER_NODE "1" CACHE STRING "Maximum number of processes per node")
SET(PLATOANALYZE_VALGRIND "" CACHE STRING "Optional Valgrind call and arguments")

function(build_mpi_test_string RESULT_NAME num_mpi_proc)
  string(REPLACE " " ";" VALGRIND "${PLATOANALYZE_VALGRIND}")
  set(${RESULT_NAME} ${PLATOANALYZE_MPIEXEC} ${PLATOANALYZE_RANKS_PER_NODE_FLAG} ${PLATOANALYZE_RANKS_PER_NODE} ${PLATOANALYZE_RANKS_FLAG} ${num_mpi_proc} ${VALGRIND} ${ARGN} PARENT_SCOPE)
endfunction(build_mpi_test_string)

ADD_SUBDIRECTORY(src)

if( PLATOANALYZE_UNIT_TEST)
  ADD_SUBDIRECTORY(unit_tests)
endif()

if(PLATOANALYZE_SMOKE_TESTS)
  add_subdirectory(tests)
endif()

set(PLATOANALYZE_KEY_INTS
    PLATOANALYZE_VERSION_MAJOR
    PLATOANALYZE_VERSION_MINOR
    PLATOANALYZE_VERSION_PATCH
   )
