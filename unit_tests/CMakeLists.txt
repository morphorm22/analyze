configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/amgx.json
                 ${CMAKE_CURRENT_BINARY_DIR}/amgx.json COPYONLY)
configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/points.xyz
                 ${CMAKE_CURRENT_BINARY_DIR}/points.xyz COPYONLY)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/BamG)

include (${CMAKE_CURRENT_SOURCE_DIR}/add_test.cmake)

## if SKIP_MATH is provided as the optional third argument below, unit tests
## will be compiled without checking for math exceptions.

#Plato_add_test(MassProperties MassPropertiesTests.cpp)
Plato_add_test(VolumeAverageCriterion VolumeAverageCriterionTests.cpp SKIP_MATH)
#TODO Plato_add_test(GeometryMisfit GeometryMisfitTests.cpp)
Plato_add_test(LinearMaterialModel LinearMaterialModelTests.cpp)
#TODO Plato_add_test(PlatoAugLagStress PlatoAugLagStressTest.cpp)
#TODO Plato_add_test(PlatoMathHelpers PlatoMathHelpersTest.cpp)
Plato_add_test(PlatoMaterialModel PlatoMaterialModelTest.cpp)
#Plato_add_test(PlatoFunctors PlatoFunctorsTest.cpp)
Plato_add_test(PlatoSpatialModel PlatoSpatialModelTests.cpp)
#TODO Plato_add_test(Electroelastic ElectroelasticTests.cpp)
Plato_add_test(Thermoelastic ThermoelasticTests.cpp)
#TODO Plato_add_test(EllipticUpdLag EllipticUpdLagTests.cpp)
#TODO Plato_add_test(TwoField TwoFieldTests.cpp)
#TODO Plato_add_test(Derivative DerivativeTests.cpp)
Plato_add_test(Thermostatic ThermostaticTests.cpp)
Plato_add_test(Elastostatic ElastostaticTests.cpp)
#TODO Plato_add_test(Derivative DerivativeTests.cpp)
Plato_add_test(Element ElementTests.cpp)
Plato_add_test(Elastic_Tet10 ElasticTests_Tet10.cpp)
#TODO Plato_add_test(SolverInterface PlatoSolverInterfaceTests.cpp)
Plato_add_test(PlatoMathTypes PlatoMathTypesTests.cpp)
#TODO Plato_add_test(MultipointConstraint PlatoMultipointConstraintTests.cpp)
#TODO Plato_add_test(AugLagStress PlatoAugLagStressTest.cpp)


if(STABILIZED)
  Plato_add_test(StabilizedMechanics StabilizedMechanicsTests.cpp)
endif()

if(PLASTICITY)
  Plato_add_test(Plasticity PlasticityTests.cpp)
  Plato_add_test(ElastoPlasticity ElastoPlasticityTests.cpp SKIP_MATH)
  Plato_add_test(Thermoplasticity ThermoplasticityTests.cpp SKIP_MATH)
endif()

if(PARABOLIC)
  Plato_add_test(TransientThermomech TransientThermomechTests.cpp)
  Plato_add_test(HeatEquation HeatEquationTests.cpp)
endif()

if(HYPERBOLIC)
  Plato_add_test(TransientDynamics TransientDynamicsTests.cpp)
  Plato_add_test(RelaxedMicromorphic RelaxedMicromorphicTests.cpp)
  Plato_add_test(ComputationalFluidDynamics ComputationalFluidDynamicsTests.cpp SKIP_MATH)
endif()

if(HELMHOLTZ)
  Plato_add_test(HelmholtzFilter PlatoHelmholtzFilterTests.cpp)
endif()

if(PLATOANALYZE_ENABLE_MPMD)
  set(AnalyzeApp_HEADERS
    ${CMAKE_SOURCE_DIR}/src/Analyze_App.hpp
    PlatoTestHelpers.hpp
  )
  set(AnalyzeApp_SOURCES
    AnalyzeAppIntxTests.cpp
    PlatoUnitTestMain.cpp
  )
  add_executable(AnalyzeAppIntxTests
    ${AnalyzeApp_SOURCES}
    ${AnalyzeApp_HEADERS}
  )
  target_link_libraries(AnalyzeAppIntxTests
    Analyze_App
    analyzelib
    ${PLATO_LIBS}
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
    )

  target_include_directories(AnalyzeAppIntxTests PRIVATE "${PLATOENGINE_PREFIX}/include")
  target_include_directories(AnalyzeAppIntxTests PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_include_directories(AnalyzeAppIntxTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/BamG")
  target_include_directories(AnalyzeAppIntxTests PRIVATE ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})

  if (PLATOANALYZE_ENABLE_ENGINEMESH)
    set(EngineMesh_HEADERS
      PlatoTestHelpers.hpp
    )
    set(EngineMesh_SOURCES
      EngineMeshIntxTests.cpp
      PlatoUnitTestMain.cpp
    )
    add_executable(EngineMeshIntxTests
      ${EngineMesh_SOURCES}
    )
    target_link_libraries(EngineMeshIntxTests
      analyzelib
      BamGlib
      ${PLATO_LIBS}
      ${Trilinos_LIBRARIES}
      ${Trilinos_TPL_LIBRARIES}
      )

    target_include_directories(EngineMeshIntxTests PRIVATE "${PLATOENGINE_PREFIX}/include")
    target_include_directories(EngineMeshIntxTests PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_include_directories(EngineMeshIntxTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/BamG")
    target_include_directories(EngineMeshIntxTests PRIVATE ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
  endif()

  if(PLATOANALYZE_ENABLE_MESHMAP)
    set(MeshMapTests_SOURCES
      PlatoMeshMapTests.cpp
      PlatoUnitTestMain.cpp
    )

    set(MeshMapTests_HEADERS
      PlatoTestHelpers.hpp
    )

    add_executable(MeshMapUnitTests
      ${MeshMapTests_SOURCES}
      ${MeshMapTests_HEADERS}
    )
    target_link_libraries(MeshMapUnitTests
      analyzelib
      ${PLATO_LIBS}
      ${Trilinos_LIBRARIES}
      ${Trilinos_TPL_LIBRARIES}
      )
    target_include_directories(MeshMapUnitTests PRIVATE "${PLATOENGINE_PREFIX}/include")
    target_include_directories(MeshMapUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(MeshMapUnitTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/BamG")
    target_include_directories(MeshMapUnitTests PRIVATE ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
    target_link_libraries(MeshMapUnitTests ArborX::ArborX BamGlib)

    build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/MeshMapUnitTests)
    add_test(NAME runMeshMapUnitTests COMMAND ${ES_MPI_TEST})
  endif()

endif()
