configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/amgx.json
                 ${CMAKE_CURRENT_BINARY_DIR}/amgx.json COPYONLY)
configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/points.xyz
                 ${CMAKE_CURRENT_BINARY_DIR}/points.xyz COPYONLY)

include (${CMAKE_CURRENT_SOURCE_DIR}/add_test.cmake)

## if SKIP_MATH is provided as the optional third argument below, unit tests
## will be compiled without checking for math exceptions.

Plato_add_test(MassProperties MassPropertiesTests.cpp)
Plato_add_test(VolumeAverageCriterion VolumeAverageCriterionTests.cpp SKIP_MATH)
Plato_add_test(GeometryMisfit GeometryMisfitTests.cpp)
Plato_add_test(LinearMaterialModel LinearMaterialModelTests.cpp)
Plato_add_test(PlatoAugLagStress PlatoAugLagStressTest.cpp)
Plato_add_test(PlatoMathHelpers PlatoMathHelpersTest.cpp)
Plato_add_test(PlatoMaterialModel PlatoMaterialModelTest.cpp)
Plato_add_test(PlatoFunctors PlatoFunctorsTest.cpp)
Plato_add_test(PlatoSpatialModel PlatoSpatialModelTests.cpp)
Plato_add_test(Electroelastic ElectroelasticTests.cpp)
Plato_add_test(Thermoelastic ThermoelasticTests.cpp)
Plato_add_test(EllipticUpdLag EllipticUpdLagTests.cpp)
Plato_add_test(TwoField TwoFieldTests.cpp)
Plato_add_test(Derivative DerivativeTests.cpp)
Plato_add_test(SolverInterface PlatoSolverInterfaceTests.cpp)
Plato_add_test(PlatoMathTypes PlatoMathTypesTests.cpp)
Plato_add_test(MultipointConstraint PlatoMultipointConstraintTests.cpp)
Plato_add_test(AugLagStress PlatoAugLagStressTest.cpp)
Plato_add_test(TensorNorm TensorNormTests.cpp)

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
      ${PLATO_LIBS}
      ${Trilinos_LIBRARIES}
      ${Trilinos_TPL_LIBRARIES}
      )

    target_include_directories(EngineMeshIntxTests PRIVATE "${PLATOENGINE_PREFIX}/include")
    target_include_directories(EngineMeshIntxTests PRIVATE "${CMAKE_SOURCE_DIR}/src")
  endif()

endif()
