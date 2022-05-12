configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AnalyzeConfig.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/AnalyzeConfig.hpp
  @ONLY)

## material model sources
set(ANALYZE_SOURCES
    ParseTools.cpp
    CustomMaterial.cpp
    AbstractLinearStress.cpp
    LinearStress.cpp
    LinearStressFactory.cpp
##    ElasticModelFactory.cpp
    LinearElasticMaterial.cpp
    CubicLinearElasticMaterial.cpp
    CustomLinearElasticMaterial.cpp
    LinearElectroelasticMaterial.cpp
    IsotropicLinearElasticMaterial.cpp
    OrthotropicLinearElasticMaterial.cpp
    LinearThermoelasticMaterial.cpp
    MaterialModel.cpp
    Rank4Voigt.cpp
    IsotropicStiffness.cpp
)

if(EXPRESSION)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
      ExpressionEvaluator.cpp
      LinearStressExpression.cpp
      YieldStressExpression.cpp
  )
endif()

## Augmented Lagrangian sources
# TODO
#set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
#    TensileEnergyDensityLocalMeasure.cpp
#    TensileEnergyDensity.cpp
#    Plato_AugLagStressCriterionQuadratic.cpp
#    Plato_AugLagStressCriterionGeneral.cpp
#    Plato_AugLagStressCriterion.cpp
#    Plato_StructuralMass.cpp
#    VonMisesLocalMeasure.cpp
#    ThermalVonMisesLocalMeasure.cpp
#)

## core sources
set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    WorkSets.cpp
    Variables.cpp
    Solutions.cpp
    AnalyzeAppUtils.cpp
    PlatoMask.cpp
    EngineMesh.cpp
    EngineMeshIO.cpp
    mesh/ExodusIO.cpp
    alg/ParseInput.cpp
    alg/Basis.cpp
    alg/Cubature.cpp
    alg/ParallelComm.cpp
    alg/AmgXConfigs.cpp
    alg/AmgXLinearSolver.cpp
    alg/PlatoSolverFactory.cpp
)

if (PLATOANALYZE_ENABLE_EPETRA)
set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    alg/EpetraLinearSolver.cpp
)
endif()

IF(OMEGA_H_PREFIX)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    OmegaHMesh.cpp
    OmegaHMeshIO.cpp
  )
ENDIF()

## Stabilized VMS sources
if(STABILIZED)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    EllipticVMSProblem.cpp
    StabilizedElastostaticEnergy.cpp
  )
endif()

## plasticity sources
if(PLASTICITY)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    AbstractYieldStress.cpp
    YieldStress.cpp
    YieldStressFactory.cpp
    J2PlasticityLocalResidual.cpp
    NewtonRaphsonSolver.cpp
    PathDependentAdjointSolver.cpp
    GlobalVectorFunctionInc.cpp
    PathDependentScalarFunctionFactory.cpp
    BasicLocalScalarFunction.cpp
    PlasticityProblem.cpp
    ComputePrincipalStresses.cpp
    WeightedLocalScalarFunction.cpp
    InfinitesimalStrainPlasticityResidual.cpp
    InfinitesimalStrainThermoPlasticityResidual.cpp
    ElasticWorkCriterion.cpp
    PlasticWorkCriterion.cpp
    TotalWorkCriterion.cpp
    ThermoplasticityThermalEnergyCriterion.cpp
  )
endif()

## geometric sources
if(PLATOANALYZE_ENABLE_MESHMAP)
  ## Mesh constraint sources
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
      MultipointConstraints.cpp
  )

  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    geometric/DivisionFunction.cpp
    geometric/GeometryScalarFunction.cpp
    geometric/LeastSquaresFunction.cpp
#TODO    geometric/MassMoment.cpp
#TODO    geometric/MassPropertiesFunction.cpp
    geometric/ScalarFunctionBaseFactory.cpp
    geometric/WeightedSumFunction.cpp
  )
endif()

## elliptic sources
if(ELLIPTIC)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    elliptic/Problem.cpp
#TODO    elliptic/SolutionFunction.cpp
#TODO    elliptic/DivisionFunction.cpp
#TODO    elliptic/LeastSquaresFunction.cpp
#TODO    elliptic/WeightedSumFunction.cpp
#TODO    elliptic/ScalarFunctionBaseFactory.cpp
    elliptic/PhysicsScalarFunction.cpp
#TODO    elliptic/MassMoment.cpp
#TODO    elliptic/MassPropertiesFunction.cpp
#TODO    elliptic/FluxPNorm.cpp
#TODO    elliptic/StressPNorm.cpp
#TODO    elliptic/TMStressPNorm.cpp
#TODO    elliptic/VolAvgStressPNormDenominator.cpp
#TODO    elliptic/EMStressPNorm.cpp
#TODO    elliptic/EffectiveEnergy.cpp
#TODO    elliptic/InternalThermalEnergy.cpp
#TODO    elliptic/InternalThermoelasticEnergy.cpp
#TODO    elliptic/InternalElectroelasticEnergy.cpp
#TODO    elliptic/InternalElasticEnergy.cpp
    elliptic/ElastostaticResidual.cpp
#TODO    elliptic/ThermostaticResidual.cpp
#TODO    elliptic/VolumeAverageCriterion.cpp
#TODO    IntermediateDensityPenalty.cpp
#TODO    elliptic/updated_lagrangian/Problem.cpp
#TODO    elliptic/updated_lagrangian/ElastostaticResidual.cpp
#TODO    elliptic/updated_lagrangian/ScalarFunctionBaseFactory.cpp
#TODO    elliptic/updated_lagrangian/PhysicsScalarFunction.cpp
#TODO    elliptic/updated_lagrangian/InternalElasticEnergy.cpp
#TODO    elliptic/updated_lagrangian/AbstractEllipticUpLagLinearStress.cpp
#TODO    elliptic/updated_lagrangian/EllipticUpLagLinearStress.cpp
#TODO    elliptic/updated_lagrangian/EllipticUpLagLinearStressFactory.cpp
  )

  if(EXPRESSION)
    set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
      elliptic/updated_lagrangian/EllipticUpLagLinearStressExpression.cpp
    )
  endif()
endif()

## parabolic sources
if(PARABOLIC)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    parabolic/Problem.cpp
    parabolic/InternalThermalEnergy.cpp
    parabolic/InternalThermoelasticEnergy.cpp
    parabolic/TemperatureAverage.cpp
    parabolic/ScalarFunctionBaseFactory.cpp
    parabolic/PhysicsScalarFunction.cpp
    parabolic/HeatEquationResidual.cpp
    parabolic/TransientThermomechResidual.cpp
  )
endif()

## hyperbolic sources
if(HYPERBOLIC)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    AbstractVolumetricSource.cpp
    hyperbolic/BrinkmanForces.cpp
    hyperbolic/CriterionVolume.cpp
    hyperbolic/ThermalBuoyancy.cpp
    hyperbolic/PressureResidual.cpp
    hyperbolic/CriterionFlowRate.cpp
    hyperbolic/TemperatureResidual.cpp
    hyperbolic/FluidsQuasiImplicit.cpp
    hyperbolic/FluidsScalarFunction.cpp
    hyperbolic/FluidsVectorFunction.cpp
    hyperbolic/FluidsThermalSources.cpp
    hyperbolic/FluidsWorkSetBuilders.cpp
    hyperbolic/InternalThermalForces.cpp
    hyperbolic/MomentumSurfaceForces.cpp
    hyperbolic/FluidsCriterionFactory.cpp
    hyperbolic/AbstractHyperbolicLinearStress.cpp
    hyperbolic/AbstractScalarFunction.cpp
    hyperbolic/AbstractVectorFunction.cpp
    hyperbolic/VelocityCorrectorResidual.cpp
    hyperbolic/VelocityPredictorResidual.cpp
    hyperbolic/CriterionThermalCompliance.cpp
    hyperbolic/FluidsUniformThermalSource.cpp
    hyperbolic/CriterionSurfaceThermalFlux.cpp
    hyperbolic/FluidsWeightedScalarFunction.cpp
    hyperbolic/HyperbolicLinearStress.cpp
    hyperbolic/HyperbolicLinearStressFactory.cpp
    hyperbolic/HyperbolicScalarFunctionFactory.cpp
    hyperbolic/HyperbolicPhysicsScalarFunction.cpp
    hyperbolic/CriterionMeanSurfacePressure.cpp
    hyperbolic/FluidsLeastSquaresScalarFunction.cpp
    hyperbolic/CriterionMeanSurfaceTemperature.cpp
    hyperbolic/FluidsStabilizedUniformThermalSource.cpp
    hyperbolic/MicromorphicLinearElasticMaterial.cpp
    hyperbolic/CubicMicromorphicLinearElasticMaterial.cpp
    hyperbolic/MicromorphicInertiaMaterial.cpp
    hyperbolic/CubicMicromorphicInertiaMaterial.cpp
  )

  if(EXPRESSION)
    set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
      hyperbolic/HyperbolicLinearStressExpression.cpp
    )
  endif()
endif()

## helmholtz sources
if(HELMHOLTZ)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    helmholtz/Problem.cpp
    helmholtz/HelmholtzResidual.cpp
  )
endif()

if (PLATOANALYZE_ENABLE_TPETRA)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES} alg/TpetraLinearSolver.cpp)
endif()

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/mesh)
include_directories(${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})

add_library( analyzelib ${ANALYZE_SOURCES})
SET_TARGET_PROPERTIES(analyzelib PROPERTIES OUTPUT_NAME analyze)
if( CMAKE_INSTALL_PREFIX )
  install( TARGETS analyzelib DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
endif()
# for AnalyzeConfig.hpp
target_include_directories(analyzelib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(analyzelib PRIVATE "${PLATOENGINE_PREFIX}/include")

set(LINK_LIBS
    ${PLATO_LIBS}
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
    ${Trilinos_EXTRA_LD_FLAGS}
    ${SOLVER_INTERFACE_LIBRARIES}
    )

IF(OMEGA_H_PREFIX)
  set(LINK_LIBS Omega_h::omega_h ${LINK_LIBS})
ENDIF()

if (PLATOANALYZE_ENABLE_MESHMAP)
  set(LINK_LIBS ArborX::ArborX ${LINK_LIBS})
endif()

target_link_libraries(analyzelib PUBLIC ${LINK_LIBS})
IF( AMGX_FOUND )
  target_include_directories(analyzelib PUBLIC ${AMGX_INCLUDE_DIRS})
  target_include_directories(analyzelib PUBLIC ${PROJECT_SOURCE_DIR}/src/solverInterfaces/AmgX/)
ENDIF()
