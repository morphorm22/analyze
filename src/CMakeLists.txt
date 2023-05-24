configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AnalyzeConfig.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/AnalyzeConfig.hpp
  @ONLY)

## material model sources
set(ANALYZE_SOURCES
    ParseTools.cpp
    InputDataUtils.cpp
    ExpressionEvaluator.cpp
    LinearElasticMaterial.cpp
    CubicLinearElasticMaterial.cpp
    LinearElectroelasticMaterial.cpp
    IsotropicLinearElasticMaterial.cpp
    OrthotropicLinearElasticMaterial.cpp
    LinearThermoelasticMaterial.cpp
    MaterialModel.cpp
    Rank4Voigt.cpp
    IsotropicStiffness.cpp
    AbstractTMKinetics.cpp
    LinearTMKinetics.cpp
    NonLinearTMKinetics.cpp
)

if(EXPRESSION)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
      CustomMaterial.cpp
      CustomLinearElasticMaterial.cpp
      LinearStressExpression.cpp
      YieldStressExpression.cpp
      ExpressionTMKinetics.cpp
  )
endif()

set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    TensileEnergyDensityLocalMeasure.cpp
    Plato_AugLagStressCriterionQuadratic.cpp
    Plato_AugLagStressCriterionGeneral.cpp
    elliptic/AugLagStrengthCriterion.cpp
    VonMisesLocalMeasure.cpp
    ThermalVonMisesLocalMeasure.cpp
    optimizer/AugLagDataMng.cpp
)

## core sources
set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    WorkSets.cpp
    Variables.cpp
    Solutions.cpp
    AnalyzeAppUtils.cpp
    PlatoMask.cpp
    EngineMesh.cpp
    EngineMeshIO.cpp
    PlatoMathHelpers.cpp
    mesh/ExodusIO.cpp
    solver/ParseInput.cpp
    solver/ParallelComm.cpp
    solver/AmgXConfigs.cpp
    solver/AmgXLinearSolver.cpp
    solver/PlatoSolverFactory.cpp
    solver/PlatoAbstractSolver.cpp
)

if (PLATOANALYZE_ENABLE_EPETRA)
set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    solver/EpetraLinearSolver.cpp
)
endif()

IF(OMEGA_H_PREFIX)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    OmegaHMesh.cpp
    OmegaHMeshIO.cpp
  )
ENDIF()

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
    geometric/MassMoment.cpp
    geometric/MassPropertiesFunction.cpp
    geometric/ScalarFunctionBaseFactory.cpp
    geometric/WeightedSumFunction.cpp
  )
endif()

## elliptic sources
if(ELLIPTIC)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    elliptic/Problem.cpp
    elliptic/SolutionFunction.cpp
    elliptic/DivisionFunction.cpp
    elliptic/LeastSquaresFunction.cpp
    elliptic/WeightedSumFunction.cpp
    elliptic/ScalarFunctionBaseFactory.cpp
    elliptic/PhysicsScalarFunction.cpp
    elliptic/MassMoment.cpp
    elliptic/MassPropertiesFunction.cpp
    elliptic/VolAvgStressPNormDenominator.cpp
    elliptic/VolumeIntegralCriterion.cpp
    elliptic/Volume.cpp
    elliptic/FluxPNorm.cpp
    elliptic/StressPNorm.cpp
    elliptic/TMStressPNorm.cpp
    elliptic/EMStressPNorm.cpp
    elliptic/EffectiveEnergy.cpp
    elliptic/InternalThermalEnergy.cpp
    elliptic/InternalThermoelasticEnergy.cpp
    elliptic/InternalElectroelasticEnergy.cpp
    elliptic/InternalElasticEnergy.cpp
    elliptic/ElastostaticResidual.cpp
    elliptic/ThermostaticResidual.cpp
    elliptic/ElectroelastostaticResidual.cpp
    elliptic/ThermoelastostaticResidual.cpp
    elliptic/VolumeAverageCriterion.cpp
    elliptic/VolumeAverageCriterionDenominator.cpp
    IntermediateDensityPenalty.cpp
    elliptic/electrical/SupportedOptionEnums.cpp
    elliptic/electrical/FactorySourceEvaluator.cpp
    elliptic/electrical/SourceWeightedSum.cpp
    elliptic/electrical/FactoryElectricalMaterial.cpp
    elliptic/electrical/MaterialDielectric.cpp
    elliptic/electrical/MaterialElectricalConductivity.cpp
    elliptic/electrical/MaterialElectricalConductivityTwoPhaseAlloy.cpp
    elliptic/electrical/FactoryCurrentDensityEvaluator.cpp
    elliptic/electrical/DarkCurrentDensityTwoPhaseAlloy.cpp
    elliptic/electrical/LightCurrentDensityTwoPhaseAlloy.cpp
  )
  
  if(HATCHING)
    set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
      elliptic/hatching/Problem.cpp
      elliptic/hatching/ElastostaticResidual.cpp
      elliptic/hatching/ScalarFunctionBaseFactory.cpp
      elliptic/hatching/PhysicsScalarFunction.cpp
      elliptic/hatching/InternalElasticEnergy.cpp
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
    hyperbolic/ElastomechanicsResidual.cpp
    hyperbolic/InternalElasticEnergy.cpp
    hyperbolic/PhysicsScalarFunction.cpp
    hyperbolic/Problem.cpp
    hyperbolic/ScalarFunctionFactory.cpp
    hyperbolic/StressPNorm.cpp
  )

  if(FLUIDS)
    set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
      hyperbolic/fluids/AbstractScalarFunction.cpp
      hyperbolic/fluids/AbstractVectorFunction.cpp
      hyperbolic/fluids/AbstractVolumetricSource.cpp
      hyperbolic/fluids/BrinkmanForces.cpp
      hyperbolic/fluids/FluidsQuasiImplicit.cpp
      hyperbolic/fluids/FluidsScalarFunction.cpp
      hyperbolic/fluids/FluidsVectorFunction.cpp
      hyperbolic/fluids/FluidsThermalSources.cpp
      hyperbolic/fluids/FluidsWorkSetBuilders.cpp
      hyperbolic/fluids/FluidsCriterionFactory.cpp
      hyperbolic/fluids/FluidsUniformThermalSource.cpp
      hyperbolic/fluids/FluidsWeightedScalarFunction.cpp
      hyperbolic/fluids/FluidsLeastSquaresScalarFunction.cpp
      hyperbolic/fluids/FluidsStabilizedUniformThermalSource.cpp
      hyperbolic/fluids/CriterionVolume.cpp
      hyperbolic/fluids/ThermalBuoyancy.cpp
      hyperbolic/fluids/PressureResidual.cpp
      hyperbolic/fluids/CriterionFlowRate.cpp
      hyperbolic/fluids/TemperatureResidual.cpp
      hyperbolic/fluids/InternalThermalForces.cpp
      hyperbolic/fluids/MomentumSurfaceForces.cpp
      hyperbolic/fluids/VelocityCorrectorResidual.cpp
      hyperbolic/fluids/VelocityPredictorResidual.cpp
      hyperbolic/fluids/CriterionThermalCompliance.cpp
      hyperbolic/fluids/CriterionSurfaceThermalFlux.cpp
      hyperbolic/fluids/CriterionMeanSurfacePressure.cpp
      hyperbolic/fluids/CriterionMeanSurfaceTemperature.cpp
    )
  endif()

  if(MICROMORPHIC)
    set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
      hyperbolic/micromorphic/MicromorphicLinearElasticMaterial.cpp
      hyperbolic/micromorphic/CubicMicromorphicLinearElasticMaterial.cpp
      hyperbolic/micromorphic/MicromorphicInertiaMaterial.cpp
      hyperbolic/micromorphic/CubicMicromorphicInertiaMaterial.cpp
      hyperbolic/micromorphic/RelaxedMicromorphicResidual.cpp
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
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES} solver/TpetraLinearSolver.cpp)
endif()

if (PLATOANALYZE_ENABLE_TACHO)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES} solver/TachoLinearSolver.cpp)
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
ENDIF()
