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
    bcs/neumann/SupportedParamOptions.cpp
    bcs/dirichlet/nitsche/NitscheEvaluator.cpp
    materials/mechanical/MaterialKirchhoff.cpp
    materials/mechanical/MaterialNeoHookean.cpp
    materials/mechanical/FactoryElasticMaterial.cpp
    materials/mechanical/MaterialIsotropicElastic.cpp
    materials/mechanical/FactoryMechanicalMaterials.cpp
    materials/mechanical/FactoryNonlinearElasticMaterial.cpp
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
    elliptic/mechanical/linear/LocalMeasureVonMises.cpp
    elliptic/mechanical/linear/LocalMeasureTensileEnergyDensity.cpp
    elliptic/mechanical/linear/Plato_AugLagStressCriterionQuadratic.cpp
    elliptic/mechanical/linear/Plato_AugLagStressCriterionGeneral.cpp
    elliptic/mechanical/linear/CriterionAugLagStrength.cpp
    elliptic/thermomechanics/linear/LocalMeasureThermalVonMises.cpp
    optimizer/AugLagDataMng.cpp
)

## core sources
set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    WorkSets.cpp
    base/Database.cpp
    base/SupportedParamOptions.cpp
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
endif()

## elliptic sources
if(ELLIPTIC)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    elliptic/Volume.cpp
    elliptic/Problem.cpp
    elliptic/base/VectorFunction.cpp
    elliptic/base/WorksetBuilder.cpp
    elliptic/evaluators/criterion/FactoryCriterionEvaluator.cpp
    elliptic/evaluators/criterion/CriterionEvaluatorDivision.cpp
    elliptic/evaluators/criterion/CriterionEvaluatorWeightedSum.cpp
    elliptic/evaluators/criterion/CriterionEvaluatorLeastSquares.cpp
    elliptic/evaluators/criterion/CriterionEvaluatorVolumeAverage.cpp
    elliptic/evaluators/criterion/CriterionEvaluatorScalarFunction.cpp
    elliptic/evaluators/criterion/CriterionEvaluatorMassProperties.cpp
    elliptic/evaluators/criterion/CriterionEvaluatorSolutionFunction.cpp
    elliptic/evaluators/problem/FactoryProblemEvaluator.cpp
    elliptic/evaluators/problem/ProblemEvaluatorVectorState.cpp
    elliptic/evaluators/problem/ProblemEvaluatorThermoMechanics.cpp
    elliptic/evaluators/problem/SupportedEllipticProblemOptions.cpp
    elliptic/mechanical/SupportedParamOptions.cpp
    elliptic/mechanical/linear/ResidualElastostatic.cpp
    elliptic/mechanical/linear/CriterionMassMoment.cpp
    elliptic/mechanical/linear/CriterionStressPNorm.cpp
    elliptic/mechanical/linear/CriterionVolumeIntegral.cpp
    elliptic/mechanical/linear/CriterionEffectiveEnergy.cpp
    elliptic/mechanical/linear/CriterionInternalElasticEnergy.cpp
    elliptic/mechanical/linear/CriterionVolAvgStressPNormDenominator.cpp
    elliptic/mechanical/linear/CriterionVolumeAverageDenominator.cpp
    elliptic/mechanical/linear/nitsche/BoundaryEvaluatorTestIsotropicElasticStress.cpp
    elliptic/mechanical/linear/nitsche/BoundaryEvaluatorTrialIsotropicElasticStress.cpp
    elliptic/mechanical/nonlinear/StressEvaluatorKirchhoff.cpp
    elliptic/mechanical/nonlinear/StressEvaluatorNeoHookean.cpp
    elliptic/mechanical/nonlinear/FactoryStressEvaluator.cpp
    elliptic/mechanical/nonlinear/ResidualElastostaticTotalLagrangian.cpp
    elliptic/mechanical/nonlinear/CriterionKirchhoffEnergyPotential.cpp
    elliptic/mechanical/nonlinear/CriterionNeoHookeanEnergyPotential.cpp
    elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTestKirchhoffStress.cpp
    elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTrialKirchhoffStress.cpp
    elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTestNeoHookeanStress.cpp
    elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTrialNeoHookeanStress.cpp
    elliptic/thermal/ResidualThermostatic.cpp
    elliptic/thermal/CriterionFluxPNorm.cpp
    elliptic/thermal/CriterionInternalThermalEnergy.cpp
    elliptic/thermomechanics/SupportedParamOptions.cpp
    elliptic/thermomechanics/linear/CriterionThermoMechStressPNorm.cpp
    elliptic/thermomechanics/linear/ResidualThermoelastostatic.cpp
    elliptic/thermomechanics/linear/CriterionInternalThermoelasticEnergy.cpp
    elliptic/thermomechanics/nonlinear/ResidualThermoElastoStaticTotalLagrangian.cpp
    elliptic/electrical/SupportedParamOptions.cpp
    elliptic/electrical/FactorySourceEvaluator.cpp
    elliptic/electrical/SourceWeightedSum.cpp
    elliptic/electrical/FactoryElectricalMaterial.cpp
    elliptic/electrical/MaterialDielectric.cpp
    elliptic/electrical/MaterialElectricalConductivity.cpp
    elliptic/electrical/MaterialElectricalConductivityTwoPhaseAlloy.cpp
    elliptic/electrical/FactoryCurrentDensitySourceEvaluator.cpp
    elliptic/electrical/DarkCurrentDensityTwoPhaseAlloy.cpp
    elliptic/electrical/LightCurrentDensityTwoPhaseAlloy.cpp
    elliptic/electrical/CriterionVolumeTwoPhase.cpp
    elliptic/electrical/CriterionPowerSurfaceDensityTwoPhase.cpp
    elliptic/electrical/CurrentDensityConstant.cpp
    elliptic/electrical/CurrentDensityTwoPhaseAlloy.cpp
    elliptic/electrical/FactoryCurrentDensityEvaluator.cpp
    elliptic/electrical/ResidualSteadyStateCurrent.cpp
    elliptic/electromechanics/EMStressPNorm.cpp
    elliptic/electromechanics/InternalElectroelasticEnergy.cpp
    elliptic/electromechanics/ElectroelastostaticResidual.cpp
  )
  
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
