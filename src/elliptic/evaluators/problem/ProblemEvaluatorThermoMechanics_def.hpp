/*
 * ProblemEvaluatorThermoMechanics_def.hpp
 *
 *  Created on: June 27, 2023
 */

#pragma once

#include "BLAS1.hpp"
#include "EssentialBCs.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoMathHelpers.hpp"
#include "ApplyConstraints.hpp"

#include "solver/PlatoSolverFactory.hpp"

#include "elliptic/base/VectorFunction.hpp"
#include "elliptic/evaluators/criterion/FactoryCriterionEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
ProblemEvaluatorThermoMechanics<PhysicsType>::
ProblemEvaluatorThermoMechanics(
  Teuchos::ParameterList & aParamList,
  Plato::SpatialModel    & aSpatialModel,
  Plato::DataMap         & aDataMap,
  Plato::Comm::Machine     aMachine
) : 
  mSpatialModel  (aSpatialModel),
  mDataMap       (aDataMap),
  mNumNewtonSteps(Plato::ParseTools::getSubParam<int>(aParamList,"Newton Iteration","Maximum Iterations",1.)),
  mNewtonIncTol  (Plato::ParseTools::getSubParam<double>(aParamList,"Newton Iteration","Increment Tolerance",0.)),
  mNewtonResTol  (Plato::ParseTools::getSubParam<double>(aParamList,"Newton Iteration","Residual Tolerance",0.)),
  mTypePDE       (aParamList.get<std::string>("PDE Constraint")),
  mPhysics       (aParamList.get<std::string>("Physics"))
{
  this->initializeThermalResidualEvaluators(aParamList);
  this->initializeMechanicalResidualEvaluators(aParamList);
  this->initializeCriterionEvaluators(aParamList);
  this->readEssentialBoundaryConditions(aParamList);
  this->initializeSolvers(mSpatialModel.Mesh,aParamList,aMachine);
}

template<typename PhysicsType>
Plato::Solutions
ProblemEvaluatorThermoMechanics<PhysicsType>::
getSolution()
{
  Plato::Solutions tSolution(mPhysics, mTypePDE);
  for( auto& tPair : mResidualEvaluators ){
    this->setSolution(tPair.second,tSolution);
  }
  return tSolution;
}

template<typename PhysicsType>
void
ProblemEvaluatorThermoMechanics<PhysicsType>::
postProcess(
  Plato::Solutions & aSolutions
)
{
  for(auto& tPair : mResidualEvaluators){
    tPair.second->postProcess(aSolutions);
  }
}

template<typename PhysicsType>
void
ProblemEvaluatorThermoMechanics<PhysicsType>::
updateProblem(
  Plato::Database & aDatabase
)
{
  this->updateDatabase(aDatabase);
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  for( auto& tCriterion : mCriterionEvaluators ) {
    tCriterion.second->updateProblem(aDatabase, tCycle);
  }
}

template<typename PhysicsType>
void
ProblemEvaluatorThermoMechanics<PhysicsType>::
analyze(
  Plato::Database & aDatabase
)
{
  // solve thermal equations
  this->analyzeThermalPhysics(aDatabase);
  // solve mechanical physics
  this->analyzeMechanicalPhysics(aDatabase);
}

template<typename PhysicsType>
void
ProblemEvaluatorThermoMechanics<PhysicsType>::
residual(
  Plato::Database & aDatabase
)
{
  this->updateDatabase(aDatabase);
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  // mechanical and thermal residuals are not saved into the database
  for(auto& tPair : mResidualEvaluators){
    auto tResidual = tPair.second->value(aDatabase,tCycle);
  }
}

template<typename PhysicsType>
Plato::Scalar
ProblemEvaluatorThermoMechanics<PhysicsType>::
criterionValue(
  const std::string     & aName,
        Plato::Database & aDatabase
)
{
  if( mCriterionEvaluators.find(aName) == mCriterionEvaluators.end() )
  {
    auto tErrMsg = this->getCriterionErrorMsg(aName);
    ANALYZE_THROWERR(tErrMsg)
  }
  this->updateDatabase(aDatabase);
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  auto tValue = mCriterionEvaluators[aName]->value(aDatabase,tCycle);
  return tValue;
}

template<typename PhysicsType>
Plato::ScalarVector
ProblemEvaluatorThermoMechanics<PhysicsType>::
criterionGradient(
  const Plato::evaluation_t & aEvalType,
  const std::string         & aName,
        Plato::Database     & aDatabase
)
{
  auto tCriterionItr = mCriterionEvaluators.find(aName);
  if( tCriterionItr == mCriterionEvaluators.end() )
  {
    auto tErrMsg = this->getCriterionErrorMsg(aName);
    ANALYZE_THROWERR(tErrMsg)
  }
  this->updateDatabase(aDatabase);
  switch (aEvalType)
  {
    case Plato::evaluation_t::GRAD_Z: 
      return ( this->criterionGradientControl(tCriterionItr->second,aDatabase) );
      break;
    case Plato::evaluation_t::GRAD_X:
      return ( this->criterionGradientConfig(tCriterionItr->second,aDatabase) );
      break;
    default:
      return ( Plato::ScalarVector("empty",0) );
      break;
  }
}

template<typename PhysicsType>
bool  
ProblemEvaluatorThermoMechanics<PhysicsType>::
criterionIsLinear(
  const std::string & aName
)
{
  if( mCriterionEvaluators.find(aName) == mCriterionEvaluators.end() )
  {
    auto tErrMsg = this->getCriterionErrorMsg(aName);
    ANALYZE_THROWERR(tErrMsg)
  }
  return ( mCriterionEvaluators.at(aName)->isLinear() );
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
analyzeThermalPhysics(
  Plato::Database & aDatabase
)
{
  // initialize database for thermal analysis
  Plato::Database tThermalDatabase(aDatabase);
  const Plato::OrdinalType tCycleIndex = tThermalDatabase.scalar("cycle index");
  auto tTempStates = Kokkos::subview(mTemperatures, tCycleIndex, Kokkos::ALL());
  Plato::blas1::fill(0.0, tTempStates);
  tThermalDatabase.vector("states", tTempStates);
  // solve for thermal states; i.e., temperatures
  auto tResidualItr = mResidualEvaluators.find(mThermalResidualType);
  if(tResidualItr == mResidualEvaluators.end() ){
    ANALYZE_THROWERR("ERROR: Did not find requested thermal residual evaluator")
  }
  auto tTemperatures = this->analyzePhysics(tResidualItr->second,tThermalDatabase);
  // update thermo-mechanical database
  aDatabase.vector("node states",tTemperatures);
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
analyzeMechanicalPhysics(
  Plato::Database & aDatabase
)
{
  // initialize database for mechanical analysis
  Plato::Database tMechanicalDatabase(aDatabase);
  const Plato::OrdinalType tCycleIndex = tMechanicalDatabase.scalar("cycle index");
  auto tDispStates = Kokkos::subview(mDisplacements, tCycleIndex, Kokkos::ALL());
  Plato::blas1::fill(0.0,tDispStates);
  tMechanicalDatabase.vector("states",tDispStates);
  // solve for thermal states; i.e., temperatures
  auto tResidualItr = mResidualEvaluators.find(mMechanicalResidualType);
  if(tResidualItr == mResidualEvaluators.end() ){
    ANALYZE_THROWERR("ERROR: Did not find requested mechanical residual evaluator")
  }
  auto tDisplacements = this->analyzePhysics(tResidualItr->second,tMechanicalDatabase);
  aDatabase.vector("states",tDisplacements);
}

template<typename PhysicsType>
Plato::ScalarVector 
ProblemEvaluatorThermoMechanics<PhysicsType>::
analyzePhysics(
  const Residual        & aResidual,
  const Plato::Database & aDatabase
)
{
  // initialize solution values
  Plato::Database tDatabase(aDatabase);
  auto tNumStates = mSpatialModel.Mesh->NumNodes() * aResidual->numStateDofsPerNode();
  Plato::ScalarVector tStates("states",tNumStates);
  Plato::blas1::fill(0.0, tStates);
  tDatabase.vector("states",tStates);
  // inner loop for non-linear models
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  for(Plato::OrdinalType tNewtonIndex = 0; tNewtonIndex < mNumNewtonSteps; tNewtonIndex++)
  {
    Plato::ScalarVector tResidual = aResidual->value(aDatabase,tCycle);
    Plato::blas1::scale(-1.0, tResidual);
    if (mNumNewtonSteps > 1) {
      auto tResidualNorm = Plato::blas1::norm(tResidual);
      std::cout << " Residual norm: " << tResidualNorm << std::endl;
      if (tResidualNorm < mNewtonResTol) {
        std::cout << " Residual norm tolerance satisfied." << std::endl;
        break;
      }
    }
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianState = 
      aResidual->jacobianState(aDatabase,tCycle,/*transpose=*/false);
    // solve linear system of equations
    Plato::Scalar tScale = (tNewtonIndex == 0) ? 1.0 : 0.0;
    if( !mWeakEBCs )
    { this->enforceStrongEssentialBoundaryConditions(tScale,aResidual,tJacobianState,tResidual); }
    Plato::ScalarVector tDelta("increment", tStates.extent(0));
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDelta);
    mLinearSolvers[aResidual->type()]->solve(*tJacobianState, tDelta, tResidual);
    Plato::blas1::axpy(1.0, tDelta, tStates);
    if (mNumNewtonSteps > 1) {
      auto tIncrementNorm = Plato::blas1::norm(tDelta);
      std::cout << " Delta norm: " << tIncrementNorm << std::endl;
      if (tIncrementNorm < mNewtonIncTol) {
        std::cout << " Solution increment norm tolerance satisfied." << std::endl;
        break;
      }
    }
  }
  return tStates;
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
enforceStrongEssentialBoundaryConditions(
  const Plato::Scalar                      & aMultiplier,
  const Residual                           & aResidual,
  const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
  const Plato::ScalarVector                & aVector
)
{
  if(aMatrix->isBlockMatrix())
  {
    Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(
      aMatrix, aVector, mDirichletDofs[aResidual->type()], mDirichletVals[aResidual->type()], aMultiplier
    );
  }
  else
  {
    Plato::applyConstraints<ElementType::mNumDofsPerNode>(
      aMatrix, aVector, mDirichletDofs[aResidual->type()], mDirichletVals[aResidual->type()], aMultiplier
    );
  }
}

template<typename PhysicsType>
Plato::ScalarVector 
ProblemEvaluatorThermoMechanics<PhysicsType>::
adjointThermalPhysics(
  const Criterion           & aCriterion,
  const Plato::ScalarVector & aMechAdjoints,
  Plato::Database           & aDatabase
)
{
  // compute gradient with respect to node states
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  auto tRightHandSideVector = aCriterion->gradientNodeState(aDatabase,tCycle);
  // compute jacobian of mechanical residual with respect to the node states
  auto tMechanicalResidualItr = mResidualEvaluators.find(mMechanicalResidualType);
  Teuchos::RCP<Plato::CrsMatrixType> tMechJacobianNodeState = 
    tMechanicalResidualItr->second->jacobianNodeState(aDatabase,tCycle,/*transpose=*/true);
  // add mechanical residual contribution to right hand side vector 
  Plato::MatrixTimesVectorPlusVector(tMechJacobianNodeState,aMechAdjoints,tRightHandSideVector);
  // compute additive inverse of the right hand side vector 
  Plato::blas1::scale(-1.0,tRightHandSideVector);
  // compute jacobian of node state residual with respect to the node states
  auto tThermalResidualItr = mResidualEvaluators.find(mThermalResidualType);
  Teuchos::RCP<Plato::CrsMatrixType> tTempJacobianState = 
    tThermalResidualItr->second->jacobianState(aDatabase,tCycle,/*transpose=*/true);
  // enforce dirichlet boundary conditons 
  if( mWeakEBCs ){ 
    this->enforceWeakEssentialAdjointBoundaryConditions(tThermalResidualItr->second,aDatabase); 
  }
  else{ 
    this->enforceStrongEssentialAdjointBoundaryConditions(
      tThermalResidualItr->second,tTempJacobianState,tRightHandSideVector
    ); 
  }
  // solve adjoint system of equations
  auto tNumTempStates = mSpatialModel.Mesh->NumNodes() * tThermalResidualItr->second->numStateDofsPerNode();
  Plato::ScalarVector tTempAdjoints("thermal adjoints",tNumTempStates);
  mLinearSolvers[tThermalResidualItr->second->type()]->solve(
    *tTempJacobianState,tTempAdjoints,tRightHandSideVector,/*isAdjointSolve=*/true
  );
  return tTempAdjoints;
}

template<typename PhysicsType>
Plato::ScalarVector 
ProblemEvaluatorThermoMechanics<PhysicsType>::
adjointMechanicalPhysics(
  const Criterion       & aCriterion,
        Plato::Database & aDatabase
)
{
  auto tResidualItr = mResidualEvaluators.find(mMechanicalResidualType);
  // compute gradient with respect to the vector states
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  auto tGradientState = aCriterion->gradientState(aDatabase,tCycle);
  Plato::blas1::scale(-1.0, tGradientState);
  // compute jacobian with respect to the state vector
  Teuchos::RCP<Plato::CrsMatrixType> tJacobianState = 
    tResidualItr->second->jacobianState(aDatabase,tCycle,/*transpose=*/true);
  // enforce dirichlet boundary conditons 
  if( mWeakEBCs )
  { this->enforceWeakEssentialAdjointBoundaryConditions(tResidualItr->second,aDatabase); }
  else
  { this->enforceStrongEssentialAdjointBoundaryConditions(tResidualItr->second,tJacobianState,tGradientState); }
  // solve adjoint system of equations
  auto tNumStates = mSpatialModel.Mesh->NumNodes() * tResidualItr->second->numStateDofsPerNode();
  Plato::ScalarVector tAdjoints("Adjoint Variables",tNumStates);
  mLinearSolvers[tResidualItr->second->type()]->solve(
    *tJacobianState,tAdjoints,tGradientState,/*isAdjointSolve=*/true
  );
  return tAdjoints;
}

template<typename PhysicsType>
Plato::ScalarVector 
ProblemEvaluatorThermoMechanics<PhysicsType>::
criterionGradientControl(
  const Criterion       & aCriterion,
        Plato::Database & aDatabase
)
{
  if(aCriterion == nullptr){ 
    ANALYZE_THROWERR("ERROR: Requested criterion is a null pointer"); 
  }
  // compute criterion contribution to the gradient
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  auto tGradientControl = aCriterion->gradientControl(aDatabase,tCycle);
  // add residual contribution to the gradient
  if( aCriterion->isLinear() == false )
  {
    // compute mechanical adjoints
    Plato::ScalarVector tMechAdjoints = this->adjointMechanicalPhysics(aCriterion,aDatabase);
    // compute jacobian of mechanical residual with respect to the controls
    auto tMechResidualItr = mResidualEvaluators.find(mMechanicalResidualType);
    Teuchos::RCP<Plato::CrsMatrixType> tMechJacobianControl = 
      tMechResidualItr->second->jacobianControl(aDatabase,tCycle,/*transpose=*/true);
    // add mechanical residual contribution to gradient with respect to the controls
    Plato::MatrixTimesVectorPlusVector(tMechJacobianControl,tMechAdjoints,tGradientControl);
    // compute thermal adjoints
    Plato::ScalarVector tTempAdjoints = this->adjointThermalPhysics(aCriterion,tMechAdjoints,aDatabase);
    // compute jacobian of thermal residual with respect to the controls
    auto tThermalResidualItr = mResidualEvaluators.find(mThermalResidualType);
    Teuchos::RCP<Plato::CrsMatrixType> tTempJacobianControl = 
      tThermalResidualItr->second->jacobianControl(aDatabase,tCycle,/*transpose=*/true);
    // add thermal residual contribution to gradient with respect to the controls
    Plato::MatrixTimesVectorPlusVector(tTempJacobianControl,tTempAdjoints,tGradientControl);
  }
  return tGradientControl; 
}

template<typename PhysicsType>
Plato::ScalarVector 
ProblemEvaluatorThermoMechanics<PhysicsType>::
criterionGradientConfig(
  const Criterion       & aCriterion,
        Plato::Database & aDatabase
)
{
  if(aCriterion == nullptr){ 
    ANALYZE_THROWERR("ERROR: Requested criterion is a null pointer"); 
  }
  // compute criterion contribution to the gradient
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  auto tGradientConfig = aCriterion->gradientConfig(aDatabase,tCycle);
  // add residual contribution to the gradient
  if( aCriterion->isLinear() == false )
  {
    // compute mechanical adjoints
    Plato::ScalarVector tMechAdjoints = this->adjointMechanicalPhysics(aCriterion,aDatabase);
    // compute jacobian of mechanical residual with respect to the controls
    auto tMechResidualItr = mResidualEvaluators.find(mMechanicalResidualType);
    Teuchos::RCP<Plato::CrsMatrixType> tMechJacobianConfig = 
      tMechResidualItr->second->jacobianConfig(aDatabase,tCycle,/*transpose=*/true);
    // add mechanical residual contribution to gradient with respect to the controls
    Plato::MatrixTimesVectorPlusVector(tMechJacobianConfig,tMechAdjoints,tGradientConfig);
    // compute thermal adjoints
    Plato::ScalarVector tTempAdjoints = this->adjointThermalPhysics(aCriterion,tMechAdjoints,aDatabase);
    // compute jacobian of thermal residual with respect to the controls
    auto tThermalResidualItr = mResidualEvaluators.find(mThermalResidualType);
    Teuchos::RCP<Plato::CrsMatrixType> tTempJacobianConfig = 
      tThermalResidualItr->second->jacobianConfig(aDatabase,tCycle,/*transpose=*/true);
    // add thermal residual contribution to gradient with respect to the controls
    Plato::MatrixTimesVectorPlusVector(tTempJacobianConfig,tTempAdjoints,tGradientConfig);
  }
  return tGradientConfig; 
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
enforceStrongEssentialAdjointBoundaryConditions(
  const Residual                           & aResidual,
  const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
  const Plato::ScalarVector                & aVector
)
{
  auto tDirichletVals = mDirichletVals[aResidual->type()];
  Plato::ScalarVector tDirichletAdjointValues("Adjoint EBCs", tDirichletVals.size());
  Plato::blas1::scale(static_cast<Plato::Scalar>(0.0), tDirichletAdjointValues);
  if(aMatrix->isBlockMatrix())
  {
    Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(
      aMatrix, aVector, mDirichletDofs[aResidual->type()], tDirichletAdjointValues
    );
  }
  else
  {
    Plato::applyConstraints<ElementType::mNumDofsPerNode>(
      aMatrix, aVector, mDirichletDofs[aResidual->type()], tDirichletAdjointValues
    );
  }
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
enforceWeakEssentialAdjointBoundaryConditions(
  const Residual        & aResidual,
        Plato::Database & aDatabase
)
{
  auto tNumStates = mSpatialModel.Mesh->NumNodes() * aResidual->numStateDofsPerNode();
  Plato::ScalarVector tAdjointDirichlet("Adjoint EBCs", tNumStates);
  Kokkos::deep_copy(tAdjointDirichlet, 0.0);
  aDatabase.vector("dirichlet",tAdjointDirichlet);
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
setSolution(
  const Residual         & aResidual,
        Plato::Solutions & aSolutions
)
{
  switch (aResidual->type()) 
  {
    case Plato::Elliptic::residual_t::LINEAR_THERMO_MECHANICAL:
    case Plato::Elliptic::residual_t::NONLINEAR_THERMO_MECHANICAL:
      aSolutions.set("Displacements",mDisplacements,aResidual->getDofNames());
      break;
    case Plato::Elliptic::residual_t::LINEAR_THERMAL:
      aSolutions.set("Temperatures" ,mTemperatures ,aResidual->getDofNames());
      break;
  }
}

template<typename PhysicsType>
std::string
ProblemEvaluatorThermoMechanics<PhysicsType>::
getCriterionErrorMsg(
  const std::string & aName
) const
{
  std::string tMsg = std::string("ERROR: Criterion parameter list with name '")
    + aName + "' is not defined. " + "Defined criterion parameter lists are: ";
  for(const auto& tPair : mCriterionEvaluators){
    tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  tSubMsg += ". The parameter list name and criterion ('Functions') arguments must match.";
  return tSubMsg;
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
updateDatabase(
  Plato::Database & aDatabase
)
{
  const Plato::OrdinalType tCycleIndex = aDatabase.scalar("cycle index");
  auto tMyDisplacement = Kokkos::subview(mDisplacements,tCycleIndex,Kokkos::ALL());
  aDatabase.vector("states", tMyDisplacement);
  auto tMyTemperature = Kokkos::subview(mTemperatures,tCycleIndex,Kokkos::ALL());
  aDatabase.vector("node states", tMyTemperature);
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
initializeCriterionEvaluators(
  Teuchos::ParameterList & aParamList
)
{
  if(aParamList.isSublist("Criteria"))
  {
    Plato::Elliptic::FactoryCriterionEvaluator<PhysicsType> tCriterionFactory;
    auto tCriteriaParams = aParamList.sublist("Criteria");
    for(Teuchos::ParameterList::ConstIterator tIndex=tCriteriaParams.begin(); tIndex!=tCriteriaParams.end(); ++tIndex)
    {
      const Teuchos::ParameterEntry & tEntry = tCriteriaParams.entry(tIndex);
      std::string tCriterionName = tCriteriaParams.name(tIndex);
      TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error,
        " Parameter in ('Criteria') block not valid. Expect parameter lists only.");
      auto tCriterion = tCriterionFactory.create(mSpatialModel,mDataMap,aParamList,tCriterionName);
      if( tCriterion != nullptr )
      {
        mCriterionEvaluators[tCriterionName] = tCriterion;
      }
    }
  }
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
initializeThermalResidualEvaluators(
  Teuchos::ParameterList & aParamList
)
{
  auto tParamList = aParamList.sublist(mTypePDE);
  if(tParamList.isSublist("Thermal Residual") == false){
    auto tErrorMsg = std::string("ERROR: ('Thermal Residual') parameter list is not defined'");
    ANALYZE_THROWERR(tErrorMsg)
  }
  // only linear thermal physics are supported/implemented at the moment
  mThermalResidualType = Plato::Elliptic::residual_t::LINEAR_THERMAL;
  mResidualEvaluators[mThermalResidualType] = 
    std::make_shared<Plato::Elliptic::VectorFunction<ThermalPhysicsType>>(mTypePDE,mSpatialModel,mDataMap,aParamList);
  // allocate memory for thermal states
  mTemperatures = Plato::ScalarMultiVector(
    "Temperatures",/*number of cycles=*/1, mResidualEvaluators[mThermalResidualType]->numStateDofsPerNode()
  );
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
initializeMechanicalResidualEvaluators(
  Teuchos::ParameterList & aParamList
)
{
  auto tParamList = aParamList.sublist(mTypePDE);
  if(tParamList.isSublist("Mechanical Residual") == false){
    auto tErrorMsg = std::string("ERROR: ('Mechanical Residual') parameter list is not defined'");
    ANALYZE_THROWERR(tErrorMsg)
  }
  Plato::Elliptic::ResidualEnum tS2E;
  auto tMechResParamList = tParamList.sublist("Mechanical Residual");
  auto tResponse = tMechResParamList.get<std::string>("Response","linear");
  auto tResidualStringType = tResponse + " thermomechanical";
  mMechanicalResidualType = tS2E.get(tResidualStringType);
  mResidualEvaluators[mMechanicalResidualType] = 
    std::make_shared<Plato::Elliptic::VectorFunction<PhysicsType>>(mTypePDE,mSpatialModel,mDataMap,aParamList);
  // allocate memory for mechanical states
  mDisplacements = Plato::ScalarMultiVector(
    "Displacements",/*number of cycles=*/1, mResidualEvaluators[mMechanicalResidualType]->numStateDofsPerNode()
  );
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
initializeSolvers(
  Plato::Mesh            & aMesh,
  Teuchos::ParameterList & aParamList,
  Comm::Machine          & aMachine
)
{
  LinearSystemType tSystemType = LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE;
  Plato::SolverFactory tSolverFactory(aParamList.sublist("Linear Solver"), tSystemType);
  for(auto& tPair : mResidualEvaluators)
  {
    mLinearSolvers[tPair.first] = 
      tSolverFactory.create(aMesh->NumNodes(), aMachine, tPair.second->numStateDofsPerNode(), nullptr);
  }
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
readEssentialBoundaryConditions(
  Teuchos::ParameterList & aParamList
)
{
  this->readThermalEssentialBoundaryConditions(aParamList);
  this->readMechanicalEssentialBoundaryConditions(aParamList);
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
readMechanicalEssentialBoundaryConditions(
  Teuchos::ParameterList & aParamList
)
{
  if(aParamList.isSublist("Mechanical Essential Boundary Conditions") == false)
  { 
    auto tErrorMsg = std::string("ERROR: Parameter list ('Mechanical Essential Boundary Conditions') ") + 
      "is not defined in the input deck";
    ANALYZE_THROWERR(tErrorMsg) 
  }
  Plato::EssentialBCs<PhysicsType> tMechanicalEBCs(
    aParamList.sublist("Mechanical Essential Boundary Conditions", false), mSpatialModel.Mesh
  );
  Plato::ScalarVector  tDirichletVals;
  Plato::OrdinalVector tDirichletDofs; 
  tMechanicalEBCs.get(tDirichletDofs,tDirichletVals);
  mDirichletDofs[mMechanicalResidualType] = tDirichletDofs;
  mDirichletVals[mMechanicalResidualType] = tDirichletVals;
  
  if(aParamList.isType<bool>("Weak Essential Boundary Conditions"))
  { mWeakEBCs = aParamList.get<bool>("Weak Essential Boundary Conditions",false); }
}

template<typename PhysicsType>
void 
ProblemEvaluatorThermoMechanics<PhysicsType>::
readThermalEssentialBoundaryConditions(
  Teuchos::ParameterList & aParamList
)
{
  if(aParamList.isSublist("Thermal Essential Boundary Conditions") == false)
  { 
    auto tErrorMsg = std::string("ERROR: Parameter list ('Thermal Essential Boundary Conditions') ") + 
      "is not defined in the input deck";
    ANALYZE_THROWERR(tErrorMsg) 
  }
  Plato::EssentialBCs<PhysicsType> tThermalEBCs(
    aParamList.sublist("Thermal Essential Boundary Conditions", false), mSpatialModel.Mesh
  );
  
  Plato::ScalarVector  tDirichletVals;
  Plato::OrdinalVector tDirichletDofs; 
  tThermalEBCs.get(tDirichletDofs,tDirichletVals);
  mDirichletDofs[mThermalResidualType] = tDirichletDofs;
  mDirichletVals[mThermalResidualType] = tDirichletVals;
  
  if(aParamList.isType<bool>("Weak Essential Boundary Conditions"))
  { mWeakEBCs = aParamList.get<bool>("Weak Essential Boundary Conditions",false); }
}

} // namespace Elliptic

} // namespace Plato
