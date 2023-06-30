/*
 * ProblemEvaluatorVectorState_def.hpp
 *
 *  Created on: June 21, 2023
 */

#pragma once


#include "BLAS1.hpp"
#include "Solutions.hpp"
#include "ParseTools.hpp"
#include "AnalyzeMacros.hpp"
#include "AnalyzeOutput.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"
#include "MultipointConstraints.hpp"
#include "solver/PlatoSolverFactory.hpp"
#include "bcs/dirichlet/EssentialBCs.hpp"
#include "elliptic/evaluators/criterion/FactoryCriterionEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
ProblemEvaluatorVectorState<PhysicsType>::
ProblemEvaluatorVectorState(
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
  mPhysics       (aParamList.get<std::string>("Physics")),
  mMPCs          (nullptr)
{
  this->initializeEvaluators(aParamList);
  this->initializeMultiPointConstraints(aParamList);
  this->readEssentialBoundaryConditions(aParamList);
  this->initializeSolver(mSpatialModel.Mesh,aParamList,aMachine);
}

template<typename PhysicsType>
Plato::Solutions
ProblemEvaluatorVectorState<PhysicsType>::
getSolution()
{
  Plato::Solutions tSolution(mPhysics, mTypePDE);
  tSolution.set("states", mStates, mResidualEvaluator->getDofNames());
  return tSolution;
}

template<typename PhysicsType>
void
ProblemEvaluatorVectorState<PhysicsType>::
postProcess(
  Plato::Solutions & aSolutions
)
{
  mResidualEvaluator->postProcess(aSolutions);
}

template<typename PhysicsType>
bool
ProblemEvaluatorVectorState<PhysicsType>::
criterionIsLinear(
  const std::string & aName
)
{
  if( mCriterionEvaluator.find(aName) == mCriterionEvaluator.end() )
  {
    auto tErrMsg = this->getErrorMsg(aName);
    ANALYZE_THROWERR(tErrMsg)
  }
  return ( mCriterionEvaluator.at(aName)->isLinear() );
}

template<typename PhysicsType>
void
ProblemEvaluatorVectorState<PhysicsType>::
analyze(
  Plato::Database & aDatabase
)
{
  this->updateDatabase(aDatabase);
  // initialize state values
  Plato::ScalarVector tMyStates = aDatabase.vector("states");
  Plato::blas1::fill(0.0, tMyStates);
  // inner loop for non-linear models
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  for(Plato::OrdinalType tNewtonIndex = 0; tNewtonIndex < mNumNewtonSteps; tNewtonIndex++)
  {
    Plato::ScalarVector tResidual = mResidualEvaluator->value(aDatabase,tCycle);
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
      mResidualEvaluator->jacobianState(aDatabase,tCycle,/*transpose=*/false);
    // solve linear system of equations
    Plato::Scalar tScale = (tNewtonIndex == 0) ? 1.0 : 0.0;
    if( !mWeakEBCs )
    { this->enforceStrongEssentialBoundaryConditions(tJacobianState,tResidual,tScale); }
    Plato::ScalarVector tDeltaD("increment", tMyStates.extent(0));
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaD);
    mSolver->solve(*tJacobianState, tDeltaD, tResidual);
    Plato::blas1::axpy(1.0, tDeltaD, tMyStates);
    if (mNumNewtonSteps > 1) {
      auto tIncrementNorm = Plato::blas1::norm(tDeltaD);
      std::cout << " Delta norm: " << tIncrementNorm << std::endl;
      if (tIncrementNorm < mNewtonIncTol) {
        std::cout << " Solution increment norm tolerance satisfied." << std::endl;
        break;
      }
    }
  }
}

template<typename PhysicsType>
void
ProblemEvaluatorVectorState<PhysicsType>::
residual(
  Plato::Database & aDatabase
)
{
  this->updateDatabase(aDatabase);
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  auto tResidual = mResidualEvaluator->value(aDatabase,tCycle);
  // residual is not saved into the database
}

template<typename PhysicsType>
Plato::Scalar
ProblemEvaluatorVectorState<PhysicsType>::
criterionValue(
  const std::string     & aName,
        Plato::Database & aDatabase
)
{
  this->updateDatabase(aDatabase);
  if( mCriterionEvaluator.find(aName) == mCriterionEvaluator.end() )
  {
    auto tErrMsg = this->getErrorMsg(aName);
    ANALYZE_THROWERR(tErrMsg)
  }
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  auto tValue = mCriterionEvaluator[aName]->value(aDatabase,/*cycle=*/tCycle);
  return tValue;
}

template<typename PhysicsType>
Plato::ScalarVector
ProblemEvaluatorVectorState<PhysicsType>::
criterionGradient(
  const Plato::evaluation_t & aEvalType,
  const std::string         & aName,
        Plato::Database     & aDatabase
)
{
  this->updateDatabase(aDatabase);
  if( mCriterionEvaluator.find(aName) == mCriterionEvaluator.end() )
  {
    auto tErrMsg = this->getErrorMsg(aName);
    ANALYZE_THROWERR(tErrMsg)
  }
  switch (aEvalType)
  {
    case Plato::evaluation_t::GRAD_Z: 
      return ( this->criterionGradientControl(mCriterionEvaluator[aName],aDatabase) );
      break;
    case Plato::evaluation_t::GRAD_X:
      return (this->criterionGradientConfig(mCriterionEvaluator[aName],aDatabase));
      break;
    default:
      return ( Plato::ScalarVector("empty",0) );
      break;
  }
}

template<typename PhysicsType>
void 
ProblemEvaluatorVectorState<PhysicsType>::
updateProblem(
  Plato::Database & aDatabase
)
{
  this->updateDatabase(aDatabase);
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  for( auto tCriterion : mCriterionEvaluator ) {
    tCriterion.second->updateProblem(aDatabase, tCycle);
  }
}

template<typename PhysicsType>
void 
ProblemEvaluatorVectorState<PhysicsType>::
updateDatabase(
  Plato::Database & aDatabase
)
{
  const Plato::OrdinalType tCycleIndex = aDatabase.scalar("cycle index");
  auto tMyStates = Kokkos::subview(mStates, tCycleIndex, Kokkos::ALL());
  aDatabase.vector("states", tMyStates);
}

template<typename PhysicsType>
std::string
ProblemEvaluatorVectorState<PhysicsType>::
getErrorMsg(
  const std::string & aName
) const
{
  std::string tMsg = std::string("ERROR: Criterion parameter list with name '")
    + aName + "' is not defined. " + "Defined criterion parameter lists are: ";
  for(const auto& tPair : mCriterionEvaluator)
  {
    tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  tSubMsg += ". The parameter list name and criterion ('Functions') criterion name argument must match.";
  return tSubMsg;
}

template<typename PhysicsType>
Plato::ScalarVector 
ProblemEvaluatorVectorState<PhysicsType>::
criterionGradientControl(
  const Criterion       & aCriterion,
        Plato::Database & aDatabase
)
{
  if(aCriterion == nullptr)
    { ANALYZE_THROWERR("ERROR: Requested criterion is a null pointer"); }
  // compute criterion contribution to the gradient
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  auto tGradientControl = aCriterion->gradientControl(aDatabase, tCycle);
  // add residual contribution to the gradient
  if( aCriterion->isLinear() == false )
  {
    // compute gradient with respect to state variables
    auto tGradientState = aCriterion->gradientState(aDatabase, tCycle);
    Plato::blas1::scale(-1.0, tGradientState);
    // compute jacobian with respect to state variables
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianState = 
      mResidualEvaluator->jacobianState(aDatabase, tCycle, /*transpose=*/true);
    if( mWeakEBCs )
    { this->enforceWeakEssentialAdjointBoundaryConditions(aDatabase); }
    else
    { this->enforceStrongEssentialAdjointBoundaryConditions(tJacobianState, tGradientState); }
    // solve adjoint system of equations
    const auto tNumDofs = mResidualEvaluator->numDofs();
    Plato::ScalarVector tAdjoints("Adjoint Variables", tNumDofs);
    mSolver->solve(*tJacobianState, tAdjoints, tGradientState, /*isAdjointSolve=*/true);
    // compute jacobian with respect to control variables
    auto tJacobianControl = mResidualEvaluator->jacobianControl(aDatabase, tCycle, /*transpose=*/true);
    // compute gradient with respect to design variables
    Plato::MatrixTimesVectorPlusVector(tJacobianControl, tAdjoints, tGradientControl);
  }
  return tGradientControl; 
}

template<typename PhysicsType>
Plato::ScalarVector 
ProblemEvaluatorVectorState<PhysicsType>:: 
criterionGradientConfig(
  const Criterion       & aCriterion,
        Plato::Database & aDatabase
)
{
  if(aCriterion == nullptr)
    { ANALYZE_THROWERR("ERROR: Requested criterion is a null pointer"); }
  // compute criterion contribution to the gradient
  const Plato::Scalar tCycle = aDatabase.scalar("cycle");
  auto tGradientConfig  = aCriterion->gradientConfig(aDatabase, tCycle);
  // add residual contribution to the gradient
  if( aCriterion->isLinear() == false )
  {
    // compute gradient with respect to state variables
    auto tGradientState = aCriterion->gradientState(aDatabase, tCycle);
    Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tGradientState);
    // compute jacobian with respect to state variables
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianState = 
      mResidualEvaluator->jacobianState(aDatabase, tCycle, /*transpose=*/true);
    if( mWeakEBCs )
    { this->enforceWeakEssentialAdjointBoundaryConditions(aDatabase); }
    else
    { this->enforceStrongEssentialAdjointBoundaryConditions(tJacobianState, tGradientState); }
    // solve adjoint system of equations
    const auto tNumDofs = mResidualEvaluator->numDofs();
    Plato::ScalarVector tAdjoints("Adjoint Variables", tNumDofs);
    mSolver->solve(*tJacobianState, tAdjoints, tGradientState, /*isAdjointSolve=*/true);
    // compute jacobian with respect to configuration variables
    auto tJacobianConfig = mResidualEvaluator->jacobianConfig(aDatabase, tCycle, /*transpose=*/true);
    // compute gradient with respect to design variables: dgdx * adjoint + dfdx
    Plato::MatrixTimesVectorPlusVector(tJacobianConfig, tAdjoints, tGradientConfig);
  }
  return tGradientConfig;
}

template<typename PhysicsType>
void 
ProblemEvaluatorVectorState<PhysicsType>:: 
initializeEvaluators(
  Teuchos::ParameterList& aParamList
)
{
  auto tTypePDE = aParamList.get<std::string>("PDE Constraint");
  mResidualEvaluator = std::make_shared<Plato::Elliptic::VectorFunction<PhysicsType>>(
    tTypePDE,mSpatialModel,mDataMap,aParamList);
  mStates = Plato::ScalarMultiVector("States",/*num_cycles=*/1, mResidualEvaluator->numDofs());
  if(aParamList.isSublist("Criteria"))
  {
    Plato::Elliptic::FactoryCriterionEvaluator<PhysicsType> tCriterionFactory;
    auto tCriteriaParams = aParamList.sublist("Criteria");
    for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); 
      tIndex != tCriteriaParams.end(); 
      ++tIndex)
    {
      const Teuchos::ParameterEntry & tEntry = tCriteriaParams.entry(tIndex);
      std::string tCriterionName = tCriteriaParams.name(tIndex);
      TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error,
        " Parameter in Criteria block not valid. Expect parameter lists only.");
      auto tCriterion = tCriterionFactory.create(mSpatialModel,mDataMap,aParamList,tCriterionName);
      if( tCriterion != nullptr )
      {
        mCriterionEvaluator[tCriterionName] = tCriterion;
      }
    }
  }
}

template<typename PhysicsType>
void 
ProblemEvaluatorVectorState<PhysicsType>:: 
readEssentialBoundaryConditions(
  Teuchos::ParameterList & aParamList
)
{
  if(aParamList.isSublist("Essential Boundary Conditions") == false)
  { ANALYZE_THROWERR("ERROR: Essential boundary conditions parameter list is not defined in input deck") }
  Plato::EssentialBCs<ElementType> tEssentialBoundaryConditions(
    aParamList.sublist("Essential Boundary Conditions", false), mSpatialModel.Mesh
  );
  tEssentialBoundaryConditions.get(mDirichletDofs, mDirichletStateVals);
  
  if(aParamList.isType<bool>("Weak Essential Boundary Conditions"))
  { mWeakEBCs = aParamList.get<bool>("Weak Essential Boundary Conditions",false); }
  if(mMPCs)
  { mMPCs->checkEssentialBcsConflicts(mDirichletDofs);}
}

template<typename PhysicsType>
void 
ProblemEvaluatorVectorState<PhysicsType>::
setEssentialBoundaryConditions(
  const Plato::OrdinalVector & aDofs, 
  const Plato::ScalarVector  & aValues
)
{
  if(aDofs.size() != aValues.size())
  {
      std::ostringstream tError;
      tError << "DIMENSION MISMATCH: THE NUMBER OF ELEMENTS IN INPUT DOFS AND VALUES ARRAY DO NOT MATCH."
          << "DOFS SIZE = " << aDofs.size() << " AND VALUES SIZE = " << aValues.size();
      ANALYZE_THROWERR(tError.str())
  }
  mDirichletDofs = aDofs;
  mDirichletStateVals = aValues;
}

template<typename PhysicsType>
void 
ProblemEvaluatorVectorState<PhysicsType>::
initializeMultiPointConstraints(
  Teuchos::ParameterList & aParamList
)
{
  if(aParamList.isSublist("Multipoint Constraints") == true)
  {
    Plato::OrdinalType tNumDofsPerNode = mResidualEvaluator->numStateDofsPerNode();
    auto & tMyParams = aParamList.sublist("Multipoint Constraints", false);
    mMPCs = std::make_shared<Plato::MultipointConstraints>(mSpatialModel, tNumDofsPerNode, tMyParams);
    mMPCs->setupTransform();
  }
}

template<typename PhysicsType>
void 
ProblemEvaluatorVectorState<PhysicsType>::
initializeSolver(
  Plato::Mesh            & aMesh,
  Teuchos::ParameterList & aParamList,
  Comm::Machine          & aMachine
)
{
  mPhysics = Plato::tolower(mPhysics);
  LinearSystemType tSystemType = LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE;
  if (mPhysics == "electromechanical" || mPhysics == "thermomechanical") {
    tSystemType = LinearSystemType::SYMMETRIC_INDEFINITE;
  }
  Plato::SolverFactory tSolverFactory(aParamList.sublist("Linear Solver"), tSystemType);
  mSolver = tSolverFactory.create(aMesh->NumNodes(), aMachine, ElementType::mNumDofsPerNode, mMPCs);
}

template<typename PhysicsType>
void 
ProblemEvaluatorVectorState<PhysicsType>::
enforceStrongEssentialBoundaryConditions(
  const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
  const Plato::ScalarVector                & aVector,
  const Plato::Scalar                      & aMultiplier
)
{
  if(aMatrix->isBlockMatrix())
  {
    Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(
      aMatrix, aVector, mDirichletDofs, mDirichletStateVals, aMultiplier
    );
  }
  else
  {
    Plato::applyConstraints<ElementType::mNumDofsPerNode>(
      aMatrix, aVector, mDirichletDofs, mDirichletStateVals, aMultiplier
    );
  }
}

template<typename PhysicsType>
void 
ProblemEvaluatorVectorState<PhysicsType>::
enforceStrongEssentialAdjointBoundaryConditions(
  const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
  const Plato::ScalarVector                & aVector
)
{
  // Essential Boundary Conditions (EBCs)
  Plato::ScalarVector tDirichletValues("Adjoint EBCs", mDirichletStateVals.size());
  Plato::blas1::scale(static_cast<Plato::Scalar>(0.0), tDirichletValues);
  if(aMatrix->isBlockMatrix())
  {
    Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, mDirichletDofs, tDirichletValues);
  }
  else
  {
    Plato::applyConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, mDirichletDofs, tDirichletValues);
  }
}

template<typename PhysicsType>
void 
ProblemEvaluatorVectorState<PhysicsType>::
enforceWeakEssentialAdjointBoundaryConditions(
  Plato::Database & aDatabase
)
{
  // Essential Boundary Conditions (EBCs)
  mDirichletAdjointVals = Plato::ScalarVector("Adjoint EBCs", mResidualEvaluator->numDofs());
  Kokkos::deep_copy(mDirichletAdjointVals, 0.0);
  aDatabase.vector("dirichlet", mDirichletAdjointVals);
}

} // namespace Elliptic

} // namespace Plato
