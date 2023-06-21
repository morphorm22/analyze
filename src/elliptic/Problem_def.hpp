#pragma once

#include "BLAS1.hpp"
#include "Solutions.hpp"
#include "ParseTools.hpp"
#include "EssentialBCs.hpp"
#include "AnalyzeMacros.hpp"
#include "AnalyzeOutput.hpp"
#include "base/Database.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"
#include "MultipointConstraints.hpp"
#include "elliptic/evaluators/criterion/FactoryCriterionEvaluator.hpp"


namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
Problem<PhysicsType>::
Problem(
  Plato::Mesh              aMesh,
  Teuchos::ParameterList & aParamList,
  Plato::Comm::Machine     aMachine
) :
  AbstractProblem(aMesh,aParamList),
  mSpatialModel  (aMesh,aParamList,mDataMap),
  mResidualEvaluator(std::make_shared<VectorFunctionType>(
    aParamList.get<std::string>("PDE Constraint"),mSpatialModel,mDataMap,aParamList)),
  mNumNewtonSteps(Plato::ParseTools::getSubParam<int>(aParamList,"Newton Iteration","Maximum Iterations",1.)),
  mNewtonIncTol(Plato::ParseTools::getSubParam<double>(aParamList,"Newton Iteration","Increment Tolerance",0.)),
  mNewtonResTol(Plato::ParseTools::getSubParam<double>(aParamList,"Newton Iteration","Residual Tolerance",0.)),
  mResidual     ("MyResidual", mResidualEvaluator->numDofs()),
  mStates       ("States", static_cast<Plato::OrdinalType>(1), mResidualEvaluator->numDofs()),
  mJacobianState(Teuchos::null),
  mTypePDE      (aParamList.get<std::string>("PDE Constraint")),
  mPhysics      (aParamList.get<std::string>("Physics")),
  mMPCs         (nullptr)
{
  this->initializeEvaluators(aParamList);
  this->initializeMultiPointConstraints(aParamList);
  this->readEssentialBoundaryConditions(aParamList);
  this->initializeSolver(aMesh,aParamList,aMachine);
  this->parseSaveOutput(aParamList);
}

template<typename PhysicsType>
Problem<PhysicsType>::
~Problem(){}

template<typename PhysicsType>
bool
Problem<PhysicsType>::
criterionIsLinear(
  const std::string & aName
)
{
  return ( mCriterionEvaluator.at(aName)->isLinear() );
}

template<typename PhysicsType>
void Problem<PhysicsType>::
output(
  const std::string & aFilepath
)
{
  auto tDataMap = this->getDataMap();
  auto tSolution = this->getSolution();
  auto tSolutionOutput = mResidualEvaluator->getSolutionStateOutputData(tSolution);
  Plato::universal_solution_output(aFilepath, tSolutionOutput, tDataMap, mSpatialModel.Mesh);
}

template<typename PhysicsType>
void Problem<PhysicsType>::
updateProblem(
  const Plato::ScalarVector & aControls, 
  const Plato::Solutions & aSolution
)
{
  // build database
  Plato::Database tDatabase;
  this->buildDatabase(aControls,tDatabase);
  // update problem
  constexpr Plato::Scalar tCYCLE = 0.;
  for( auto tCriterion : mCriterionEvaluator ) {
    tCriterion.second->updateProblem(tDatabase, tCYCLE);
  }
}

template<typename PhysicsType>
void
Problem<PhysicsType>::
buildDatabase(
  const Plato::ScalarVector & aControls,
        Plato::Database     & aDatabase)
{
  constexpr size_t tCYCLE_INDEX = 0;
  aDatabase.scalar("cycle_index",tCYCLE_INDEX);
  auto tMyStates = Kokkos::subview(mStates, tCYCLE_INDEX, Kokkos::ALL());
  aDatabase.vector("states"  , tMyStates);
  aDatabase.vector("controls", aControls);
}

template<typename PhysicsType>
void
Problem<PhysicsType>::
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
Plato::Solutions
Problem<PhysicsType>::
solution(
  const Plato::ScalarVector & aControls
)
{
  // clear output database
  mDataMap.clearStates();
  // build database
  Plato::Database tDatabase;
  this->buildDatabase(aControls,tDatabase);
  // save controls to output database
  mDataMap.scalarNodeFields["Topology"] = aControls;
  // initialize state values
  Plato::ScalarVector tMyStates = tDatabase.vector("states");
  Plato::blas1::fill(0.0, tMyStates);
  // inner loop for non-linear models
  constexpr Plato::Scalar tCYCLE = 0.0;
  for(Plato::OrdinalType tNewtonIndex = 0; tNewtonIndex < mNumNewtonSteps; tNewtonIndex++)
  {
    mResidual = mResidualEvaluator->value(tDatabase,tCYCLE);
    Plato::blas1::scale(-1.0, mResidual);
    if (mNumNewtonSteps > 1) {
      auto tResidualNorm = Plato::blas1::norm(mResidual);
      std::cout << " Residual norm: " << tResidualNorm << std::endl;
      if (tResidualNorm < mNewtonResTol) {
        std::cout << " Residual norm tolerance satisfied." << std::endl;
        break;
      }
    }
    mJacobianState = mResidualEvaluator->jacobianState(tDatabase,tCYCLE,false/*transpose=*/);
    // solve linear system of equations
    Plato::Scalar tScale = (tNewtonIndex == 0) ? 1.0 : 0.0;
    if( !mWeakEBCs )
    { this->enforceStrongEssentialBoundaryConditions(mJacobianState,mResidual,tScale); }
    Plato::ScalarVector tDeltaD("increment", tMyStates.extent(0));
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaD);
    mSolver->solve(*mJacobianState, tDeltaD, mResidual);
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
  if ( mSaveState )
  {
    // evaluate at new state
    mResidual = mResidualEvaluator->value(tDatabase,tCYCLE);
    mDataMap.saveState();
  }
  auto tSolution = this->getSolution();
  return tSolution;
}

template<typename PhysicsType>
std::string
Problem<PhysicsType>::
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
  tSubMsg += ". The parameter list name and criterion 'Type' argument must match.";
  return tSubMsg;
}

template<typename PhysicsType>
Plato::Scalar
Problem<PhysicsType>::
criterionValue(
    const Plato::ScalarVector & aControls,
    const Plato::Solutions    & aSolution,
    const std::string         & aName
)
{
  return( this->criterionValue(aControls,aName) );
}

template<typename PhysicsType>
Plato::Scalar
Problem<PhysicsType>::
criterionValue(
    const Plato::ScalarVector & aControls,
    const std::string         & aName
)
{
  Plato::Database tDatabase;
  this->buildDatabase(aControls,tDatabase);
  if( mCriterionEvaluator.count(aName) )
  {
    auto tValue = mCriterionEvaluator[aName]->value(tDatabase,/*cycle=*/0.);
    return tValue;
  }
  else
  {
    auto tErrMsg = this->getErrorMsg(aName);
    ANALYZE_THROWERR(tErrMsg)
  }
}

template<typename PhysicsType>
Plato::ScalarVector
Problem<PhysicsType>::
criterionGradient(
  const Plato::ScalarVector & aControls,
  const Plato::Solutions    & aSolution,
  const std::string         & aName
)
{
  return ( this->criterionGradient(aControls,aName) );
}

template<typename PhysicsType>
Plato::ScalarVector
Problem<PhysicsType>::
criterionGradient(
    const Plato::ScalarVector & aControls,
    const std::string         & aName
)
{
  if( mCriterionEvaluator.find(aName) == mCriterionEvaluator.end() )
  {
    auto tErrMsg = this->getErrorMsg(aName);
    ANALYZE_THROWERR(tErrMsg)
  }
  // build database
  Plato::Database tDatabase;
  this->buildDatabase(aControls,tDatabase);
  // compute gradient
  if(mCriterionEvaluator.at(aName)->isLinear() )
  {
    return ( mCriterionEvaluator.at(aName)->gradientControl(tDatabase,/*cycle=*/0.0) );
  }
  else
  {
    return ( this->computeCriterionGradientControl(tDatabase, mCriterionEvaluator[aName]) );
  }
}

template<typename PhysicsType>
void 
Problem<PhysicsType>::
enforceWeakEssentialAdjointBoundaryConditions(
  Plato::Database & aDatabase
)
{
  // Essential Boundary Conditions (EBCs)
  mDirichletAdjointVals = Plato::ScalarVector("Adjoint EBCs", mResidualEvaluator->numDofs());
  Kokkos::deep_copy(mDirichletAdjointVals, 0.0);
  aDatabase.vector("dirichlet", mDirichletAdjointVals);
}

template<typename PhysicsType>
void 
Problem<PhysicsType>::
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
Plato::ScalarVector
Problem<PhysicsType>::
computeCriterionGradientControl(
  Plato::Database & aDatabase,
  Criterion       & aCriterion
)
{
  if(aCriterion == nullptr)
  {
    ANALYZE_THROWERR("ERROR: Requested criterion is a null pointer");
  }
  if(static_cast<Plato::OrdinalType>(mAdjoints.size()) <= static_cast<Plato::OrdinalType>(0))
  {
    const auto tNumDofs = mResidualEvaluator->numDofs();
    mAdjoints = Plato::ScalarMultiVector("Adjoint Variables", 1, tNumDofs);
  }
  // compute criterion contribution to the gradient
  constexpr Plato::Scalar tCYCLE = 0.0;
  auto tGradientControl = aCriterion->gradientControl(aDatabase, tCYCLE);
  // add residual contribution to the gradient
  {
    // compute gradient with respect to state variables
    auto tGradientState = aCriterion->gradientState(aDatabase, tCYCLE);
    Plato::blas1::scale(-1.0, tGradientState);
    // compute jacobian with respect to state variables
    mJacobianState = mResidualEvaluator->jacobianState(aDatabase, tCYCLE, /*transpose=*/ true);
    if( mWeakEBCs )
    { this->enforceWeakEssentialAdjointBoundaryConditions(aDatabase); }
    else
    { this->enforceStrongEssentialAdjointBoundaryConditions(mJacobianState, tGradientState); }
    // solve adjoint system of equations
    constexpr size_t tCYCLE_INDEX = 0;
    Plato::ScalarVector tMyAdjoints = Kokkos::subview(mAdjoints, tCYCLE_INDEX, Kokkos::ALL());
    mSolver->solve(*mJacobianState, tMyAdjoints, tGradientState, /*isAdjointSolve=*/ true);
    // compute jacobian with respect to control variables
    auto tJacobianControl = mResidualEvaluator->jacobianControl(aDatabase, tCYCLE, /*transpose=*/ true);
    // compute gradient with respect to design variables
    Plato::MatrixTimesVectorPlusVector(tJacobianControl, tMyAdjoints, tGradientControl);
  }
  return tGradientControl;
}

template<typename PhysicsType>
Plato::ScalarVector
Problem<PhysicsType>::
criterionGradientX(
    const Plato::ScalarVector & aControls,
    const Plato::Solutions    & aSolution,
    const std::string         & aName
)
{
  return (this->criterionGradientX(aControls,aName));
}

template<typename PhysicsType>
Plato::ScalarVector
Problem<PhysicsType>::
criterionGradientX(
  const Plato::ScalarVector & aControls,
  const std::string         & aName
)
{
  if( mCriterionEvaluator.find(aName) == mCriterionEvaluator.end() )
  {
    auto tErrMsg = this->getErrorMsg(aName);
    ANALYZE_THROWERR(tErrMsg)
  }
  // build database
  Plato::Database tDatabase;
  this->buildDatabase(aControls,tDatabase);
  // compute gradient
  if( mCriterionEvaluator.at(aName)->isLinear() )
  {
    return ( mCriterionEvaluator.at(aName)->gradientConfig(tDatabase,/*cycle=*/0.0) );
  }
  else
  {
    return ( this->computeCriterionGradientConfig(tDatabase, mCriterionEvaluator[aName]) );
  }
}

template<typename PhysicsType>
Plato::ScalarVector
Problem<PhysicsType>::
computeCriterionGradientConfig(
  Plato::Database & aDatabase,
  Criterion       & aCriterion
)
{
  if(aCriterion == nullptr)
  {
    ANALYZE_THROWERR("ERROR: Requested criterion is a null pointer");
  }
  if(static_cast<Plato::OrdinalType>(mAdjoints.size()) <= static_cast<Plato::OrdinalType>(0))
  {
    const auto tNumDofs = mResidualEvaluator->numDofs();
    mAdjoints = Plato::ScalarMultiVector("Adjoint Variables", 1, tNumDofs);
  }
  // compute criterion contribution to the gradient
  constexpr Plato::Scalar tCYCLE = 0.0;
  auto tGradientConfig  = aCriterion->gradientConfig(aDatabase, tCYCLE);
  // add residual contribution to the gradient
  {
    // compute gradient with respect to state variables
    auto tGradientState = aCriterion->gradientState(aDatabase, tCYCLE);
    Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tGradientState);
    // compute jacobian with respect to state variables
    mJacobianState = mResidualEvaluator->jacobianState(aDatabase, tCYCLE, /*transpose=*/true);
    if( mWeakEBCs )
    { this->enforceWeakEssentialAdjointBoundaryConditions(aDatabase); }
    else
    { this->enforceStrongEssentialAdjointBoundaryConditions(mJacobianState, tGradientState); }
    // solve adjoint system of equations
    constexpr size_t tCYCLE_INDEX = 0;
    Plato::ScalarVector tMyAdjoints = Kokkos::subview(mAdjoints, tCYCLE_INDEX, Kokkos::ALL());
    mSolver->solve(*mJacobianState, tMyAdjoints, tGradientState, /*isAdjointSolve=*/ true);
    // compute jacobian with respect to configuration variables
    auto tJacobianConfig = mResidualEvaluator->jacobianConfig(aDatabase, tCYCLE, /*transpose=*/ true);
    // compute gradient with respect to design variables: dgdx * adjoint + dfdx
    Plato::MatrixTimesVectorPlusVector(tJacobianConfig, tMyAdjoints, tGradientConfig);
  }
  return tGradientConfig;
}

template<typename PhysicsType>
void Problem<PhysicsType>::
readEssentialBoundaryConditions(
  Teuchos::ParameterList& aParamList
)
{
  if(aParamList.isSublist("Essential Boundary Conditions") == false)
  { ANALYZE_THROWERR("ERROR: Essential boundary conditions parameter list is not defined in input deck") }
  Plato::EssentialBCs<ElementType>
  tEssentialBoundaryConditions(aParamList.sublist("Essential Boundary Conditions", false), mSpatialModel.Mesh);
  tEssentialBoundaryConditions.get(mDirichletDofs, mDirichletStateVals);
  
  if(aParamList.isType<bool>("Weak Essential Boundary Conditions"))
  { mWeakEBCs = aParamList.get<bool>("Weak Essential Boundary Conditions",false); }
  if(mMPCs)
  { mMPCs->checkEssentialBcsConflicts(mDirichletDofs);}
}

template<typename PhysicsType>
void Problem<PhysicsType>::
setEssentialBoundaryConditions(
    const Plato::OrdinalVector & aDofs, 
    const Plato::ScalarVector & aValues
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
Problem<PhysicsType>::
initializeEvaluators(
  Teuchos::ParameterList& aParamList
)
{
  auto tTypePDE = aParamList.get<std::string>("PDE Constraint");
  mResidualEvaluator = std::make_shared<Plato::Elliptic::VectorFunction<PhysicsType>>(
    tTypePDE,mSpatialModel,mDataMap,aParamList);
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
        " Parameter in Criteria block not valid.  Expect lists only.");
      auto tCriterion = tCriterionFactory.create(mSpatialModel,mDataMap,aParamList,tCriterionName);
      if( tCriterion != nullptr )
      {
        mCriterionEvaluator[tCriterionName] = tCriterion;
      }
    }
    if( mCriterionEvaluator.size() )
    {
      auto tTotalNumDofs = mResidualEvaluator->numDofs();
      mAdjoints = Plato::ScalarMultiVector("Adjoint Variables", 1, tTotalNumDofs);
    }
  }
}

template<typename PhysicsType>
void 
Problem<PhysicsType>::
initializeMultiPointConstraints(
  Teuchos::ParameterList& aParamList
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
Problem<PhysicsType>::
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
Problem<PhysicsType>::
parseSaveOutput(
  Teuchos::ParameterList & aParamList
)
{
  if( aParamList.isSublist("Output") ){
    auto tOutputParamList = aParamList.sublist("Output");
    if( tOutputParamList.isType<Teuchos::Array<std::string>>("Plottable") ){
      auto tPlottable = tOutputParamList.get<Teuchos::Array<std::string>>("Plottable");
      if( !tPlottable.empty() ){
        mSaveState = true;
      }
    }
  }
}

template<typename PhysicsType>
Plato::Solutions Problem<PhysicsType>::
getSolution() 
const
{
  Plato::Solutions tSolution(mPhysics, mTypePDE);
  tSolution.set("State", mStates, mResidualEvaluator->getDofNames());
  return tSolution;
}

} // namespace Elliptic

} // namespace Plato
