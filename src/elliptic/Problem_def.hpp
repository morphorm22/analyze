/*
 * Problem_def.hpp
 *
 *  Created on: June 21, 2023
 */

#pragma once

#include "AnalyzeOutput.hpp"
#include "elliptic/evaluators/problem/ProblemEvaluatorVectorState.hpp"

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
  Plato::AbstractProblem(aMesh,aParamList),
  mSpatialModel(aMesh,aParamList,mDataMap),
  mTypePDE(aParamList.get<std::string>("PDE Constraint")),
  mPhysics(aParamList.get<std::string>("Physics"))
{
  mProblemEvaluator = std::make_shared<Plato::Elliptic::ProblemEvaluatorVectorState<PhysicsType>>(
    aParamList,mSpatialModel,mDataMap,aMachine
  );
  this->parseSaveOutput(aParamList);
}

template<typename PhysicsType>
Plato::Solutions 
Problem<PhysicsType>::
getSolution() 
const
{
  return ( mProblemEvaluator->getSolution() );
}

template<typename PhysicsType>
bool
Problem<PhysicsType>::
criterionIsLinear(
  const std::string & aName
)
{
  return ( mProblemEvaluator->criterionIsLinear(aName) );
}

template<typename PhysicsType>
void 
Problem<PhysicsType>::
output(
  const std::string & aFilepath
)
{
  auto tDataMap = this->getDataMap();
  auto tSolution = mProblemEvaluator->getSolution();
  Plato::universal_solution_output(aFilepath, tSolution, tDataMap, mSpatialModel.Mesh);
}

template<typename PhysicsType>
void 
Problem<PhysicsType>::
updateProblem(
  const Plato::ScalarVector & aControls, 
  const Plato::Solutions    & aSolution
)
{
  Plato::Database tDatabase;
  this->buildDatabase(aControls,tDatabase);
  mProblemEvaluator->updateProblem(tDatabase);
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
  // save controls to output database
  mDataMap.scalarNodeFields["Topology"] = aControls;
  // analyze physics
  Plato::Database tDatabase;
  this->buildDatabase(aControls,tDatabase);
  mProblemEvaluator->analyze(tDatabase);
  // save state data if requested
  if ( mSaveState )
  {
    // evaluate at new state
    mProblemEvaluator->residual(tDatabase);
    mDataMap.saveState();
  }
  Plato::Solutions tSolution = mProblemEvaluator->getSolution();
  return tSolution;
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
  return ( mProblemEvaluator->criterionValue(aName, tDatabase) );
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
  Plato::Database tDatabase;
  this->buildDatabase(aControls,tDatabase);
  return ( mProblemEvaluator->criterionGradient(Plato::evaluation_t::GRAD_Z,aName,tDatabase) );
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
  Plato::Database tDatabase;
  this->buildDatabase(aControls,tDatabase);
  return ( mProblemEvaluator->criterionGradient(Plato::evaluation_t::GRAD_X,aName,tDatabase) );
}

template<typename PhysicsType>
void
Problem<PhysicsType>::
buildDatabase(
  const Plato::ScalarVector & aControls,
        Plato::Database     & aDatabase)
{
  aDatabase.scalar("cycle",0.0);
  aDatabase.scalar("cycle index",0);
  aDatabase.vector("controls", aControls);
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

} // namespace Elliptic

} // namespace Plato
