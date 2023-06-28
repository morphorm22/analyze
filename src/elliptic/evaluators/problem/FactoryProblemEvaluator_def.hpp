/*
 * FactoryProblemEvaluator_def.hpp
 *
 *  Created on: June 27, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "base/SupportedParamOptions.hpp"
#include "elliptic/evaluators/problem/ProblemEvaluatorVectorState.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
FactoryProblemEvaluator<PhysicsType>::
FactoryProblemEvaluator(){}

template<typename PhysicsType>
std::shared_ptr<Plato::ProblemEvaluatorBase>
FactoryProblemEvaluator<PhysicsType>::
create(
  Teuchos::ParameterList & aParamList,
  Plato::SpatialModel    & aSpatialModel,
  Plato::DataMap         & aDataMap,
  Plato::Comm::Machine   & aMachine
)
{
  auto tPhysics  = aParamList.get<std::string>("Physics","");
  auto tCoupling = aParamList.get<std::string>("Coupling","monolithic");
  if( tPhysics.empty() ){
    ANALYZE_THROWERR("ERROR: Problem ('Physics') were not defined, analysis cannot be performed")
  }
  Plato::PhysicsEnum tS2E;
  auto tLowerPhysics = Plato::tolower(tPhysics);
  auto tMyPhysics = tS2E.physics(tLowerPhysics);
  switch ( tMyPhysics )
  {
  case Plato::physics_t::THERMAL:
  case Plato::physics_t::ELECTRICAL:
  case Plato::physics_t::MECHANICAL:
  case Plato::physics_t::THERMOMECHANICAL:
    return (
      std::make_shared<Plato::Elliptic::ProblemEvaluatorVectorState<PhysicsType>>(
        aParamList,aSpatialModel,aDataMap,aMachine) 
    );
    break;
  default:
    ANALYZE_THROWERR("ERROR: Fail to create an elliptic problem evaluator, check input physics")
    break;
  }
}

} // namespace Elliptic

} // namespace Plato
