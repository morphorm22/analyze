/*
 * FactoryProblemEvaluator_def.hpp
 *
 *  Created on: June 27, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "base/SupportedParamOptions.hpp"
#include "elliptic/evaluators/problem/ProblemEvaluatorVectorState.hpp"
#include "elliptic/evaluators/problem/ProblemEvaluatorThermoMechanics.hpp"

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
    return (
      std::make_shared<Plato::Elliptic::ProblemEvaluatorVectorState<PhysicsType>>(
        aParamList,aSpatialModel,aDataMap,aMachine) 
    );
    break;
  case Plato::physics_t::THERMOMECHANICAL:
    return ( this->createThermoMechanicalProblemEvaluator(aParamList,aSpatialModel,aDataMap,aMachine) );
    break;
  default:
    ANALYZE_THROWERR("ERROR: Fail to create an elliptic problem evaluator, check input physics")
    break;
  }
}

template<typename PhysicsType>
std::shared_ptr<Plato::ProblemEvaluatorBase>
FactoryProblemEvaluator<PhysicsType>::
createThermoMechanicalProblemEvaluator(
  Teuchos::ParameterList & aParamList,
  Plato::SpatialModel    & aSpatialModel,
  Plato::DataMap         & aDataMap,
  Plato::Comm::Machine   & aMachine
)
{
  Plato::PhysicsEnum tS2E;
  auto tCoupling = aParamList.get<std::string>("Coupling","monolithic");
  auto tLowerCoupling = Plato::tolower(tCoupling);
  auto tMyCoupling = tS2E.coupling(tLowerCoupling);
  switch (tMyCoupling)
  {
    case Plato::coupling_t::STAGGERED:
      return (
        std::make_shared<Plato::Elliptic::ProblemEvaluatorThermoMechanics<PhysicsType>>(
          aParamList,aSpatialModel,aDataMap,aMachine) 
      );
      break;
    default:
    case Plato::coupling_t::MONOLITHIC:
      return (
        std::make_shared<Plato::Elliptic::ProblemEvaluatorVectorState<PhysicsType>>(
          aParamList,aSpatialModel,aDataMap,aMachine) 
      );
      break;
  }
}

} // namespace Elliptic

} // namespace Plato
