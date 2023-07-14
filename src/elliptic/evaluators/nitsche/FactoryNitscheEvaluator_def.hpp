/*
 * FactoryNitscheEvaluator_def.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "base/SupportedParamOptions.hpp"

#include "elliptic/thermal/nitsche/NitscheLinearThermoStatics.hpp"
#include "elliptic/mechanical/linear/nitsche/NitscheLinearMechanics.hpp"
#include "elliptic/mechanical/nonlinear/nitsche/NitscheNonLinearMechanics.hpp"
#include "elliptic/thermomechanics/nonlinear/nitsche/NitscheNonlinearThermoMechanics.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
std::shared_ptr<Plato::NitscheEvaluator>
FactoryNitscheEvaluator<EvaluationType>::
create(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
)
{
  if( !aParamList.isParameter("Physics") ){
    ANALYZE_THROWERR("ERROR: Argument ('Physics') is not defined, nitsche's evaluator cannot be created")
  }
  Plato::PhysicsEnum tS2E;
  auto tResponse = aParamList.get<std::string>("Response","Linear");
  auto tResponseEnum = tS2E.response(tResponse);
  switch (tResponseEnum)
  {
  case Plato::response_t::LINEAR:
    return ( this->createLinearNitscheEvaluator(aParamList,aNitscheParams) );
    break;
  case Plato::response_t::NONLINEAR:
    return ( this->createNonlinearNitscheEvaluator(aParamList,aNitscheParams) );
    break;
  default:
    ANALYZE_THROWERR(std::string("ERROR: Response '") + tResponse 
      + "' does not support weak enforcement of Dirichlet boundary conditions")
    break;
  }
}

template<typename EvaluationType>
std::shared_ptr<Plato::NitscheEvaluator>
FactoryNitscheEvaluator<EvaluationType>::
createLinearNitscheEvaluator(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
)
{
  if( !aParamList.isParameter("Physics") ){
    ANALYZE_THROWERR("ERROR: Argument ('Physics') is not defined, nitsche's evaluator cannot be created")
  }
  Plato::PhysicsEnum tS2E;
  auto tPhysics = aParamList.get<std::string>("Physics");
  auto tPhysicsEnum = tS2E.physics(tPhysics);
  switch (tPhysicsEnum)
  {
  case Plato::physics_t::THERMAL:
    return ( std::make_shared<Plato::Elliptic::NitscheLinearThermoStatics<EvaluationType>>(
      aParamList,aNitscheParams) 
    );
    break;
  case Plato::physics_t::MECHANICAL:
    return ( std::make_shared<Plato::Elliptic::NitscheLinearMechanics<EvaluationType>>(
      aParamList,aNitscheParams) 
    );
    break;
  default:
    ANALYZE_THROWERR(std::string("ERROR: Physics '") + tPhysics 
      + "' does not support weak enforcement of Dirichlet boundary conditions")
    break;
  }
}

template<typename EvaluationType>
std::shared_ptr<Plato::NitscheEvaluator>
FactoryNitscheEvaluator<EvaluationType>::
createNonlinearNitscheEvaluator(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
)
{
  if( !aParamList.isParameter("Physics") ){
    ANALYZE_THROWERR("ERROR: Argument ('Physics') is not defined, nitsche's evaluator cannot be created")
  }
  Plato::PhysicsEnum tS2E;
  auto tPhysics = aParamList.get<std::string>("Physics");
  auto tPhysicsEnum = tS2E.physics(tPhysics);
  switch (tPhysicsEnum)
  {
  case Plato::physics_t::MECHANICAL:
    return 
    ( 
      std::make_shared<Plato::Elliptic::NitscheNonLinearMechanics<EvaluationType>>(
        aParamList,aNitscheParams) 
    );
    break;
  case Plato::physics_t::THERMOMECHANICAL:
    return 
    ( 
      std::make_shared<Plato::Elliptic::NitscheNonlinearThermoMechanics<EvaluationType>>(
        aParamList,aNitscheParams) 
    );
    break;
  default:
    ANALYZE_THROWERR(std::string("ERROR: Physics '") + tPhysics 
      + "' does not support weak enforcement of Dirichlet boundary conditions")
    break;
  }
}

} // namespace Elliptic

} // namespace Plato