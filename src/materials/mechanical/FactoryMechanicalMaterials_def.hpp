/*
 * FactoryMechanicalMaterials_def.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "base/SupportedParamOptions.hpp"
#include "materials/mechanical/FactoryElasticMaterial.hpp"
#include "materials/mechanical/FactoryNonlinearElasticMaterial.hpp"

namespace Plato
{

template<typename EvaluationType>
std::shared_ptr<Plato::MaterialModel<EvaluationType>>
FactoryMechanicalMaterials<EvaluationType>::
create(
  const std::string            & aMaterialName,
        Teuchos::ParameterList & aParamList
)
{
  Plato::PhysicsEnum tS2E;
  auto tResponse = aParamList.get<std::string>("Response","Linear");
  auto tResponseEnum = tS2E.response(tResponse);
  switch (tResponseEnum)
  {
  case Plato::response_t::LINEAR:
  {
    Plato::FactoryElasticMaterial<EvaluationType> tFactory(aParamList);
    return ( tFactory.create(aMaterialName) );
    break;
  }
  case Plato::response_t::NONLINEAR:
  {
    Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
    return ( tFactory.create(aMaterialName) );
    break;
  }
  default:
    ANALYZE_THROWERR(std::string("ERROR: Response '") + tResponse 
      + "' does not support weak enforcement of Dirichlet boundary conditions")
    break;
  }
}

} // namespace Plato