/*
 * NitscheBoundaryCondition_def.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "elliptic/evaluators/nitsche/FactoryNitscheEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
NitscheBoundaryCondition<EvaluationType>::
NitscheBoundaryCondition(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams)
{
  Plato::Elliptic::FactoryNitscheEvaluator<EvaluationType> tFactory;
  mNitscheBC = tFactory.create(aParamList,aNitscheParams);
}

template<typename EvaluationType>
void 
NitscheBoundaryCondition<EvaluationType>::
evaluate(
  const Plato::SpatialModel & aSpatialModel,
  const Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aCycle,
        Plato::Scalar         aScale
)
{
  mNitscheBC->evaluate(aSpatialModel,aWorkSets,aCycle,aScale);
}

} // namespace Elliptic

} // namespace Plato