/*
 * NitscheLinearThermoStatics_decl.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "elliptic/thermal/nitsche/NitscheTempMisfitEvaluator.hpp"
#include "elliptic/thermal/nitsche/NitscheTestHeatFluxEvaluator.hpp"
#include "elliptic/thermal/nitsche/NitscheTrialHeatFluxEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
NitscheLinearThermoStatics<EvaluationType>::
NitscheLinearThermoStatics(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams)
{
  // trial heat flux evaluator
  //
  mEvaluators.push_back(
    std::make_shared<Plato::Elliptic::NitscheTrialHeatFluxEvaluator<EvaluationType>>(
      aParamList,aNitscheParams
    )
  );
  // test heat flux evaluator
  //
  mEvaluators.push_back(
    std::make_shared<Plato::Elliptic::NitscheTestHeatFluxEvaluator<EvaluationType>>(
      aParamList,aNitscheParams
    )
  );
  // temperature misfit evaluator
  //
  mEvaluators.push_back(
    std::make_shared<Plato::Elliptic::NitscheTempMisfitEvaluator<EvaluationType>>(
      aParamList,aNitscheParams
    )
  );
}

template<typename EvaluationType>
void 
NitscheLinearThermoStatics<EvaluationType>::
evaluate(
  const Plato::SpatialModel & aSpatialModel,
  const Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aCycle,
        Plato::Scalar         aScale
)
{
  if(mEvaluators.empty()){
    ANALYZE_THROWERR( std::string("ERROR: Found an empty list of Nitsche evaluators, weak Dirichlet boundary " )
      + "conditions cannot be enforced" )
  }
  for(auto& tEvaluator : mEvaluators){
    tEvaluator->evaluate(aSpatialModel,aWorkSets,aCycle,aScale);
  }
}

} // namespace Elliptic

} // namespace Plato
