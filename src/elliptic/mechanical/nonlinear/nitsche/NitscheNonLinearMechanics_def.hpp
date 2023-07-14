/*
 * NitscheNonLinearMechanics_def.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "elliptic/mechanical/NitscheDispMisfitEvaluator.hpp"
#include "elliptic/mechanical/nonlinear/nitsche/NitscheTestHyperElasticStressEvaluator.hpp"
#include "elliptic/mechanical/nonlinear/nitsche/NitscheTrialHyperElasticStressEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
NitscheNonLinearMechanics<EvaluationType>::
NitscheNonLinearMechanics(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams)
{
  // trial stress evaluator
  //
  mEvaluators.push_back(
    std::make_shared<Plato::Elliptic::NitscheTrialHyperElasticStressEvaluator<EvaluationType>>(
      aParamList,aNitscheParams
    )
  );
  // test stress evaluator
  //
  mEvaluators.push_back(
    std::make_shared<Plato::Elliptic::NitscheTestHyperElasticStressEvaluator<EvaluationType>>(
      aParamList,aNitscheParams
    )
  );
  // displacement misfit evaluator
  //
  mEvaluators.push_back(
    std::make_shared<Plato::Elliptic::NitscheDispMisfitEvaluator<EvaluationType>>(
      aParamList,aNitscheParams
    )
  );
}

template<typename EvaluationType>
void 
NitscheNonLinearMechanics<EvaluationType>::
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