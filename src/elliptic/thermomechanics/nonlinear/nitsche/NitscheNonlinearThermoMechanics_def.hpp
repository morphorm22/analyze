/*
 * NitscheNonlinearThermoMechanics_def.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "elliptic/mechanical/NitscheDispMisfitEvaluator.hpp"
#include "elliptic/thermomechanics/nonlinear/nitsche/NitscheTestThermalHyperElasticStressEvaluator.hpp"
#include "elliptic/thermomechanics/nonlinear/nitsche/NitscheTrialThermalHyperElasticStressEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
NitscheNonlinearThermoMechanics<EvaluationType>::
NitscheNonlinearThermoMechanics(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams)
{
  // trial stress evaluator
  //
  mEvaluators.push_back(
    std::make_shared<Plato::Elliptic::NitscheTrialThermalHyperElasticStressEvaluator<EvaluationType>>(
      aParamList,aNitscheParams
    )
  );
  // test stress evaluator
  //
  mEvaluators.push_back(
    std::make_shared<Plato::Elliptic::NitscheTestThermalHyperElasticStressEvaluator<EvaluationType>>(
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
NitscheNonlinearThermoMechanics<EvaluationType>::
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