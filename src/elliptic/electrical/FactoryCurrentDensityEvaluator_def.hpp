/*
 * FactoryCurrentDensityEvaluator_deF.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "elliptic/electrical/SupportedOptionEnums.hpp"
#include "elliptic/electrical/DarkCurrentDensityTwoPhaseAlloy.hpp"
#include "elliptic/electrical/LightCurrentDensityTwoPhaseAlloy.hpp"

namespace Plato
{

template<typename EvaluationType>
FactoryCurrentDensityEvaluator<EvaluationType>::
FactoryCurrentDensityEvaluator(){}

template<typename EvaluationType>
FactoryCurrentDensityEvaluator<EvaluationType>::
~FactoryCurrentDensityEvaluator(){}

template<typename EvaluationType>
std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>> 
FactoryCurrentDensityEvaluator<EvaluationType>::
create(
  const std::string            & aMaterialName,
  const std::string            & aFunctionName,
        Teuchos::ParameterList & aParamList
)
{
  if( !aParamList.isSublist("Source Terms") )
  {
    auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
    ANALYZE_THROWERR(tMsg)
  }
  auto tSourceTermsParamList = aParamList.sublist("Source Terms");
  if( !tSourceTermsParamList.isSublist(aFunctionName) )
  {
    auto tMsg = std::string("Parameter is not valid. Argument ('") + aFunctionName + "') is not a parameter list";
    ANALYZE_THROWERR(tMsg)
  }
  auto mCurrentDensityEvaluatorParamList = tSourceTermsParamList.sublist(aFunctionName);
  if( !mCurrentDensityEvaluatorParamList.isParameter("Function") )
  {
    auto tMsg = std::string("Parameter ('Function') is not defined in parameter list ('") 
      + aFunctionName + "'), current density evaluator cannot be determined";
    ANALYZE_THROWERR(tMsg)
  }
  Plato::electrical::CurrentDensityEvaluatorEnum tS2E;
  auto tType = mCurrentDensityEvaluatorParamList.get<std::string>("Function");
  auto tLowerType = Plato::tolower(tType);
  auto tSupportedSourceTermEnum = tS2E.get(tLowerType);
  switch (tSupportedSourceTermEnum)
  {
    case Plato::electrical::current_density_evaluator::TWO_PHASE_DARK_CURRENT_DENSITY:
      return std::make_shared<Plato::DarkCurrentDensityTwoPhaseAlloy<EvaluationType>>(
        aMaterialName,aFunctionName,aParamList);
      break;
    case Plato::electrical::current_density_evaluator::TWO_PHASE_LIGHT_GENERATED_CURRENT_DENSITY:
      return std::make_shared<Plato::LightCurrentDensityTwoPhaseAlloy<EvaluationType>>(
        aMaterialName,aFunctionName,aParamList);
      break;
    default:
      return nullptr;
      break;
  }
}

}