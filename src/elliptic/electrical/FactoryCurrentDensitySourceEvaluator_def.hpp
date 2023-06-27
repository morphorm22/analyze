/*
 * FactoryCurrentDensitySourceEvaluator_deF.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "elliptic/electrical/SupportedParamOptions.hpp"
#include "elliptic/electrical/DarkCurrentDensityTwoPhaseAlloy.hpp"
#include "elliptic/electrical/LightCurrentDensityTwoPhaseAlloy.hpp"

namespace Plato
{

template<typename EvaluationType>
FactoryCurrentDensitySourceEvaluator<EvaluationType>::
FactoryCurrentDensitySourceEvaluator(){}

template<typename EvaluationType>
FactoryCurrentDensitySourceEvaluator<EvaluationType>::
~FactoryCurrentDensitySourceEvaluator(){}

template<typename EvaluationType>
std::shared_ptr<Plato::CurrentDensitySourceEvaluator<EvaluationType>> 
FactoryCurrentDensitySourceEvaluator<EvaluationType>::
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
  auto mCurrentDensitySourceEvaluatorParamList = tSourceTermsParamList.sublist(aFunctionName);
  if( !mCurrentDensitySourceEvaluatorParamList.isParameter("Function") )
  {
    auto tMsg = std::string("Parameter ('Function') is not defined in parameter list ('") 
      + aFunctionName + "'), current density evaluator cannot be determined";
    ANALYZE_THROWERR(tMsg)
  }
  Plato::Elliptic::electrical::CurrentDensitySourceEvaluatorEnum tS2E;
  auto tType = mCurrentDensitySourceEvaluatorParamList.get<std::string>("Function");
  auto tLowerType = Plato::tolower(tType);
  auto tSupportedSourceTermEnum = tS2E.get(tLowerType);
  switch (tSupportedSourceTermEnum)
  {
    case Plato::Elliptic::electrical::current_density_evaluator::TWO_PHASE_DARK_CURRENT_DENSITY:
      return std::make_shared<Plato::DarkCurrentDensityTwoPhaseAlloy<EvaluationType>>(
        aMaterialName,aFunctionName,aParamList);
      break;
    case Plato::Elliptic::electrical::current_density_evaluator::TWO_PHASE_LIGHT_GENERATED_CURRENT_DENSITY:
      return std::make_shared<Plato::LightCurrentDensityTwoPhaseAlloy<EvaluationType>>(
        aMaterialName,aFunctionName,aParamList);
      break;
    default:
      return nullptr;
      break;
  }
}

}