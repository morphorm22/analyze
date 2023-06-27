/*
 *  FactorySourceEvaluator_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "PlatoUtilities.hpp"
#include "elliptic/electrical/SourceWeightedSum.hpp"
#include "elliptic/electrical/SupportedParamOptions.hpp"

namespace Plato
{

template<typename EvaluationType>
FactorySourceEvaluator<EvaluationType>::
FactorySourceEvaluator(){}

template<typename EvaluationType>
FactorySourceEvaluator<EvaluationType>::
~FactorySourceEvaluator(){}

template<typename EvaluationType>
std::shared_ptr<Plato::SourceEvaluator<EvaluationType>> 
FactorySourceEvaluator<EvaluationType>::
create(
  const std::string            & aMaterialName,
        Teuchos::ParameterList & aParamList
)
{
  if( !aParamList.isSublist("Source Terms") )
  {
    /* auto tMsg = std::string("WARNING: Parameter list ('Source Terms') is not defined, ") 
      + "program assumes no source term is needed in the analysis";
    WARNING(tMsg)*/
    return nullptr;
  }
  auto tSourceTermsParamList = aParamList.sublist("Source Terms");
  if( !tSourceTermsParamList.isSublist("Source") )
  {
    auto tMsg = std::string("Parameter is not valid. Argument ('Source') is not a parameter list");
    ANALYZE_THROWERR(tMsg)
  }
  auto tSourceParamList = tSourceTermsParamList.sublist("Source");
  if( !tSourceParamList.isParameter("Type") )
  {
    auto tMsg = std::string("Parameter ('Type') is not defined in parameter list ('Source'), ") 
      + "source evaluator cannot be determined";
    ANALYZE_THROWERR(tMsg)
  }
  Plato::Elliptic::electrical::SourceEvaluatorEnum tS2E;
  auto tType = tSourceParamList.get<std::string>("Type");
  auto tLowerType = Plato::tolower(tType);
  auto tSupportedSourceEvaluatorEnum = tS2E.get(tLowerType);
  switch (tSupportedSourceEvaluatorEnum)
  {
    case Plato::Elliptic::electrical::source_evaluator::WEIGHTED_SUM:
      return std::make_shared<Plato::SourceWeightedSum<EvaluationType>>(aMaterialName,aParamList);
      break;
    default:
      return nullptr;
      break;
  }
}

}