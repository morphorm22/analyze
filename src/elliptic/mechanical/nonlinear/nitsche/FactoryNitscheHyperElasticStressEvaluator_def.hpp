/*
 *  FactoryNitscheHyperElasticStressEvaluator_def.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "elliptic/mechanical/SupportedParamOptions.hpp"

#include "elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTestKirchhoffStress.hpp"
#include "elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTrialKirchhoffStress.hpp"
#include "elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTestNeoHookeanStress.hpp"
#include "elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTrialNeoHookeanStress.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
std::shared_ptr<Plato::BoundaryFluxEvaluator<EvaluationType>> 
FactoryNitscheHyperElasticStressEvaluator<EvaluationType>::
createTrialEvaluator(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
)
{
  if (!aParamList.isSublist("Material Models"))
  {
    ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
  }
  else
  {
    auto tMaterialName = this->getMaterialName(aNitscheParams);
    auto tMaterialModelParamList = aParamList.get<Teuchos::ParameterList>("Material Models");
    if (!tMaterialModelParamList.isSublist(tMaterialName))
    {
        auto tMsg = std::string("Requested a material model with name ('") + tMaterialName 
          + "') that is not defined in the input deck";
        ANALYZE_THROWERR(tMsg);
    }
    auto tMaterialParamList = tMaterialModelParamList.sublist(tMaterialName);
    if(tMaterialParamList.isSublist("Hyperelastic Kirchhoff")){
      auto tStressEvaluator = 
        std::make_shared<Plato::Elliptic::BoundaryEvaluatorTrialKirchhoffStress<EvaluationType>>(
          aParamList,aNitscheParams
        );
      return tStressEvaluator;
    }
    else
    if(tMaterialParamList.isSublist("Hyperelastic Neo-Hookean")){
      auto tStressEvaluator = 
        std::make_shared<Plato::Elliptic::BoundaryEvaluatorTrialNeoHookeanStress<EvaluationType>>(
          aParamList,aNitscheParams
        );
      return tStressEvaluator;
    }
    else{
      Plato::Elliptic::mechanical::MaterialEnum tS2E;
      tS2E.get("Not Supported"); // throws
      return nullptr;
    }
  }
}

template<typename EvaluationType>
std::shared_ptr<Plato::BoundaryFluxEvaluator<EvaluationType>> 
FactoryNitscheHyperElasticStressEvaluator<EvaluationType>::
createTestEvaluator(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
)
{
  if (!aParamList.isSublist("Material Models"))
  {
    ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
  }
  else
  {
    auto tMaterialName = this->getMaterialName(aNitscheParams);
    auto tMaterialModelParamList = aParamList.get<Teuchos::ParameterList>("Material Models");
    if (!tMaterialModelParamList.isSublist(tMaterialName))
    {
        auto tMsg = std::string("Requested a material model with name ('") + tMaterialName 
          + "') that is not defined in the input deck";
        ANALYZE_THROWERR(tMsg);
    }
    auto tMaterialParamList = tMaterialModelParamList.sublist(tMaterialName);
    if(tMaterialParamList.isSublist("Hyperelastic Kirchhoff")){
      auto tStressEvaluator = 
        std::make_shared<Plato::Elliptic::BoundaryEvaluatorTestKirchhoffStress<EvaluationType>>(
          aParamList,aNitscheParams
        );
      return tStressEvaluator;
    }
    else
    if(tMaterialParamList.isSublist("Hyperelastic Neo-Hookean")){
      auto tStressEvaluator = 
        std::make_shared<Plato::Elliptic::BoundaryEvaluatorTestNeoHookeanStress<EvaluationType>>(
          aParamList,aNitscheParams
        );
      return tStressEvaluator;
    }
    else{
      Plato::Elliptic::mechanical::MaterialEnum tS2E;
      tS2E.get("Not Supported"); // throws
      return nullptr;
    }
  }
}

template<typename EvaluationType>
std::string 
FactoryNitscheHyperElasticStressEvaluator<EvaluationType>::
getMaterialName(
  const Teuchos::ParameterList & aParamList
)
{
  if( !aParamList.isParameter("Material Model") ){
    ANALYZE_THROWERR( std::string("ERROR: Input argument ('Material Model') is not defined, ") + 
      "material constitutive model for Nitsche's method cannot be determined" )
  }
  auto tMaterialName = aParamList.get<std::string>("Material Model");
  return (tMaterialName);
}

} // namespace Elliptic

} // namespace Plato
