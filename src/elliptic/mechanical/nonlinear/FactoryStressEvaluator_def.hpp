/*
 * FactoryStressEvaluator_def.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "elliptic/mechanical/nonlinear/StressEvaluatorKirchhoff.hpp"
#include "elliptic/mechanical/nonlinear/StressEvaluatorNeoHookean.hpp"

namespace Plato
{

template<typename EvaluationType>
FactoryStressEvaluator<EvaluationType>::
FactoryStressEvaluator(
  const std::string & aMaterialName
) :
  mMaterialName(aMaterialName)
{}

template<typename EvaluationType>
std::shared_ptr<Plato::StressEvaluator<EvaluationType>> 
FactoryStressEvaluator<EvaluationType>::
create(
        Teuchos::ParameterList & aParamList,
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap
)
{
  if (!aParamList.isSublist("Material Models"))
  {
    ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
  }
  else
  {
    auto tMaterialModelParamList = aParamList.get<Teuchos::ParameterList>("Material Models");
    if (!tMaterialModelParamList.isSublist(mMaterialName))
    {
        auto tMsg = std::string("Requested a material model with name ('") + mMaterialName 
          + "') that is not defined in the input deck";
        ANALYZE_THROWERR(tMsg);
    }
    auto tMaterialParamList = tMaterialModelParamList.sublist(mMaterialName);
    if(tMaterialParamList.isSublist("Hyperelastic Kirchhoff")){
      auto tStressEvaluator = std::make_shared<Plato::StressEvaluatorKirchhoff<EvaluationType>>
                              (mMaterialName,aParamList,aSpatialDomain,aDataMap);
      return tStressEvaluator;
    }
    else
    if(tMaterialParamList.isSublist("Hyperelastic Neo-Hookean")){
      auto tStressEvaluator = std::make_shared<Plato::StressEvaluatorNeoHookean<EvaluationType>>
                              (mMaterialName,aParamList,aSpatialDomain,aDataMap);
      return tStressEvaluator;
    }
    else{
      mS2E.get("Not Supported"); // throws
      return nullptr;
    }
  }
}

} // namespace Plato
