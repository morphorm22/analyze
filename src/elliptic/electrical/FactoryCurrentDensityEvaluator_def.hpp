/*
 *  FactoryCurrentDensityEvaluator_def.hpp
 *
 *  Created on: June 2, 2023
 */

#include "AnalyzeMacros.hpp"
#include "elliptic/electrical/CurrentDensityConstant.hpp"
#include "elliptic/electrical/CurrentDensityTwoPhaseAlloy.hpp"

namespace Plato
{

template<typename EvaluationType>
FactoryCurrentDensityEvaluator<EvaluationType>::
FactoryCurrentDensityEvaluator(
  const std::string            & aMaterialName,
        Teuchos::ParameterList & aParamList
) :
  mMaterialName(aMaterialName),
  mParamList(aParamList)
{}

template<typename EvaluationType>
std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>> 
FactoryCurrentDensityEvaluator<EvaluationType>::
create(
  const Plato::SpatialDomain & aSpatialDomain,
        Plato::DataMap       & aDataMap
)
{
  if (!mParamList.isSublist("Material Models"))
  {
    ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
  }
  else
  {
    auto tModelsParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");
    if (!tModelsParamList.isSublist(mMaterialName))
    {
      auto tMsg = std::string("Requested a material model with name ('") + mMaterialName 
                  + "') that is not defined in the input deck";
      ANALYZE_THROWERR(tMsg);
    }
    Teuchos::ParameterList tMaterialParamList = tModelsParamList.sublist(mMaterialName);
    if(tMaterialParamList.isSublist("Two Phase Conductive"))
    {
      auto tCurrentDensity = std::make_shared<Plato::CurrentDensityTwoPhaseAlloy<EvaluationType>>
                               (mMaterialName,mParamList,aSpatialDomain,aDataMap);
      return tCurrentDensity;
    }
    else
    if(tMaterialParamList.isSublist("Conductive"))
    {
      auto tCurrentDensity = std::make_shared<Plato::CurrentDensityConstant<EvaluationType>>
                               (mMaterialName,mParamList,aSpatialDomain,aDataMap);
      return tCurrentDensity;
    }
    else 
    if(tMaterialParamList.isSublist("Dielectric"))
    {
      auto tCurrentDensity = std::make_shared<Plato::CurrentDensityConstant<EvaluationType>>
                               (mMaterialName,mParamList,aSpatialDomain,aDataMap);
      return tCurrentDensity;
    }
    else
    {
      mS2E.get("Not Supported"); // throws error 
      return nullptr;
    }
  }
}

} // namespace Plato
