/*
 * FactoryNonlinearElasticMaterial_def.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "materials/mechanical/MaterialKirchhoff.hpp"
#include "materials/mechanical/MaterialNeoHookean.hpp"

namespace Plato
{

template<typename EvaluationType>
FactoryNonlinearElasticMaterial<EvaluationType>::
FactoryNonlinearElasticMaterial(
    Teuchos::ParameterList& aParamList
) :
    mParamList(aParamList)
{}

template<typename EvaluationType>
std::shared_ptr<Plato::MaterialModel<EvaluationType>> 
FactoryNonlinearElasticMaterial<EvaluationType>::
create(std::string aMaterialName)
{
  if (!mParamList.isSublist("Material Models"))
  {
    ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
  }
  else
  {
    auto tMaterialModelParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");
    if (!tMaterialModelParamList.isSublist(aMaterialName))
    {
        auto tMsg = std::string("Requested a material model with name ('") + aMaterialName 
                    + "') that is not defined in the input deck";
        ANALYZE_THROWERR(tMsg);
    }
    auto tMaterialParamList = tMaterialModelParamList.sublist(aMaterialName);
    if(tMaterialParamList.isSublist("Hyperelastic Kirchhoff")){
      auto tMaterial = std::make_shared<Plato::MaterialKirchhoff<EvaluationType>>
                        (aMaterialName, tMaterialParamList.sublist("Hyperelastic Kirchhoff"));
      return tMaterial;
    }
    else
    if(tMaterialParamList.isSublist("Hyperelastic Neo-Hookean")){
      auto tMaterial = std::make_shared<Plato::MaterialNeoHookean<EvaluationType>>
                        (aMaterialName, tMaterialParamList.sublist("Hyperelastic Neo-Hookean"));
      return tMaterial;
    }
    else{
      mS2E.get("Not Supported"); // throws
      return nullptr;
    }
  }
}

}