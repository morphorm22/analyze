/*
 * FactoryNonlinearElasticMaterial_def.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "elliptic/mechanical/nonlinear/MaterialKirchhoff.hpp"
#include "elliptic/mechanical/nonlinear/MaterialNeoHookean.hpp"

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
    if(tMaterialParamList.isSublist("Kirchhoff")){
      auto tMaterial = std::make_shared<Plato::MaterialKirchhoff<EvaluationType>>
                        (aMaterialName, tMaterialParamList.sublist("Kirchhoff"));
      return tMaterial;
    }
    else
    if(tMaterialParamList.isSublist("Neo-Hookean")){
      auto tMaterial = std::make_shared<Plato::MaterialNeoHookean<EvaluationType>>
                        (aMaterialName, tMaterialParamList.sublist("Neo-Hookean"));
      return tMaterial;
    }
    else{
      mS2E.get("Not Supported"); // throws
      return nullptr;
    }
  }
}

}