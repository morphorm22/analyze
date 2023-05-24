/*
 * FactoryElectricalMaterial_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

#include "elliptic/electrical/MaterialDielectric.hpp"
#include "elliptic/electrical/MaterialElectricalConductivity.hpp"
#include "elliptic/electrical/MaterialElectricalConductivityTwoPhaseAlloy.hpp"

namespace Plato
{

template<typename EvaluationType>
FactoryElectricalMaterial<EvaluationType>::
FactoryElectricalMaterial(
    Teuchos::ParameterList& aParamList
) :
  mParamList(aParamList)
{}

template<typename EvaluationType>
FactoryElectricalMaterial<EvaluationType>::
~FactoryElectricalMaterial()
{}

template<typename EvaluationType>
std::shared_ptr<MaterialModel<EvaluationType>> 
FactoryElectricalMaterial<EvaluationType>::
create(std::string aModelName)
{
    if (!mParamList.isSublist("Material Models"))
    {
        ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
    }
    else
    {
        auto tModelsParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");
        if (!tModelsParamList.isSublist(aModelName))
        {
            auto tMsg = std::string("Requested a material model with name ('") + aModelName 
                        + "') that is not defined in the input deck";
            ANALYZE_THROWERR(tMsg);
        }

        auto tModelParamList = tModelsParamList.sublist(aModelName);
        if(tModelParamList.isSublist("Two Phase Electrical Conductivity"))
        {
            auto tMaterial = std::make_shared<Plato::MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>>
                             (aModelName, tModelParamList.sublist("Two Phase Electrical Conductivity"));
            tMaterial->model("Two Phase Electrical Conductivity");
            return tMaterial;
        }
        else
        if(tModelParamList.isSublist("Electrical Conductivity"))
        {
            auto tMaterial = std::make_shared<Plato::MaterialElectricalConductivity<EvaluationType>>
                             (aModelName, tModelParamList.sublist("Electrical Conductivity"));
            tMaterial->model("Electrical Conductivity");
            return tMaterial;
        }
        else 
        if(tModelParamList.isSublist("Dielectric"))
        {
            auto tMaterial = std::make_shared<Plato::MaterialDielectric<EvaluationType>>
                            (aModelName, tModelParamList.sublist("Dielectric"));
            tMaterial->model("Dielectric");
            return tMaterial;
        }
        else
        {
            auto tErrMsg = this->getErrorMsg();
            ANALYZE_THROWERR(tErrMsg);
        }
    }
}

template<typename EvaluationType>
std::string
FactoryElectricalMaterial<EvaluationType>::
getErrorMsg()
const
{
    std::string tMsg = std::string("ERROR: Requested material constitutive model is not supported. ")
        + "Supported material constitutive models for a steady state current analysis are: ";
    for(const auto& tElement : mSupportedMaterials)
    {
        tMsg = tMsg + "'" + tElement + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
}

}
// namespace Plato