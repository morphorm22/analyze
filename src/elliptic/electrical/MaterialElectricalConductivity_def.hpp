/*
 * MaterialElectricalConductivity_def.hpp
 *
 *  Created on: May 23, 2023
 */

#pragma once

#include "elliptic/electrical/MaterialElectricalConductivity_decl.hpp"

namespace Plato
{

template<typename EvaluationType>
MaterialElectricalConductivity<EvaluationType>::
MaterialElectricalConductivity(
    const std::string            & aMaterialName,
    const Teuchos::ParameterList & aParamList
)
{
    this->name(aMaterialName);
    this->parseScalar("Electrical Conductivity", aParamList);
    auto tElectricConductivity = this->getScalarConstant("Electrical Conductivity");
    this->setTensorConstant("material tensor",Plato::TensorConstant<mNumSpatialDims>(tElectricConductivity));
    mProperties[mS2E.get("Electrical Conductivity")].push_back( std::to_string(tElectricConductivity) );
}

template<typename EvaluationType>
std::vector<std::string> 
MaterialElectricalConductivity<EvaluationType>::
property(const std::string & aPropertyID)
const 
{
    auto tEnum = mS2E.get(aPropertyID);
    auto tItr = mProperties.find(tEnum);
    if( tItr == mProperties.end() ){
        return {};
    }
    return tItr->second;
}

}
// namespace Plato