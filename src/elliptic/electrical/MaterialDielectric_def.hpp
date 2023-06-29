/*
 * MaterialDielectric_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

namespace Plato
{

template<typename EvaluationType>
MaterialDielectric<EvaluationType>::
MaterialDielectric(
    const std::string            & aMaterialName, 
          Teuchos::ParameterList & aParamList
)
{
    this->name(aMaterialName);
    this->initialize(aParamList);
}

template<typename EvaluationType>
MaterialDielectric<EvaluationType>::
~MaterialDielectric(){}

template<typename EvaluationType>
std::vector<std::string> 
MaterialDielectric<EvaluationType>::
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

template<typename EvaluationType>
void 
MaterialDielectric<EvaluationType>::
initialize(
    Teuchos::ParameterList & aParamList
)
{
  if(aParamList.isParameter("Electrical Resistance")){
    this->parseScalar("Electrical Resistance", aParamList);
    auto tElectricalResistance = this->getScalarConstant("Electrical Resistance");
    mProperties[mS2E.get("Electrical Resistance")].push_back(std::to_string(tElectricalResistance));
  }
  this->parseScalar("Electrical Constant", aParamList);
  this->parseScalar("Relative Static Permittivity", aParamList);
  auto tElectricConstant = this->getScalarConstant("Electrical Constant");
  auto tRelativeStaticPermittivity = this->getScalarConstant("Relative Static Permittivity");
  auto tValue = tElectricConstant * tRelativeStaticPermittivity;
  this->setTensorConstant("material tensor",Plato::TensorConstant<mNumSpatialDims>(tValue));
  mProperties[mS2E.get("Electrical Constant")].push_back(std::to_string(tElectricConstant));
  mProperties[mS2E.get("Relative Static Permittivity")].push_back(std::to_string(tRelativeStaticPermittivity));
}

}
// namespace Plato