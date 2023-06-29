/*
 * MaterialElectricalConductivity_def.hpp
 *
 *  Created on: May 23, 2023
 */

#pragma once

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
  if(aParamList.isParameter("Electrical Resistance")){
    this->parseScalar("Electrical Resistance", aParamList);
    auto tElectricalResistance = this->getScalarConstant("Electrical Resistance");
    mProperties[mS2E.get("Electrical Resistance")].push_back(std::to_string(tElectricalResistance));
  }
  this->parseScalar("Electrical Conductivity", aParamList);
  auto tElectricalConductivity = this->getScalarConstant("Electrical Conductivity");
  this->setTensorConstant("material tensor",Plato::TensorConstant<mNumSpatialDims>(tElectricalConductivity));
  mProperties[mS2E.get("Electrical Conductivity")].push_back( std::to_string(tElectricalConductivity) );
}

template<typename EvaluationType>
std::vector<std::string> 
MaterialElectricalConductivity<EvaluationType>::
property(
  const std::string & aPropertyID
)
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