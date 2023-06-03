/*
 * MaterialElectricalConductivityTwoPhaseAlloy_def.hpp
 *
 *  Created on: May 23, 2023
 */

#pragma once

#include "Simp.hpp"
#include "AnalyzeMacros.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

template<typename EvaluationType>
MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>::
MaterialElectricalConductivityTwoPhaseAlloy(
  const std::string            & aMaterialName, 
        Teuchos::ParameterList & aParamList
)
{
  this->name(aMaterialName);
  this->initialize(aParamList);
}

template<typename EvaluationType>
MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>::
~MaterialElectricalConductivityTwoPhaseAlloy(){}

template<typename EvaluationType>
std::vector<std::string> 
MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>::
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
MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>::
initialize(
    Teuchos::ParameterList & aParamList
)
{
  this->parseMaterialProperties(aParamList);
  this->parseOutofPlaneThickness(aParamList);
  this->parseMaterialNames(aParamList);
  this->parsePenaltyModel(aParamList);
  this->setMaterialTensors();
}

template<typename EvaluationType>
void
MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>::
parseMaterialProperties(
    Teuchos::ParameterList & aParamList
)
{
  bool tIsArray = aParamList.isType<Teuchos::Array<Plato::Scalar>>("Electrical Conductivity");
  if (tIsArray)
  {
      // parse inputs
      Teuchos::Array<Plato::Scalar> tConductivities = 
        aParamList.get<Teuchos::Array<Plato::Scalar>>("Electrical Conductivity");
      if(tConductivities.size() != 2){
        auto tMaterialName = this->name();
        auto tMsg = std::string("Size of electrical conductivity array must equal two. ") 
          + "Check electrical conductivity inputs in material block with name '" + tMaterialName
          + "'. Material tensor cannnot be computed.";
        ANALYZE_THROWERR(tMsg)
      }
      // create mirror 
      for(size_t tIndex = 0; tIndex < tConductivities.size(); tIndex++)
      {
        mProperties[mS2E.get("Electrical Conductivity")].push_back( std::to_string(tConductivities[tIndex]) );
        mConductivities.push_back(tConductivities[tIndex]);
      }
  }
  else
  {
      auto tMaterialName = this->name();
      auto tMsg = std::string("Array of electrical conductivities is not defined in material block with name '") 
        + tMaterialName + "'. Material tensor for a two-phase alloy cannnot be computed.";
      ANALYZE_THROWERR(tMsg)
  }
}

template<typename EvaluationType>
void
MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>::
parseMaterialNames(
    Teuchos::ParameterList & aParamList
)
{
  bool tIsArray = aParamList.isType<Teuchos::Array<Plato::Scalar>>("Material Name");
  if (tIsArray)
  {
      // parse inputs
      Teuchos::Array<std::string> tMaterialNames = 
        aParamList.get<Teuchos::Array<std::string>>("Material Name");
      // create mirror 
      for(size_t tIndex = 0; tIndex < mMaterialNames.size(); tIndex++){
          mMaterialNames.push_back(tMaterialNames[tIndex]);
      }
      if( mConductivities.size() > mMaterialNames.size() ){
          // assume default values for missing names
          for(Plato::OrdinalType tIndex = mMaterialNames.size() - 1u; tIndex < mConductivities.size(); tIndex++){
              auto tName = std::string("material ") + std::to_string(tIndex);
              mProperties[mS2E.get("Material Name")].push_back(tName);
              mMaterialNames.push_back(tName);
          }
      }
  }
  else
  {
      // assuming default names
      for(Plato::OrdinalType tIndex = 0; tIndex < mConductivities.size(); tIndex++)
      {
          auto tName = std::string("material ") + std::to_string(tIndex);
          mProperties[mS2E.get("Material Name")].push_back(tName);
          mMaterialNames.push_back(tName);
      }
  }
}

template<typename EvaluationType>
void
MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>::
parseOutofPlaneThickness(
    Teuchos::ParameterList &aParamList
)
{
  if(mNumSpatialDims >= 3){
      return;
  }
  bool tIsArray = aParamList.isType<Teuchos::Array<Plato::Scalar>>("Out-of-Plane Thickness");
  if (tIsArray)
  {
      // parse inputs
      Teuchos::Array<Plato::Scalar> tOutofPlaneThickness = 
        aParamList.get<Teuchos::Array<Plato::Scalar>>("Out-of-Plane Thickness");
      if(tOutofPlaneThickness.size() != 2){
        auto tMaterialName = this->name();
        auto tMsg = std::string("Size of out-of-plane thickness array must equal two. ") 
          + "Check out-of-plane thickness inputs in material block with name '" + tMaterialName
          + "'. Material tensor cannnot be computed.";
        ANALYZE_THROWERR(tMsg)
      }
      // create mirror 
      mOutofPlaneThickness.clear();
      for(size_t tIndex = 0; tIndex < tOutofPlaneThickness.size(); tIndex++)
      {
        mProperties[mS2E.get("Out-of-Plane Thickness")].push_back(std::to_string(tOutofPlaneThickness[tIndex]));
        mOutofPlaneThickness.push_back(tOutofPlaneThickness[tIndex]);
      }
  }
  else
  {
      auto tMsg = std::string("Requested an electrical conductivity material constitutive model for ") 
        + "a two-phase alloy modeled in two dimensions but array of out-of-plane thicknesses is not defined. "
        + "Physics of interest cannot be accurately simulated.";
      ANALYZE_THROWERR(tMsg)
  }
}

template<typename EvaluationType>
void
MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>::
parsePenaltyModel(
    Teuchos::ParameterList & aParamList
)
{
  mPenaltyExponent = aParamList.get<Plato::Scalar>("Penalty Exponent", 3.0);
  mProperties[mS2E.get("Penalty Exponent")].push_back( std::to_string(mPenaltyExponent) );
  mMinErsatzMaterialValue = aParamList.get<Plato::Scalar>("Minimum Value", 0.0);
  mProperties[mS2E.get("Minimum Value")].push_back( std::to_string(mMinErsatzMaterialValue) );
}

template<typename EvaluationType>
void
MaterialElectricalConductivityTwoPhaseAlloy<EvaluationType>::
setMaterialTensors()
{
  for(const auto& tConductivity : mConductivities){
    Plato::OrdinalType tIndex = &tConductivity - &mConductivities[0];
    this->setTensorConstant(mMaterialNames[tIndex],Plato::TensorConstant<mNumSpatialDims>(tConductivity));
  }
}

}
// namespace Plato