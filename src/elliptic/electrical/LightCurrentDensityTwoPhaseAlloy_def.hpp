/*
 * LightCurrentDensityTwoPhaseAlloy_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "elliptic/electrical/FactoryElectricalMaterial.hpp"

namespace Plato
{

template<typename EvaluationType>
LightCurrentDensityTwoPhaseAlloy<EvaluationType>::
LightCurrentDensityTwoPhaseAlloy(
  const std::string            & aMaterialName,
  const std::string            & aCurrentDensityName,
        Teuchos::ParameterList & aParamList
) : 
  mMaterialName(aMaterialName),
  mCurrentDensityName(aCurrentDensityName)
{
    this->initialize(aParamList);
}

template<typename EvaluationType>
LightCurrentDensityTwoPhaseAlloy<EvaluationType>::
~LightCurrentDensityTwoPhaseAlloy(){}

template<typename EvaluationType>
void 
LightCurrentDensityTwoPhaseAlloy<EvaluationType>::
evaluate(
    const Plato::SpatialDomain                         & aSpatialDomain,
    const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
    const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
    const Plato::Scalar                                & aScale
) const
{
    // compute current density
    Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
    Plato::ScalarMultiVectorT<Plato::Scalar> 
      tCurrentDensity("current density",tNumCells,mNumGaussPoints);
    mLightGeneratedCurrentDensity->evaluate(aState,tCurrentDensity);
    this->evaluate<Plato::Scalar>(aSpatialDomain,tCurrentDensity,aState,aControl,aConfig,aResult,aScale);
}

template<typename EvaluationType>
void 
LightCurrentDensityTwoPhaseAlloy<EvaluationType>::
evaluate(
    const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
    const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
    const Plato::Scalar                                & aScale
) const
{
    // compute current density
    Plato::OrdinalType tNumCells = aResult.extent(0);
    Plato::ScalarMultiVectorT<Plato::Scalar> 
      tCurrentDensity("current density",tNumCells,mNumGaussPoints);
    mLightGeneratedCurrentDensity->evaluate(aState,tCurrentDensity);
    this->evaluate<Plato::Scalar>(tCurrentDensity,aState,aControl,aConfig,aResult,aScale);
}

template<typename EvaluationType>
void 
LightCurrentDensityTwoPhaseAlloy<EvaluationType>::
initialize(
  Teuchos::ParameterList &aParamList
)
{
    if( !aParamList.isSublist("Source Terms") ){
      auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
      ANALYZE_THROWERR(tMsg)
    }
    auto tSourceTermsSublist = aParamList.sublist("Source Terms");
    if( !tSourceTermsSublist.isSublist(mCurrentDensityName) ){
      auto tMsg = std::string("Parameter is not valid. Argument ('") + mCurrentDensityName 
        + "') is not a parameter list";
      ANALYZE_THROWERR(tMsg)
    }
    auto tCurrentDensitySublist = tSourceTermsSublist.sublist(mCurrentDensityName);
    mPenaltyExponent = tCurrentDensitySublist.get<Plato::Scalar>("Penalty Exponent", 3.0);
    mMinErsatzMaterialValue = tCurrentDensitySublist.get<Plato::Scalar>("Minimum Value", 0.0);
    // set current density model
    mCurrentDensityModel = tCurrentDensitySublist.get<std::string>("Model","Constant");
    this->buildCurrentDensityModel(aParamList);
    // set out-of-plane thickness array
    Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
    auto tMaterialModel = tMaterialFactory.create(mMaterialName);
    this->setOutofPlaneThickness(tMaterialModel.operator*());
}

template<typename EvaluationType>
void 
LightCurrentDensityTwoPhaseAlloy<EvaluationType>::
setOutofPlaneThickness(
    Plato::MaterialModel<EvaluationType> & aMaterialModel
)
{
    std::vector<std::string> tThickness = aMaterialModel.property("out-of-plane thickness");
    if ( tThickness.empty() )
    {
        auto tMsg = std::string("Array of out-of-plane thicknesses is empty. ") 
          + "Light-generated current density cannnot be computed.";
        ANALYZE_THROWERR(tMsg)
    }
    else
    {
        mOutofPlaneThickness.clear();
        for(size_t tIndex = 0; tIndex < tThickness.size(); tIndex++)
        {
          Plato::Scalar tMyThickness = std::stod(tThickness[tIndex]);
          mOutofPlaneThickness.push_back(tMyThickness);
        }
    }
}

template<typename EvaluationType>
void 
LightCurrentDensityTwoPhaseAlloy<EvaluationType>::
buildCurrentDensityModel(
  Teuchos::ParameterList &aParamList
)
{
  mLightGeneratedCurrentDensity = 
    std::make_shared<Plato::LightGeneratedCurrentDensityConstant<EvaluationType>>(mCurrentDensityName,aParamList);
}

}
// namespace Plato