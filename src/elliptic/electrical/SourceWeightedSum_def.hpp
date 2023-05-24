/*
 *  SourceWeightedSum_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "elliptic/electrical/FactoryCurrentDensityEvaluator.hpp"

namespace Plato
{

template<typename EvaluationType>
SourceWeightedSum<EvaluationType>::
SourceWeightedSum(
  const std::string            & aMaterialName,
        Teuchos::ParameterList & aParamList
) : 
  mMaterialName(aMaterialName)
{
  this->initialize(aParamList);
}

template<typename EvaluationType>
SourceWeightedSum<EvaluationType>::
~SourceWeightedSum()
{}

template<typename EvaluationType>
void 
SourceWeightedSum<EvaluationType>::
evaluate(
    const Plato::SpatialDomain                         & aSpatialDomain,
    const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
    const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
    const Plato::Scalar                                & aScale
) const
{
  for(auto& tFunction : mCurrentDensityEvaluators)
  {
    Plato::OrdinalType tFunctionIndex = &tFunction - &mCurrentDensityEvaluators[0];
    Plato::Scalar tScalarMultiplier = mFunctionWeights[tFunctionIndex] * aScale;
    tFunction->evaluate(aSpatialDomain,aState,aControl,aConfig,aResult,tScalarMultiplier);
  }
}

template<typename EvaluationType>
void 
SourceWeightedSum<EvaluationType>::
initialize(
  Teuchos::ParameterList & aParamList
)
{
  if( !aParamList.isSublist("Source Terms") ){
    auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
    ANALYZE_THROWERR(tMsg)
  }
  auto tSourceTermsParamList = aParamList.sublist("Source Terms");
  if( !tSourceTermsParamList.isSublist("Source") ){
    auto tMsg = std::string("Parameter is not valid. Argument ('Source') is not a parameter list");
    ANALYZE_THROWERR(tMsg)
  }
  auto tSourceParamList = tSourceTermsParamList.sublist("Source");
  this->parseFunctions(tSourceParamList);
  this->parseWeights(tSourceParamList);
  this->createCurrentDensityEvaluators(aParamList);
}

template<typename EvaluationType>
void 
SourceWeightedSum<EvaluationType>::
parseFunctions(
  Teuchos::ParameterList & aParamList
)
{
  bool tIsArray = aParamList.isType<Teuchos::Array<std::string>>("Functions");
  if( !tIsArray)
  {
    auto tMsg = std::string("Argument ('Functions') is not defined in ('Source') parameter list");
    ANALYZE_THROWERR(tMsg)
  }
  Teuchos::Array<std::string> tFunctions = aParamList.get<Teuchos::Array<std::string>>("Functions");
  for( Plato::OrdinalType tIndex = 0; tIndex < tFunctions.size(); tIndex++ )
    { mFunctions.push_back(tFunctions[tIndex]); }
}

template<typename EvaluationType>
void 
SourceWeightedSum<EvaluationType>::
parseWeights(
  Teuchos::ParameterList & aParamList      
)
{
  bool tIsArray = aParamList.isType<Teuchos::Array<Plato::Scalar>>("Weights");
  if( !tIsArray)
  {
    if(mFunctions.size() <= 0)
    {
      auto tMsg = std::string("Argument ('Functions') has not been parsed, ") 
        + "default number of weights cannot be determined";
      ANALYZE_THROWERR(tMsg)
    }
    for( Plato::OrdinalType tIndex = 0; tIndex < mFunctions.size(); tIndex++ )
      { mFunctionWeights.push_back(1.0); }
  }
  else
  {
    Teuchos::Array<Plato::Scalar> tWeights = aParamList.get<Teuchos::Array<Plato::Scalar>>("Weights");
    for( Plato::OrdinalType tIndex = 0; tIndex < tWeights.size(); tIndex++ )
      { mFunctionWeights.push_back(tWeights[tIndex]); }
  }
}

template<typename EvaluationType>
void 
SourceWeightedSum<EvaluationType>::
createCurrentDensityEvaluators(
  Teuchos::ParameterList & aParamList  
)
{
  for(auto& tFunctionName : mFunctions)
  {
    Plato::FactoryCurrentDensityEvaluator<EvaluationType> tFactoryCurrentDensityEvaluator;
    std::shared_ptr<Plato::CurrentDensityEvaluator<EvaluationType>> tEvaluator = 
      tFactoryCurrentDensityEvaluator.create(mMaterialName,tFunctionName,aParamList);
    mCurrentDensityEvaluators.push_back(tEvaluator);
  }
}

}