/*
 * MaterialNeoHookean_def.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

namespace Plato
{

template<typename EvaluationType>
MaterialNeoHookean<EvaluationType>::
MaterialNeoHookean(
    const std::string            & aMaterialName,
    const Teuchos::ParameterList & aParamList
)
{
  // set material input parameter list name
  this->name(aMaterialName);
  // parse youngs modulus
  this->parseScalar("Youngs Modulus", aParamList);
  auto tYoungsModulus = this->getScalarConstant("Youngs Modulus");
  mProperties[mS2E.get("Youngs Modulus")].push_back( std::to_string(tYoungsModulus) );
  // parse poissons ratio
  this->parseScalar("Poissons Ratio", aParamList);
  auto tPoissonsRatio = this->getScalarConstant("Poissons Ratio");
  mProperties[mS2E.get("Poissons Ratio")].push_back( std::to_string(tPoissonsRatio) );
  // compute Lame constants
  auto tMu = tYoungsModulus / ( 2.*(1.+tPoissonsRatio) );
  mProperties[mS2E.get("Lame Mu")].push_back( std::to_string(tMu) );
  auto tLambda = (tPoissonsRatio * tYoungsModulus) / ( (1.+tPoissonsRatio)*(1.-2.*tPoissonsRatio) );
  mProperties[mS2E.get("Lame Lambda")].push_back( std::to_string(tLambda) );
}

template<typename EvaluationType>
std::vector<std::string> 
MaterialNeoHookean<EvaluationType>::
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