/*
 * SupportedParamOptions.cpp
 *
 *  Created on: May 30, 2023
 */


#include "AnalyzeMacros.hpp"
#include "PlatoUtilities.hpp"
#include "elliptic/mechanical/SupportedParamOptions.hpp"

namespace Plato
{

namespace mechanical
{

Plato::mechanical::property 
PropEnum::get(
  const std::string &aInput
) const
{
  auto tLower = Plato::tolower(aInput);
  auto tItr = s2e.find(tLower);
  if( tItr == s2e.end() ){
      auto tMsg = this->getErrorMsg(tLower);
      ANALYZE_THROWERR(tMsg)
  }
  return tItr->second;
}

std::string
PropEnum::getErrorMsg(
  const std::string & aInProperty
) const
{
  auto tMsg = std::string("Did not find matching enum for input mechanical property '") 
          + aInProperty + "'. Supported mechanical property keywords are: ";
  for(const auto& tPair : s2e)
  {
      tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

Plato::mechanical::material 
MaterialEnum::get(
  const std::string &aInput
) const
{
  auto tLower = Plato::tolower(aInput);
  auto tItr = s2e.find(tLower);
  if( tItr == s2e.end() ){
      auto tMsg = this->getErrorMsg(tLower);
      ANALYZE_THROWERR(tMsg)
  }
  return tItr->second;
}

std::string
MaterialEnum::getErrorMsg(
  const std::string & aInProperty
) const
{
  auto tMsg = std::string("Did not find matching enum for input mechanical material '") 
          + aInProperty + "'. Supported mechanical material keywords are: ";
  for(const auto& tPair : s2e)
  {
      tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

Plato::mechanical::criterion
CriterionEnum::get(
    const std::string &aInput
) const
{
  auto tLower = Plato::tolower(aInput);
  auto tItr = s2e.find(tLower);
  if( tItr == s2e.end() ){
    auto tMsg = this->getErrorMsg(tLower);
    ANALYZE_THROWERR(tMsg)
  }
  return tItr->second;
}

std::string
CriterionEnum::getErrorMsg(
    const std::string & aInProperty
) const
{
  auto tMsg = std::string("Did not find matching enum for input nonlinear mechanical criterion '") 
    + aInProperty + "'. Supported nonlinear mechanical criterion keywords are: ";
  for(const auto& tPair : s2e){
    tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

}

}