/*
 * SupportedOptionEnums.cpp
 *
 *  Created on: May 23, 2023
 */

#include "AnalyzeMacros.hpp"
#include "PlatoUtilities.hpp"
#include "elliptic/electrical/SupportedOptionEnums.hpp"

namespace Plato
{

namespace electrical 
{

// begin: functions associated with PropEnum struct
Plato::electrical::property 
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
    auto tMsg = std::string("Did not find matching enum for input electrical property '") 
            + aInProperty + "'. Supported electrical property keywords are: ";
    for(const auto& tPair : s2e)
    {
        tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
}
// end: functions associated with PropEnum struct

// begin: functions associated with SourceEvaluatorEnum struct
Plato::electrical::source_evaluator 
SourceEvaluatorEnum::get(
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
SourceEvaluatorEnum::getErrorMsg(
    const std::string & aInProperty
) const
{
    auto tMsg = std::string("Did not find matching enum for input electrical source evaluator '") 
            + aInProperty + "'. Supported electrical source evaluator keywords are: ";
    for(const auto& tPair : s2e)
    {
        tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
}

// end: functions associated with SourceEvaluatorEnum struct

// begin: functions associated with CurrentDensityEvaluatorEnum struct

Plato::electrical::current_density_evaluator 
CurrentDensityEvaluatorEnum::get(
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
CurrentDensityEvaluatorEnum::getErrorMsg(
    const std::string & aInProperty
)
const
{
    auto tMsg = std::string("Did not find matching enum for input current density evaluator '") 
            + aInProperty + "'. Supported current density evaluator keywords are: ";
    for(const auto& tPair : s2e)
    {
        tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
}

// end: functions associated with CurrentDensityEvaluatorEnum struct

// begin: functions associated with CurrentDensityEnum struct

Plato::electrical::current_density 
CurrentDensityEnum::current_density(
  const std::string & aFunction,
  const std::string & aModel
) const
{
  Plato::electrical::current_density_evaluator tSourceEnum = mSourceTermEnums.get(aFunction);
  auto tLowerModel = Plato::tolower(aModel);
  auto tItr = s2e.find(tSourceEnum)->second.find(tLowerModel);
  if( tItr == s2e.find(tSourceEnum)->second.end() ){
    auto tMsg = this->getErrorMsg(aFunction,aModel);
    ANALYZE_THROWERR(tMsg)
  }
  return tItr->second.first;
}

Plato::electrical::response
CurrentDensityEnum::response(
  const std::string & aFunction,
  const std::string & aModel
) const
{
  Plato::electrical::current_density_evaluator tSourceEnum = mSourceTermEnums.get(aFunction);
  auto tLowerModel = Plato::tolower(aModel);
  auto tItr = s2e.find(tSourceEnum)->second.find(tLowerModel);
  if( tItr == s2e.find(tSourceEnum)->second.end() ){
    auto tMsg = this->getErrorMsg(aFunction,aModel);
    ANALYZE_THROWERR(tMsg)
  }
  return tItr->second.second;
}

std::string
CurrentDensityEnum::getErrorMsg(
  const std::string & aFunction,
  const std::string & aModel
) const
{
    auto tMsg = std::string("Requested current density evaluator ('") + aFunction 
      + "') does not support current density model of type ('" + aModel 
      + "'), supported current density models for current density evaluator ('" 
      + aFunction + "') are: ";
    Plato::electrical::current_density_evaluator tSourceEnum = mSourceTermEnums.get(aFunction);
    for(const auto& tPair : s2e.find(tSourceEnum)->second)
    {
        tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
}

// end: functions associated with CurrentDensityEnum struct

}
// namespace electrical 

}
// namespace Plato