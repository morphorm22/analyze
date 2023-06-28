/*
 * SupportedParamOptions.cpp
 *
 *  Created on: June 27, 2023
 */

#include "AnalyzeMacros.hpp"
#include "PlatoUtilities.hpp"
#include "base/SupportedParamOptions.hpp"

namespace Plato
{

Plato::physics_t 
PhysicsEnum::
physics(
  const std::string & aPhysics
) 
const
{
  auto tLowerPhysics = Plato::tolower(aPhysics);
  auto tItr = sp2e.find(tLowerPhysics);
  if( tItr == sp2e.end() ){
    auto tMsg = this->getErrorMsgPhysics(tLowerPhysics);
    ANALYZE_THROWERR(tMsg)
  }
  return tItr->second;
}

Plato::coupling_t
PhysicsEnum::
coupling(
  const std::string & aCoupling
) 
const
{
  auto tLowerCoupling = Plato::tolower(aCoupling);
  auto tItr = sc2e.find(tLowerCoupling);
  if( tItr == sc2e.end() ){
    auto tMsg = this->getErrorMsgCoupling(tLowerCoupling);
    ANALYZE_THROWERR(tMsg)
  }
  return tItr->second;
}

std::string
PhysicsEnum::
getErrorMsgPhysics(
  const std::string & aPhysics
)
const
{
  auto tMsg = std::string("Did not find input physics '") + aPhysics
    + "'. Supported response options are: ";
  for(const auto& tPair : sp2e){
    tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

std::string
PhysicsEnum::
getErrorMsgCoupling(
  const std::string & aCoupling
)
const
{
  auto tMsg = std::string("Did not find physics coupling option '") + aCoupling
    + "'. Supported physics coupling options are: ";
  for(const auto& tPair : sc2e){
    tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

} // namespace Plato
