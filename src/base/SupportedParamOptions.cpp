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
) const
{
  auto tLowerPhysics = Plato::tolower(aPhysics);
  auto tItr = sp2e.find(tLowerPhysics);
  if( tItr == sp2e.end() ){
    auto tMsg = this->getErrorMsgPhysics(tLowerPhysics);
    ANALYZE_THROWERR(tMsg)
  }
  return tItr->second.first;
}

Plato::coupling_t
PhysicsEnum::
coupling(
  const std::string & aCoupling
) const
{
  auto tLowerCoupling = Plato::tolower(aCoupling);
  auto tItr = sc2e.find(tLowerCoupling);
  if( tItr == sc2e.end() ){
    auto tMsg = this->getErrorMsgCoupling(tLowerCoupling);
    ANALYZE_THROWERR(tMsg)
  }
  return tItr->second;
}

Plato::response_t
PhysicsEnum::
response(
  const std::string & aResponse
) const
{
  auto tLowerResponse = Plato::tolower(aResponse);
  auto tItr = sr2e.find(tLowerResponse);
  if( tItr == sr2e.end() ){
    auto tMsg = this->getErrorMsgResponse(tLowerResponse);
    ANALYZE_THROWERR(tMsg)
  }
  return tItr->second;
}

Plato::pde_t
PhysicsEnum::
pde(
  const std::string & aPDE
) const
{
  auto tLowerPDE = Plato::tolower(aPDE);
  auto tItr = se2e.find(tLowerPDE);
  if( tItr == se2e.end() ){
    auto tMsg = this->getErrorMsgPDE(tLowerPDE);
    ANALYZE_THROWERR(tMsg)
  }
  return tItr->second;
}

bool
PhysicsEnum::
production(
  const std::string & aPhysics
) const
{
  auto tLowerPhysics = Plato::tolower(aPhysics);
  auto tItr = sp2e.find(tLowerPhysics);
  if( tItr == sp2e.end() ){
    auto tMsg = this->getErrorMsgPhysics(tLowerPhysics);
    ANALYZE_THROWERR(tMsg)
  }
  bool tIsProduction = tItr->second.second == Plato::production_t::SUPPORTED ? true : false;
  return tIsProduction;
}

std::string
PhysicsEnum::
getErrorMsgPhysics(
  const std::string & aPhysics
) const
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
) const
{
  auto tMsg = std::string("Did not find physics coupling option '") + aCoupling
    + "'. Supported physics coupling options are: ";
  for(const auto& tPair : sc2e){
    tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

std::string
PhysicsEnum::
getErrorMsgResponse(
  const std::string & aResponse
) const
{
  auto tMsg = std::string("Did not find physics response option '") + aResponse
    + "'. Supported physics response options are: ";
  for(const auto& tPair : sr2e){
    tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

std::string
PhysicsEnum::
getErrorMsgPDE(
  const std::string & aPDE
) const
{
  auto tMsg = std::string("Did not find partial differential equation option '") + aPDE
    + "'. Supported partial differentical equation options are: ";
  for(const auto& tPair : se2e){
    tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

} // namespace Plato
