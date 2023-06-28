/*
 * SupportedParamOptions.cpp
 *
 *  Created on: June 27, 2023
 */

#include "AnalyzeMacros.hpp"
#include "PlatoUtilities.hpp"
#include "elliptic/thermomechanics/SupportedParamOptions.hpp"

namespace Plato
{

namespace Elliptic
{

namespace thermomechanical
{

Plato::Elliptic::thermomechanical::residual 
ResidualEnum::
get(
  const std::string & aResponse
) 
const
{
  auto tLowerResponse = Plato::tolower(aResponse);
  auto tItrResponse = s2e.find(tLowerResponse);
  if( tItrResponse == s2e.end() ){
    auto tMsg = this->getErrorMsg(tLowerResponse);
    ANALYZE_THROWERR(tMsg)
  }
  return tItrResponse->second;
}

std::string
ResidualEnum::
getErrorMsg(
  const std::string & aResponse
)
const
{
  auto tMsg = std::string("Did not find response '") + aResponse 
    + "'. Supported response options are: ";
  for(const auto& tPair : s2e){
    tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

} // namespace thermomechanical

} // namespace Elliptic

} // namespace Plato
