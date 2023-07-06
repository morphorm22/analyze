/*
 * SupportedParamOptions.cpp
 *
 *  Created on: July 5, 2023
 */

#include "AnalyzeMacros.hpp"
#include "PlatoUtilities.hpp"

#include "bcs/neumann/SupportedParamOptions.hpp"

namespace Plato
{

Plato::neumann_bc 
NeumannEnum::
bc(
  const std::string & aNBC
) const
{
  auto tLowerNBC = Plato::tolower(aNBC);
  auto tItr = s2e.find(tLowerNBC);
  if( tItr == s2e.end() ){
    auto tMsg = this->getErrorMsg(tLowerNBC);
    ANALYZE_THROWERR(tMsg)
  }
  return tItr->second;
}

std::string
NeumannEnum::
getErrorMsg(
  const std::string & aNBC
) const
{
  auto tMsg = std::string("Did not find input Neumann boundary condition '") + aNBC
    + "'. Supported Neumann boundary conditions are: ";
  for(const auto& tPair : s2e){
    tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

} // namespace Plato
