/*
 * SupportedEllipticProblemOptions.cpp
 *
 *  Created on: June 22, 2023
 */

#include "AnalyzeMacros.hpp"
#include "PlatoUtilities.hpp"
#include "elliptic/evaluators/problem/SupportedEllipticProblemOptions.hpp"

namespace Plato
{

namespace Elliptic
{

Plato::Elliptic::residual_t
ResidualEnum::
get(
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
ResidualEnum::
getErrorMsg(
  const std::string & aInProperty
) const
{
  auto tMsg = std::string("Did not find matching enum for input physics '")
          + aInProperty + "'. Supported physics keywords are: ";
  for(const auto& tPair : s2e)
  {
      tMsg = tMsg + "'" + tPair.first + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

} // namespace Elliptic

} // namespace Plato
