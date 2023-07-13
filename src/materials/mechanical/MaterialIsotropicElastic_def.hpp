/*
 * MaterialIsotropicElastic_def.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"

namespace Plato
{

template<typename EvaluationType>
MaterialIsotropicElastic<EvaluationType>::
MaterialIsotropicElastic(
  const Teuchos::ParameterList& aParamList
)
{
  this->parse(aParamList);
  this->computeLameConstants();
}

template<typename EvaluationType>
Plato::Scalar 
MaterialIsotropicElastic<EvaluationType>::
mu() 
{ 
  return this->getScalarConstant("mu"); 
}

template<typename EvaluationType>
void 
MaterialIsotropicElastic<EvaluationType>::
mu(
  const Plato::Scalar & aValue
)
{ 
  this->setScalarConstant("mu",aValue); 
}

template<typename EvaluationType>
Plato::Scalar 
MaterialIsotropicElastic<EvaluationType>::
lambda()
{ 
  return this->getScalarConstant("lambda"); 
}

template<typename EvaluationType>
void 
MaterialIsotropicElastic<EvaluationType>::
lambda(
  const Plato::Scalar & aValue
)
{ 
  this->setScalarConstant("lambda",aValue); 
}

template<typename EvaluationType>
void 
MaterialIsotropicElastic<EvaluationType>::
parse(
  const Teuchos::ParameterList& aParamList
)
{
  this->parseScalarConstant("Youngs Modulus", aParamList);
  this->parseScalarConstant("Poissons Ratio", aParamList);
}

template<typename EvaluationType>
void 
MaterialIsotropicElastic<EvaluationType>::
computeLameConstants()
{
  auto tYoungsModulus = this->getScalarConstant("youngs modulus");
  if(tYoungsModulus <= std::numeric_limits<Plato::Scalar>::epsilon())
  {
    ANALYZE_THROWERR(std::string("ERROR: The Young's Modulus is less than the machine epsilon. ")
      + "The input material properties were not parsed properly.");
  }
  auto tPoissonsRatio = this->getScalarConstant("poissons ratio");
  if(tPoissonsRatio <= std::numeric_limits<Plato::Scalar>::epsilon())
  {
    ANALYZE_THROWERR(std::string("ERROR: The Poisson's Ratio is less than the machine epsilon. ")
      + "The input material properties were not parsed properly.");
  }
  auto tMu = tYoungsModulus / (2.0 * (1.0 + tPoissonsRatio) );
  this->setScalarConstant("mu",tMu);
  auto tLambda = (tYoungsModulus * tPoissonsRatio) / ( (1.0 + tPoissonsRatio) * (1.0 - 2.0 * tPoissonsRatio) );
  this->setScalarConstant("lambda",tLambda);
}

} // namespace Plato
