/*
 * NitscheEvaluator.cpp
 *
 *  Created on: July 13, 2023
 */

#include "AnalyzeMacros.hpp"
#include "bcs/dirichlet/nitsche/NitscheEvaluator.hpp"

namespace Plato
{

NitscheEvaluator::
NitscheEvaluator(
  Teuchos::ParameterList & aParamList
)
{
  this->initialize(aParamList);
}

NitscheEvaluator::
~NitscheEvaluator()
{}

std::string 
NitscheEvaluator::
sideset() 
const 
{ return mSideSetName; }

std::string 
NitscheEvaluator::
material() 
const 
{ return mMaterialName; }

void 
NitscheEvaluator::
initialize(
  Teuchos::ParameterList & aParamList
)
{
  if( !aParamList.isParameter("Sides") ){
    ANALYZE_THROWERR( std::string("ERROR: Input argument ('Sides') is not defined, ") + 
      "side set for Nitsche's method cannot be determined" )
  }
  mSideSetName = aParamList.get<std::string>("Sides");
  
  if( !aParamList.isParameter("Material Model") ){
    ANALYZE_THROWERR( std::string("ERROR: Input argument ('Material Model') is not defined, ") + 
      "material constitutive model for Nitsche's method cannot be determined" )
  }
  mMaterialName = aParamList.get<std::string>("Material Model");
}

} // namespace Plato
