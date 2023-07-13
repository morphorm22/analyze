/*
 * FactoryElasticMaterial_def.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "materials/mechanical/MaterialIsotropicElastic.hpp"

namespace Plato
{

template<typename EvaluationType>
FactoryElasticMaterial<EvaluationType>::
FactoryElasticMaterial(
  const Teuchos::ParameterList& aParamList
) :
  mParamList(aParamList)
{}

template<typename EvaluationType>
std::shared_ptr<Plato::MaterialModel<EvaluationType>>
FactoryElasticMaterial<EvaluationType>::
create(
  std::string aModelName
) const
{
  if (!mParamList.isSublist("Material Models"))
  {
    ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
  }
  else
  {
    auto tModelsParamList = mParamList.get<Teuchos::ParameterList>("Material Models");
    if (!tModelsParamList.isSublist(aModelName))
    {
      std::stringstream tSS;
      tSS << "ERROR: Requested a material model ('" << aModelName << "') that isn't defined";
      ANALYZE_THROWERR(tSS.str());
    }
    auto tModelParamList = tModelsParamList.sublist(aModelName);
    if(tModelParamList.isSublist("Isotropic Linear Elastic"))
    {
      return ( std::make_shared<Plato::MaterialIsotropicElastic<EvaluationType>>( 
        tModelParamList.sublist("Isotropic Linear Elastic") ) 
      );
    }
    else
    {
      auto tErrMsg = this->getErrorMsg();
      ANALYZE_THROWERR(tErrMsg);
    }
  }
}

template<typename EvaluationType>
std::string
FactoryElasticMaterial<EvaluationType>::
getErrorMsg()
const
{
  std::string tMsg = std::string("ERROR: Requested material constitutive model is not supported. ")
    + "Supported material constitutive models for mechanical analyses are: ";
  for(const auto& tElement : mSupportedMaterials)
  {
    tMsg = tMsg + "'" + tElement + "', ";
  }
  auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
  return tSubMsg;
}

} // namespace Plato
