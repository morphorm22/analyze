/*
 * FactoryThermalConductionMaterial.hpp
 *
 *  Created on: July 8, 2023
 */

#include "AnalyzeMacros.hpp"
#include "MaterialModel.hpp"
#include "ThermalConductivityMaterial.hpp"

#pragma once

namespace Plato
{

/// @class FactoryThermalConductionMaterial
/// @brief factory for linear thermal constitutive models
/// @tparam EvaluationType 
template<typename EvaluationType>
class FactoryThermalConductionMaterial
{
public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  FactoryThermalConductionMaterial(
    const Teuchos::ParameterList & aParamList
  ) :
    mParamList(aParamList)
  {}
  
  /// @brief create thermal material constitutive model interface 
  /// @param [in] aModelName name of input material parameter list
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> 
  create(
    std::string aModelName
  );

private:
  /// @brief input problem parameters
  const Teuchos::ParameterList& mParamList;
};

template<typename EvaluationType>
std::shared_ptr<Plato::MaterialModel<EvaluationType>>
FactoryThermalConductionMaterial<EvaluationType>::
create(
  std::string aModelName
)
{
  if (!mParamList.isSublist("Material Models"))
  {
    REPORT("'Material Models' list not found! Returning 'nullptr'");
    return nullptr;
  }
  else
  {
    auto tModelsParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");
    if (!tModelsParamList.isSublist(aModelName))
    {
      std::stringstream ss;
      ss << "Requested a material model ('" << aModelName << "') that isn't defined";
      ANALYZE_THROWERR(ss.str());
    }
    auto tModelParamList = tModelsParamList.sublist(aModelName);
    if(tModelParamList.isSublist("Thermal Conduction"))
    {
      return ( std::make_shared<Plato::ThermalConductionModel<EvaluationType>>(
        tModelParamList.sublist("Thermal Conduction"))
      );
    }
    else
      ANALYZE_THROWERR("Expected 'Thermal Conduction' ParameterList");
  }
}

} // namespace Plato
