#ifndef LINEARTHERMALMATERIAL_HPP
#define LINEARTHERMALMATERIAL_HPP

#include <Teuchos_ParameterList.hpp>
#include "MaterialModel.hpp"

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/// @class ThermalConductionModel
/// @brief base class for linear thermally conductive material constitutive models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class ThermalConductionModel : public MaterialModel<EvaluationType>
{
public:
  /// @brief class constructor
  /// @param [in] aParamList input material parameter list
  ThermalConductionModel(
    const Teuchos::ParameterList & aParamList
  );
}; // class ThermalConductionModel

template<typename EvaluationType>
ThermalConductionModel<EvaluationType>::
ThermalConductionModel(
  const Teuchos::ParameterList& aParamList
) : 
  MaterialModel<EvaluationType>(aParamList)
{
 
  this->parseScalar("Thermal Expansivity", aParamList);
  this->parseScalar("Reference Temperature", aParamList);
  this->parseTensor("Thermal Conductivity", aParamList);

} // constructor ThermalConductionModel

/// @class ThermalConductionModelFactory
/// @brief factory for linear thermal constitutive models
/// @tparam EvaluationType 
template<typename EvaluationType>
class ThermalConductionModelFactory
{
public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  ThermalConductionModelFactory(
    const Teuchos::ParameterList & aParamList
  ) :
    mParamList(aParamList)
  {}
  
  /// @brief create thermal material constitutive model interface 
  /// @param [in] aModelName name of input material parameter list
  Teuchos::RCP<MaterialModel<EvaluationType>> 
  create(
    std::string aModelName
  );

private:
  /// @brief input problem parameters
  const Teuchos::ParameterList& mParamList;
};

template<typename EvaluationType>
Teuchos::RCP<MaterialModel<EvaluationType>>
ThermalConductionModelFactory<EvaluationType>::
create(
  std::string aModelName
)
{
  if (!mParamList.isSublist("Material Models"))
  {
    REPORT("'Material Models' list not found! Returning 'nullptr'");
    return Teuchos::RCP<Plato::MaterialModel<EvaluationType>>(nullptr);
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
      return Teuchos::rcp(new ThermalConductionModel<EvaluationType>(
          tModelParamList.sublist("Thermal Conduction")));
    }
    else
      ANALYZE_THROWERR("Expected 'Thermal Conduction' ParameterList");
  }
}

}

#endif
