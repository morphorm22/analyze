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
  if(aParamList.isParameter("Thermal Expansivity")){
    this->parseScalar("Thermal Expansivity", aParamList);
  }
  if(aParamList.isParameter("Reference Temperature")){
    this->parseScalar("Reference Temperature", aParamList);
  }
  this->parseTensor("Thermal Conductivity", aParamList);

} // constructor ThermalConductionModel

}

#endif
