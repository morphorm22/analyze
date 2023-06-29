/*
 * ThermoMechanics.hpp
 *
 *  Created on: June 27, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "base/ResidualBase.hpp"
#include "base/CriterionBase.hpp"

#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/thermal/Thermal.hpp"
#include "elliptic/thermomechanics/SupportedParamOptions.hpp"
#include "elliptic/thermomechanics/nonlinear/ResidualThermoElastoStaticTotalLagrangian.hpp"

namespace Plato
{

namespace Elliptic
{

namespace NonlinearThermoMechanics
{

struct FunctionFactory
{
  template<typename EvaluationType>
  std::shared_ptr<Plato::ResidualBase>
  createVectorFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList,
          std::string              aTypePDE
  )
  {  
    if( !aParamList.sublist(aTypePDE).isSublist("Mechanical Residual") ){ 
      ANALYZE_THROWERR("ERROR: 'Mechanical Residual' parameter list not found!"); 
    }
    else{
      auto tMechResidualParamList = aParamList.sublist(aTypePDE).sublist("Mechanical Residual");
      auto tResponse = tMechResidualParamList.get<std::string>("Response","linear");
      Plato::Elliptic::thermomechanical::ResidualEnum tSupportedResidual;
      auto tResidual = tSupportedResidual.get(tResponse);
      switch (tResidual)
      {
      case Plato::Elliptic::thermomechanical::residual::NONLINEAR_THERMO_MECHANICS:
        return 
          (std::make_shared<Plato::Elliptic::ResidualThermoElastoStaticTotalLagrangian<EvaluationType>>(
            aSpatialDomain, aDataMap, aParamList
          ));
        break;
      case Plato::Elliptic::thermomechanical::residual::LINEAR_THERMO_MECHANICS:
      default:
        ANALYZE_THROWERR("ERROR: Requested 'Mechanical Residual' is not supported!"); 
        break;
      }
    }
  }

  template<typename EvaluationType>
  std::shared_ptr<Plato::CriterionBase>
  createScalarFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap, 
          Teuchos::ParameterList & aProblemParams, 
          std::string              aFuncType,
          std::string              aFuncName
  )
  {  
    ANALYZE_THROWERR("ERROR: Requested 'Scalar Function' is not supported");
  }
};

} // namespace NonlinearThermoMechanics

} // namespace Elliptic

} // namespace Plato

#include "element/ThermoElasticElement.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Nonlinear
{

/// @brief concrete class use to define elliptic thermomechanics physics
/// @tparam TopoElementType topological element typename 
template<typename TopoElementType>
class ThermoMechanics : public Plato::ThermoElasticElement<TopoElementType>
{
public:
  /// @brief residual and criteria factory for elliptic thermomechanics physics
  typedef Plato::Elliptic::NonlinearThermoMechanics::FunctionFactory FunctionFactory;
  /// @brief topological element type with additional physics related information 
  using ElementType = ThermoElasticElement<TopoElementType>;
  /// @brief typename for possible extra physics in the analysis; e.g., coupled physics analysis
  using OtherPhysics = Plato::Elliptic::Linear::Thermal<TopoElementType>;
};

} // namespace Nonlinear

} // namespace Elliptic

} // namespace Plato

