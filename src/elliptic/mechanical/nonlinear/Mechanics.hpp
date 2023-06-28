/*
 *  NonlinearMechanics.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

/// @include standard cpp includes
#include <memory>

// mechanics related
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/mechanical/SupportedParamOptions.hpp"
// residuals related
#include "base/ResidualBase.hpp"
#include "elliptic/mechanical/nonlinear/ResidualElastostaticTotalLagrangian.hpp"
// criteria related
#include "base/CriterionBase.hpp"
#include "elliptic/mechanical/nonlinear/CriterionKirchhoffEnergyPotential.hpp"
#include "elliptic/mechanical/nonlinear/CriterionNeoHookeanEnergyPotential.hpp"

namespace Plato
{

namespace Elliptic
{

namespace NonlinearMechanics
{
  
struct FunctionFactory
{
  /// @fn createVectorFunction
  /// @brief create elliptic vector function
  /// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
  /// @param [in] aSpatialDomain contains mech and model information
  /// @param [in] aDataMap       output database
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aTypePDE       partial differential equation type
  /// @return shared pointer to residual base class
  template<typename EvaluationType>
  std::shared_ptr<Plato::ResidualBase>
  createVectorFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap, 
          Teuchos::ParameterList & aParamList,
          std::string              aTypePDE)
  {
    auto tLowerPDE = Plato::tolower(aTypePDE);
    if(tLowerPDE == "elliptic")
    {
      auto tResidual = std::make_shared<Plato::Elliptic::ResidualElastostaticTotalLagrangian<EvaluationType>>
                        (aSpatialDomain,aDataMap, aParamList,aTypePDE);
      return tResidual;
    }
    else
    {
      auto tMsg = std::string("Invalid input parameter argument: Requested ('PDE Constraint') is not supported") 
        + "Supported ('PDE constraint') options for nonlinear mechanical problems are: 'elliptic'";
        ANALYZE_THROWERR(tMsg)
    }
  }

  /// @fn createScalarFunction
  /// @brief create elliptic scalar function
  /// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
  /// @param [in] aSpatialDomain contains mech and model information
  /// @param [in] aDataMap       output database
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aFuncType      scalar function type
  /// @param [in] aFuncName      scalar function name
  /// @return shared pointer to criterion base class
  template<typename EvaluationType>
  std::shared_ptr<Plato::CriterionBase>
  createScalarFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap, 
          Teuchos::ParameterList & aParamList,
          std::string              aFuncType,
          std::string              aFuncName
  )
  {
    Plato::Elliptic::mechanical::CriterionEnum tSupportedCriterion;
    auto tCriterion = tSupportedCriterion.get(aFuncType);
    switch (tCriterion)
    {
      case Plato::Elliptic::mechanical::criterion::KIRCHHOFF_ENERGY_POTENTIAL:
        return ( std::make_shared<Plato::Elliptic::CriterionKirchhoffEnergyPotential<EvaluationType>>(
          aSpatialDomain, aDataMap, aParamList, aFuncName) );
        break;
      case Plato::Elliptic::mechanical::criterion::NEO_HOOKEAN_ENERGY_POTENTIAL:
        return ( std::make_shared<Plato::Elliptic::CriterionNeoHookeanEnergyPotential<EvaluationType>>(
          aSpatialDomain, aDataMap, aParamList, aFuncName) );
        break;  
      default:
        ANALYZE_THROWERR("Error while constructing criterion in scalar function factory");
        return nullptr;
        break;
    }
  }
};

} // namespace NonlinearMechanics

} // namespace Elliptic

} // Plato

/// @include analyze includes
#include "MechanicsElement.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Nonlinear
{

/// @brief concrete class use to define elliptic nonlinear mechanical physics
/// @tparam TopoElementType topological element typename
template<typename TopoElementType>
class Mechanics : public Plato::MechanicsElement<TopoElementType>
{
public:
  /// @brief residual and criteria factory for elliptic linear mechanical physics
  typedef Plato::Elliptic::NonlinearMechanics::FunctionFactory FunctionFactory;
  /// @brief physics-based topological element typename
  using ElementType = Plato::MechanicsElement<TopoElementType>;
  /// @brief typename for possible extra physics in the analysis; e.g., coupled physics analysis
  using OtherPhysics = Plato::Elliptic::Nonlinear::Mechanics<TopoElementType>;
};

} // namespace Nonlinear

} // namespace Elliptic

} // namespace Plato