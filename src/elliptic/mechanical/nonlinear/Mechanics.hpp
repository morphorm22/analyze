/*
 *  NonlinearMechanics.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

/// @include standard cpp includes
#include <memory>
/// @include analyze includes
#include "MechanicsElement.hpp"
// mechanics related
#include "elliptic/mechanical/SupportedParamOptions.hpp"
// residuals related
#include "elliptic/mechanical/nonlinear/ResidualElastostaticTotalLagrangian.hpp"
// criteria related
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
  /// @brief 
  /// @tparam EvaluationType 
  /// @param aSpatialDomain 
  /// @param aDataMap 
  /// @param aParamList 
  /// @param aTypePDE 
  /// @return 
  template<typename EvaluationType>
  std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
  createVectorFunction(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap, 
            Teuchos::ParameterList & aParamList,
            std::string              aTypePDE)
  {
      auto tLowerPDE = Plato::tolower(aTypePDE);
      if(tLowerPDE == "elliptic")
      {
        auto tResidual = std::make_shared<Plato::ResidualElastostaticTotalLagrangian<EvaluationType>>
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

  /// @brief 
  /// @tparam EvaluationType 
  /// @param aSpatialDomain 
  /// @param aDataMap 
  /// @param aParamList 
  /// @param aFuncType 
  /// @param aFuncName 
  /// @return 
  template<typename EvaluationType>
  std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
  createScalarFunction(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap, 
            Teuchos::ParameterList & aParamList,
            std::string              aFuncType,
            std::string              aFuncName
  )
  {
    Plato::mechanical::CriterionEnum tSupportedCriterion;
    auto tCriterion = tSupportedCriterion.get(aFuncType);
    switch (tCriterion)
    {
      case Plato::mechanical::criterion::KIRCHHOFF_ENERGY_POTENTIAL:
        return ( std::make_shared<Plato::CriterionKirchhoffEnergyPotential<EvaluationType>>(
          aSpatialDomain, aDataMap, aParamList, aFuncName) );
        break;
      case Plato::mechanical::criterion::NEO_HOOKEAN_ENERGY_POTENTIAL:
        return ( std::make_shared<Plato::CriterionNeoHookeanEnergyPotential<EvaluationType>>(
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

namespace Plato
{

namespace Elliptic
{

namespace Nonlinear
{

/// @brief concrete class use to define elliptic nonlinear mechanical physics
/// @tparam TopoElementType topological element typename
template<typename TopoElementType>
class Mechanics
{
public:
  /// @brief residual and criteria factory for elliptic linear mechanical physics
  typedef Plato::Elliptic::NonlinearMechanics::FunctionFactory FunctionFactory;
  /// @brief physics-based topological element typename
  using ElementType = MechanicsElement<TopoElementType>;
};

} // namespace Nonlinear

} // namespace Elliptic

} // namespace Plato