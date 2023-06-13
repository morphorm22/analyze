#pragma once

#include "base/ResidualBase.hpp"
#include "base/CriterionBase.hpp"

#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/thermal/ResidualThermostatic.hpp"
#include "elliptic/thermal/CriterionInternalThermalEnergy.hpp"
#include "elliptic/thermal/CriterionFluxPNorm.hpp"

#include "MakeFunctions.hpp"

namespace Plato 
{

namespace Elliptic 
{

namespace LinearThermal
{
  
struct FunctionFactory
{
  template <typename EvaluationType>
  std::shared_ptr<Plato::ResidualBase>
  createVectorFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string              aPDE
  )
  {  
    auto tLowerPDE = Plato::tolower(aPDE);
    if(tLowerPDE == "elliptic")
    {
      return Plato::makeVectorFunction<EvaluationType, Plato::Elliptic::ResidualThermostatic>
               (aSpatialDomain, aDataMap, aProblemParams, aPDE);
    }
    else
    {
      ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
    }
  }

  template <typename EvaluationType>
  std::shared_ptr<Plato::CriterionBase>
  createScalarFunction( 
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string              aFuncType,
          std::string              aFuncName
  )
  {
    auto tLowerFuncType = Plato::tolower(aFuncType);
    if(tLowerFuncType == "internal thermal energy")
    {
      return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::CriterionInternalThermalEnergy>
          (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    } 
    else
    if( tLowerFuncType == "flux p-norm" )
    {
      return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::CriterionFluxPNorm>
          (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    }
    else
    {
      ANALYZE_THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
    }
  }
};

} // namespace LinearThermal

} // namespace Elliptic

} // namespace Plato

#include "ThermalElement.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Linear
{
  
/// @brief concrete class use to define elliptic thermal physics
/// @tparam TopoElementType topological element typename 
template <typename TopoElementType>
class Thermal : public Plato::ThermalElement<TopoElementType>
{
public:
  /// @brief residual and criteria factory for elliptic thermal physics
  typedef Plato::Elliptic::LinearThermal::FunctionFactory FunctionFactory;
  /// @brief topological element type with additional physics related information 
  using ElementType = ThermalElement<TopoElementType>;
};

} // namespace Linear

} // namespace Elliptic

} //namespace Plato
