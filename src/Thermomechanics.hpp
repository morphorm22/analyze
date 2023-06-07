#pragma once

#include <memory>

#ifdef PLATO_PARABOLIC
  #include "parabolic/AbstractScalarFunction.hpp"
  #include "parabolic/TransientThermomechResidual.hpp"
  #include "parabolic/InternalThermoelasticEnergy.hpp"
#endif

#include "MakeFunctions.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Parabolic
{

namespace LinearThermoMechanics
{
  
struct FunctionFactory
{

#ifdef PLATO_PARABOLIC
  template <typename EvaluationType>
  std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<EvaluationType>>
  createVectorFunction(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aProblemParams,
            std::string              aPDE
  )
  {
    auto tLowerPDE = Plato::tolower(aPDE);
    if( tLowerPDE == "parabolic" )
    {
      return Plato::makeVectorFunction<EvaluationType, Plato::Parabolic::TransientThermomechResidual>
               (aSpatialDomain, aDataMap, aProblemParams, aPDE);
    }
    else
    {
      ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
    }
  }
#endif

#ifdef PLATO_PARABOLIC
  template <typename EvaluationType>
  std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>>
  createScalarFunction(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aProblemParams,
            std::string              aFuncType,
            std::string              aFuncName
  )
  {
    auto tLowerFuncType = Plato::tolower(aFuncType);
    if(tLowerFuncType == "internal thermoelastic energy")
    {
      return Plato::makeScalarFunction<EvaluationType, Plato::Parabolic::InternalThermoelasticEnergy>
          (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    }
    else
    {
      ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
    }
  }
#endif

}; // struct FunctionFactory

} // namespace LinearThermoMechanics

} // namespace Parabolic

} // namespace Plato

#include "ThermomechanicsElement.hpp"

namespace Plato
{

namespace Parabolic
{

namespace Linear
{
  
/// @brief concrete class use to define parabolic thermomechanics physics
/// @tparam TopoElementType topological element typename 
template<typename TopoElementType>
class Thermomechanics
{
public:
  /// @brief residual and criteria factory for parabolic thermomechanics physics
  typedef Plato::Parabolic::LinearThermoMechanics::FunctionFactory FunctionFactory;
  /// @brief topological element type with additional physics related information 
  using ElementType = ThermomechanicsElement<TopoElementType>;
};

} // namespace Linear

} // namespace Parabolic

} // namespace Plato
