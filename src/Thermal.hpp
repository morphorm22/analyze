#ifndef PLATO_THERMAL_HPP
#define PLATO_THERMAL_HPP

#include "parabolic/AbstractVectorFunction.hpp"
#include "parabolic/AbstractScalarFunction.hpp"

#ifdef PLATO_PARABOLIC
  #include "parabolic/HeatEquationResidual.hpp"
  #include "parabolic/InternalThermalEnergy.hpp"
  #include "parabolic/TemperatureAverage.hpp"
#endif

#include "MakeFunctions.hpp"

namespace Plato 
{

namespace Parabolic 
{

namespace LinearThermal
{
  
struct FunctionFactory
{
  template <typename EvaluationType>
  std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<EvaluationType>>
  createVectorFunction(
      const Plato::SpatialDomain   & aSpatialDomain,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aProblemParams,
            std::string              aPDE
  )
  {
#ifdef PLATO_PARABOLIC
    auto tLowerPDE = Plato::tolower(aPDE);
    if(tLowerPDE == "parabolic")
    {
      return Plato::makeVectorFunction<EvaluationType, Plato::Parabolic::HeatEquationResidual>(
        aSpatialDomain, aDataMap, aProblemParams, aPDE);
    }
    else
    {
      ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
    }
#else
      ANALYZE_THROWERR("Plato Analyze was not compiled with parabolic physics.");
#endif
  }

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
#ifdef PLATO_PARABOLIC
    auto tLowerFuncType = Plato::tolower(aFuncType);
    if(tLowerFuncType == "internal thermal energy")
    {
      return Plato::makeScalarFunction<EvaluationType, Plato::Parabolic::InternalThermalEnergy>
          (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    }
    else
    if( tLowerFuncType == "temperature average" )
    {
      return Plato::makeScalarFunction<EvaluationType, Plato::Parabolic::TemperatureAverage>
          (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
    }
    else
    {
      ANALYZE_THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
    }
#else
      ANALYZE_THROWERR("Plato Analyze was not compiled with parabolic physics.");
#endif
  }
};

} // namespace LinearThermal

} // namespace Parabolic

} // namespace Plato

#include "ThermalElement.hpp"

namespace Plato
{

namespace Parabolic
{

namespace Linear
{
  
/// @brief concrete class use to define parabolic thermal physics
/// @tparam TopoElementType topological element typename 
template <typename TopoElementType>
class Thermal
{
public:
  /// @brief residual and criteria factory for parabolic thermal physics
  typedef Plato::Parabolic::LinearThermal::FunctionFactory FunctionFactory;
  /// @brief topological element type with additional physics related information 
  using ElementType = ThermalElement<TopoElementType>;
};

} // namespace Linear

} // namespace Parabolic

} //namespace Plato

#endif
