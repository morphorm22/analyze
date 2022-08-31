#ifndef PLATO_THERMAL_HPP
#define PLATO_THERMAL_HPP

#include "parabolic/AbstractVectorFunction.hpp"
#include "parabolic/AbstractScalarFunction.hpp"

#ifdef PLATO_PARABOLIC
  #include "parabolic/HeatEquationResidual.hpp"
  #include "parabolic/InternalThermalEnergy.hpp"
  #include "parabolic/TemperatureAverage.hpp"
#endif

#include "elliptic/AbstractVectorFunction.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/ThermostaticResidual.hpp"
#include "elliptic/InternalThermalEnergy.hpp"
#include "elliptic/FluxPNorm.hpp"

#include "MakeFunctions.hpp"

namespace Plato {

namespace ThermalFactory {
/******************************************************************************/
struct FunctionFactory{
/******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
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
            return Plato::makeVectorFunction<EvaluationType, Plato::Elliptic::ThermostaticResidual>
                     (aSpatialDomain, aDataMap, aProblemParams, aPDE);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<EvaluationType>>
    createVectorFunctionParabolic(
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
            return Plato::makeVectorFunction<EvaluationType, Plato::Parabolic::HeatEquationResidual>
                     (aSpatialDomain, aDataMap, aProblemParams, aPDE);
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
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
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
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::InternalThermalEnergy>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        } else
        if( tLowerFuncType == "flux p-norm" )
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::FluxPNorm>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }

    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>>
    createScalarFunctionParabolic( 
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

} // namespace ThermalFactory

} // namespace Plato

#include "ThermalElement.hpp"

namespace Plato
{
/******************************************************************************//**
 * \brief Concrete class for use as the Physics template argument in
 *        Plato::Elliptic::Problem
**********************************************************************************/
template <typename TopoElementType>
class Thermal
{
public:
    typedef Plato::ThermalFactory::FunctionFactory FunctionFactory;
    using ElementType = ThermalElement<TopoElementType>;
};
} //namespace Plato

#endif
