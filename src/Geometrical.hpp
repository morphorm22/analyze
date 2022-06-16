#pragma once

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"
#include "MakeFunctions.hpp"
#include "PlatoUtilities.hpp"

#include "geometric/Volume.hpp"
#include "geometric/GeometryMisfit.hpp"
#include "geometric/AbstractScalarFunction.hpp"

namespace Plato {

namespace GeometryFactory {
/******************************************************************************/
struct FunctionFactory{
/******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Geometric::AbstractScalarFunction<EvaluationType>>
    createScalarFunction( 
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string              aFuncType,
              std::string              aFuncName
    )
    {
        auto tLowerFuncType = Plato::tolower(aFuncType);
        if( tLowerFuncType == "volume" )
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Geometric::Volume>
                (aSpatialDomain, aDataMap, aParamList, aFuncName);
        }
// TODO    if( tLowerFuncType == "geometry misfit" )
// TODO        {
// TODO            return std::make_shared<Plato::Geometric::GeometryMisfit<EvaluationType>>
// TODO               (aSpatialDomain, aDataMap, aParamList, aStrScalarFunctionName);
        else
        {
            ANALYZE_THROWERR(std::string("Unknown 'Objective' of type '") + tLowerFuncType + "' specified in 'Plato Problem' ParameterList");
        }
    }
};

} // namespace GeometryFactory

} // namespace Plato

#include "geometric/GeometricalElement.hpp"

namespace Plato
{
template <typename TopoElementType>
class Geometrical
{
  public:
    typedef Plato::GeometryFactory::FunctionFactory FunctionFactory;
    using ElementType = GeometricalElement<TopoElementType>;
};
// class Geometrical

} //namespace Plato
