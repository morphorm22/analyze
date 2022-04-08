#pragma once

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"
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
              std::string              aStrScalarFunctionType,
              std::string              aStrScalarFunctionName
    )
    {
        auto tLowerScalarFunc = Plato::tolower(aStrScalarFunctionType);
        if( tLowerScalarFunc == "volume" )
        {
            auto penaltyParams = aParamList.sublist("Criteria").sublist(aStrScalarFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = penaltyParams.get<std::string>("Type");
            auto tLowerPenaltyType = Plato::tolower(tPenaltyType);
            if( tLowerPenaltyType == "simp" )
            {
                return std::make_shared<Plato::Geometric::Volume<EvaluationType, Plato::MSIMP>>
                   (aSpatialDomain, aDataMap, aParamList, penaltyParams, aStrScalarFunctionName);
            }
            else
            if( tLowerPenaltyType == "ramp" )
            {
                return std::make_shared<Plato::Geometric::Volume<EvaluationType, Plato::RAMP>>
                   (aSpatialDomain, aDataMap, aParamList, penaltyParams, aStrScalarFunctionName);
            }
            else
            if( tLowerPenaltyType == "heaviside" )
            {
                return std::make_shared<Plato::Geometric::Volume<EvaluationType, Plato::Heaviside>>
                   (aSpatialDomain, aDataMap, aParamList, penaltyParams, aStrScalarFunctionName);
            }
            else
            if( tLowerPenaltyType == "nopenalty" )
            {
                return std::make_shared<Plato::Geometric::Volume<EvaluationType, Plato::NoPenalty>>
                   (aSpatialDomain, aDataMap, aParamList, penaltyParams, aStrScalarFunctionName);
            }
            else
            {
                ANALYZE_THROWERR(std::string("Unknown 'Penalty Function' of type '") + tLowerPenaltyType + "' specified in ParameterList");
            }
// TODO        } else
// TODO        if( tLowerScalarFunc == "geometry misfit" )
// TODO        {
// TODO            return std::make_shared<Plato::Geometric::GeometryMisfit<EvaluationType>>
// TODO               (aSpatialDomain, aDataMap, aParamList, aStrScalarFunctionName);
        }
        else
        {
            ANALYZE_THROWERR(std::string("Unknown 'Objective' of type '") + tLowerScalarFunc + "' specified in 'Plato Problem' ParameterList");
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
