#pragma once

#include "SpatialModel.hpp"
#include "MechanicsElement.hpp"

#include "MakeFunctions.hpp"

#include "hyperbolic/ElastomechanicsResidual.hpp"
#include "hyperbolic/AbstractScalarFunction.hpp"
#include "hyperbolic/InternalElasticEnergy.hpp"
#include "hyperbolic/StressPNorm.hpp"

namespace Plato
{

namespace Hyperbolic
{

struct FunctionFactory
{
    template <typename EvaluationType>
    std::shared_ptr<::Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string              aPDE
    )
    {
        auto tLowerPDE = Plato::tolower(aPDE);
        if(tLowerPDE == "hyperbolic")
        {
            return Plato::makeVectorFunction<EvaluationType, Plato::Hyperbolic::TransientMechanicsResidual>
                     (aSpatialDomain, aDataMap, aProblemParams, aPDE);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    template <typename EvaluationType>
    std::shared_ptr<::Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string              aFuncType,
              std::string              aFuncName
    )
    {
        auto tLowerFuncType = Plato::tolower(aFuncType);

        if(tLowerFuncType == "internal elastic energy")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Hyperbolic::InternalElasticEnergy>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if(tLowerFuncType == "stress p-norm")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Hyperbolic::StressPNorm>
                (aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        {
          const std::string tErrorString = std::string("Function '") + tLowerFuncType + "' not implemented yet in transient mechanics.";
          ANALYZE_THROWERR(tErrorString)
        }
    }
};

/******************************************************************************//**
 * \brief Concrete class for use as the Physics template argument in
 *        Plato::Hyperbolic::Problem
**********************************************************************************/
template<typename TopoElementType>
class Mechanics
{
public:
    typedef Plato::Hyperbolic::FunctionFactory FunctionFactory;
    using ElementType = MechanicsElement<TopoElementType>;
};
} 

} 
