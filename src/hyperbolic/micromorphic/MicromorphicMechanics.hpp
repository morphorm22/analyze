#pragma once

#include "SpatialModel.hpp"

#include "hyperbolic/micromorphic/MicromorphicMechanicsElement.hpp"
#include "hyperbolic/micromorphic/RelaxedMicromorphicResidual.hpp"

namespace Plato
{

namespace Hyperbolic
{

struct MicromorphicFunctionFactory
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
            return Plato::makeVectorFunction<EvaluationType, Plato::Hyperbolic::RelaxedMicromorphicResidual>
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
              Plato::DataMap&          aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string              strScalarFunctionType,
              std::string              strScalarFunctionName
    )
    {
        ANALYZE_THROWERR("No criteria supported for micromorphic mechanics currently.")
    }
};

/******************************************************************************//**
 * \brief Concrete class for use as the Physics template argument in
 *        Plato::Hyperbolic::Problem
**********************************************************************************/
template<typename TopoElementType>
class MicromorphicMechanics
{
public:
    typedef Plato::Hyperbolic::MicromorphicFunctionFactory FunctionFactory;
    using ElementType = MicromorphicMechanicsElement<TopoElementType>;
};

}

} 

