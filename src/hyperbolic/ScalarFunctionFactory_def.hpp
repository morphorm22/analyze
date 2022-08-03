#pragma once

#include "hyperbolic/ScalarFunctionBase.hpp"
#include "hyperbolic/PhysicsScalarFunction.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Hyperbolic
{
    /******************************************************************************//**
     * \brief Create method
     * \param [in] aMesh mesh database
     * \param [in] aDataMap Plato Engine and Analyze data map
     * \param [in] aInputParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsType>
    std::shared_ptr<Plato::Hyperbolic::ScalarFunctionBase>
    ScalarFunctionFactory<PhysicsType>::create(
        Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aInputParams,
        std::string            & aFunctionName
    )
    {
        auto tProblemFunction = aInputParams.sublist("Criteria").sublist(aFunctionName);
        auto tFunctionType = tProblemFunction.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<Hyperbolic::PhysicsScalarFunction<PhysicsType>>(aSpatialModel, aDataMap, aInputParams, aFunctionName);
        }
        else
        {
            const std::string tErrorString = std::string("Unknown function Type '") + tFunctionType +
                            "' specified in function name " + aFunctionName + " ParameterList";
            ANALYZE_THROWERR(tErrorString)
        }
    }
} //namespace Hyperbolic

} // namespace Plato
