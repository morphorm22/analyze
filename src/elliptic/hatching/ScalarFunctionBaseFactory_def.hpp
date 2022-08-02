#pragma once

#include "elliptic/hatching/ScalarFunctionBase.hpp"
#include "elliptic/hatching/PhysicsScalarFunction.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato and Analyze data map
     * \param [in] aProblemParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsType>
    std::shared_ptr<Plato::Elliptic::Hatching::ScalarFunctionBase> 
    ScalarFunctionBaseFactory<PhysicsType>::create(
              Plato::SpatialModel                                & aSpatialModel,
        const Plato::Sequence<typename PhysicsType::ElementType> & aSequence,
              Plato::DataMap                                     & aDataMap,
              Teuchos::ParameterList                             & aProblemParams,
              std::string                                        & aFunctionName
    ) 
    {
        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(aFunctionName);
        auto tFunctionType = tFunctionParams.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<PhysicsScalarFunction<PhysicsType>>(aSpatialModel, aSequence, aDataMap, aProblemParams, aFunctionName);
        }
        else
        {
            return nullptr;
        }
    }
} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
