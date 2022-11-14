#pragma once

#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/GeometryScalarFunction.hpp"
#include "geometric/WeightedSumFunction.hpp"
#include "geometric/DivisionFunction.hpp"
#include "geometric/LeastSquaresFunction.hpp"
#include "geometric/MassPropertiesFunction.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Geometric
{

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aProblemParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsType>
    std::shared_ptr<Plato::Geometric::ScalarFunctionBase> 
    ScalarFunctionBaseFactory<PhysicsType>::create(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aFunctionName
    )
    {
        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(aFunctionName);
        auto tFunctionType = tFunctionParams.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Weighted Sum")
        {
            return std::make_shared<WeightedSumFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Division")
        {
            return std::make_shared<DivisionFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Least Squares")
        {
            return std::make_shared<LeastSquaresFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Mass Properties")
        {
            return std::make_shared<MassPropertiesFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<GeometryScalarFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        {
            // don't throw an exception.  The calling function must decide if this is a fatal error.
            return nullptr;
        }
    }

} // namespace Geometric

} // namespace Plato
