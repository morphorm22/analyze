#pragma once

#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/VolumeAverageCriterion.hpp"
#include "elliptic/DivisionFunction.hpp"
#include "elliptic/SolutionFunction.hpp"
#include "elliptic/WeightedSumFunction.hpp"
#include "elliptic/LeastSquaresFunction.hpp"

#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/MassPropertiesFunction.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato and Analyze data map
     * \param [in] aProblemParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsType>
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> 
    ScalarFunctionBaseFactory<PhysicsType>::create(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aFunctionName
    ) 
    {
        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(aFunctionName);
        auto tFunctionType = tFunctionParams.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Mass Properties")
        {
            return std::make_shared<MassPropertiesFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Least Squares")
        {
            return std::make_shared<LeastSquaresFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Weighted Sum")
        {
            return std::make_shared<WeightedSumFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Solution")
        {
            return std::make_shared<SolutionFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Division")
        {
            return std::make_shared<DivisionFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Volume Average Criterion")
        {
            return std::make_shared<VolumeAverageCriterion<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<PhysicsScalarFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        {
            return nullptr;
        }
        return nullptr;
    }

} // namespace Elliptic

} // namespace Plato
