#pragma once

#include "AnalyzeMacros.hpp"

#include "elliptic/evaluators/criterion/CriterionEvaluatorDivision.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorWeightedSum.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorLeastSquares.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorVolumeAverage.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorScalarFunction.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorSolutionFunction.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorMassProperties.hpp"

namespace Plato
{

namespace Elliptic
{

/// @brief create criterion evaluators
/// @tparam PhysicsType physics typename
/// \param [in] aSpatialModel  contains mesh and model information
/// \param [in] aDataMap       output database 
/// \param [in] aProblemParams input problem parameters
/// \param [in] aFunctionName  name of function in parameter list
/// @return shared pointer to criterion evaluators
template <typename PhysicsType>
std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase> 
FactoryCriterionEvaluator<PhysicsType>::create(
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
    return std::make_shared<CriterionEvaluatorMassProperties<PhysicsType>>(
        aSpatialModel, aDataMap, aProblemParams, aFunctionName);
  }
  else
  if(tFunctionType == "Least Squares")
  {
    return std::make_shared<CriterionEvaluatorLeastSquares<PhysicsType>>(
        aSpatialModel, aDataMap, aProblemParams, aFunctionName);
  }
  else
  if(tFunctionType == "Weighted Sum")
  {
    return std::make_shared<CriterionEvaluatorWeightedSum<PhysicsType>>(
        aSpatialModel, aDataMap, aProblemParams, aFunctionName);
  }
  else
  if(tFunctionType == "Solution")
  {
    return std::make_shared<CriterionEvaluatorSolutionFunction<PhysicsType>>(
        aSpatialModel, aDataMap, aProblemParams, aFunctionName);
  }
  else
  if(tFunctionType == "Division")
  {
    return std::make_shared<CriterionEvaluatorDivision<PhysicsType>>(
        aSpatialModel, aDataMap, aProblemParams, aFunctionName);
  }
  else
  if(tFunctionType == "Volume Average Criterion")
  {
    return std::make_shared<CriterionEvaluatorVolumeAverage<PhysicsType>>(
        aSpatialModel, aDataMap, aProblemParams, aFunctionName);
  }
  else
  if(tFunctionType == "Scalar Function")
  {
    return std::make_shared<CriterionEvaluatorScalarFunction<PhysicsType>>(
        aSpatialModel, aDataMap, aProblemParams, aFunctionName);
  }
  else
  {
    return nullptr;
  }
  return nullptr;
}

} // namespace Elliptic

} // namespace Plato
