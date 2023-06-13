#pragma once

#include "AnalyzeMacros.hpp"

#include "elliptic/criterioneval/CriterionEvaluatorDivision.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorWeightedSum.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorLeastSquares.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorVolumeAverage.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorScalarFunction.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorSolutionFunction.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorMassProperties.hpp"

namespace Plato
{

namespace Elliptic
{

/// @brief create criterion evaluator
/// @tparam PhysicsType physics typename
/// \param [in] aSpatialModel  contains mesh and model information
/// \param [in] aDataMap       output database 
/// \param [in] aProblemParams input problem parameters
/// \param [in] aFunctionName  name of function in parameter list
/// @return shared pointer to criterion evaluator
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
