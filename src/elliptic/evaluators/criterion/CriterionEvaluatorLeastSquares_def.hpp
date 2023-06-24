#pragma once

#include "BLAS1.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/evaluators/criterion/FactoryCriterionEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
void CriterionEvaluatorLeastSquares<PhysicsType>::
initialize (
  Teuchos::ParameterList & aProblemParams
)
{
  Plato::Elliptic::FactoryCriterionEvaluator<PhysicsType> tFactory;

  mScalarFunctionBaseContainer.clear();
  mFunctionWeights.clear();
  mFunctionGoldValues.clear();
  mFunctionNormalization.clear();

  auto tFunctionParams = aProblemParams.sublist("Criterial").sublist(mFunctionName);
  auto tFunctionNamesArray = tFunctionParams.get<Teuchos::Array<std::string>>("Functions");
  auto tFunctionWeightsArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Weights");
  auto tFunctionGoldValuesArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Gold Values");
  auto tFunctionNames      = tFunctionNamesArray.toVector();
  auto tFunctionWeights    = tFunctionWeightsArray.toVector();
  auto tFunctionGoldValues = tFunctionGoldValuesArray.toVector();

  if (tFunctionNames.size() != tFunctionWeights.size())
  {
    const std::string tErrorString = std::string("Number of 'Functions' in '") + mFunctionName + 
                                                 "' parameter list does not equal the number of 'Weights'";
    ANALYZE_THROWERR(tErrorString)
  }
  if (tFunctionNames.size() != tFunctionGoldValues.size())
  {
    const std::string tErrorString = std::string("Number of 'Gold Values' in '") + mFunctionName + 
                                                 "' parameter list does not equal the number of 'Functions'";
    ANALYZE_THROWERR(tErrorString)
  }
  for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
  {
    mScalarFunctionBaseContainer.push_back(
      tFactory.create(mSpatialModel, mDataMap, aProblemParams, tFunctionNames[tFunctionIndex])
    );
    mFunctionWeights.push_back(tFunctionWeights[tFunctionIndex]);
    appendGoldFunctionValue(tFunctionGoldValues[tFunctionIndex]);
  }
}

template<typename PhysicsType>
CriterionEvaluatorLeastSquares<PhysicsType>::
CriterionEvaluatorLeastSquares(
  const Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
        std::string            & aName
) :
  mSpatialModel (aSpatialModel),
  mDataMap      (aDataMap),
  mFunctionName (aName)
{
  initialize(aProblemParams);
}

template<typename PhysicsType>
CriterionEvaluatorLeastSquares<PhysicsType>::
CriterionEvaluatorLeastSquares(
  const Plato::SpatialModel & aSpatialModel,
        Plato::DataMap      & aDataMap
) :
  mSpatialModel (aSpatialModel),
  mDataMap      (aDataMap),
  mFunctionName ("Least Squares")
{
}

template<typename PhysicsType>
void CriterionEvaluatorLeastSquares<PhysicsType>::
appendFunctionWeight(
  Plato::Scalar aWeight
)
{
  mFunctionWeights.push_back(aWeight);
}

template<typename PhysicsType>
void CriterionEvaluatorLeastSquares<PhysicsType>::
appendGoldFunctionValue(
  Plato::Scalar aGoldValue, 
  bool aUseAsNormalization
)
{
  mFunctionGoldValues.push_back(aGoldValue);
  if (aUseAsNormalization)
  {
    if (std::abs(aGoldValue) > mFunctionNormalizationCutoff)
        mFunctionNormalization.push_back(std::abs(aGoldValue));
    else
        mFunctionNormalization.push_back(1.0);
  }
}

template<typename PhysicsType>
void 
CriterionEvaluatorLeastSquares<PhysicsType>::
appendFunctionNormalization(
  Plato::Scalar aFunctionNormalization
)
{
  // Dont allow the function normalization to be "too small"
  if (std::abs(aFunctionNormalization) > mFunctionNormalizationCutoff)
    mFunctionNormalization.push_back(std::abs(aFunctionNormalization));
  else
    mFunctionNormalization.push_back(mFunctionNormalizationCutoff);
}

template<typename PhysicsType>
void 
CriterionEvaluatorLeastSquares<PhysicsType>::
allocateScalarFunctionBase(
  const std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>& aInput
)
{
  mScalarFunctionBaseContainer.push_back(aInput);
}

template<typename PhysicsType>
bool 
CriterionEvaluatorLeastSquares<PhysicsType>::
isLinear() 
const
{
  bool tIsLinear = true;
  for(auto& tEvaluator : mScalarFunctionBaseContainer){
    if( !tEvaluator->isLinear() ){
      tIsLinear = false;
      break;
    }
  }
  return tIsLinear;
}

template<typename PhysicsType>
void 
CriterionEvaluatorLeastSquares<PhysicsType>::
updateProblem(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
  {
    mScalarFunctionBaseContainer[tFunctionIndex]->updateProblem(aDatabase,aCycle);
  }
}

template<typename PhysicsType>
Plato::Scalar 
CriterionEvaluatorLeastSquares<PhysicsType>::
value(const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
) const
{
  assert(mFunctionWeights.size() == mScalarFunctionBaseContainer.size());
  assert(mFunctionGoldValues.size() == mScalarFunctionBaseContainer.size());
  assert(mFunctionNormalization.size() == mScalarFunctionBaseContainer.size());
  Plato::Scalar tResult = 0.0;
  for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
  {
    const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
    const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
    const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
    Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aDatabase,aCycle);
    tResult += tFunctionWeight * 
               std::pow((tFunctionValue - tFunctionGoldValue) / tFunctionScale, 2);
    Plato::Scalar tPercentDiff = std::abs(tFunctionGoldValue) > 0.0 ? 
                                 100.0 * (tFunctionValue - tFunctionGoldValue) / tFunctionGoldValue :
                                 (tFunctionValue - tFunctionGoldValue);
    printf("%20s = %12.4e * ((%12.4e - %12.4e) / %12.4e)^2 =  %12.4e (PercDiff = %10.1f)\n", 
           mScalarFunctionBaseContainer[tFunctionIndex]->name().c_str(),
           tFunctionWeight,
           tFunctionValue, 
           tFunctionGoldValue,
           tFunctionScale,
           tFunctionWeight * 
                     std::pow((tFunctionValue - tFunctionGoldValue) / tFunctionScale, 2),
           tPercentDiff);
  }
  return tResult;
}

template<typename PhysicsType>
Plato::ScalarVector 
CriterionEvaluatorLeastSquares<PhysicsType>::
gradientConfig(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumNodes = mSpatialModel.Mesh->NumNodes();
  const Plato::OrdinalType tNumDofs = mNumSpatialDims * tNumNodes;
  Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);
  for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
  {
    const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
    const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
    const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
    Plato::Scalar tFunctionValue = 
      mScalarFunctionBaseContainer[tFunctionIndex]->value(aDatabase,aCycle);
    Plato::ScalarVector tFunctionGradX = 
      mScalarFunctionBaseContainer[tFunctionIndex]->gradientConfig(aDatabase,aCycle);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
    {
        tGradientX(tDof) += 2.0 * tFunctionWeight * (tFunctionValue - tFunctionGoldValue) 
                                * tFunctionGradX(tDof) / (tFunctionScale * tFunctionScale);
    },"Least Squares Function Summation Grad X");
  }
  return tGradientX;
}

template<typename PhysicsType>
Plato::ScalarVector 
CriterionEvaluatorLeastSquares<PhysicsType>::
gradientState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumNodes = mSpatialModel.Mesh->NumNodes();
  const Plato::OrdinalType tNumDofs = mNumDofsPerNode * tNumNodes;
  Plato::ScalarVector tGradientU ("gradient state", tNumDofs);
  if (mGradientWRTStateIsZero)
  {
    Plato::blas1::fill(0.0, tGradientU);
  }
  else
  {
    for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
    {
      const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
      const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
      const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
      Plato::Scalar tFunctionValue = 
        mScalarFunctionBaseContainer[tFunctionIndex]->value(aDatabase,aCycle);
      Plato::ScalarVector tFunctionGradU = 
        mScalarFunctionBaseContainer[tFunctionIndex]->gradientState(aDatabase,aCycle);
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
      {
        tGradientU(tDof) += 2.0 * tFunctionWeight * (tFunctionValue - tFunctionGoldValue) 
                                * tFunctionGradU(tDof) / (tFunctionScale * tFunctionScale);
      },"Least Squares Function Summation Grad U");
    }
  }
  return tGradientU;
}

template<typename PhysicsType>
Plato::ScalarVector 
CriterionEvaluatorLeastSquares<PhysicsType>::
gradientNodeState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumNodeStates = mSpatialModel.Mesh->NumNodes() * mNumNodeStatePerNode;
  Plato::ScalarVector tGradientN ("gradient wrt node states", tNumNodeStates);
  for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
  {
    const Plato::Scalar tFunctionScale     = mFunctionNormalization[tFunctionIndex];
    const Plato::Scalar tFunctionWeight    = mFunctionWeights[tFunctionIndex];
    const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
    Plato::Scalar tFunctionValue = 
      mScalarFunctionBaseContainer[tFunctionIndex]->value(aDatabase,aCycle);
    Plato::ScalarVector tFunctionGradN = 
      mScalarFunctionBaseContainer[tFunctionIndex]->gradientNodeState(aDatabase,aCycle);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodeStates), KOKKOS_LAMBDA(const Plato::OrdinalType & tNode)
    {
      tGradientN(tNode) += 2.0 * tFunctionWeight * (tFunctionValue - tFunctionGoldValue) 
                              * tFunctionGradN(tNode) / (tFunctionScale * tFunctionScale);
    },"Least Squares Function Summation Grad N");
  }
  return tGradientN;
}

template<typename PhysicsType>
Plato::ScalarVector 
CriterionEvaluatorLeastSquares<PhysicsType>::
gradientControl(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumNodes = mSpatialModel.Mesh->NumNodes();
  Plato::ScalarVector tGradientZ ("gradient control", tNumNodes);
  for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
  {
    const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
    const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
    const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
    Plato::Scalar tFunctionValue = 
      mScalarFunctionBaseContainer[tFunctionIndex]->value(aDatabase,aCycle);
    Plato::ScalarVector tFunctionGradZ = 
      mScalarFunctionBaseContainer[tFunctionIndex]->gradientControl(aDatabase,aCycle);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), KOKKOS_LAMBDA(const Plato::OrdinalType & tNode)
    {
      tGradientZ(tNode) += 2.0 * tFunctionWeight * (tFunctionValue - tFunctionGoldValue) 
                              * tFunctionGradZ(tNode) / (tFunctionScale * tFunctionScale);
    },"Least Squares Function Summation Grad Z");
  }
  return tGradientZ;
}

template<typename PhysicsType>
std::string 
CriterionEvaluatorLeastSquares<PhysicsType>::
name() const
{
  return mFunctionName;
}

template<typename PhysicsType>
void 
CriterionEvaluatorLeastSquares<PhysicsType>::
setGradientWRTStateIsZeroFlag(
  bool aGradientWRTStateIsZero
)
{
  mGradientWRTStateIsZero = aGradientWRTStateIsZero;
}
} // namespace Elliptic

} // namespace Plato
