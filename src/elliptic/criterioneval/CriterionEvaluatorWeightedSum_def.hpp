#pragma once

#include "AnalyzeMacros.hpp"
#include "elliptic/criterioneval/FactoryCriterionEvaluator.hpp"


namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
void
CriterionEvaluatorWeightedSum<PhysicsType>::
initialize(
    Teuchos::ParameterList & aProblemParams
)
{
  Plato::Elliptic::FactoryCriterionEvaluator<PhysicsType> tFactory;

  mScalarFunctionBaseContainer.clear();
  mFunctionWeights.clear();

  auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);
  auto tFunctionNamesArray = tFunctionParams.get<Teuchos::Array<std::string>>("Functions");
  auto tFunctionWeightsArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Weights");
  auto tFunctionNames = tFunctionNamesArray.toVector();
  auto tFunctionWeights = tFunctionWeightsArray.toVector();

  if (tFunctionNames.size() != tFunctionWeights.size())
  {
    const std::string tErrorString = std::string("Number of 'Functions' in '") + mFunctionName + 
                                                 "' parameter list does not equal the number of 'Weights'";
    ANALYZE_THROWERR(tErrorString)
  }
  for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
  {
    mScalarFunctionBaseContainer.push_back(
      tFactory.create(mSpatialModel, mDataMap, aProblemParams, tFunctionNames[tFunctionIndex]) 
    );
    mFunctionWeights.push_back(tFunctionWeights[tFunctionIndex]);
    mFunctionNames.push_back(tFunctionNames[tFunctionIndex]);
  }
}

template<typename PhysicsType>
CriterionEvaluatorWeightedSum<PhysicsType>::
CriterionEvaluatorWeightedSum(
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
CriterionEvaluatorWeightedSum<PhysicsType>::
CriterionEvaluatorWeightedSum(
    const Plato::SpatialModel & aSpatialModel,
          Plato::DataMap      & aDataMap
) :
  mSpatialModel (aSpatialModel),
  mDataMap      (aDataMap),
  mFunctionName ("Weighted Sum")
{
}

template<typename PhysicsType>
void
CriterionEvaluatorWeightedSum<PhysicsType>::
appendFunctionWeight(Plato::Scalar aWeight)
{
  mFunctionWeights.push_back(aWeight);
}

template<typename PhysicsType>
void
CriterionEvaluatorWeightedSum<PhysicsType>::
appendFunctionName(const std::string & aName)
{
  mFunctionNames.push_back(aName);
}

template<typename PhysicsType>
void 
CriterionEvaluatorWeightedSum<PhysicsType>::
allocateScalarFunctionBase(const std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>& aInput)
{
  mScalarFunctionBaseContainer.push_back(aInput);
}

template<typename PhysicsType>
bool 
CriterionEvaluatorWeightedSum<PhysicsType>::
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
CriterionEvaluatorWeightedSum<PhysicsType>::
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

namespace Private
{
  inline std::string name(
      const Plato::OrdinalType       & aIndex,
      const std::vector<std::string> & aList
  )
  {
    std::string tOut = "";
    try 
    {
      // Set element 6
      tOut = aList.at(aIndex);
      return tOut;
    }catch (std::out_of_range const& exc) 
    {
      return tOut;
    }
  }
}

template<typename PhysicsType>
Plato::Scalar 
CriterionEvaluatorWeightedSum<PhysicsType>::
value(const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
) const
{
  assert(mScalarFunctionBaseContainer.size() == mFunctionWeights.size());
  Plato::Scalar tResult = 0.0;
  for (Plato::OrdinalType tFunctionIndex = 0; 
       tFunctionIndex < mScalarFunctionBaseContainer.size(); 
       ++tFunctionIndex)
  {
    const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
    Plato::Scalar tFunctionValue = 
      mScalarFunctionBaseContainer[tFunctionIndex]->value(aDatabase,aCycle);
    std::string tFuncName = Plato::Elliptic::Private::name(tFunctionIndex, mFunctionNames);
    tFuncName = tFuncName.empty() ? std::string("F-") + std::to_string(tFunctionIndex) : tFuncName;
    std::cout << "Function: " << tFuncName << " Value: " << std::to_string(tFunctionValue) << "\n";
    tResult += tFunctionWeight * tFunctionValue;
  }
  return tResult;
}

template<typename PhysicsType>
Plato::ScalarVector CriterionEvaluatorWeightedSum<PhysicsType>::
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
    Plato::ScalarVector tFunctionGradX = 
      mScalarFunctionBaseContainer[tFunctionIndex]->gradientConfig(aDatabase,aCycle);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
    {
      tGradientX(tDof) += tFunctionWeight * tFunctionGradX(tDof);
    },"Weighted Sum Function Summation Grad X");
  }
  return tGradientX;
}

template<typename PhysicsType>
Plato::ScalarVector 
CriterionEvaluatorWeightedSum<PhysicsType>::
gradientState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumNodes = mSpatialModel.Mesh->NumNodes();
  const Plato::OrdinalType tNumDofs = mNumDofsPerNode * tNumNodes;
  Plato::ScalarVector tGradientU ("gradient state", tNumDofs);
  for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
  {
    const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
    Plato::ScalarVector tFunctionGradU = 
      mScalarFunctionBaseContainer[tFunctionIndex]->gradientState(aDatabase,aCycle);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
    {
      tGradientU(tDof) += tFunctionWeight * tFunctionGradU(tDof);
    },"Weighted Sum Function Summation Grad U");
  }
  return tGradientU;
}

template<typename PhysicsType>
Plato::ScalarVector CriterionEvaluatorWeightedSum<PhysicsType>::
gradientControl(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumDofs = mSpatialModel.Mesh->NumNodes();
  Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
  for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
  {
    const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
    Plato::ScalarVector tFunctionGradZ = 
      mScalarFunctionBaseContainer[tFunctionIndex]->gradientControl(aDatabase,aCycle);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
    {
      tGradientZ(tDof) += tFunctionWeight * tFunctionGradZ(tDof);
    },"Weighted Sum Function Summation Grad Z");
  }
  return tGradientZ;
}

template<typename PhysicsType>
void CriterionEvaluatorWeightedSum<PhysicsType>::
setFunctionName(
  const std::string aFunctionName
)
{
  mFunctionName = aFunctionName;
}

template<typename PhysicsType>
std::string 
CriterionEvaluatorWeightedSum<PhysicsType>::
name() 
const
{
  return mFunctionName;
}
} // namespace Elliptic

} // namespace Plato
