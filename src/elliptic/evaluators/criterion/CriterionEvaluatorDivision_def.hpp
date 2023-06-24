#pragma once

namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
void CriterionEvaluatorDivision<PhysicsType>::
initialize(
  Teuchos::ParameterList & aProblemParams
)
{
  Plato::Elliptic::FactoryCriterionEvaluator<PhysicsType> tFactory;
  auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);
  auto tNumeratorFunctionName = tFunctionParams.get<std::string>("Numerator");
  auto tDenominatorFunctionName = tFunctionParams.get<std::string>("Denominator");
  mScalarFunctionBaseNumerator = 
       tFactory.create(mSpatialModel, mDataMap, aProblemParams, tNumeratorFunctionName);
  mScalarFunctionBaseDenominator = 
       tFactory.create(mSpatialModel, mDataMap, aProblemParams, tDenominatorFunctionName);
}

template<typename PhysicsType>
CriterionEvaluatorDivision<PhysicsType>::
CriterionEvaluatorDivision(
  const Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
  const std::string            & aName
) :
  mSpatialModel (aSpatialModel),
  mDataMap      (aDataMap),
  mFunctionName (aName)
{
  initialize(aProblemParams);
}

template<typename PhysicsType>
CriterionEvaluatorDivision<PhysicsType>::
CriterionEvaluatorDivision(
  const Plato::SpatialModel & aSpatialModel,
        Plato::DataMap      & aDataMap
) :
  mSpatialModel (aSpatialModel),
  mDataMap      (aDataMap),
  mFunctionName ("Division Function")
{
}

template<typename PhysicsType>
void 
CriterionEvaluatorDivision<PhysicsType>::
allocateNumeratorFunction(
  const std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>& aInput
)
{
  mScalarFunctionBaseNumerator = aInput;
}

template<typename PhysicsType>
void 
CriterionEvaluatorDivision<PhysicsType>::
allocateDenominatorFunction(
  const std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>& aInput
)
{
  mScalarFunctionBaseDenominator = aInput;
}

template<typename PhysicsType>
bool 
CriterionEvaluatorDivision<PhysicsType>::
isLinear() 
const
{
  bool tIsLinear = mScalarFunctionBaseNumerator->isLinear() && mScalarFunctionBaseDenominator->isLinear();
  return tIsLinear;
}

template<typename PhysicsType>
void 
CriterionEvaluatorDivision<PhysicsType>::
updateProblem(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  mScalarFunctionBaseNumerator->updateProblem(aDatabase,aCycle);
  mScalarFunctionBaseDenominator->updateProblem(aDatabase,aCycle);
}


template<typename PhysicsType>
Plato::Scalar
CriterionEvaluatorDivision<PhysicsType>::
value(const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
) const
{
  Plato::Scalar tNumeratorValue = 
    mScalarFunctionBaseNumerator->value(aDatabase,aCycle);
  Plato::Scalar tDenominatorValue = 
    mScalarFunctionBaseDenominator->value(aDatabase,aCycle);
  Plato::Scalar tResult = tNumeratorValue / tDenominatorValue;
  if (tDenominatorValue == 0.0)
  {
    ANALYZE_THROWERR("Denominator of division function evaluated to 0!")
  }
  
  return tResult;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorDivision<PhysicsType>::
gradientConfig(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumNodes = mSpatialModel.Mesh->NumNodes();
  const Plato::OrdinalType tNumDofs = mNumSpatialDims * tNumNodes;
  Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);
  Plato::Scalar tNumeratorValue = 
    mScalarFunctionBaseNumerator->value(aDatabase,aCycle);
  Plato::Scalar tDenominatorValue = 
    mScalarFunctionBaseDenominator->value(aDatabase,aCycle);
  Plato::Scalar tDenominatorValueSquared = tDenominatorValue * tDenominatorValue;
  Plato::ScalarVector tNumeratorGradX = 
    mScalarFunctionBaseNumerator->gradientConfig(aDatabase,aCycle);
  Plato::ScalarVector tDenominatorGradX = 
    mScalarFunctionBaseDenominator->gradientConfig(aDatabase,aCycle);
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
  {
    tGradientX(tDof) = (tNumeratorGradX(tDof) * tDenominatorValue - 
                        tDenominatorGradX(tDof) * tNumeratorValue) 
                       / (tDenominatorValueSquared);
  },"Division Function Grad X");
  return tGradientX;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorDivision<PhysicsType>::
gradientState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumNodes = mSpatialModel.Mesh->NumNodes();
  const Plato::OrdinalType tNumDofs = mNumDofsPerNode * tNumNodes;
  Plato::ScalarVector tGradientU ("gradient state", tNumDofs);
  Plato::Scalar tNumeratorValue = 
    mScalarFunctionBaseNumerator->value(aDatabase,aCycle);
  Plato::Scalar tDenominatorValue = 
    mScalarFunctionBaseDenominator->value(aDatabase,aCycle);
  Plato::Scalar tDenominatorValueSquared = tDenominatorValue * tDenominatorValue;
  Plato::ScalarVector tNumeratorGradU = 
    mScalarFunctionBaseNumerator->gradientState(aDatabase,aCycle);
  Plato::ScalarVector tDenominatorGradU = 
    mScalarFunctionBaseDenominator->gradientState(aDatabase,aCycle);
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
  {
    tGradientU(tDof) = (tNumeratorGradU(tDof) * tDenominatorValue - 
                        tDenominatorGradU(tDof) * tNumeratorValue) 
                       / (tDenominatorValueSquared);
  },"Division Function Grad U");
  return tGradientU;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorDivision<PhysicsType>::
gradientNodeState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumDofs = mSpatialModel.Mesh->NumNodes() * mNumNodeStatePerNode;
  Plato::ScalarVector tGradientN ("gradient wrt node states", tNumDofs);
  
  Plato::Scalar tNumeratorValue = 
    mScalarFunctionBaseNumerator->value(aDatabase,aCycle);
  Plato::Scalar tDenominatorValue = 
    mScalarFunctionBaseDenominator->value(aDatabase,aCycle);
  Plato::Scalar tDenominatorValueSquared = tDenominatorValue * tDenominatorValue;
  Plato::ScalarVector tNumeratorGradN = 
    mScalarFunctionBaseNumerator->gradientNodeState(aDatabase,aCycle);
  Plato::ScalarVector tDenominatorGradN = 
    mScalarFunctionBaseDenominator->gradientNodeState(aDatabase,aCycle);
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
  {
    tGradientN(tDof) = (tNumeratorGradN(tDof) * tDenominatorValue - 
                        tDenominatorGradN(tDof) * tNumeratorValue) 
                       / (tDenominatorValueSquared);
  },"Division Function Grad N");
  return tGradientN;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorDivision<PhysicsType>::
gradientControl(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumDofs = mSpatialModel.Mesh->NumNodes();
  Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
  
  Plato::Scalar tNumeratorValue = 
    mScalarFunctionBaseNumerator->value(aDatabase,aCycle);
  Plato::Scalar tDenominatorValue = 
    mScalarFunctionBaseDenominator->value(aDatabase,aCycle);
  Plato::Scalar tDenominatorValueSquared = tDenominatorValue * tDenominatorValue;
  Plato::ScalarVector tNumeratorGradZ = 
    mScalarFunctionBaseNumerator->gradientControl(aDatabase,aCycle);
  Plato::ScalarVector tDenominatorGradZ = 
    mScalarFunctionBaseDenominator->gradientControl(aDatabase,aCycle);
  Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
  {
    tGradientZ(tDof) = (tNumeratorGradZ(tDof) * tDenominatorValue - 
                        tDenominatorGradZ(tDof) * tNumeratorValue) 
                       / (tDenominatorValueSquared);
  },"Division Function Grad Z");
  return tGradientZ;
}

template<typename PhysicsType>
void 
CriterionEvaluatorDivision<PhysicsType>::
setFunctionName(
  const std::string aFunctionName
)
{
  mFunctionName = aFunctionName;
}

template<typename PhysicsType>
std::string 
CriterionEvaluatorDivision<PhysicsType>::
name() const
{
  return mFunctionName;
}

} // namespace Elliptic

} // namespace Plato
