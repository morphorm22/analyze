#pragma once

#include "PlatoMeshExpr.hpp"
#include "base/CriterionBase.hpp"
#include "base/SupportedParamOptions.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorDivision.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorScalarFunction.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
void
CriterionEvaluatorVolumeAverage<PhysicsType>::
initialize(
  Teuchos::ParameterList & aInputParams
)
{
  auto params = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(mFunctionName);
  if (params.isType<std::string>("Function"))
    mSpatialWeightingFunctionString = params.get<std::string>("Function");
  createDivisionFunction(mSpatialModel, aInputParams);
}

template<typename PhysicsType>
std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>>
CriterionEvaluatorVolumeAverage<PhysicsType>::
getVolumeFunction(
  const Plato::SpatialModel & aSpatialModel,
  Teuchos::ParameterList & aInputParams
)
{
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>> tVolumeFunction =
    std::make_shared<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>>(aSpatialModel, mDataMap);
  tVolumeFunction->setFunctionName("Volume Function");
  typename PhysicsType::FunctionFactory tFactory;
  std::string tFunctionType = "volume average criterion denominator";
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    auto tName = tDomain.getDomainName();
    std::shared_ptr<Plato::CriterionBase> tValue = 
      tFactory.template createScalarFunction<Residual>(tDomain,mDataMap,aInputParams,tFunctionType,mFunctionName);
    tValue->setSpatialWeightFunction(mSpatialWeightingFunctionString);
    tVolumeFunction->setEvaluator(Plato::evaluation_t::VALUE, tValue, tName);
    std::shared_ptr<Plato::CriterionBase> tGradientU = 
      tFactory.template createScalarFunction<GradientU>(tDomain,mDataMap,aInputParams,tFunctionType,mFunctionName);
    tGradientU->setSpatialWeightFunction(mSpatialWeightingFunctionString);
    tVolumeFunction->setEvaluator(Plato::evaluation_t::GRAD_U, tGradientU, tName);
    std::shared_ptr<Plato::CriterionBase> tGradientZ = 
      tFactory.template createScalarFunction<GradientZ>(tDomain,mDataMap,aInputParams,tFunctionType,mFunctionName);
    tGradientZ->setSpatialWeightFunction(mSpatialWeightingFunctionString);
    tVolumeFunction->setEvaluator(Plato::evaluation_t::GRAD_Z, tGradientZ, tName);
    std::shared_ptr<Plato::CriterionBase> tGradientX = 
      tFactory.template createScalarFunction<GradientX>(tDomain,mDataMap,aInputParams,tFunctionType,mFunctionName);
    tGradientX->setSpatialWeightFunction(mSpatialWeightingFunctionString);
    tVolumeFunction->setEvaluator(Plato::evaluation_t::GRAD_X, tGradientX, tName);
  }
  return tVolumeFunction;
}

template<typename PhysicsType>
void
CriterionEvaluatorVolumeAverage<PhysicsType>::
createDivisionFunction(
  const Plato::SpatialModel & aSpatialModel,
  Teuchos::ParameterList & aInputParams
)
{
  const std::string tNumeratorName = "Volume Average Criterion Numerator";
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>> tNumerator =
    std::make_shared<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>>(aSpatialModel, mDataMap);
  tNumerator->setFunctionName(tNumeratorName);

  typename PhysicsType::FunctionFactory tFactory;
  std::string tFunctionType = "volume average criterion numerator";
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    auto tName = tDomain.getDomainName();
    std::shared_ptr<Plato::CriterionBase> tNumeratorValue = 
      tFactory.template createScalarFunction<Residual>(tDomain,mDataMap,aInputParams,tFunctionType,mFunctionName);
    tNumeratorValue->setSpatialWeightFunction(mSpatialWeightingFunctionString);
    tNumerator->setEvaluator(Plato::evaluation_t::VALUE, tNumeratorValue, tName);
    std::shared_ptr<Plato::CriterionBase> tNumeratorGradientU = 
      tFactory.template createScalarFunction<GradientU>(tDomain,mDataMap,aInputParams,tFunctionType,mFunctionName);
    tNumeratorGradientU->setSpatialWeightFunction(mSpatialWeightingFunctionString);
    tNumerator->setEvaluator(Plato::evaluation_t::GRAD_U, tNumeratorGradientU, tName);
    std::shared_ptr<Plato::CriterionBase> tNumeratorGradientZ = 
      tFactory.template createScalarFunction<GradientZ>(tDomain,mDataMap,aInputParams,tFunctionType,mFunctionName);
    tNumeratorGradientZ->setSpatialWeightFunction(mSpatialWeightingFunctionString);
    tNumerator->setEvaluator(Plato::evaluation_t::GRAD_Z, tNumeratorGradientZ, tName);
    std::shared_ptr<Plato::CriterionBase> tNumeratorGradientX = 
      tFactory.template createScalarFunction<GradientX>(tDomain,mDataMap,aInputParams,tFunctionType,mFunctionName);
    tNumeratorGradientX->setSpatialWeightFunction(mSpatialWeightingFunctionString);
    tNumerator->setEvaluator(Plato::evaluation_t::GRAD_X, tNumeratorGradientX, tName);
  }
  const std::string tDenominatorName = "Volume Function";
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>> tDenominator = 
       getVolumeFunction(aSpatialModel, aInputParams);
  tDenominator->setFunctionName(tDenominatorName);
  mDivisionFunction =
       std::make_shared<Plato::Elliptic::CriterionEvaluatorDivision<PhysicsType>>(aSpatialModel, mDataMap);
  mDivisionFunction->allocateNumeratorFunction(tNumerator);
  mDivisionFunction->allocateDenominatorFunction(tDenominator);
  mDivisionFunction->setFunctionName("Volume Average Criterion Division Function");
}

template<typename PhysicsType>
CriterionEvaluatorVolumeAverage<PhysicsType>::
CriterionEvaluatorVolumeAverage(
  const Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aInputParams,
        std::string            & aName
) :
  mSpatialModel (aSpatialModel),
  mDataMap      (aDataMap),
  mFunctionName (aName)
{
  initialize(aInputParams);
}

template<typename PhysicsType>
bool 
CriterionEvaluatorVolumeAverage<PhysicsType>::
isLinear() 
const
{
  return ( mDivisionFunction->isLinear() );
}

template<typename PhysicsType>
void
CriterionEvaluatorVolumeAverage<PhysicsType>::
updateProblem(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  mDivisionFunction->updateProblem(aDatabase,aCycle);
}

template<typename PhysicsType>
Plato::Scalar
CriterionEvaluatorVolumeAverage<PhysicsType>::
value(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  Plato::Scalar tFunctionValue = mDivisionFunction->value(aDatabase,aCycle);
  return tFunctionValue;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorVolumeAverage<PhysicsType>::
gradientState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  Plato::ScalarVector tGradientU = mDivisionFunction->gradientState(aDatabase,aCycle);
  return tGradientU;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorVolumeAverage<PhysicsType>::
gradientConfig(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  Plato::ScalarVector tGradientX = mDivisionFunction->gradientConfig(aDatabase,aCycle);
  return tGradientX;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorVolumeAverage<PhysicsType>::
gradientControl(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  Plato::ScalarVector tGradientZ = mDivisionFunction->gradientControl(aDatabase,aCycle);
  return tGradientZ;
}

template<typename PhysicsType>
std::string
CriterionEvaluatorVolumeAverage<PhysicsType>::
name() const
{
  return mFunctionName;
}

} // namespace Elliptic

} // namespace Plato
