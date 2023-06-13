#pragma once

#include "BLAS2.hpp"
#include "MetaData.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "ScalarProduct.hpp"
#include "PlatoMeshExpr.hpp"
#include "GradientMatrix.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, 
         typename IndicatorFunctionType>
CriterionVolAvgStressPNormDenominator<EvaluationType, IndicatorFunctionType>::
CriterionVolAvgStressPNormDenominator(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap, 
        Teuchos::ParameterList & aProblemParams, 
        Teuchos::ParameterList & aPenaltyParams,
  const std::string            & aFunctionName
) :
  Plato::CriterionBase(aFunctionName, aSpatialDomain, aDataMap, aProblemParams),
  mIndicatorFunction (aPenaltyParams),
  mApplyWeighting    (mIndicatorFunction)
{
  auto tParams = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);
  TensorNormFactory<mNumVoigtTerms, EvaluationType> tNormFactory;
  mNorm = tNormFactory.create(tParams);
  if (tParams.isType<std::string>("Function"))
    mSpatialWeightFunction = tParams.get<std::string>("Function");
}

template<typename EvaluationType, typename IndicatorFunctionType>
bool 
CriterionVolAvgStressPNormDenominator<EvaluationType, IndicatorFunctionType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
CriterionVolAvgStressPNormDenominator<EvaluationType, IndicatorFunctionType>::
evaluateConditional(
  const Plato::WorkSets & aWorkSets,
  const Plato::Scalar   & aCycle
) const
{
  // unpack worksets
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<ControlScalarType> tControlWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));
  Plato::ScalarVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarVectorT<ResultScalarType>>(aWorkSets.get("result"));

  auto tSpatialWeights = 
    Plato::computeSpatialWeights<ConfigScalarType, ElementType>(mSpatialDomain, tConfigWS, mSpatialWeightFunction);
  auto tNumCells = mSpatialDomain.numCells();
  Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight", tNumCells);
  Plato::ScalarMultiVectorT<ResultScalarType> tCellWeights("weighted one", tNumCells, mNumVoigtTerms);
  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();
  auto applyWeighting = mApplyWeighting;

  Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint  = tCubPoints(iGpOrdinal);
    auto tCubWeight = tCubWeights(iGpOrdinal);
    // compute cell volume
    //
    auto tJacobian = ElementType::jacobian(tCubPoint, tConfigWS, iCellOrdinal);
    ConfigScalarType tVolume = 
      Plato::determinant(tJacobian) * tCubWeight * tSpatialWeights(iCellOrdinal*tNumPoints + iGpOrdinal, 0);
    Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
    // apply weighting
    //
    ResultScalarType tWeightedOne(1.0);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    applyWeighting(iCellOrdinal, tControlWS, tBasisValues, tWeightedOne);
    Kokkos::atomic_add(&tCellWeights(iCellOrdinal, 0), tWeightedOne);
  });
  mNorm->evaluate(tResultWS, tCellWeights, tControlWS, tCellVolume);
}

template<typename EvaluationType, 
         typename IndicatorFunctionType>
void
CriterionVolAvgStressPNormDenominator<EvaluationType, IndicatorFunctionType>::
postEvaluate( 
  Plato::ScalarVector resultVector,
  Plato::Scalar       resultScalar)
{
  mNorm->postEvaluate(resultVector, resultScalar);
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
CriterionVolAvgStressPNormDenominator<EvaluationType, IndicatorFunctionType>::
postEvaluate( 
  Plato::Scalar& resultValue 
)
{
  mNorm->postEvaluate(resultValue);
}

} // namespace Elliptic

} // namespace Plato
