#pragma once

#include "ToMap.hpp"
#include "MetaData.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
CriterionInternalElasticEnergy<EvaluationType, IndicatorFunctionType>::
CriterionInternalElasticEnergy(
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
  Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
  mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());
}

template<typename EvaluationType, typename IndicatorFunctionType>
bool
CriterionInternalElasticEnergy<EvaluationType, IndicatorFunctionType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
CriterionInternalElasticEnergy<EvaluationType, IndicatorFunctionType>::
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
  Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
  Plato::ScalarVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarVectorT<ResultScalarType>>(aWorkSets.get("result"));

  auto tNumCells = mSpatialDomain.numCells();
  Plato::ComputeGradientMatrix<ElementType>   tComputeGradient;
  Plato::SmallStrain<ElementType>             tComputeVoigtStrain;
  Plato::ScalarProduct<mNumVoigtTerms>        tComputeScalarProduct;
  Plato::LinearStress<EvaluationType, ElementType> tComputeVoigtStress(mMaterialModel);
  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();
  auto tApplyWeighting = mApplyWeighting;
  Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    ConfigScalarType tVolume(0.0);
    Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;
    Plato::Array<mNumVoigtTerms, StrainScalarType> tStrain(0.0);
    Plato::Array<mNumVoigtTerms, ResultScalarType> tStress(0.0);
    auto tCubPoint = tCubPoints(iGpOrdinal);
    tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    tComputeVoigtStrain(iCellOrdinal, tStrain, tStateWS, tGradient);
    tComputeVoigtStress(tStress, tStrain);
    tVolume *= tCubWeights(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tStress);
    tComputeScalarProduct(iCellOrdinal, tResultWS, tStress, tStrain, tVolume, 0.5);
  });
}

} // namespace Elliptic

} // namespace Plato
