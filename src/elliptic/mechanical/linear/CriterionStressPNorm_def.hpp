#pragma once

#include "FadTypes.hpp"
#include "MetaData.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "PlatoMeshExpr.hpp"
#include "GradientMatrix.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
CriterionStressPNorm<EvaluationType, IndicatorFunctionType>::
CriterionStressPNorm(
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
  Plato::ElasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
  mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
  auto params = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);
  TensorNormFactory<mNumVoigtTerms, EvaluationType> normFactory;
  mNorm = normFactory.create(params);
  if (params.isType<std::string>("Function"))
    mFuncString = params.get<std::string>("Function");
}

template<typename EvaluationType, typename IndicatorFunctionType>
bool 
CriterionStressPNorm<EvaluationType, IndicatorFunctionType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
CriterionStressPNorm<EvaluationType, IndicatorFunctionType>::
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

  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  auto tNumCells = mSpatialDomain.numCells();
  
  Plato::ScalarMultiVectorT<ConfigScalarType> tFxnValues("function values", tNumCells*tNumPoints, 1);
  
  if (mFuncString == "1.0")
  {
    Kokkos::deep_copy(tFxnValues, 1.0);
  }
  else
  {
    Plato::ScalarArray3DT<ConfigScalarType> tPhysicalPoints("physical points", tNumCells, tNumPoints, mNumSpatialDims);
    Plato::mapPoints<ElementType>(tConfigWS, tPhysicalPoints);
    Plato::getFunctionValues<mNumSpatialDims>(tPhysicalPoints, mFuncString, tFxnValues);
  }

  using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
  Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
  Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume", tNumCells);
  Plato::ComputeGradientMatrix<ElementType> computeGradient;
  Plato::SmallStrain<ElementType>           computeVoigtStrain;
  Plato::LinearStress<EvaluationType, ElementType> computeVoigtStress(mMaterialModel);
  auto applyWeighting = mApplyWeighting;

  Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    ConfigScalarType tVolume(0.0);
    Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;
    Plato::Array<mNumVoigtTerms, StrainScalarType> tStrain(0.0);
    Plato::Array<mNumVoigtTerms, ResultScalarType> tStress(0.0);
    auto tCubPoint = tCubPoints(iGpOrdinal);
    computeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    computeVoigtStrain(iCellOrdinal, tStrain, tStateWS, tGradient);
    computeVoigtStress(tStress, tStrain);
    tVolume *= tCubWeights(iGpOrdinal);
    tVolume *= tFxnValues(iCellOrdinal*tNumPoints + iGpOrdinal, 0);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    applyWeighting(iCellOrdinal, tControlWS, tBasisValues, tStress);
    for(int i=0; i<mNumVoigtTerms; i++)
    {
      Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
    }
    Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
  });

  Kokkos::parallel_for("compute cell quantities", 
    Kokkos::RangePolicy<>(0, tNumCells),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
  {
    for(int i=0; i<mNumVoigtTerms; i++)
    {
      tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
    }
  });

  mNorm->evaluate(tResultWS, tCellStress, tControlWS, tCellVolume);
}

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    CriterionStressPNorm<EvaluationType, IndicatorFunctionType>::postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
        mNorm->postEvaluate(resultVector, resultScalar);
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    CriterionStressPNorm<EvaluationType, IndicatorFunctionType>::postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
        mNorm->postEvaluate(resultValue);
    }

} // namespace Elliptic

} // namespace Plato
