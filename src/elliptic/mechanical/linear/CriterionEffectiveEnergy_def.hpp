#pragma once

#include "ToMap.hpp"
#include "FadTypes.hpp"
#include "SmallStrain.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"
#include "HomogenizedStress.hpp"
#include "ElasticModelFactory.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
CriterionEffectiveEnergy<EvaluationType, IndicatorFunctionType>::
CriterionEffectiveEnergy(
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
  auto materialModel = mmfactory.create(aSpatialDomain.getMaterialName());
  mCellStiffness = materialModel->getStiffnessMatrix();
  Teuchos::ParameterList& tParams = aProblemParams.sublist("Criteria").sublist(aFunctionName);
  auto tAssumedStrain = tParams.get<Teuchos::Array<Plato::Scalar>>("Assumed Strain");
  assert(tAssumedStrain.size() == mNumVoigtTerms);
  for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
  {
    mAssumedStrain[iVoigt] = tAssumedStrain[iVoigt];
  }
  // parse cell problem forcing
  //
  if(aProblemParams.isSublist("Cell Problem Forcing"))
  {
    mColumnIndex = aProblemParams.sublist("Cell Problem Forcing").get<Plato::OrdinalType>("Column Index");
  }
  else
  {
    ANALYZE_THROWERR("Required input missing: 'Column Index' not given in the 'Cell Problem Forcing' block");
  }
  if( tParams.isType<Teuchos::Array<std::string>>("Plottable") )
    mPlottable = tParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
}

template<typename EvaluationType, typename IndicatorFunctionType>
bool 
CriterionEffectiveEnergy<EvaluationType, IndicatorFunctionType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType, typename IndicatorFunctionType>
void 
CriterionEffectiveEnergy<EvaluationType, IndicatorFunctionType>::
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

  Plato::SmallStrain<ElementType>           tVoigtStrain;
  Plato::ScalarProduct<mNumVoigtTerms>      tScalarProduct;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::HomogenizedStress<ElementType>     tHomogenizedStress(mCellStiffness, mColumnIndex);

  auto tNumCells = mSpatialDomain.numCells();
  Plato::ScalarVectorT<ConfigScalarType>      tCellVolume("volume", tNumCells);
  Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);

  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();

  auto tApplyWeighting = mApplyWeighting;
  auto tAssumedStrain  = mAssumedStrain;
  Kokkos::parallel_for("elastic energy", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    ConfigScalarType tVolume(0.0);
    Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;
    Plato::Array<mNumVoigtTerms, StrainScalarType> tStrain(0.0);
    Plato::Array<mNumVoigtTerms, ResultScalarType> tStress(0.0);
    auto tCubPoint = tCubPoints(iGpOrdinal);
    tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    // compute strain
    //
    tVoigtStrain(iCellOrdinal, tStrain, tStateWS, tGradient);
    // compute stress
    //
    tHomogenizedStress(iCellOrdinal, tStress, tStrain);
    // apply weighting
    //
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tStress);
    // compute element internal energy (inner product of strain and weighted stress)
    //
    tVolume *= tCubWeights(iGpOrdinal);
    tScalarProduct(iCellOrdinal, tResultWS, tStress, tAssumedStrain, tVolume);
    for(int i=0; i< mNumVoigtTerms; i++)
    {
        Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
    }
    Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
  });

  Kokkos::parallel_for("compute cell quantities", 
    Kokkos::RangePolicy<>(0, tNumCells),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
  {
    for(int i = 0; i < mNumVoigtTerms; i++)
    {
      tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
    }
  });

  if( std::count(mPlottable.begin(),mPlottable.end(),"effective stress") ) 
    toMap(mDataMap, tCellStress, "effective stress", mSpatialDomain);
  if( std::count(mPlottable.begin(),mPlottable.end(),"cell volume") ) 
    toMap(mDataMap, tCellVolume, "cell volume", mSpatialDomain);
}

} // namespace Elliptic

} // namespace Plato
