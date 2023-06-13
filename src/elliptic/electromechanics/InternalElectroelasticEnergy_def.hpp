#pragma once

#include "ToMap.hpp"
#include "MetaData.hpp"
#include "EMKinetics.hpp"
#include "EMKinematics.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
InternalElectroelasticEnergy<EvaluationType, IndicatorFunctionType>::
InternalElectroelasticEnergy(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
        Teuchos::ParameterList & aPenaltyParams,
  const std::string            & aFunctionName
) :
  FunctionBaseType      (aFunctionName, aSpatialDomain, aDataMap, aProblemParams),
  mIndicatorFunction    (aPenaltyParams),
  mApplyStressWeighting (mIndicatorFunction),
  mApplyEDispWeighting  (mIndicatorFunction)
{
  Plato::ElectroelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
  mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
}

template<typename EvaluationType, typename IndicatorFunctionType>
bool
InternalElectroelasticEnergy<EvaluationType, IndicatorFunctionType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
InternalElectroelasticEnergy<EvaluationType, IndicatorFunctionType>::
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

  auto tNumCells   = mSpatialDomain.numCells();
  auto tNumPoints  = mNumGaussPoints;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();

  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::EMKinematics<ElementType>          tKinematics;
  Plato::EMKinetics<ElementType>            tKinetics(mMaterialModel);
  Plato::ScalarProduct<mNumVoigtTerms>      tMechanicalScalarProduct;
  Plato::ScalarProduct<mNumSpatialDims>     tElectricalScalarProduct;

  auto& tApplyStressWeighting = mApplyStressWeighting;
  auto& tApplyEDispWeighting  = mApplyEDispWeighting;
  Kokkos::parallel_for("compute internal energy", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    ConfigScalarType tVolume(0.0);
    Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;
    Plato::Array<mNumVoigtTerms,  GradScalarType>   tStrain(0.0);
    Plato::Array<mNumSpatialDims, GradScalarType>   tEField(0.0);
    Plato::Array<mNumVoigtTerms,  ResultScalarType> tStress(0.0);
    Plato::Array<mNumSpatialDims, ResultScalarType> tEDisp (0.0);
    auto tCubPoint = tCubPoints(iGpOrdinal);
    tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    tVolume *= tCubWeights(iGpOrdinal);
    // compute strain and electric field
    //
    tKinematics(iCellOrdinal, tStrain, tEField, tStateWS, tGradient);
    // compute stress and electric displacement
    //
    tKinetics(tStress, tEDisp, tStrain, tEField);
    // apply weighting
    //
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tApplyStressWeighting(iCellOrdinal, tControlWS, tBasisValues, tStress);
    tApplyEDispWeighting (iCellOrdinal, tControlWS, tBasisValues, tEDisp);
    // compute element internal energy
    //
    tMechanicalScalarProduct(iCellOrdinal, tResultWS, tStress, tStrain, tVolume);
    tElectricalScalarProduct(iCellOrdinal, tResultWS, tEDisp,  tEField, tVolume, -1.0);
  });
}

} // namespace Elliptic

} // namespace Plato
