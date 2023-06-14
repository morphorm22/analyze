#pragma once

#include "MetaData.hpp"
#include "TMKinematics.hpp"
#include "PlatoMeshExpr.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"
#include "TMKineticsFactory.hpp"
#include "InterpolateFromNodal.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
CriterionThermoMechStressPNorm<EvaluationType, IndicatorFunctionType>::
CriterionThermoMechStressPNorm(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap, 
        Teuchos::ParameterList & aProblemParams, 
        Teuchos::ParameterList & aPenaltyParams,
  const std::string            & aFunctionName
) :
  FunctionBaseType      (aFunctionName, aSpatialDomain, aDataMap, aProblemParams),
  mIndicatorFunction    (aPenaltyParams),
  mApplyStressWeighting (mIndicatorFunction)
{
  Plato::ThermoelasticModelFactory<EvaluationType> tFactory(aProblemParams);
  mMaterialModel = tFactory.create(aSpatialDomain.getMaterialName());
  auto tParams = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);
  TensorNormFactory<mNumVoigtTerms, EvaluationType> tNormFactory;
  mNorm = tNormFactory.create(tParams);
  if (tParams.isType<std::string>("Function"))
    mFuncString = tParams.get<std::string>("Function");
}

template<typename EvaluationType, typename IndicatorFunctionType>
bool 
CriterionThermoMechStressPNorm<EvaluationType, IndicatorFunctionType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
CriterionThermoMechStressPNorm<EvaluationType, IndicatorFunctionType>::
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

  Plato::ScalarMultiVectorT<ConfigScalarType> 
    tFxnValues("function values", tNumCells*tNumPoints, 1);
  if (mFuncString == "1.0")
  {
    Kokkos::deep_copy(tFxnValues, 1.0);
  }
  else
  {
    Plato::ScalarArray3DT<ConfigScalarType> 
      tPhysicalPoints("physical points", tNumCells, tNumPoints, mNumSpatialDims);
    Plato::mapPoints<ElementType>(tConfigWS, tPhysicalPoints);
    Plato::getFunctionValues<mNumSpatialDims>(tPhysicalPoints, mFuncString, tFxnValues);
  }
  
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::TMKinematics<ElementType>          tKinematics;
  Plato::TMKineticsFactory< EvaluationType, ElementType > tTMKineticsFactory;
  auto tTMKinetics = tTMKineticsFactory.create(mMaterialModel, mSpatialDomain, mDataMap);
  Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> tInterpolateFromNodal;

  Plato::ScalarArray3DT<ResultScalarType>     tStress("stress", tNumCells, tNumPoints, mNumVoigtTerms);
  Plato::ScalarArray3DT<ResultScalarType>     tFlux  ("flux",   tNumCells, tNumPoints, mNumSpatialDims);
  Plato::ScalarArray3DT<GradScalarType>       tStrain("strain", tNumCells, tNumPoints, mNumVoigtTerms);
  Plato::ScalarArray3DT<GradScalarType>       tTGrad ("tgrad",  tNumCells, tNumPoints, mNumSpatialDims);
  Plato::ScalarMultiVectorT<ConfigScalarType> tVolume("volume", tNumCells, tNumPoints);
  Plato::ScalarMultiVectorT<StateScalarType>  tTemperature("temperature", tNumCells, tNumPoints);
  Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
  Plato::ScalarVectorT<ConfigScalarType>      tCellVolume("volume", tNumCells);
  Plato::ScalarArray4DT<ConfigScalarType> 
    tGradient("gradient", tNumCells, tNumPoints, mNumNodesPerCell, mNumSpatialDims);
  
  auto tApplyStressWeighting = mApplyStressWeighting;
  Kokkos::parallel_for("compute internal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint = tCubPoints(iGpOrdinal);
    tComputeGradient(iCellOrdinal, iGpOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    tVolume(iCellOrdinal, iGpOrdinal) *= tCubWeights(iGpOrdinal);
    tVolume(iCellOrdinal, iGpOrdinal) *= tFxnValues(iCellOrdinal*tNumPoints + iGpOrdinal, 0);
    // compute strain and electric field
    //
    tKinematics(iCellOrdinal, iGpOrdinal, tStrain, tTGrad, tStateWS, tGradient);
    // compute stress and electric displacement
    //
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tTemperature(iCellOrdinal, iGpOrdinal) = tInterpolateFromNodal(iCellOrdinal, tBasisValues, tStateWS);
  });
  // compute element state
  //
  (*tTMKinetics)(tStress, tFlux, tStrain, tTGrad, tTemperature, tControlWS);
  Kokkos::parallel_for("compute internal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // apply weighting
    //
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tApplyStressWeighting(iCellOrdinal, iGpOrdinal, tControlWS, tBasisValues, tStress);
    for(int i=0; i<ElementType::mNumVoigtTerms; i++)
    {
      Kokkos::atomic_add(
        &tCellStress(iCellOrdinal,i), tVolume(iCellOrdinal, iGpOrdinal)*tStress(iCellOrdinal, iGpOrdinal, i)
      );
    }
    Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume(iCellOrdinal, iGpOrdinal));
  });

  Kokkos::parallel_for("compute cell stress", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, mNumVoigtTerms}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iVoigtOrdinal)
  {
    tCellStress(iCellOrdinal, iVoigtOrdinal) /= tCellVolume(iCellOrdinal);
  });
  mNorm->evaluate(tResultWS, tCellStress, tControlWS, tCellVolume);
}

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    CriterionThermoMechStressPNorm<EvaluationType, IndicatorFunctionType>::postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
      mNorm->postEvaluate(resultVector, resultScalar);
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    CriterionThermoMechStressPNorm<EvaluationType, IndicatorFunctionType>::postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
      mNorm->postEvaluate(resultValue);
    }
} // namespace Elliptic

} // namespace Plato
