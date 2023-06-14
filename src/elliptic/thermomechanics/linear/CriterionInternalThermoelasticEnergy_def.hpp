#pragma once

#include "ToMap.hpp"
#include "MetaData.hpp"
#include "TMKinetics.hpp"
#include "TMKinematics.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"
#include "TMKineticsFactory.hpp"
#include "InterpolateFromNodal.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
CriterionInternalThermoelasticEnergy<EvaluationType, IndicatorFunctionType>::
CriterionInternalThermoelasticEnergy(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
        Teuchos::ParameterList & aPenaltyParams,
  const std::string            & aFunctionName
) :
  FunctionBaseType      (aFunctionName, aSpatialDomain, aDataMap, aProblemParams),
  mIndicatorFunction    (aPenaltyParams),
  mApplyStressWeighting (mIndicatorFunction),
  mApplyFluxWeighting   (mIndicatorFunction)
{
  Teuchos::ParameterList tProblemParams(aProblemParams);
  auto tMaterialName = aSpatialDomain.getMaterialName();
  if( aProblemParams.isSublist("Material Models") == false )
  {
    ANALYZE_THROWERR("Required input list ('Material Models') is missing.");
  }
  if( aProblemParams.sublist("Material Models").isSublist(tMaterialName) == false )
  {
    std::stringstream ss;
    ss << "Specified material model ('" << tMaterialName << "') is not defined";
    ANALYZE_THROWERR(ss.str());
  }
  auto& tParams = aProblemParams.sublist(aFunctionName);
  if( tParams.get<bool>("Include Thermal Strain", true) == false )
  {
    auto tMaterialParams = tProblemParams.sublist("Material Models").sublist(tMaterialName);
    tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a11",0.0);
    tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a22",0.0);
    tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a33",0.0);
  }
  Plato::ThermoelasticModelFactory<EvaluationType> mmfactory(tProblemParams);
  mMaterialModel = mmfactory.create(tMaterialName);
}

template<typename EvaluationType, typename IndicatorFunctionType>
bool 
CriterionInternalThermoelasticEnergy<EvaluationType, IndicatorFunctionType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
CriterionInternalThermoelasticEnergy<EvaluationType, IndicatorFunctionType>::
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

  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::TMKinematics<ElementType>          tKinematics;
  Plato::TMKineticsFactory< EvaluationType, ElementType > tTMKineticsFactory;
  Teuchos::RCP<Plato::AbstractTMKinetics<EvaluationType, ElementType>> tTMKinetics = 
    tTMKineticsFactory.create(mMaterialModel, mSpatialDomain, mDataMap);
  Plato::ScalarProduct<mNumVoigtTerms>      tMechanicalScalarProduct;
  Plato::ScalarProduct<mNumSpatialDims>     tThermalScalarProduct;
  Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> tInterpolateFromNodal;
  
  auto tNumCells   = mSpatialDomain.numCells();
  auto tNumPoints  = mNumGaussPoints;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  
  Plato::ScalarArray3DT<ResultScalarType>     tStress("stress", tNumCells, tNumPoints, mNumVoigtTerms);
  Plato::ScalarArray3DT<ResultScalarType>     tFlux  ("flux",   tNumCells, tNumPoints, mNumSpatialDims);
  Plato::ScalarArray3DT<GradScalarType>       tStrain("strain", tNumCells, tNumPoints, mNumVoigtTerms);
  Plato::ScalarArray3DT<GradScalarType>       tTGrad ("tgrad",  tNumCells, tNumPoints, mNumSpatialDims);
  Plato::ScalarMultiVectorT<ConfigScalarType> tVolume("volume", tNumCells, tNumPoints);
  Plato::ScalarMultiVectorT<StateScalarType>  tTemperature("temperature", tNumCells, tNumPoints);
  Plato::ScalarArray4DT<ConfigScalarType> 
    tGradient("gradient", tNumCells, tNumPoints, mNumNodesPerCell, mNumSpatialDims);
  
  auto& tApplyFluxWeighting   = mApplyFluxWeighting;
  auto& tApplyStressWeighting = mApplyStressWeighting;
  Kokkos::parallel_for("compute internal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint = tCubPoints(iGpOrdinal);
    tComputeGradient(iCellOrdinal, iGpOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    tVolume(iCellOrdinal, iGpOrdinal) *= tCubWeights(iGpOrdinal);
    // compute strain and temperature gradient
    //
    tKinematics(iCellOrdinal, iGpOrdinal, tStrain, tTGrad, tStateWS, tGradient);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tTemperature(iCellOrdinal, iGpOrdinal) = tInterpolateFromNodal(iCellOrdinal, tBasisValues, tStateWS);
  });
  // compute element state
  //
  (*tTMKinetics)(tStress, tFlux, tStrain, tTGrad, tTemperature, tControlWS);
  Kokkos::parallel_for("compute product", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // apply weighting
    //
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tApplyStressWeighting(iCellOrdinal, iGpOrdinal, tControlWS, tBasisValues, tStress);
    tApplyFluxWeighting  (iCellOrdinal, iGpOrdinal, tControlWS, tBasisValues, tFlux);
    // compute element internal energy
    //
    tMechanicalScalarProduct(iCellOrdinal, iGpOrdinal, tResultWS, tStress, tStrain, tVolume);
    tThermalScalarProduct   (iCellOrdinal, iGpOrdinal, tResultWS, tFlux,   tTGrad,  tVolume);
  });
}

} // namespace Elliptic

} // namespace Plato
