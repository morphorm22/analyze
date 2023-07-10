#pragma once

#include "MetaData.hpp"
#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "elliptic/thermal/FactoryThermalConductionMaterial.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
CriterionInternalThermalEnergy<EvaluationType, IndicatorFunctionType>::
CriterionInternalThermalEnergy(
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
  Plato::FactoryThermalConductionMaterial<EvaluationType> tMaterialModelFactory(aProblemParams);
  mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());
}

template<typename EvaluationType, typename IndicatorFunctionType>
bool 
CriterionInternalThermalEnergy<EvaluationType, IndicatorFunctionType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType, typename IndicatorFunctionType>
void 
CriterionInternalThermalEnergy<EvaluationType, IndicatorFunctionType>::
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
  Plato::ScalarGrad<ElementType>            tComputeScalarGrad;
  Plato::ScalarProduct<mNumSpatialDims>     tComputeScalarProduct;
  Plato::ThermalFlux<EvaluationType>        tComputeThermalFlux(mMaterialModel);
  Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> tInterpolateFromNodal;

  auto tNumCells   = mSpatialDomain.numCells();
  auto tNumPoints  = mNumGaussPoints;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  
  auto tApplyWeighting = mApplyWeighting;
  Kokkos::parallel_for("thermal energy", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // compute interpolation function gradients
    //
    ConfigScalarType tVolume(0.0);
    Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;
    Plato::Array<mNumSpatialDims, GradScalarType> tGrad(0.0);
    Plato::Array<mNumSpatialDims, ResultScalarType> tFlux(0.0);
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    tVolume *= tCubWeights(iGpOrdinal);
    // compute temperature gradient
    //
    tComputeScalarGrad(iCellOrdinal, tGrad, tStateWS, tGradient);
    // compute flux
    //
    StateScalarType tTemperature = tInterpolateFromNodal(iCellOrdinal, tBasisValues, tStateWS);
    tComputeThermalFlux(tFlux, tGrad, tTemperature);
    // apply weighting
    //
    tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tFlux);
    // compute element internal energy (inner product of tgrad and weighted tflux)
    //
    tComputeScalarProduct(iCellOrdinal, tResultWS, tFlux, tGrad, tVolume, -1.0);
  });
}

} // namespace Elliptic

} // namespace Plato
