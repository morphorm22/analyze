#pragma once

#include "MetaData.hpp"
#include "ScalarGrad.hpp"
#include "VectorPNorm.hpp"
#include "ThermalFlux.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "elliptic/thermal/FactoryThermalConductionMaterial.hpp"


namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
CriterionFluxPNorm<EvaluationType, IndicatorFunctionType>::
CriterionFluxPNorm(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap, 
        Teuchos::ParameterList & aProblemParams, 
        Teuchos::ParameterList & aPenaltyParams,
        std::string              aFunctionName
) :
  Plato::CriterionBase(aFunctionName, aSpatialDomain, aDataMap, aProblemParams),
  mIndicatorFunction(aPenaltyParams),
  mApplyWeighting(mIndicatorFunction)
{
  Plato::FactoryThermalConductionMaterial<EvaluationType> mmfactory(aProblemParams);
  mMaterialModel = mmfactory.create(mSpatialDomain.getMaterialName());
  auto params = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);
  mExponent = params.get<Plato::Scalar>("Exponent");
}

template<typename EvaluationType, typename IndicatorFunctionType>
bool 
CriterionFluxPNorm<EvaluationType, IndicatorFunctionType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType, typename IndicatorFunctionType>
void 
CriterionFluxPNorm<EvaluationType, IndicatorFunctionType>::
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
  Plato::ScalarGrad<ElementType>            tScalarGrad;
  Plato::ThermalFlux<EvaluationType>        thermalFlux(mMaterialModel);
  Plato::VectorPNorm<mNumSpatialDims>       tVectorPNorm;

  auto tNumCells   = mSpatialDomain.numCells();
  auto tNumPoints  = mNumGaussPoints;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();

  Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> tInterpolateFromNodal;

  auto& tApplyWeighting = mApplyWeighting;
  auto tExponent        = mExponent;
  Kokkos::parallel_for("thermal energy", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // compute integration function gradients
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
    tScalarGrad(iCellOrdinal, tGrad, tStateWS, tGradient);
    // compute flux
    //
    StateScalarType tTemperature = tInterpolateFromNodal(iCellOrdinal, tBasisValues, tStateWS);
    thermalFlux(tFlux, tGrad, tTemperature);
    // apply weighting
    //
    tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tFlux);
    // compute vector p-norm of flux
    //
    tVectorPNorm(iCellOrdinal, tResultWS, tFlux, tExponent, tVolume);
  });
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
CriterionFluxPNorm<EvaluationType, IndicatorFunctionType>::
postEvaluate( 
  Plato::ScalarVector resultVector,
  Plato::Scalar       resultScalar)
{
  auto scale = pow(resultScalar,(1.0-mExponent)/mExponent)/mExponent;
  auto numEntries = resultVector.size();
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numEntries), KOKKOS_LAMBDA(int entryOrdinal)
  {
    resultVector(entryOrdinal) *= scale;
  },"scale vector");
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
CriterionFluxPNorm<EvaluationType, IndicatorFunctionType>::
postEvaluate( Plato::Scalar& resultValue )
{
  resultValue = pow(resultValue, 1.0/mExponent);
}

} // namespace Elliptic

} // namespace Plato
