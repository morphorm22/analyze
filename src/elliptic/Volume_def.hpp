#pragma once

#include "MetaData.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename PenaltyFunctionType>
Volume<EvaluationType, PenaltyFunctionType>::
Volume(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap, 
        Teuchos::ParameterList & aProblemParams, 
        Teuchos::ParameterList & aPenaltyParams,
  const std::string            & aFunctionName
) :
  Plato::CriterionBase(aFunctionName, aSpatialDomain, aDataMap, aProblemParams),
  mPenaltyFunction (aPenaltyParams),
  mApplyWeighting  (mPenaltyFunction)
{}

template<typename EvaluationType, typename PenaltyFunctionType>
bool 
Volume<EvaluationType, PenaltyFunctionType>::
isLinear() 
const
{
  return true;
}

template<typename EvaluationType, typename PenaltyFunctionType>
void
Volume<EvaluationType, PenaltyFunctionType>::
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

  auto tNumCells = mSpatialDomain.numCells();
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  auto tApplyWeighting  = mApplyWeighting;
  Kokkos::parallel_for("compute volume", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint  = tCubPoints(iGpOrdinal);
    auto tCubWeight = tCubWeights(iGpOrdinal);
    auto tJacobian = ElementType::jacobian(tCubPoint, tConfigWS, iCellOrdinal);
    ResultScalarType tCellVolume = Plato::determinant(tJacobian);
    tCellVolume *= tCubWeight;
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tCellVolume);
    Kokkos::atomic_add(&tResultWS(iCellOrdinal), tCellVolume);
  });
}

} // namespace Elliptic

} // namespace Plato
