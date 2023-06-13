#pragma once

#include "MetaData.hpp"
#include "PlatoMeshExpr.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
CriterionVolumeAverageDenominator<EvaluationType>::
CriterionVolumeAverageDenominator(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap, 
        Teuchos::ParameterList & aProblemParams, 
        std::string            & aFunctionName
) :
  CriterionBaseType(aFunctionName, aSpatialDomain, aDataMap, aProblemParams),
  mSpatialWeightFunction("1.0")
{}

template<typename EvaluationType>
void
CriterionVolumeAverageDenominator<EvaluationType>::
setSpatialWeightFunction(
  std::string aWeightFunctionString
)
{
  mSpatialWeightFunction = aWeightFunctionString;
}

template<typename EvaluationType>
bool 
CriterionVolumeAverageDenominator<EvaluationType>::
isLinear() 
const
{
  return true;
}

template<typename EvaluationType>
void
CriterionVolumeAverageDenominator<EvaluationType>::
evaluateConditional(
  const Plato::WorkSets & aWorkSets,
  const Plato::Scalar   & aCycle
) const
{
  // unpack worksets
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarVectorT<ResultScalarType>>(aWorkSets.get("result"));

  auto tSpatialWeights  = 
    Plato::computeSpatialWeights<ConfigScalarType, ElementType>(mSpatialDomain, tConfigWS, mSpatialWeightFunction);
  auto tNumCells = mSpatialDomain.numCells();
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint  = tCubPoints(iGpOrdinal);
    auto tCubWeight = tCubWeights(iGpOrdinal);
    auto tJacobian = ElementType::jacobian(tCubPoint, tConfigWS, iCellOrdinal);
    ResultScalarType tCellVolume = Plato::determinant(tJacobian);
    tCellVolume *= tCubWeight;
    Kokkos::atomic_add(
      &tResultWS(iCellOrdinal), tCellVolume*tSpatialWeights(iCellOrdinal * tNumPoints + iGpOrdinal, 0)
    );
  });
}

} // namespace Elliptic

} // namespace Plato

