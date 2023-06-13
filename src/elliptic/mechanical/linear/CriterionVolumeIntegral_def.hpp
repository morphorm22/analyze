#pragma once

#include "Simp.hpp"
#include "MetaData.hpp"
#include "PlatoMeshExpr.hpp"
#include "ApplyWeighting.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
void
CriterionVolumeIntegral<EvaluationType>::
initialize(
  Teuchos::ParameterList & aInputParams
)
{
  this->readInputs(aInputParams);
}

template<typename EvaluationType>
void
CriterionVolumeIntegral<EvaluationType>::
readInputs(
  Teuchos::ParameterList & aInputParams
)
{
  Teuchos::ParameterList & tParams = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
  auto tPenaltyParams = tParams.sublist("Penalty Function");
  std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
  if (tPenaltyType != "SIMP")
  {
    ANALYZE_THROWERR("A penalty function type other than SIMP is not yet implemented for the CriterionVolumeIntegral.")
  }
  mPenalty        = tPenaltyParams.get<Plato::Scalar>("Exponent", 3.0);
  mMinErsatzValue = tPenaltyParams.get<Plato::Scalar>("Minimum Value", 1e-9);
}

template<typename EvaluationType>
CriterionVolumeIntegral<EvaluationType>::
CriterionVolumeIntegral(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aInputParams,
  const std::string            & aFuncName
) :
  CriterionBaseType(aFuncName, aSpatialDomain, aDataMap, aInputParams),
  mSpatialWeightFunction("1.0"),
  mPenalty(3),
  mMinErsatzValue(1.0e-9)
{
  this->initialize(aInputParams);
}

template<typename EvaluationType>
CriterionVolumeIntegral<EvaluationType>::
CriterionVolumeIntegral(
  const Plato::SpatialDomain & aSpatialDomain,
        Plato::DataMap       & aDataMap
) :
  CriterionBaseType("Volume Integral Criterion", aSpatialDomain, aDataMap),
  mPenalty(3),
  mMinErsatzValue(0.0),
  mLocalMeasure(nullptr)
{
}

template<typename EvaluationType>
CriterionVolumeIntegral<EvaluationType>::
~CriterionVolumeIntegral()
{
}

template<typename EvaluationType>
void
CriterionVolumeIntegral<EvaluationType>::
setVolumeIntegratedQuantity(
  const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInput
)
{
  mLocalMeasure = aInput;
}

template<typename EvaluationType>
void
CriterionVolumeIntegral<EvaluationType>::
setSpatialWeightFunction(
  std::string aWeightFunctionString
)
{
  mSpatialWeightFunction = aWeightFunctionString;
}

template<typename EvaluationType>
bool 
CriterionVolumeIntegral<EvaluationType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType>
void
CriterionVolumeIntegral<EvaluationType>::
updateProblem(
  const Plato::WorkSets & aWorkSets,
  const Plato::Scalar   & aCycle
)
{
  // Perhaps update penalty exponent?
  WARNING("Penalty exponents not yet updated in CriterionVolumeIntegral.")
}

template<typename EvaluationType>
void
CriterionVolumeIntegral<EvaluationType>::
evaluateConditional(
  const Plato::WorkSets & aWorkSets,
  const Plato::Scalar   & aCycle
) const
{
  // unpack worksets
  Plato::ScalarArray3DT<ConfigT> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<ControlT> tControlWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("controls"));
  Plato::ScalarMultiVectorT<StateT> tStateWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateT>>(aWorkSets.get("states"));
  Plato::ScalarVectorT<ResultT> tResultWS = 
    Plato::unpack<Plato::ScalarVectorT<ResultT>>(aWorkSets.get("result"));

  // preprocess data before criterion evaluation
  auto tSpatialWeights = 
    Plato::computeSpatialWeights<ConfigT, ElementType>(mSpatialDomain, tConfigWS, mSpatialWeightFunction);
  auto tNumCells = mSpatialDomain.numCells();
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);
  Plato::ApplyWeighting<mNumNodesPerCell, /*num_terms=*/1, Plato::MSIMP> tApplyWeighting(tSIMP);

  // ****** COMPUTE VOLUME AVERAGED QUANTITIES AND STORE ON DEVICE ******
  Plato::ScalarVectorT<ResultT> tVolumeIntegratedQuantity("volume integrated quantity", tNumCells);
  (*mLocalMeasure)(tStateWS, tControlWS, tConfigWS, tVolumeIntegratedQuantity);
    
  Kokkos::parallel_for("compute volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint  = tCubPoints(iGpOrdinal);
    auto tCubWeight = tCubWeights(iGpOrdinal);
    auto tJacobian = ElementType::jacobian(tCubPoint, tConfigWS, iCellOrdinal);
    ResultT tCellVolume = Plato::determinant(tJacobian);
    tCellVolume *= tCubWeight;
    ResultT tValue = 
      tVolumeIntegratedQuantity(iCellOrdinal) * tCellVolume * tSpatialWeights(iCellOrdinal*tNumPoints + iGpOrdinal, 0);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tValue);
    Kokkos::atomic_add(&tResultWS(iCellOrdinal), tValue);
  });
}
}
//namespace Elliptic

}
//namespace Plato
