/*
 * CriterionAugLagStrength_def.hpp
 *
 *  Created on: May 4, 2023
 */

#pragma once

#include "Simp.hpp"
#include "ToMap.hpp"
#include "BLAS1.hpp"
#include "MetaData.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

namespace Elliptic
{
  
template<typename EvaluationType>
CriterionAugLagStrength<EvaluationType>::
CriterionAugLagStrength(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aParams,
  const std::string            & aFuncName
) :
  FunctionBaseType(aFuncName, aSpatialDomain, aDataMap, aParams)
{
  this->initialize(aParams);
}

template<typename EvaluationType>
void 
CriterionAugLagStrength<EvaluationType>::
setLocalMeasure(
  const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInputEvaluationType,
  const std::shared_ptr<AbstractLocalMeasure<Residual>>       & aInputPODType
)
{
  mLocalMeasurePODType        = aInputPODType;
  mLocalMeasureEvaluationType = aInputEvaluationType;
}

template<typename EvaluationType>
void 
CriterionAugLagStrength<EvaluationType>::
updateProblem(
  const Plato::WorkSets & aWorkSets,
  const Plato::Scalar   & aCycle
)
{
  // unpack worksets
  Plato::ScalarArray3DT<Plato::Scalar> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<Plato::Scalar>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<Plato::Scalar>>(aWorkSets.get("controls"));
  Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<Plato::Scalar>>(aWorkSets.get("states"));
  // evaluate local contraints and update Lagrange multipliers and local penalty paramaters
  this->evaluateCurrentConstraints(tStateWS,tControlWS,tConfigWS);
  mAugLagDataMng.updateLagrangeMultipliers();
  mAugLagDataMng.updatePenaltyValues();
}

template<typename EvaluationType>
bool 
CriterionAugLagStrength<EvaluationType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType>
void 
CriterionAugLagStrength<EvaluationType>::
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
  // compute local measure and store on device
  const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
  const Plato::OrdinalType tNumConstraints = tNumCells * mAugLagDataMng.mNumLocalConstraints;
  Plato::ScalarVectorT<ResultScalarType> tLocalMeasureValues("local measure values", tNumConstraints);
  (*mLocalMeasureEvaluationType)(tStateWS, tControlWS, tConfigWS, tLocalMeasureValues);
  Plato::ScalarVectorT<ResultScalarType> 
    tOutputPenalizedLocalMeasure("output penalized local measure values",tNumConstraints);
  // transfer member data to local scope
  auto tPenaltyMultipliers = mAugLagDataMng.mPenaltyValues;
  auto tLocalMeasureLimits = mAugLagDataMng.mLocalMeasureLimits;
  auto tLagrangeMultipliers = mAugLagDataMng.mLagrangeMultipliers;
  // access cubature/integration rule data
  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();
  // evaluate augmented lagrangian penalty term 
  Plato::MSIMP tSIMP(mMaterialPenalty,mMinErsatzMaterialValue);
  auto tNumLocalConstraints = mAugLagDataMng.mNumLocalConstraints;
  Plato::Scalar tNormalization = static_cast<Plato::Scalar>(1.0/tNumCells);
  Kokkos::parallel_for("augmented lagrangian penalty", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // evaluate material penalty model
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, tControlWS, tBasisValues);
    ControlScalarType tStiffnessPenalty = tSIMP(tDensity);
    // evaluate local equality constraint
    for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
    {
      // evaluate local inequality constraint 
      const Plato::OrdinalType tLocalIndex = 
        (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;
      const ResultScalarType tLocalMeasureValueOverLimit = 
        tLocalMeasureValues(tLocalIndex) / tLocalMeasureLimits(tConstraintIndex);
      const ResultScalarType tLocalMeasureValueOverLimitMinusOne = 
        tLocalMeasureValueOverLimit - static_cast<Plato::Scalar>(1.0);
      const ResultScalarType tConstraintValue = pow(tLocalMeasureValueOverLimitMinusOne, 2.0);
      // penalized local inequality constraint
      const ResultScalarType tPenalizedConstraintValue = tStiffnessPenalty * tConstraintValue;
      // evaluate condition: \frac{-\lambda}{\theta}
      const Plato::Scalar tLagMultOverPenalty = ( static_cast<Plato::Scalar>(-1.0) * 
        tLagrangeMultipliers(tLocalIndex) ) / tPenaltyMultipliers(tLocalIndex);
      // evaluate augmented Lagrangian equality constraint
      const ResultScalarType tConstraint = tPenalizedConstraintValue > tLagMultOverPenalty ?
        tPenalizedConstraintValue : static_cast<ResultScalarType>(tLagMultOverPenalty);
      // evaluate augmented Lagrangian penalty term
      const ResultScalarType tResult = tNormalization * ( ( tLagrangeMultipliers(tLocalIndex) *
        tConstraint ) + ( static_cast<Plato::Scalar>(0.5) * tPenaltyMultipliers(tLocalIndex) *
        tConstraint * tConstraint ) );
      Kokkos::atomic_add(&tResultWS(iCellOrdinal), tResult);
      // evaluate outputs
      const ResultScalarType tOutput = tStiffnessPenalty * tLocalMeasureValues(iCellOrdinal);
      Kokkos::atomic_add(&tOutputPenalizedLocalMeasure(iCellOrdinal),tOutput);
    }
  });
  // save penalized output local measure values to output data map 
  Plato::toMap(mDataMap,tOutputPenalizedLocalMeasure,"p-vonmises",mSpatialDomain);
}

template<typename EvaluationType>
void 
CriterionAugLagStrength<EvaluationType>::
initialize(Teuchos::ParameterList & aParams)
{
  this->parseNumerics(aParams);
  this->parseLimits(aParams); 
  auto tNumCells = mSpatialDomain.Mesh->NumElements();
  mAugLagDataMng.allocateContainers(tNumCells);
  mAugLagDataMng.initialize();
}

template<typename EvaluationType>
void
CriterionAugLagStrength<EvaluationType>::
parseNumerics(Teuchos::ParameterList & aParams)
{
  Teuchos::ParameterList & tParams = 
    aParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
  mMaterialPenalty = tParams.get<Plato::Scalar>("Exponent", 3.0);
  mMinErsatzMaterialValue = tParams.get<Plato::Scalar>("Minimum Value", 1e-9);
  mAugLagDataMng.parseNumerics(tParams);
}

template<typename EvaluationType>
void 
CriterionAugLagStrength<EvaluationType>::
parseLimits(Teuchos::ParameterList & aParams)
{
  Teuchos::ParameterList &tParams = 
    aParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
  mAugLagDataMng.parseLimits(tParams);
}

template<typename EvaluationType>
void 
CriterionAugLagStrength<EvaluationType>::
evaluateCurrentConstraints(
  const Plato::ScalarMultiVector &tStateWS,
  const Plato::ScalarMultiVector &tControlWS,
  const Plato::ScalarArray3D     &tConfigWS
)
{
  // update previous constraint container
  Plato::blas1::copy(mAugLagDataMng.mCurrentConstraintValues,mAugLagDataMng.mPreviousConstraintValues);
  // evaluate local measure
  auto tNumCells = mAugLagDataMng.mNumCells;
  auto tNumConstraints = tNumCells * mAugLagDataMng.mNumLocalConstraints;
  Plato::ScalarVector tLocalMeasureValues("local measure values", tNumConstraints);
  (*mLocalMeasurePODType)(tStateWS, tControlWS, tConfigWS, tLocalMeasureValues);
  // copy global member data to local scope
  auto tLocalMeasureLimits  = mAugLagDataMng.mLocalMeasureLimits;
  auto tCurrentConstraints  = mAugLagDataMng.mCurrentConstraintValues;
  auto tPenaltyValues = mAugLagDataMng.mPenaltyValues;
  auto tLagrangeMultipliers = mAugLagDataMng.mLagrangeMultipliers;
  // gather integration rule data
  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();
  // evaluate local constraints
  Plato::MSIMP tSIMP(mMaterialPenalty,mMinErsatzMaterialValue);
  auto tNumLocalConstraints = mAugLagDataMng.mNumLocalConstraints;
  Kokkos::parallel_for("local constraints",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells,tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    // evaluate material penalty model
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    Plato::Scalar tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, tControlWS, tBasisValues);
    Plato::Scalar tStiffnessPenalty = tSIMP(tDensity);
    // evaluate local equality constraint
    for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
    {
      // evaluate local inequality constraint 
      auto tLocalIndex = (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;
      const Plato::Scalar tLocalMeasureValueOverLimit = 
        tLocalMeasureValues(tLocalIndex) / tLocalMeasureLimits(tConstraintIndex);
      const Plato::Scalar tLocalMeasureValueOverLimitMinusOne = 
        tLocalMeasureValueOverLimit - static_cast<Plato::Scalar>(1.0);
      const Plato::Scalar tConstraintValue = pow(tLocalMeasureValueOverLimitMinusOne, 2.0);
      // penalized local inequality constraint
      const Plato::Scalar tPenalizedConstraintValue = tStiffnessPenalty * tConstraintValue;
      // evaluate condition: \frac{-\lambda}{\theta}
      const Plato::Scalar tLagMultOverPenalty = ( static_cast<Plato::Scalar>(-1.0) * 
        tLagrangeMultipliers(tLocalIndex) ) / tPenaltyValues(tLocalIndex);
      // evaluate equality constraint
      tCurrentConstraints(tLocalIndex) = tPenalizedConstraintValue > tLagMultOverPenalty ?
        tPenalizedConstraintValue : tLagMultOverPenalty;
    }
  });
}

} // namespace Elliptic

}
// namespace Plato
