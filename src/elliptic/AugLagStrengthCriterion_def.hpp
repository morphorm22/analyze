/*
 * AugLagStrengthCriterion_def.hpp
 *
 *  Created on: May 4, 2023
 */

#pragma once

#include "elliptic/AugLagStrengthCriterion_decl.hpp"

#include "Simp.hpp"
#include "ToMap.hpp"
#include "BLAS1.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

template<typename EvaluationType>
AugLagStrengthCriterion<EvaluationType>::
AugLagStrengthCriterion(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParams,
    const std::string            & aFuncName
) :
    FunctionBaseType(aSpatialDomain, aDataMap, aParams, aFuncName)
{
    this->initialize(aParams);
}

template<typename EvaluationType>
void 
AugLagStrengthCriterion<EvaluationType>::
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
AugLagStrengthCriterion<EvaluationType>::
updateProblem(
    const Plato::ScalarMultiVector & aStateWS,
    const Plato::ScalarMultiVector & aControlWS,
    const Plato::ScalarArray3D     & aConfigWS
)
{
    this->evaluateCurrentConstraints(aStateWS,aControlWS,aConfigWS);
    mAugLagDataMng.updateLagrangeMultipliers();
    mAugLagDataMng.updatePenaltyValues();
}

template<typename EvaluationType>
void 
AugLagStrengthCriterion<EvaluationType>::
evaluate_conditional(
    const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
    const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
    const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
          Plato::ScalarVectorT      <ResultT>  & aResultWS,
          Plato::Scalar aTimeStep
) const
{
    using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;
    // compute local measure and store on device
    const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
    const Plato::OrdinalType tNumConstraints = tNumCells * mAugLagDataMng.mNumLocalConstraints;
    Plato::ScalarVectorT<ResultT> tLocalMeasureValues("local measure values", tNumConstraints);
    (*mLocalMeasureEvaluationType)(aStateWS, aControlWS, aConfigWS, tLocalMeasureValues);
    Plato::ScalarVectorT<ResultT> 
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
        ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControlWS, tBasisValues);
        ControlT tStiffnessPenalty = tSIMP(tDensity);
        // evaluate local equality constraint
        for(Plato::OrdinalType tConstraintIndex = 0; tConstraintIndex < tNumLocalConstraints; tConstraintIndex++)
        {
            // evaluate local inequality constraint 
            const Plato::OrdinalType tLocalIndex = 
                (iCellOrdinal * tNumLocalConstraints) + tConstraintIndex;
            const ResultT tLocalMeasureValueOverLimit = 
                tLocalMeasureValues(tLocalIndex) / tLocalMeasureLimits(tConstraintIndex);
            const ResultT tLocalMeasureValueOverLimitMinusOne = 
                tLocalMeasureValueOverLimit - static_cast<Plato::Scalar>(1.0);
            const ResultT tConstraintValue = pow(tLocalMeasureValueOverLimitMinusOne, 2.0);
            // penalized local inequality constraint
            const ResultT tPenalizedConstraintValue = tStiffnessPenalty * tConstraintValue;
            // evaluate condition: \frac{-\lambda}{\theta}
            const Plato::Scalar tLagMultOverPenalty = ( static_cast<Plato::Scalar>(-1.0) * 
                tLagrangeMultipliers(tLocalIndex) ) / tPenaltyMultipliers(tLocalIndex);
            // evaluate augmented Lagrangian equality constraint
            const ResultT tConstraint = tPenalizedConstraintValue > tLagMultOverPenalty ?
                tPenalizedConstraintValue : static_cast<ResultT>(tLagMultOverPenalty);
            // evaluate augmented Lagrangian penalty term
            const ResultT tResult = tNormalization * ( ( tLagrangeMultipliers(tLocalIndex) *
                tConstraint ) + ( static_cast<Plato::Scalar>(0.5) * tPenaltyMultipliers(tLocalIndex) *
                tConstraint * tConstraint ) );
            Kokkos::atomic_add(&aResultWS(iCellOrdinal), tResult);
            // evaluate outputs
            const ResultT tOutput = tStiffnessPenalty * tLocalMeasureValues(iCellOrdinal);
            Kokkos::atomic_add(&tOutputPenalizedLocalMeasure(iCellOrdinal),tOutput);
        }
    });
    // save penalized output local measure values to output data map 
    Plato::toMap(mDataMap,tOutputPenalizedLocalMeasure,"p-vonmises",mSpatialDomain);
}

template<typename EvaluationType>
void 
AugLagStrengthCriterion<EvaluationType>::
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
AugLagStrengthCriterion<EvaluationType>::
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
AugLagStrengthCriterion<EvaluationType>::
parseLimits(Teuchos::ParameterList & aParams)
{
    Teuchos::ParameterList &tParams = 
        aParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
    mAugLagDataMng.parseLimits(tParams);
}

template<typename EvaluationType>
void 
AugLagStrengthCriterion<EvaluationType>::
evaluateCurrentConstraints(
    const Plato::ScalarMultiVector &aStateWS,
    const Plato::ScalarMultiVector &aControlWS,
    const Plato::ScalarArray3D     &aConfigWS
)
{
    // update previous constraint container
    Plato::blas1::copy(mAugLagDataMng.mCurrentConstraintValues,mAugLagDataMng.mPreviousConstraintValues);
    // evaluate local measure
    auto tNumCells = mAugLagDataMng.mNumCells;
    auto tNumConstraints = tNumCells * mAugLagDataMng.mNumLocalConstraints;
    Plato::ScalarVector tLocalMeasureValues("local measure values", tNumConstraints);
    (*mLocalMeasurePODType)(aStateWS, aControlWS, aConfigWS, tLocalMeasureValues);
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
        Plato::Scalar tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControlWS, tBasisValues);
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

}
// namespace Plato
