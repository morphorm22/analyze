/*
 * Plato_AugLagStressCriterionGeneral.hpp
 *
 *  Created on: Feb 12, 2019
 */

#pragma once

#include <algorithm>

#include "Simp.hpp"
#include "ToMap.hpp"
#include "BLAS1.hpp"
#include "FadTypes.hpp"
#include "MetaData.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathHelpers.hpp"
#include "base/WorksetBase.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/mechanical/linear/VonMisesYieldFunction.hpp"

namespace Plato
{

template<typename EvaluationType>
void
AugLagStressCriterionGeneral<EvaluationType>::
initialize(
  Teuchos::ParameterList & aInputParams
)
{
  Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aInputParams);
  auto tMaterialModel = tMaterialModelFactory.create(mSpatialDomain.getMaterialName());
  mCellStiffMatrix = tMaterialModel->getStiffnessMatrix();
  Teuchos::ParameterList tMaterialModelsInputs = aInputParams.sublist("Material Models");
  Teuchos::ParameterList tMaterialModelInputs  = tMaterialModelsInputs.sublist(mSpatialDomain.getMaterialName());
  mCellMaterialDensity = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);
  this->readInputs(aInputParams);
  Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
}

template<typename EvaluationType>
void
AugLagStressCriterionGeneral<EvaluationType>::
readInputs(
  Teuchos::ParameterList & aInputParams
)
{
  Teuchos::ParameterList & tParams = 
    aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
  mPenalty = tParams.get<Plato::Scalar>("SIMP penalty", 3.0);
  mStressLimit = tParams.get<Plato::Scalar>("Stress Limit", 1.0);
  mAugLagPenalty = tParams.get<Plato::Scalar>("Initial Penalty", 0.25);
  mMinErsatzValue = tParams.get<Plato::Scalar>("Min. Ersatz Material", 1e-9);
  mMassCriterionWeight = tParams.get<Plato::Scalar>("Mass Criterion Weight", 1.0);
  mAugLagPenaltyUpperBound = tParams.get<Plato::Scalar>("Penalty Upper Bound", 500.0);
  mStressCriterionWeight = tParams.get<Plato::Scalar>("Stress Criterion Weight", 1.0);
  mMassNormalizationMultiplier = tParams.get<Plato::Scalar>("Mass Normalization Multiplier", 1.0);
  mInitialLagrangeMultipliersValue = tParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.01);
  mAugLagPenaltyExpansionMultiplier = tParams.get<Plato::Scalar>("Penalty Expansion Multiplier", 1.5);
}

template<typename EvaluationType>
void
AugLagStressCriterionGeneral<EvaluationType>::
updateAugLagPenaltyMultipliers()
{
  mAugLagPenalty = mAugLagPenaltyExpansionMultiplier * mAugLagPenalty;
  mAugLagPenalty = std::min(mAugLagPenalty, mAugLagPenaltyUpperBound);
}

template<typename EvaluationType>
AugLagStressCriterionGeneral<EvaluationType>::
AugLagStressCriterionGeneral(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aInputParams,
  const std::string            & aFuncName
) :
  CriterionBaseType(aFuncName, aSpatialDomain, aDataMap, aInputParams),
  mPenalty(3),
  mStressLimit(1),
  mAugLagPenalty(0.1),
  mMinErsatzValue(0.0),
  mCellMaterialDensity(1.0),
  mMassCriterionWeight(1.0),
  mStressCriterionWeight(1.0),
  mAugLagPenaltyUpperBound(100),
  mMassNormalizationMultiplier(1.0),
  mInitialLagrangeMultipliersValue(0.01),
  mAugLagPenaltyExpansionMultiplier(1.05),
  mLagrangeMultipliers("Lagrange Multipliers", aSpatialDomain.Mesh->NumElements())
{
  this->initialize(aInputParams);
  this->computeStructuralMass();
}

template<typename EvaluationType>
AugLagStressCriterionGeneral<EvaluationType>::
AugLagStressCriterionGeneral(
  const Plato::SpatialDomain & aSpatialDomain,
        Plato::DataMap       & aDataMap
) :
  CriterionBaseType("AugLagStressCriterionGeneral", aSpatialDomain, aDataMap),
  mPenalty(3),
  mStressLimit(1),
  mAugLagPenalty(0.1),
  mMinErsatzValue(0.0),
  mCellMaterialDensity(1.0),
  mMassCriterionWeight(1.0),
  mStressCriterionWeight(1.0),
  mAugLagPenaltyUpperBound(100),
  mMassNormalizationMultiplier(1.0),
  mInitialLagrangeMultipliersValue(0.01),
  mAugLagPenaltyExpansionMultiplier(1.05),
  mLagrangeMultipliers("Lagrange Multipliers", aSpatialDomain.Mesh->NumElements())
{
  Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
}

template<typename EvaluationType>
AugLagStressCriterionGeneral<EvaluationType>::
~AugLagStressCriterionGeneral()
{
}

template<typename EvaluationType>
Plato::Scalar
AugLagStressCriterionGeneral<EvaluationType>::
getAugLagPenalty() 
const
{
  return (mAugLagPenalty);
}

template<typename EvaluationType>
Plato::Scalar
AugLagStressCriterionGeneral<EvaluationType>::
getMassNormalizationMultiplier() 
const
{
  return (mMassNormalizationMultiplier);
}

template<typename EvaluationType>
Plato::ScalarVector
AugLagStressCriterionGeneral<EvaluationType>::
getLagrangeMultipliers() const
{
  return (mLagrangeMultipliers);
}

template<typename EvaluationType>
void
AugLagStressCriterionGeneral<EvaluationType>::
setStressLimit(
  const Plato::Scalar & aInput
)
{
  mStressLimit = aInput;
}

template<typename EvaluationType>
void
AugLagStressCriterionGeneral<EvaluationType>::
setAugLagPenalty(
  const Plato::Scalar & aInput
)
{
  mAugLagPenalty = aInput;
}

template<typename EvaluationType>
void
AugLagStressCriterionGeneral<EvaluationType>::
setCellMaterialDensity(
  const Plato::Scalar & aInput
)
{
  mCellMaterialDensity = aInput;
}

template<typename EvaluationType>
void
AugLagStressCriterionGeneral<EvaluationType>::
setLagrangeMultipliers(
  const Plato::ScalarVector & aInput
)
{
  assert(aInput.size() == mLagrangeMultipliers.size());
  Plato::blas1::copy(aInput, mLagrangeMultipliers);
}

template<typename EvaluationType>
void
AugLagStressCriterionGeneral<EvaluationType>::
setCellStiffMatrix(
  const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> & aInput
)
{
  mCellStiffMatrix = aInput;
}

template<typename EvaluationType>
void
AugLagStressCriterionGeneral<EvaluationType>::
updateProblem(
  const Plato::WorkSets & aWorkSets,
  const Plato::Scalar   & aCycle
) 
{
  this->updateLagrangeMultipliers(aWorkSets,aCycle);
  this->updateAugLagPenaltyMultipliers();
}

template<typename EvaluationType>
bool 
AugLagStressCriterionGeneral<EvaluationType>::
isLinear() 
const
{
  return false;
}

  template<typename EvaluationType>
  void
  AugLagStressCriterionGeneral<EvaluationType>::
  evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const
    {
        using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;

        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();

        Plato::SmallStrain<ElementType> tCauchyStrain;
        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);
        Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;
        Plato::LinearStress<EvaluationType, ElementType> tCauchyStress(mCellStiffMatrix);
        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;

        Plato::ScalarVectorT<ResultT> tOutputVonMises("output von mises", tNumCells);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tStressLimit = mStressLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tMaterialDensity = mCellMaterialDensity;
        auto tLagrangeMultipliers = mLagrangeMultipliers;
        auto tMassCriterionWeight = mMassCriterionWeight;
        auto tStressCriterionWeight = mStressCriterionWeight;
        auto tMassNormalizationMultiplier = mMassNormalizationMultiplier;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        // unpack worksets
        Plato::ScalarArray3DT<ConfigT> tConfigWS  = 
          Plato::unpack<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        Plato::ScalarMultiVectorT<ControlT> tControlWS = 
          Plato::unpack<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("controls"));
        Plato::ScalarMultiVectorT<StateT> tStateWS = 
          Plato::unpack<Plato::ScalarMultiVectorT<StateT>>(aWorkSets.get("states"));
        Plato::ScalarVectorT<ResultT> tResultWS = 
          Plato::unpack<Plato::ScalarVectorT<ResultT>>(aWorkSets.get("result"));

        Plato::Scalar tLagrangianMultiplier = static_cast<Plato::Scalar>(1.0 / tNumCells);
        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigT tVolume(0.0);
            ResultT tVonMises(0.0);

            Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigT> tGradient;
            Plato::Array<mNumVoigtTerms, StrainT> tStrain(0.0);
            Plato::Array<mNumVoigtTerms, ResultT> tStress(0.0);
            auto tCubPoint = tCubPoints(iGpOrdinal);
            tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
            tVolume *= tCubWeights(iGpOrdinal);

            tCauchyStrain(iCellOrdinal, tStrain, tStateWS, tGradient);
            tCauchyStress(tStress, tStrain);
            tComputeVonMises(iCellOrdinal, tStress, tVonMises);

            // Compute Von Mises stress constraint residual
            ResultT tVonMisesOverStressLimit = tVonMises / tStressLimit;
            ResultT tVonMisesOverLimitMinusOne = tVonMisesOverStressLimit - static_cast<Plato::Scalar>(1.0);
            ResultT tConstraintValue = ( tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne
                    * tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne )
                    + ( tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne );

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ControlT tCellDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, tControlWS, tBasisValues);
            ControlT tMaterialPenalty = tSIMP(tCellDensity);
            tOutputVonMises(iCellOrdinal) = tMaterialPenalty * tVonMises;
            ResultT tTrialConstraintValue = tMaterialPenalty * tConstraintValue;
            ResultT tTrueConstraintValue = tVonMisesOverStressLimit > static_cast<ResultT>(1.0) ?
                    tTrialConstraintValue : static_cast<ResultT>(0.0);

            // Compute constraint contribution to augmented Lagrangian function
            ResultT tConstraint = tLagrangianMultiplier * ( ( tLagrangeMultipliers(iCellOrdinal) *
                    tTrueConstraintValue ) + ( static_cast<Plato::Scalar>(0.5) * tAugLagPenalty *
                            tTrueConstraintValue * tTrueConstraintValue ) );

            // Compute objective contribution to augmented Lagrangian function
            ResultT tObjective = ( Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, tControlWS) *
                    tMaterialDensity * tVolume ) / tMassNormalizationMultiplier;

            // Compute augmented Lagrangian function
            tResultWS(iCellOrdinal) = (tMassCriterionWeight * tObjective)
                    + (tStressCriterionWeight * tConstraint);
        });
       Plato::toMap(mDataMap, tOutputVonMises, "vonmises", mSpatialDomain);
    }

    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    updateLagrangeMultipliers(
      const Plato::WorkSets & aWorkSets,
      const Plato::Scalar   & aCycle
    )
    {
        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();

        // Create Cauchy stress functors
        Plato::SmallStrain<ElementType> tCauchyStrain;
        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);
        Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;

        Plato::LinearStress<Plato::Elliptic::ResidualTypes<typename EvaluationType::ElementType>, ElementType>
          tCauchyStress(mCellStiffMatrix);

        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tStressLimit = mStressLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tLagrangeMultipliers = mLagrangeMultipliers;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        // unpack worksets
        Plato::ScalarArray3DT<Plato::Scalar> tConfigWS  = 
          Plato::unpack<Plato::ScalarArray3DT<Plato::Scalar>>(aWorkSets.get("configuration"));
        Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS = 
          Plato::unpack<Plato::ScalarMultiVectorT<Plato::Scalar>>(aWorkSets.get("controls"));
        Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS = 
          Plato::unpack<Plato::ScalarMultiVectorT<Plato::Scalar>>(aWorkSets.get("states"));

        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            Plato::Scalar tVolume(0.0), tVonMises(0.0);
            Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tGradient;
            Plato::Array<mNumVoigtTerms> tStrain(0.0);
            Plato::Array<mNumVoigtTerms> tStress(0.0);
            auto tCubPoint = tCubPoints(iGpOrdinal);

            // Compute 3D Cauchy Stress
            tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
            tVolume *= tCubWeights(iGpOrdinal);
            tCauchyStrain(iCellOrdinal, tStrain, tStateWS, tGradient);
            tCauchyStress(tStress, tStrain);
            tComputeVonMises(iCellOrdinal, tStress, tVonMises);

            // Compute Von Mises stress constraint residual
            const Plato::Scalar tVonMisesOverStressLimit = tVonMises / tStressLimit;
            const Plato::Scalar tVonMisesOverLimitMinusOne = tVonMisesOverStressLimit - static_cast<Plato::Scalar>(1.0);
            const Plato::Scalar tConstraintValue = ( tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne
                    * tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne )
                    + ( tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne );

            // Compute penalized Von Mises stress constraint
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            Plato::Scalar tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, tControlWS, tBasisValues);
            auto tPenalty = tSIMP(tDensity);
            const Plato::Scalar tTrialConstraint = tPenalty * tConstraintValue;
            const Plato::Scalar tTrueConstraint = tVonMisesOverStressLimit > static_cast<Plato::Scalar>(1.0) ?
                    tTrialConstraint : static_cast<Plato::Scalar>(0.0);

            // Compute Lagrange multiplier
            const Plato::Scalar tTrialMultiplier = 
              tLagrangeMultipliers(iCellOrdinal) + ( tAugLagPenalty * tTrueConstraint );
            tLagrangeMultipliers(iCellOrdinal) = Plato::max2(tTrialMultiplier, static_cast<Plato::Scalar>(0.0));
        });
    }

template<typename EvaluationType>
void
AugLagStressCriterionGeneral<EvaluationType>::
computeStructuralMass()
{
    auto tNumCells = mSpatialDomain.numCells();
    Plato::NodeCoordinate<mNumSpatialDims, mNumNodesPerCell> tCoordinates(mSpatialDomain.Mesh);
    Plato::ScalarArray3D tConfig("configuration", tNumCells, mNumNodesPerCell, mNumSpatialDims);
    Plato::workset_config_scalar<mNumSpatialDims, mNumNodesPerCell>(tNumCells, tCoordinates, tConfig);
    Plato::ScalarVector tTotalMass("total mass", tNumCells);
    Plato::ScalarMultiVector tDensities("densities", tNumCells, mNumNodesPerCell);
    Kokkos::deep_copy(tDensities, 1.0);
    auto tCubPoints = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints = tCubWeights.size();
    auto tCellMaterialDensity = mCellMaterialDensity;
    Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
        auto tCubPoint  = tCubPoints(iGpOrdinal);
        auto tCubWeight = tCubWeights(iGpOrdinal);
        auto tJacobian = ElementType::jacobian(tCubPoint, tConfig, iCellOrdinal);
        auto tVolume = Plato::determinant(tJacobian);
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, tDensities);
        Kokkos::atomic_add(&tTotalMass(iCellOrdinal), tCellMass * tCellMaterialDensity * tVolume * tCubWeight);
    });
    Plato::blas1::local_sum(tTotalMass, mMassNormalizationMultiplier);
}

}// namespace Plato
