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
#include "SmallStrain.hpp"
#include "WorksetBase.hpp"
#include "LinearStress.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathHelpers.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "VonMisesYieldFunction.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    initialize(Teuchos::ParameterList & aInputParams)
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

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    readInputs(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
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

    /******************************************************************************//**
     * \brief Update augmented Lagrangian penalty and upper bound on mass multipliers.
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    updateAugLagPenaltyMultipliers()
    {
        mAugLagPenalty = mAugLagPenaltyExpansionMultiplier * mAugLagPenalty;
        mAugLagPenalty = std::min(mAugLagPenalty, mAugLagPenaltyUpperBound);
    }

    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     **********************************************************************************/
    template<typename EvaluationType>
    AugLagStressCriterionGeneral<EvaluationType>::
    AugLagStressCriterionGeneral(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aInputParams, aFuncName),
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

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    template<typename EvaluationType>
    AugLagStressCriterionGeneral<EvaluationType>::
    AugLagStressCriterionGeneral(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, "Stress Constraint"),
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

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    template<typename EvaluationType>
    AugLagStressCriterionGeneral<EvaluationType>::
    ~AugLagStressCriterionGeneral()
    {
    }

    /******************************************************************************//**
     * \brief Return augmented Lagrangian penalty multiplier
     * \return augmented Lagrangian penalty multiplier
    **********************************************************************************/
    template<typename EvaluationType>
    Plato::Scalar
    AugLagStressCriterionGeneral<EvaluationType>::
    getAugLagPenalty() const
    {
        return (mAugLagPenalty);
    }

    /******************************************************************************//**
     * \brief Return multiplier used to normalized mass contribution to the objective function
     * \return upper mass normalization multiplier
    **********************************************************************************/
    template<typename EvaluationType>
    Plato::Scalar
    AugLagStressCriterionGeneral<EvaluationType>::
    getMassNormalizationMultiplier() const
    {
        return (mMassNormalizationMultiplier);
    }

    /******************************************************************************//**
     * \brief Return Lagrange multipliers
     * \return 1D view of Lagrange multipliers
    **********************************************************************************/
    template<typename EvaluationType>
    Plato::ScalarVector
    AugLagStressCriterionGeneral<EvaluationType>::
    getLagrangeMultipliers() const
    {
        return (mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Set stress constraint limit/upper bound
     * \param [in] aInput stress constraint limit
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    setStressLimit(const Plato::Scalar & aInput)
    {
        mStressLimit = aInput;
    }

    /******************************************************************************//**
     * \brief Set augmented Lagrangian function penalty multiplier
     * \param [in] aInput penalty multiplier
     **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    setAugLagPenalty(const Plato::Scalar & aInput)
    {
        mAugLagPenalty = aInput;
    }

    /******************************************************************************//**
     * \brief Set cell material density
     * \param [in] aInput material density
     **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    setCellMaterialDensity(const Plato::Scalar & aInput)
    {
        mCellMaterialDensity = aInput;
    }

    /******************************************************************************//**
     * \brief Set Lagrange multipliers
     * \param [in] aInput Lagrange multipliers
     **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    setLagrangeMultipliers(const Plato::ScalarVector & aInput)
    {
        assert(aInput.size() == mLagrangeMultipliers.size());
        Plato::blas1::copy(aInput, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Set cell material stiffness matrix
     * \param [in] aInput cell material stiffness matrix
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    setCellStiffMatrix(const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> & aInput)
    {
        mCellStiffMatrix = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    updateProblem(const Plato::ScalarMultiVector & aStateWS,
                       const Plato::ScalarMultiVector & aControlWS,
                       const Plato::ScalarArray3D & aConfigWS)
    {
        this->updateLagrangeMultipliers(aStateWS, aControlWS, aConfigWS);
        this->updateAugLagPenaltyMultipliers();
    }

    /******************************************************************************//**
     * \brief Evaluate augmented Lagrangian stress constraint criterion
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS,
              Plato::Scalar aTimeStep
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

            tComputeGradient(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);

            tVolume *= tCubWeights(iGpOrdinal);

            tCauchyStrain(iCellOrdinal, tStrain, aStateWS, tGradient);
            tCauchyStress(tStress, tStrain);

            // Compute 3D Von Mises Yield Criterion
            tComputeVonMises(iCellOrdinal, tStress, tVonMises);

            // Compute Von Mises stress constraint residual
            ResultT tVonMisesOverStressLimit = tVonMises / tStressLimit;
            ResultT tVonMisesOverLimitMinusOne = tVonMisesOverStressLimit - static_cast<Plato::Scalar>(1.0);
            ResultT tConstraintValue = ( tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne
                    * tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne )
                    + ( tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne );

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ControlT tCellDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControlWS, tBasisValues);
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
            ResultT tObjective = ( Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, aControlWS) *
                    tMaterialDensity * tVolume ) / tMassNormalizationMultiplier;

            // Compute augmented Lagrangian function
            aResultWS(iCellOrdinal) = (tMassCriterionWeight * tObjective)
                    + (tStressCriterionWeight * tConstraint);
        });

       Plato::toMap(mDataMap, tOutputVonMises, "Vonmises", mSpatialDomain);
    }

    /******************************************************************************//**
     * \brief Update Lagrange multipliers
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionGeneral<EvaluationType>::
    updateLagrangeMultipliers(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
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

        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            Plato::Scalar tVolume(0.0), tVonMises(0.0);

            Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tGradient;

            Plato::Array<mNumVoigtTerms> tStrain(0.0);
            Plato::Array<mNumVoigtTerms> tStress(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            // Compute 3D Cauchy Stress
            tComputeGradient(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);
            tVolume *= tCubWeights(iGpOrdinal);
            tCauchyStrain(iCellOrdinal, tStrain, aStateWS, tGradient);
            tCauchyStress(tStress, tStrain);

            // Compute 3D Von Mises Yield Criterion
            tComputeVonMises(iCellOrdinal, tStress, tVonMises);

            // Compute Von Mises stress constraint residual
            const Plato::Scalar tVonMisesOverStressLimit = tVonMises / tStressLimit;
            const Plato::Scalar tVonMisesOverLimitMinusOne = tVonMisesOverStressLimit - static_cast<Plato::Scalar>(1.0);
            const Plato::Scalar tConstraintValue = ( tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne
                    * tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne )
                    + ( tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne );

            // Compute penalized Von Mises stress constraint
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            Plato::Scalar tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControlWS, tBasisValues);
            auto tPenalty = tSIMP(tDensity);
            const Plato::Scalar tTrialConstraint = tPenalty * tConstraintValue;
            const Plato::Scalar tTrueConstraint = tVonMisesOverStressLimit > static_cast<Plato::Scalar>(1.0) ?
                    tTrialConstraint : static_cast<Plato::Scalar>(0.0);

            // Compute Lagrange multiplier
            const Plato::Scalar tTrialMultiplier = tLagrangeMultipliers(iCellOrdinal) + ( tAugLagPenalty * tTrueConstraint );
            tLagrangeMultipliers(iCellOrdinal) = Plato::max2(tTrialMultiplier, static_cast<Plato::Scalar>(0.0));
        });
    }

    /******************************************************************************//**
     * \brief Compute structural mass (i.e. structural mass with ersatz densities set to one)
    **********************************************************************************/
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
