/*
 * Plato_AugLagStressCriterion.hpp
 *
 *  Created on: Feb 12, 2019
 */

#pragma once

#include <algorithm>

#include "ImplicitFunctors.hpp"
#include "ElasticModelFactory.hpp"

#include "Simp.hpp"
#include "BLAS1.hpp"
#include "ToMap.hpp"
#include "FadTypes.hpp"
#include "SmallStrain.hpp"
#include "WorksetBase.hpp"
#include "LinearStress.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathHelpers.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "VonMisesYieldFunction.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Augmented Lagrangian stress constraint criterion
 *
 * This implementation is based on recent work by Prof. Glaucio Paulino research
 * group at Georgia Institute of Technology. Reference will be provided as soon as
 * it becomes available.
 *
**********************************************************************************/
template<typename EvaluationType>
class AugLagStressCriterion :
        public EvaluationType::ElementType,
        public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mFunctionName;

    using StateT   = typename EvaluationType::StateScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    Plato::Scalar mPenalty; /*!< penalty parameter in SIMP model */
    Plato::Scalar mStressLimit; /*!< stress limit/upper bound */
    Plato::Scalar mAugLagPenalty; /*!< augmented Lagrangian penalty */
    Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */
    Plato::Scalar mCellMaterialDensity; /*!< material density */
    Plato::Scalar mAugLagPenaltyUpperBound; /*!< upper bound on augmented Lagrangian penalty */
    Plato::Scalar mMassMultipliersLowerBound; /*!< lower bound on mass multipliers */
    Plato::Scalar mMassMultipliersUpperBound; /*!< upper bound on mass multipliers */
    Plato::Scalar mMassNormalizationMultiplier; /*!< normalization multipliers for mass criterion */
    Plato::Scalar mInitialMassMultipliersValue; /*!< initial value for mass multipliers */
    Plato::Scalar mInitialLagrangeMultipliersValue; /*!< initial value for Lagrange multipliers */
    Plato::Scalar mAugLagPenaltyExpansionMultiplier; /*!< expansion parameter for augmented Lagrangian penalty */
    Plato::Scalar mMassMultiplierUpperBoundReductionParam; /*!< reduction parameter for upper bound on mass multipliers */

    Plato::ScalarVector mMassMultipliers; /*!< mass multipliers */
    Plato::ScalarVector mLagrangeMultipliers; /*!< Lagrange multipliers */
    Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix; /*!< cell/element Lame constants matrix */

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();

        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aInputParams);
        auto tMaterialModel = tMaterialModelFactory.create(tMaterialName);
        mCellStiffMatrix = tMaterialModel->getStiffnessMatrix();

        Teuchos::ParameterList tMaterialModelsInputs = aInputParams.sublist("Material Models");
        Teuchos::ParameterList tMaterialModelInputs = tMaterialModelsInputs.sublist(tMaterialName);
        mCellMaterialDensity = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);

        this->readInputs(aInputParams);

        Plato::blas1::fill(mInitialMassMultipliersValue, mMassMultipliers);
        Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(mFunctionName);
        mPenalty = tParams.get<Plato::Scalar>("SIMP penalty", 3.0);
        mStressLimit = tParams.get<Plato::Scalar>("Stress Limit", 1.0);
        mAugLagPenalty = tParams.get<Plato::Scalar>("Initial Penalty", 0.1);
        mMinErsatzValue = tParams.get<Plato::Scalar>("Min. Ersatz Material", 1e-9);
        mAugLagPenaltyUpperBound = tParams.get<Plato::Scalar>("Penalty Upper Bound", 100.0);
        mMassMultipliersLowerBound = tParams.get<Plato::Scalar>("Mass Multiplier Lower Bound", 0.0);
        mMassMultipliersUpperBound = tParams.get<Plato::Scalar>("Mass Multiplier Upper Bound", 4.0);
        mInitialMassMultipliersValue = tParams.get<Plato::Scalar>("Initial Mass Multiplier", 1.0);
        mMassNormalizationMultiplier = tParams.get<Plato::Scalar>("Mass Normalization Multiplier", 1.0);
        mInitialLagrangeMultipliersValue = tParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.01);
        mAugLagPenaltyExpansionMultiplier = tParams.get<Plato::Scalar>("Penalty Expansion Multiplier", 1.05);
        mMassMultiplierUpperBoundReductionParam = tParams.get<Plato::Scalar>("Mass Multiplier Reduction Multiplier", 0.95);
    }

    /******************************************************************************//**
     * \brief Update augmented Lagrangian penalty and upper bound on mass multipliers.
    **********************************************************************************/
    void updateAugLagPenaltyMultipliers()
    {
        mAugLagPenalty = mAugLagPenaltyExpansionMultiplier * mAugLagPenalty;
        mAugLagPenalty = std::min(mAugLagPenalty, mAugLagPenaltyUpperBound);
        mMassMultipliersUpperBound = mMassMultipliersUpperBound * mMassMultiplierUpperBoundReductionParam;
    }

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aInputParams input parameters database
     **********************************************************************************/
    AugLagStressCriterion(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aInputParams, aFunctionName),
        mPenalty(3),
        mStressLimit(1),
        mAugLagPenalty(0.1),
        mMinErsatzValue(0.0),
        mCellMaterialDensity(1.0),
        mAugLagPenaltyUpperBound(100),
        mMassMultipliersLowerBound(0),
        mMassMultipliersUpperBound(4),
        mMassNormalizationMultiplier(1.0),
        mInitialMassMultipliersValue(1.0),
        mInitialLagrangeMultipliersValue(0.01),
        mAugLagPenaltyExpansionMultiplier(1.05),
        mMassMultiplierUpperBoundReductionParam(0.95),
        mMassMultipliers("Mass Multipliers", aSpatialDomain.Mesh->NumElements()),
        mLagrangeMultipliers("Lagrange Multipliers", aSpatialDomain.Mesh->NumElements())
    {
        this->initialize(aInputParams);
        this->computeStructuralMass();
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze data map
     **********************************************************************************/
    AugLagStressCriterion(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, "Stress Constraint"),
        mPenalty(3),
        mStressLimit(1),
        mAugLagPenalty(0.1),
        mMinErsatzValue(0.0),
        mCellMaterialDensity(1.0),
        mAugLagPenaltyUpperBound(100),
        mMassMultipliersLowerBound(0),
        mMassMultipliersUpperBound(4),
        mMassNormalizationMultiplier(1.0),
        mInitialMassMultipliersValue(0.0),
        mInitialLagrangeMultipliersValue(0.01),
        mAugLagPenaltyExpansionMultiplier(1.05),
        mMassMultiplierUpperBoundReductionParam(0.95),
        mMassMultipliers("Mass Multipliers", aSpatialDomain.Mesh->NumElements()),
        mLagrangeMultipliers("Lagrange Multipliers", aSpatialDomain.Mesh->NumElements())
    {
        Plato::blas1::fill(mInitialMassMultipliersValue, mMassMultipliers);
        Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~AugLagStressCriterion()
    {
    }

    /******************************************************************************//**
     * \brief Return augmented Lagrangian penalty multiplier
     * \return augmented Lagrangian penalty multiplier
    **********************************************************************************/
    Plato::Scalar getAugLagPenalty() const
    {
        return (mAugLagPenalty);
    }

    /******************************************************************************//**
     * \brief Return upper bound on mass multipliers
     * \return upper bound on mass multipliers
    **********************************************************************************/
    Plato::Scalar getMassMultipliersUpperBound() const
    {
        return (mMassMultipliersUpperBound);
    }

    /******************************************************************************//**
     * \brief Return multiplier used to normalized mass contribution to the objective function
     * \return upper mass normalization multiplier
    **********************************************************************************/
    Plato::Scalar getMassNormalizationMultiplier() const
    {
        return (mMassNormalizationMultiplier);
    }

    /******************************************************************************//**
     * \brief Return mass multipliers
     * \return 1D view of mass multipliers
    **********************************************************************************/
    Plato::ScalarVector getMassMultipliers() const
    {
        return (mMassMultipliers);
    }

    /******************************************************************************//**
     * \brief Return Lagrange multipliers
     * \return 1D view of Lagrange multipliers
    **********************************************************************************/
    Plato::ScalarVector getLagrangeMultipliers() const
    {
        return (mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Set stress constraint limit/upper bound
     * \param [in] aInput stress constraint limit
    **********************************************************************************/
    void setStressLimit(const Plato::Scalar & aInput)
    {
        mStressLimit = aInput;
    }

    /******************************************************************************//**
     * \brief Set augmented Lagrangian function penalty multiplier
     * \param [in] aInput penalty multiplier
     **********************************************************************************/
    void setAugLagPenalty(const Plato::Scalar & aInput)
    {
        mAugLagPenalty = aInput;
    }

    /******************************************************************************//**
     * \brief Set cell material density
     * \param [in] aInput material density
     **********************************************************************************/
    void setCellMaterialDensity(const Plato::Scalar & aInput)
    {
        mCellMaterialDensity = aInput;
    }

    /******************************************************************************//**
     * \brief Set mass multipliers
     * \param [in] aInput mass multipliers
     **********************************************************************************/
    void setMassMultipliers(const Plato::ScalarVector & aInput)
    {
        assert(aInput.size() == mMassMultipliers.size());
        Plato::blas1::copy(aInput, mMassMultipliers);
    }

    /******************************************************************************//**
     * \brief Set Lagrange multipliers
     * \param [in] aInput Lagrange multipliers
     **********************************************************************************/
    void setLagrangeMultipliers(const Plato::ScalarVector & aInput)
    {
        assert(aInput.size() == mLagrangeMultipliers.size());
        Plato::blas1::copy(aInput, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Set cell material stiffness matrix
     * \param [in] aInput cell material stiffness matrix
    **********************************************************************************/
    void setCellStiffMatrix(const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> & aInput)
    {
        mCellStiffMatrix = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aStateWS,
                       const Plato::ScalarMultiVector & aControlWS,
                       const Plato::ScalarArray3D & aConfigWS) override
    {
        this->updateMultipliers(aStateWS, aControlWS, aConfigWS);
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
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;

        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();

        Plato::SmallStrain<ElementType> tCauchyStrain;
        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);
        Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;

        Plato::LinearStress<EvaluationType, ElementType> tCauchyStress(mCellStiffMatrix);
        Plato::ComputeGradientMatrix<ElementType> tComputeGradient;

        Plato::ScalarVectorT<ResultT> tOutputCellVonMises("output von mises", tNumCells);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tStressLimit = mStressLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tMassMultipliers = mMassMultipliers;
        auto tCellMaterialDensity = mCellMaterialDensity;
        auto tLagrangeMultipliers = mLagrangeMultipliers;
        auto tMassNormalizationMultiplier = mMassNormalizationMultiplier;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
            ResultT tCellConstraintValue = tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne;

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ControlT tCellDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControlWS, tBasisValues);
            ControlT tPenalizedCellDensity = tSIMP(tCellDensity);
            tOutputCellVonMises(iCellOrdinal) = tPenalizedCellDensity * tVonMises;
            ResultT tSuggestedPenalizedStressConstraint = tPenalizedCellDensity * tCellConstraintValue;
            ResultT tPenalizedStressConstraint = tVonMisesOverStressLimit > static_cast<ResultT>(1.0) ?
                    tSuggestedPenalizedStressConstraint : static_cast<ResultT>(0.0);

            // Compute relaxed Von Mises stress constraint
            Plato::Scalar tLambdaOverPenalty =
                    static_cast<Plato::Scalar>(-1.0) * tLagrangeMultipliers(iCellOrdinal) / tAugLagPenalty;
            ResultT tRelaxedStressConstraint = (tPenalizedStressConstraint > tLambdaOverPenalty) ?
                    tPenalizedStressConstraint : tLambdaOverPenalty;

            // Compute Von Mises stress contribution to augmented Lagrangian function
            ResultT tStressContribution = ( tLagrangeMultipliers(iCellOrdinal) +
                    static_cast<Plato::Scalar>(0.5) * tAugLagPenalty * tRelaxedStressConstraint ) * tRelaxedStressConstraint;

            // Compute mass contribution to augmented Lagrangian function
            ResultT tCellMass = Plato::cell_mass<mNumNodesPerCell>(iCellOrdinal, tBasisValues, aControlWS);
            tCellMass *= tVolume;
            ResultT tMassContribution = (tMassMultipliers(iCellOrdinal) * tCellMaterialDensity * tCellMass) / tMassNormalizationMultiplier;

            // Compute augmented Lagrangian
            aResultWS(iCellOrdinal) = tMassContribution + ( static_cast<Plato::Scalar>(1.0 / tNumCells) * tStressContribution );
        });

        Plato::toMap(mDataMap, tOutputCellVonMises, "Vonmises");
    }

    /******************************************************************************//**
     * \brief Update Lagrange and mass multipliers
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void
    updateMultipliers(
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
        auto tMassMultipliers = mMassMultipliers;
        auto tLagrangeMultipliers = mLagrangeMultipliers;
        auto tMassMultiplierLowerBound = mMassMultipliersLowerBound;
        auto tMassMultiplierUpperBound = mMassMultipliersUpperBound;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            Plato::Scalar tVolume(0.0), tVonMises(0.0), tMassMultiplierMeasure(0.0);

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
            const Plato::Scalar tVonMisesOverStressLimit = tVonMises / tStressLimit;

            // Compute mass multiplier measure
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            Plato::Scalar tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControlWS, tBasisValues);
            tMassMultiplierMeasure = tVonMisesOverStressLimit * pow(tDensity, static_cast<Plato::Scalar>(0.5));

            // Update mass multipliers
            const Plato::Scalar tOptionOne =
                    static_cast<Plato::Scalar>(0.7) * tMassMultipliers(iCellOrdinal) - static_cast<Plato::Scalar>(0.1);
            const Plato::Scalar tOptionTwo =
                    static_cast<Plato::Scalar>(2.5) * tMassMultipliers(iCellOrdinal) + static_cast<Plato::Scalar>(0.5);
            tMassMultipliers(iCellOrdinal) = tMassMultiplierMeasure > static_cast<Plato::Scalar>(1.0) ?
                    Plato::max2(tOptionOne, tMassMultiplierLowerBound) : Plato::min2(tOptionTwo, tMassMultiplierUpperBound);

            // Compute Von Mises stress constraint residual
            const Plato::Scalar tVonMisesOverLimitMinusOne = tVonMisesOverStressLimit - static_cast<Plato::Scalar>(1.0);
            const Plato::Scalar tPenalizedCellDensity = tSIMP(tDensity);
            const Plato::Scalar tSuggestedPenalizedStressConstraint =
                    tPenalizedCellDensity * tVonMisesOverLimitMinusOne * tVonMisesOverLimitMinusOne;
            const Plato::Scalar tPenalizedStressConstraint = tVonMisesOverStressLimit > static_cast<Plato::Scalar>(1.0) ?
                    tSuggestedPenalizedStressConstraint : static_cast<Plato::Scalar>(0.0);

            // Compute relaxed stress constraint
            const Plato::Scalar tLambdaOverPenalty =
                    static_cast<Plato::Scalar>(-1.0) * tLagrangeMultipliers(iCellOrdinal) / tAugLagPenalty;
            const Plato::Scalar tRelaxedStressConstraint = Plato::max2(tPenalizedStressConstraint, tLambdaOverPenalty);

            // Update Lagrange multipliers
            const Plato::Scalar tSuggestedLagrangeMultiplier =
                    tLagrangeMultipliers(iCellOrdinal) + tAugLagPenalty * tRelaxedStressConstraint;
            tLagrangeMultipliers(iCellOrdinal) = Plato::max2(tSuggestedLagrangeMultiplier, static_cast<Plato::Scalar>(0.0));
        });
    }

    /******************************************************************************//**
     * \brief Compute structural mass (i.e. structural mass with ersatz densities set to one)
    **********************************************************************************/
    void computeStructuralMass()
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
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
};
// class AugLagStressCriterion

}// namespace Plato

//TODO #include "Mechanics.hpp"

#ifdef PLATOANALYZE_1D
//TODO extern template class Plato::AugLagStressCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
//TODO extern template class Plato::AugLagStressCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
//TODO extern template class Plato::AugLagStressCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
//TODO extern template class Plato::AugLagStressCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATOANALYZE_2D
//TODO extern template class Plato::AugLagStressCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
//TODO extern template class Plato::AugLagStressCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
//TODO extern template class Plato::AugLagStressCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
//TODO extern template class Plato::AugLagStressCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATOANALYZE_3D
//TODO extern template class Plato::AugLagStressCriterion<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
//TODO extern template class Plato::AugLagStressCriterion<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
//TODO extern template class Plato::AugLagStressCriterion<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
//TODO extern template class Plato::AugLagStressCriterion<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
