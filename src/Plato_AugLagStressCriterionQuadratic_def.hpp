#pragma once

#include <algorithm>
#include <memory>

#include "Simp.hpp"
#include "BLAS1.hpp"
#include "ToMap.hpp"
#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionQuadratic<EvaluationType>::
    initialize(Teuchos::ParameterList & aInputParams)
    {
        this->readInputs(aInputParams);

        Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionQuadratic<EvaluationType>::
    readInputs(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
        mPenalty = tParams.get<Plato::Scalar>("SIMP penalty", 3.0);
        mLocalMeasureLimit = tParams.get<Plato::Scalar>("Local Measure Limit", 1.0);
        mAugLagPenalty = tParams.get<Plato::Scalar>("Initial Penalty", 0.25);
        mMinErsatzValue = tParams.get<Plato::Scalar>("Min. Ersatz Material", 1e-9);
        mAugLagPenaltyUpperBound = tParams.get<Plato::Scalar>("Penalty Upper Bound", 500.0);
        mInitialLagrangeMultipliersValue = tParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.01);
        mAugLagPenaltyExpansionMultiplier = tParams.get<Plato::Scalar>("Penalty Expansion Multiplier", 1.5);
    }

    /******************************************************************************//**
     * \brief Update Augmented Lagrangian penalty
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionQuadratic<EvaluationType>::
    updateAugLagPenaltyMultipliers()
    {
        mAugLagPenalty = mAugLagPenaltyExpansionMultiplier * mAugLagPenalty;
        mAugLagPenalty = std::min(mAugLagPenalty, mAugLagPenaltyUpperBound);
    }

    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aPlatoDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aFuncName user defined function name
     **********************************************************************************/
    template<typename EvaluationType>
    AugLagStressCriterionQuadratic<EvaluationType>::
    AugLagStressCriterionQuadratic(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aInputParams, aFuncName),
        mPenalty(3),
        mLocalMeasureLimit(1),
        mAugLagPenalty(0.1),
        mMinErsatzValue(0.0),
        mAugLagPenaltyUpperBound(100),
        mInitialLagrangeMultipliersValue(0.01),
        mAugLagPenaltyExpansionMultiplier(1.05),
        mLagrangeMultipliers("Lagrange Multipliers", aSpatialDomain.Mesh->NumElements())
    {
        this->initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    template<typename EvaluationType>
    AugLagStressCriterionQuadratic<EvaluationType>::
    AugLagStressCriterionQuadratic(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, "Local Constraint Quadratic"),
        mPenalty(3),
        mLocalMeasureLimit(1),
        mAugLagPenalty(0.1),
        mMinErsatzValue(0.0),
        mAugLagPenaltyUpperBound(100),
        mInitialLagrangeMultipliersValue(0.01),
        mAugLagPenaltyExpansionMultiplier(1.05),
        mLagrangeMultipliers("Lagrange Multipliers", aSpatialDomain.Mesh->NumElements()),
        mLocalMeasureEvaluationType(nullptr),
        mLocalMeasurePODType(nullptr)
    {
        Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    template<typename EvaluationType>
    AugLagStressCriterionQuadratic<EvaluationType>::
    ~AugLagStressCriterionQuadratic()
    {
    }

    /******************************************************************************//**
     * \brief Return augmented Lagrangian penalty multiplier
     * \return augmented Lagrangian penalty multiplier
    **********************************************************************************/
    template<typename EvaluationType>
    Plato::Scalar
    AugLagStressCriterionQuadratic<EvaluationType>::
    getAugLagPenalty() const
    {
        return (mAugLagPenalty);
    }

    /******************************************************************************//**
     * \brief Return Lagrange multipliers
     * \return 1D view of Lagrange multipliers
    **********************************************************************************/
    template<typename EvaluationType>
    Plato::ScalarVector
    AugLagStressCriterionQuadratic<EvaluationType>::
    getLagrangeMultipliers() const
    {
        return (mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Set local measure function
     * \param [in] aInputEvaluationType evaluation type local measure
     * \param [in] aInputPODType pod type local measure
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionQuadratic<EvaluationType>::
    setLocalMeasure(const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInputEvaluationType,
                         const std::shared_ptr<AbstractLocalMeasure<Residual>> & aInputPODType)
    {
        mLocalMeasureEvaluationType = aInputEvaluationType;
        mLocalMeasurePODType        = aInputPODType;
    }

    /******************************************************************************//**
     * \brief Set local constraint limit/upper bound
     * \param [in] aInput local constraint limit
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionQuadratic<EvaluationType>::
    setLocalMeasureValueLimit(const Plato::Scalar & aInput)
    {
        mLocalMeasureLimit = aInput;
    }

    /******************************************************************************//**
     * \brief Set augmented Lagrangian function penalty multiplier
     * \param [in] aInput penalty multiplier
     **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionQuadratic<EvaluationType>::
    setAugLagPenalty(const Plato::Scalar & aInput)
    {
        mAugLagPenalty = aInput;
    }

    /******************************************************************************//**
     * \brief Set Lagrange multipliers
     * \param [in] aInput Lagrange multipliers
     **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionQuadratic<EvaluationType>::
    setLagrangeMultipliers(const Plato::ScalarVector & aInput)
    {
        assert(aInput.size() == mLagrangeMultipliers.size());
        Plato::blas1::copy(aInput, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionQuadratic<EvaluationType>::
    updateProblem(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    )
    {
        this->updateLagrangeMultipliers(aStateWS, aControlWS, aConfigWS);
        this->updateAugLagPenaltyMultipliers();
    }

    /******************************************************************************//**
     * \brief Evaluate augmented Lagrangian local constraint criterion
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionQuadratic<EvaluationType>::
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

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);

        // ****** COMPUTE LOCAL MEASURE VALUES AND STORE ON DEVICE ******
        Plato::ScalarVectorT<ResultT> tLocalMeasureValue("local measure value", tNumCells);
        (*mLocalMeasureEvaluationType)(aStateWS, aControlWS, aConfigWS, tLocalMeasureValue);
        
        Plato::ScalarVectorT<ResultT> tOutputPenalizedLocalMeasure("output penalized local measure", tNumCells);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tLocalMeasureValueLimit = mLocalMeasureLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tLagrangeMultipliers = mLagrangeMultipliers;

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        Plato::Scalar tLagrangianMultiplier = static_cast<Plato::Scalar>(1.0 / tNumCells);
        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            const ResultT tLocalMeasureValueOverLimit = tLocalMeasureValue(iCellOrdinal) / tLocalMeasureValueLimit;
            const ResultT tLocalMeasureValueOverLimitMinusOne = tLocalMeasureValueOverLimit - static_cast<Plato::Scalar>(1.0);
            const ResultT tConstraintValue = ( //pow(tLocalMeasureValueOverLimitMinusOne, 4) +
                                               pow(tLocalMeasureValueOverLimitMinusOne, 2) );

            auto tCubPoint = tCubPoints(iGpOrdinal);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControlWS, tBasisValues);
            ControlT tMaterialPenalty = tSIMP(tDensity);
            const ResultT tTrialConstraintValue = tMaterialPenalty * tConstraintValue;
            const ResultT tTrueConstraintValue = tLocalMeasureValueOverLimit > static_cast<ResultT>(1.0) ?
                                                 tTrialConstraintValue : static_cast<ResultT>(0.0);

            // Compute constraint contribution to augmented Lagrangian function
            const ResultT tResult = tLagrangianMultiplier * ( ( tLagrangeMultipliers(iCellOrdinal) *
                                    tTrueConstraintValue ) + ( static_cast<Plato::Scalar>(0.5) * tAugLagPenalty *
                                    tTrueConstraintValue * tTrueConstraintValue ) );
            Kokkos::atomic_add(&aResultWS(iCellOrdinal), tResult);

            Kokkos::atomic_add(&tOutputPenalizedLocalMeasure(iCellOrdinal), tMaterialPenalty * tLocalMeasureValue(iCellOrdinal));
        });

         Plato::toMap(mDataMap, tOutputPenalizedLocalMeasure, mLocalMeasureEvaluationType->getName(), mSpatialDomain);
    }

    /******************************************************************************//**
     * \brief Update Lagrange multipliers
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    template<typename EvaluationType>
    void
    AugLagStressCriterionQuadratic<EvaluationType>::
    updateLagrangeMultipliers(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    )
    {
        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);

        // ****** COMPUTE LOCAL MEASURE VALUES AND STORE ON DEVICE ******
        Plato::ScalarVector tLocalMeasureValue("local measure value", tNumCells);
        (*mLocalMeasurePODType)(aStateWS, aControlWS, aConfigWS, tLocalMeasureValue);
        
        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tLocalMeasureValueLimit = mLocalMeasureLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tLagrangeMultipliers = mLagrangeMultipliers;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            // Compute local constraint residual
            const Plato::Scalar tLocalMeasureValueOverLimit = tLocalMeasureValue(iCellOrdinal) / tLocalMeasureValueLimit;
            const Plato::Scalar tLocalMeasureValueOverLimitMinusOne = tLocalMeasureValueOverLimit - static_cast<Plato::Scalar>(1.0);
            const Plato::Scalar tConstraintValue = ( //pow(tLocalMeasureValueOverLimitMinusOne, 4) +
                                               pow(tLocalMeasureValueOverLimitMinusOne, 2) );

            auto tCubPoint = tCubPoints(iGpOrdinal);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            Plato::Scalar tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal, aControlWS, tBasisValues);
            Plato::Scalar tMaterialPenalty = tSIMP(tDensity);
            const Plato::Scalar tTrialConstraintValue = tMaterialPenalty * tConstraintValue;
            const Plato::Scalar tTrueConstraintValue = tLocalMeasureValueOverLimit > static_cast<Plato::Scalar>(1.0) ?
                                                       tTrialConstraintValue : static_cast<Plato::Scalar>(0.0);

            // Compute Lagrange multiplier
            const Plato::Scalar tTrialMultiplier = tLagrangeMultipliers(iCellOrdinal) + 
                                           ( tAugLagPenalty * tTrueConstraintValue );
            tLagrangeMultipliers(iCellOrdinal) = (tTrialMultiplier < static_cast<Plato::Scalar>(0.0)) ?
                                                 static_cast<Plato::Scalar>(0.0) : tTrialMultiplier;
        });
    }
}
//namespace Plato
