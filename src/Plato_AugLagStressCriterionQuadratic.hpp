#pragma once

#include <algorithm>
#include <memory>

#include "Simp.hpp"
#include "ToMap.hpp"
#include "BLAS1.hpp"
#include "WorksetBase.hpp"
#include "PlatoMathHelpers.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/ExpInstMacros.hpp"
#include "AbstractLocalMeasure.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Augmented Lagrangian local constraint criterion tailored for general problems
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class AugLagStressCriterionQuadratic :
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

    using StateT   = typename EvaluationType::StateScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    using Residual = typename Plato::Elliptic::ResidualTypes<ElementType>;

    Plato::Scalar mPenalty; /*!< penalty parameter in SIMP model */
    Plato::Scalar mLocalMeasureLimit; /*!< local measure limit/upper bound */
    Plato::Scalar mAugLagPenalty; /*!< augmented Lagrangian penalty */
    Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */
    Plato::Scalar mAugLagPenaltyUpperBound; /*!< upper bound on augmented Lagrangian penalty */
    Plato::Scalar mInitialLagrangeMultipliersValue; /*!< initial value for Lagrange multipliers */
    Plato::Scalar mAugLagPenaltyExpansionMultiplier; /*!< expansion parameter for augmented Lagrangian penalty */

    Plato::ScalarVector mLagrangeMultipliers; /*!< Lagrange multipliers */

    /*!< Local measure with evaluation type */
    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType>> mLocalMeasureEvaluationType;

    /*!< Local measure with POD type */
    std::shared_ptr<Plato::AbstractLocalMeasure<Residual>> mLocalMeasurePODType;

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        this->readInputs(aInputParams);

        Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams)
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
    void updateAugLagPenaltyMultipliers()
    {
        mAugLagPenalty = mAugLagPenaltyExpansionMultiplier * mAugLagPenalty;
        mAugLagPenalty = std::min(mAugLagPenalty, mAugLagPenaltyUpperBound);
    }

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aPlatoDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aFuncName user defined function name
     **********************************************************************************/
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
    virtual ~AugLagStressCriterionQuadratic()
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
     * \brief Return Lagrange multipliers
     * \return 1D view of Lagrange multipliers
    **********************************************************************************/
    Plato::ScalarVector getLagrangeMultipliers() const
    {
        return (mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Set local measure function
     * \param [in] aInputEvaluationType evaluation type local measure
     * \param [in] aInputPODType pod type local measure
    **********************************************************************************/
    void setLocalMeasure(const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInputEvaluationType,
                         const std::shared_ptr<AbstractLocalMeasure<Residual>> & aInputPODType)
    {
        mLocalMeasureEvaluationType = aInputEvaluationType;
        mLocalMeasurePODType        = aInputPODType;
    }

    /******************************************************************************//**
     * \brief Set local constraint limit/upper bound
     * \param [in] aInput local constraint limit
    **********************************************************************************/
    void setLocalMeasureValueLimit(const Plato::Scalar & aInput)
    {
        mLocalMeasureLimit = aInput;
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
     * \brief Set Lagrange multipliers
     * \param [in] aInput Lagrange multipliers
     **********************************************************************************/
    void setLagrangeMultipliers(const Plato::ScalarVector & aInput)
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
    void
    updateProblem(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    ) override
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
    void evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;

        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);

        // ****** COMPUTE LOCAL MEASURE VALUES AND STORE ON DEVICE ******
        Plato::ScalarVectorT<ResultT> tLocalMeasureValue("local measure value", tNumCells);
        (*mLocalMeasureEvaluationType)(aStateWS, aConfigWS, tLocalMeasureValue);
        
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
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
    void
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
        (*mLocalMeasurePODType)(aStateWS, aConfigWS, tLocalMeasureValue);
        
        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tLocalMeasureValueLimit = mLocalMeasureLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tLagrangeMultipliers = mLagrangeMultipliers;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
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
};
// class AugLagStressCriterionQuadratic

}
//namespace Plato

#include "MechanicsElement.hpp"
#include "ThermomechanicsElement.hpp"

PLATO_ELLIPTIC_DEC_3(Plato::AugLagStressCriterionQuadratic, Plato::MechanicsElement)
PLATO_ELLIPTIC_DEC_3(Plato::AugLagStressCriterionQuadratic, Plato::ThermomechanicsElement)
