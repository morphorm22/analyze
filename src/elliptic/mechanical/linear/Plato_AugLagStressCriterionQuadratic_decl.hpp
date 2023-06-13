#pragma once

#include <algorithm>

#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractLocalMeasure.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Augmented Lagrangian local constraint criterion tailored for general problems
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class AugLagStressCriterionQuadratic : public Plato::CriterionBase
{
private:
    using ElementType = typename EvaluationType::ElementType;

    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
    static constexpr auto mNumVoigtTerms   = ElementType::mNumVoigtTerms;

    using CriterionBaseType = typename Plato::CriterionBase;
    using CriterionBaseType::mSpatialDomain;
    using CriterionBaseType::mDataMap;

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
    void initialize(Teuchos::ParameterList & aInputParams);

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams);

    /******************************************************************************//**
     * \brief Update Augmented Lagrangian penalty
    **********************************************************************************/
    void updateAugLagPenaltyMultipliers();

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
    );

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    AugLagStressCriterionQuadratic(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    );

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~AugLagStressCriterionQuadratic();

    /******************************************************************************//**
     * \brief Return augmented Lagrangian penalty multiplier
     * \return augmented Lagrangian penalty multiplier
    **********************************************************************************/
    Plato::Scalar getAugLagPenalty() const;

    /******************************************************************************//**
     * \brief Return Lagrange multipliers
     * \return 1D view of Lagrange multipliers
    **********************************************************************************/
    Plato::ScalarVector getLagrangeMultipliers() const;

    /******************************************************************************//**
     * \brief Set local measure function
     * \param [in] aInputEvaluationType evaluation type local measure
     * \param [in] aInputPODType pod type local measure
    **********************************************************************************/
    void setLocalMeasure(const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInputEvaluationType,
                         const std::shared_ptr<AbstractLocalMeasure<Residual>> & aInputPODType);

    /******************************************************************************//**
     * \brief Set local constraint limit/upper bound
     * \param [in] aInput local constraint limit
    **********************************************************************************/
    void setLocalMeasureValueLimit(const Plato::Scalar & aInput);

    /******************************************************************************//**
     * \brief Set augmented Lagrangian function penalty multiplier
     * \param [in] aInput penalty multiplier
     **********************************************************************************/
    void setAugLagPenalty(const Plato::Scalar & aInput);

    /******************************************************************************//**
     * \brief Set Lagrange multipliers
     * \param [in] aInput Lagrange multipliers
     **********************************************************************************/
    void setLagrangeMultipliers(const Plato::ScalarVector & aInput);

    /// @fn updateProblem
    /// @brief update criterion parameters at runtime
    /// @param [in] aWorkSets function domain and range workset database
    /// @param [in] aCycle    scalar 
    void 
    updateProblem(
      const Plato::WorkSets & aWorkSets,
      const Plato::Scalar   & aCycle
    ) override;

    /// @fn isLinear
    /// @brief returns true if criterion is linear
    /// @return boolean
    bool 
    isLinear() 
    const;

    void evaluateConditional(
      const Plato::WorkSets & aWorkSets,
      const Plato::Scalar   & aCycle
    ) const;

    void
    updateLagrangeMultipliers(
      const Plato::WorkSets & aWorkSets,
      const Plato::Scalar   & aCycle
    );
};
// class AugLagStressCriterionQuadratic

}
//namespace Plato
