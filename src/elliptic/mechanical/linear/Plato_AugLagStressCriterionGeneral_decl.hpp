/*
 * Plato_AugLagStressCriterionGeneral.hpp
 *
 *  Created on: Feb 12, 2019
 */

#pragma once

#include "ElasticModelFactory.hpp"
#include "base/CriterionBase.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Augmented Lagrangian stress constraint criterion tailored for general problems
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class AugLagStressCriterionGeneral : public Plato::CriterionBase
{
private:
    /// @brief local topological element typename
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

    Plato::Scalar mPenalty; /*!< penalty parameter in SIMP model */
    Plato::Scalar mStressLimit; /*!< stress limit/upper bound */
    Plato::Scalar mAugLagPenalty; /*!< augmented Lagrangian penalty */
    Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */
    Plato::Scalar mCellMaterialDensity; /*!< material density */
    Plato::Scalar mMassCriterionWeight; /*!< weight for mass term, i.e. /f$ \alpha_{\mbox{mass}} * f_{\mbox{mass}} /f$ */
    Plato::Scalar mStressCriterionWeight; /*!< weight for constraint term, i.e. /f$ \alpha_{\mbox{constraint}} * f_{\mbox{constraint}} /f$ */
    Plato::Scalar mAugLagPenaltyUpperBound; /*!< upper bound on augmented Lagrangian penalty */
    Plato::Scalar mMassNormalizationMultiplier; /*!< normalization multipliers for mass criterion */
    Plato::Scalar mInitialLagrangeMultipliersValue; /*!< initial value for Lagrange multipliers */
    Plato::Scalar mAugLagPenaltyExpansionMultiplier; /*!< expansion parameter for augmented Lagrangian penalty */

    Plato::ScalarVector mLagrangeMultipliers; /*!< Lagrange multipliers */
    Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix; /*!< cell/element Lame constants matrix */

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
     * \brief Update augmented Lagrangian penalty and upper bound on mass multipliers.
    **********************************************************************************/
    void updateAugLagPenaltyMultipliers();

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     **********************************************************************************/
    AugLagStressCriterionGeneral(
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
    AugLagStressCriterionGeneral(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    );

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~AugLagStressCriterionGeneral();

    /******************************************************************************//**
     * \brief Return augmented Lagrangian penalty multiplier
     * \return augmented Lagrangian penalty multiplier
    **********************************************************************************/
    Plato::Scalar getAugLagPenalty() const;

    /******************************************************************************//**
     * \brief Return multiplier used to normalized mass contribution to the objective function
     * \return upper mass normalization multiplier
    **********************************************************************************/
    Plato::Scalar getMassNormalizationMultiplier() const;

    /******************************************************************************//**
     * \brief Return Lagrange multipliers
     * \return 1D view of Lagrange multipliers
    **********************************************************************************/
    Plato::ScalarVector getLagrangeMultipliers() const;

    /******************************************************************************//**
     * \brief Set stress constraint limit/upper bound
     * \param [in] aInput stress constraint limit
    **********************************************************************************/
    void setStressLimit(const Plato::Scalar & aInput);

    /******************************************************************************//**
     * \brief Set augmented Lagrangian function penalty multiplier
     * \param [in] aInput penalty multiplier
     **********************************************************************************/
    void setAugLagPenalty(const Plato::Scalar & aInput);

    /******************************************************************************//**
     * \brief Set cell material density
     * \param [in] aInput material density
     **********************************************************************************/
    void setCellMaterialDensity(const Plato::Scalar & aInput);

    /******************************************************************************//**
     * \brief Set Lagrange multipliers
     * \param [in] aInput Lagrange multipliers
     **********************************************************************************/
    void setLagrangeMultipliers(const Plato::ScalarVector & aInput);

  /// @brief set fourth-order material constitutive matrix
  /// @param [in] aInput plato matrix VoigTerms X VoigtTerms 
  void 
  setCellStiffMatrix(
    const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> & aInput
  );

  /// @fn isLinear
  /// @brief returns true if criterion is linear
  /// @return boolean
  bool 
  isLinear() 
  const;
  
  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in] aWorkSets function domain and range workset database
  /// @param [in] aCycle    scalar 
  void 
  updateProblem(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) override;

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in] aWorkSets function domain and range workset database
  /// @param [in] aCycle    scalar 
  void
  evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) 
  const;

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in] aWorkSets function domain and range workset database
  /// @param [in] aCycle    scalar 
  void updateLagrangeMultipliers(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  );

  /// @fn computeStructuralMass
  /// @brief compute structral mass
  void 
  computeStructuralMass();

};
// class AugLagStressCriterionGeneral

}// namespace Plato
