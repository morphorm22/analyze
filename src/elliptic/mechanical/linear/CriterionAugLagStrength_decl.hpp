/*
 * CriterionAugLagStrength_decl.hpp
 *
 *  Created on: May 4, 2023
 */

#pragma once

#include <string>
#include <memory>

#include <Teuchos_ParameterList.hpp>

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/AbstractLocalMeasure.hpp"

#include "base/CriterionBase.hpp"
#include "optimizer/AugLagDataMng.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Apply augmented Lagragian method to model local constraints.
 * \tparam EvaluationType evaluation type for automatic differentiation tools; e.g., 
 *         type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class CriterionAugLagStrength : public Plato::CriterionBase
{
private:
  /// @brief topologcial element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of voigt stress-strain terms 
  static constexpr auto mNumVoigtTerms   = ElementType::mNumVoigtTerms;

  using FunctionBaseType = typename Plato::CriterionBase;
  using FunctionBaseType::mSpatialDomain;
  using FunctionBaseType::mDataMap;

  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
  /// @brief local measure evaluation type
  using Residual = typename Plato::Elliptic::ResidualTypes<ElementType>;

  Plato::Scalar mMaterialPenalty = 3.0;         /*!< penalty for material penalty model */
  Plato::Scalar mMinErsatzMaterialValue = 1e-9; /*!< minimum ersatz material stiffness for material penalty model*/

  Plato::AugLagDataMng mAugLagDataMng;    /*!< contains all relevant data associated with the AL method*/

  /*!< Local measure with FAD evaluation type */
  std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType>> mLocalMeasureEvaluationType;

  /*!< Local measure with POD type */
  std::shared_ptr<Plato::AbstractLocalMeasure<Residual>> mLocalMeasurePODType;

  /*!< plot table with output quantities of interests */
  std::vector<std::string> mPlotTable;

public:
  /// @brief class constructor
  /// @param aSpatialDomain contains mesh and model information
  /// @param aDataMap       output database
  /// @param aParams        input problem parameters
  /// @param aFuncName      criterion parameter list name
  CriterionAugLagStrength(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParams,
    const std::string            & aFuncName
  );

  /// @brief class destructor
  ~CriterionAugLagStrength(){}

  void 
  setLocalMeasure(
    const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInputEvaluationType,
    const std::shared_ptr<AbstractLocalMeasure<Residual>>       & aInputPODType
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

  void
  evaluateConditional(
    const Plato::WorkSets & aWorkSets,
    const Plato::Scalar   & aCycle
  ) const;

  void 
  evaluateCurrentConstraints(
    const Plato::ScalarMultiVector &aStateWS,
    const Plato::ScalarMultiVector &aControlWS,
    const Plato::ScalarArray3D     &aConfigWS
  );

private:
  /******************************************************************************//**
   * \brief Allocate member data
   * \param [in] aParams input parameters database
  **********************************************************************************/
  void 
  initialize(
    Teuchos::ParameterList & aParams
  );
  /******************************************************************************//**
   * \brief Parse numeric inputs from input file
   * \param [in] aParams input parameters database
  **********************************************************************************/
  void 
  parseNumerics(
    Teuchos::ParameterList & aParams
  );
  /******************************************************************************//**
   * \brief Parse limits on strength constraint
   * \param [in] aParams input parameters database
  **********************************************************************************/
  void 
  parseLimits(
    Teuchos::ParameterList & aParams
  );

};

} // namespace Elliptic

} // namespace Plato
