#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Teuchos_ParameterList.hpp>

#include "base/WorksetBase.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorBase.hpp"
#include "elliptic/evaluators/criterion/FactoryCriterionEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Division function class \f$ F(x) = numerator(x) / denominator(x) \f$
 **********************************************************************************/
template<typename PhysicsType>
class CriterionEvaluatorDivision :
    public Plato::Elliptic::CriterionEvaluatorBase
{
private:
  /// @brief topological element type
  using ElementType = typename PhysicsType::ElementType;
  /// @brief number of state degrees of freedom per node
  static constexpr auto mNumDofsPerNode      = ElementType::mNumDofsPerNode;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims      = ElementType::mNumSpatialDims;
  /// @brief number of node state degrees of freedom per node
  static constexpr auto mNumNodeStatePerNode = ElementType::mNumNodeStatePerNode;
  /// @brief numerator criterion evaluator
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase> mScalarFunctionBaseNumerator;
  /// @brief denominator criterion evaluator
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase> mScalarFunctionBaseDenominator;
  /// @brief contains mesh and model information
  const Plato::SpatialModel & mSpatialModel;
  /// @brief output database
  Plato::DataMap& mDataMap;
  /// @brief criterion function name
  std::string mFunctionName;
  
  /// @fn initialize
  /// @brief initialize member data
  /// @param aProblemParams input problem parameters
  void 
  initialize(
    Teuchos::ParameterList & aProblemParams
  );

public:
  /******************************************************************************//**
   * \brief Primary division function constructor
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aDataMap Plato Analyze data map
   * \param [in] aInputParams input parameters database
   * \param [in] aName user defined function name
  **********************************************************************************/
  CriterionEvaluatorDivision(
      const Plato::SpatialModel    & aSpatialModel,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aProblemParams,
      const std::string            & aName
  );

  /******************************************************************************//**
   * \brief Secondary division function constructor, used for unit testing
   * \param [in] aMesh mesh database
  **********************************************************************************/
  CriterionEvaluatorDivision(
      const Plato::SpatialModel & aSpatialModel,
            Plato::DataMap      & aDataMap
  );

  /******************************************************************************//**
   * \brief Allocate numerator function base using the residual automatic differentiation type
   * \param [in] aInput scalar function
  **********************************************************************************/
  void 
  allocateNumeratorFunction(
    const std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>& aInput
  );

  /******************************************************************************//**
   * \brief Allocate denominator function base using the residual automatic differentiation type
   * \param [in] aInput scalar function
  **********************************************************************************/
  void 
  allocateDenominatorFunction(
    const std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>& aInput
  );

  /// @fn isLinear
  /// @brief return true if scalar function is linear
  /// @return boolean
  bool 
  isLinear() 
  const;

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  void 
  updateProblem(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn value
  /// @brief evaluate division function criterion
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return scalar
  Plato::Scalar
  value(const Plato::Database & aDatabase,
        const Plato::Scalar   & aCycle
  ) const;

  /// @fn gradientConfig
  /// @brief compute partial derivative of the division function with respect to the configuration
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientConfig(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn gradientState
  /// @brief compute partial derivative of the division function with respect to the states
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn gradientNodeState
  /// @brief compute partial derivative with respect to the node states
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientNodeState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn gradientControl
  /// @brief compute partial derivative of the division function with respect to the controls
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientControl(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn setFunctionName
  /// @brief set criterion function name 
  /// @param aFunctionName user defined function name
  void 
  setFunctionName(
    const std::string aFunctionName
  );

  /// @fn name
  /// @brief return criterion function name
  /// @return string
  std::string 
  name() 
  const;
};
// class CriterionEvaluatorDivision

} // namespace Elliptic

} // namespace Plato
