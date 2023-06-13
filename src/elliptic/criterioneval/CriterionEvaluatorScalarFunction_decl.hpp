#pragma once

#include "base/WorksetBase.hpp"
#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Physics scalar function class
 **********************************************************************************/
template<typename PhysicsType>
class CriterionEvaluatorScalarFunction : 
  public Plato::Elliptic::CriterionEvaluatorBase
{
private:
  /// @brief local topological element typename
  using ElementType = typename PhysicsType::ElementType;

  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  static constexpr auto mNumNodesPerFace = ElementType::mNumNodesPerFace;
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  static constexpr auto mNumControl      = ElementType::mNumControl;

  using ValueEvalType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using GradUEvalType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
  using GradXEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
  using GradZEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;

  /*!< scalar function value interface */
  std::map<std::string, std::shared_ptr<Plato::CriterionBase>> mValueFunctions;     
  /*!< scalar function value partial wrt states */
  std::map<std::string, std::shared_ptr<Plato::CriterionBase>> mGradientUFunctions; 
  /*!< scalar function value partial wrt configuration */
  std::map<std::string, std::shared_ptr<Plato::CriterionBase>> mGradientXFunctions;
  /*!< scalar function value partial wrt controls */
  std::map<std::string, std::shared_ptr<Plato::CriterionBase>> mGradientZFunctions; 

  /// @brief contains mesh and model information
  const Plato::SpatialModel & mSpatialModel;
  /// @brief interface to workset builders
  Plato::WorksetBase<ElementType> mWorksetFuncs;
  /// @brief output database
  Plato::DataMap& mDataMap; 
  /// @brief criterion function name
  std::string mFunctionName;

private:
  /// @brief initialize member data
  /// @param [in] aProblemParams input problem parameters
  void 
  initialize(
    Teuchos::ParameterList & aProblemParams
  );

public:
  /// @brief class constructor
  /// @param [in] aSpatialModel  contains mesh and model information
  /// @param [in] aDataMap       output database
  /// @param [in] aProblemParams input problem parameters 
  /// @param [in] aName          criterion function name
  CriterionEvaluatorScalarFunction(
    const Plato::SpatialModel    & aSpatialModel,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string            & aName
  );

  /// @brief class constructor
  /// @param [in] aSpatialModel contains mesh and model information
  /// @param [in] aDataMap      output database
  CriterionEvaluatorScalarFunction(
    const Plato::SpatialModel & aSpatialModel,
          Plato::DataMap      & aDataMap
  );

  /// @fn isLinear
  /// @brief return true if scalar function is linear
  /// @return boolean
  bool 
  isLinear() 
  const;

  /// @fn setEvaluator
  /// @brief set criterion evaluator type. function used in composite criterion evaluators
  /// @param aEvalType   evaluation type
  /// @param aCriterion  criterion evaluator
  /// @param aDomainName domain name, i.e.; element block name
  void
  setEvaluator(
    const Plato::Elliptic::evaluator_t          & aEvalType,
    const std::shared_ptr<Plato::CriterionBase> & aCriterion,
    const std::string                           & aDomainName
  );

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
  /// @brief evaluate criterion
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return scalar
  Plato::Scalar
  value(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn gradientConfig
  /// @brief compute partial derivative with respect to the configuration
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientConfig(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn gradientState
  /// @brief compute partial derivative with respect to the states
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn gradientControl
  /// @brief compute partial derivative with respect to the controls
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientControl(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @brief set criterion function name
  /// @param aFunctionName function name
  void 
  setFunctionName(
    const std::string aFunctionName
  );

  /// @brief set criterion function name
  /// @return string
  std::string 
  name() 
  const;
};
//class CriterionEvaluatorScalarFunction

} // namespace Elliptic

} // namespace Plato
