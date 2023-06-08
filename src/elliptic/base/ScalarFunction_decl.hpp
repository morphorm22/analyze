/*
 * ScalarFunction_decl.hpp
 *
 *  Created on: June 7, 2023
 */

#pragma once

#include "SpatialModel.hpp"
#include "base/WorksetBase.hpp"
#include "base/CriterionBase.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/base/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Elliptic
{

/// @class ScalarFunction
/// @brief interface for the evaluation of scalar functions in elliptic problems
/// @tparam PhysicsType defines physics-based quantity of interests 
template<typename PhysicsType>
class ScalarFunction : public ScalarFunctionBase
{
private:
  /// @brief topological element type
  using ElementType = typename PhysicsType::ElementType;
  /// @brief number of nodes per element
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of nodes per element face
  static constexpr auto mNumNodesPerFace = ElementType::mNumNodesPerFace;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per element
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of control degree of freedoms per node
  static constexpr auto mNumControl      = ElementType::mNumControl;
  /// @brief scalar types for a given evaluation type
  using ValueEvalType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using GradUEvalType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
  using GradXEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
  using GradZEvalType = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
  /// @brief map from domain name to criterion value evaluator
  std::unordered_map<std::string, std::shared_ptr<Plato::CriterionBase>> mValueFunctions;
  /// @brief map from domain name to criterion gradient with respect to states evaluator
  std::unordered_map<std::string, std::shared_ptr<Plato::CriterionBase>> mGradientUFunctions; 
  /// @brief map from domain name to criterion gradient with respect to configuration evaluator
  std::unordered_map<std::string, std::shared_ptr<Plato::CriterionBase>> mGradientXFunctions; 
  /// @brief map from domain name to criterion gradient with respect to control evaluator
  std::unordered_map<std::string, std::shared_ptr<Plato::CriterionBase>> mGradientZFunctions; 
  /// @brief local typename for a criterion base standard shared pointer
  using CriterionType = std::shared_ptr<Plato::CriterionBase>;
  /// @brief output database
  Plato::DataMap & mDataMap;
  /// @brief contains mesh and model information
  const Plato::SpatialModel & mSpatialModel;
  /// @brief interface to workset constructors 
  Plato::WorksetBase<ElementType> mWorksetFuncs;
  /// @brief criterion name
  std::string mName;

public:
  /// @brief class constructor
  /// @param aFuncName     criterion name
  /// @param aSpatialModel contains mesh and model information
  /// @param aDataMap      output database
  /// @param aProbParams   input problem parameters
  ScalarFunction(
    const std::string            & aFuncName,
    const Plato::SpatialModel    & aSpatialModel,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProbParams
  );

  /// class destructor
  ~ScalarFunction(){}

  /// @fn setEvaluator
  /// @brief set criterion evaluator type. function used in composite criterion evaluators
  /// @param aEvalType   evaluation type
  /// @param aCriterion  criterion evaluator
  /// @param aDomainName domain name, i.e.; element block name
  void
  setEvaluator(
    const evaluator_t   & aEvalType,
    const CriterionType & aCriterion,
    const std::string   & aDomainName
  );

  /// @fn name
  /// @brief return criterion name
  /// @return string
  std::string name() const;

  /// @fn isLinear
  /// @brief return true if scalar function is linear
  /// @return boolean
  bool isLinear() const;

  /// @fn value
  /// @brief evaluate scalar function
  /// @param aDatabase output database
  /// @param aCycle    scalar, e.g.; time step
  /// @return scalar
  Plato::Scalar
  value(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  );

  /// @fn gradientConfig
  /// @brief compute the partial derivative of the scalar function with respect to configuration
  /// @param aDatabase output database
  /// @param aCycle    scalar, e.g.; time step
  /// @return scalar plato vector
  Plato::ScalarVector
  gradientConfig(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  );

  /// @fn gradientState
  /// @brief compute the partial derivative of the scalar function with respect to states
  /// @param aDatabase output database
  /// @param aCycle    scalar, e.g.; time step
  /// @return scalar plato vector
  Plato::ScalarVector
  gradientState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  );

  /// @fn gradientControl
  /// @brief compute the partial derivative of the scalar function with respect to controls
  /// @param aDatabase output database
  /// @param aCycle    scalar, e.g.; time step
  /// @return scalar plato vector
  Plato::ScalarVector
  gradientControl(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  );

private:
  /// @fn initialize
  /// @brief initialize member data
  /// @param aProbParams input problem parameters
  void initialize(
    Teuchos::ParameterList & aProbParams
  );
};

} // namespace Elliptic
    
} // namespace Plato