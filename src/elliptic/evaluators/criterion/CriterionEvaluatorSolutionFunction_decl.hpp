#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include "SpatialModel.hpp"
#include "base/WorksetBase.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Solution function class
 **********************************************************************************/
template<typename PhysicsType>
class CriterionEvaluatorSolutionFunction :
  public Plato::Elliptic::CriterionEvaluatorBase
{
    enum solution_type_t
    {
        UNKNOWN_TYPE = 0,
        SOLUTION_IN_DIRECTION = 1,
        SOLUTION_MAG_IN_DIRECTION = 2,
        DIFF_BETWEEN_SOLUTION_MAG_IN_DIRECTION_AND_TARGET = 3,
        DIFF_BETWEEN_SOLUTION_VECTOR_AND_TARGET_VECTOR = 4,
        DIFF_BETWEEN_SOLUTION_IN_DIRECTION_AND_TARGET_SOLUTION_IN_DIRECTION = 5
    };

private:
  using ElementType = typename PhysicsType::ElementType;

  static constexpr auto mNumNodesPerCell     = ElementType::mNumNodesPerCell;
  static constexpr auto mNumNodesPerFace     = ElementType::mNumNodesPerFace;
  static constexpr auto mNumDofsPerNode      = ElementType::mNumDofsPerNode;
  static constexpr auto mNumDofsPerCell      = ElementType::mNumDofsPerCell;
  static constexpr auto mNumSpatialDims      = ElementType::mNumSpatialDims;
  static constexpr auto mNumControl          = ElementType::mNumControl;
  /// @brief number of node state degrees of freedom per node
  static constexpr auto mNumNodeStatePerNode = ElementType::mNumNodeStatePerNode;

  std::string mFunctionName; /*!< User defined function name */
  std::string mDomainName;   /*!< Name of the node set that represents the domain of interest */
  
  Plato::Array<mNumDofsPerNode> mNormal;  /*!< Direction of solution criterion */
  Plato::Array<mNumDofsPerNode> mTargetSolutionVector;  /*!< Target solution vector */
  
  Plato::Scalar mTargetMagnitude; /*!< Target magnitude */
  Plato::Scalar mTargetSolution; /*!< Target solution */
  
  bool mMagnitudeSpecified;
  bool mNormalSpecified;
  bool mTargetSolutionVectorSpecified;
  bool mTargetMagnitudeSpecified;
  bool mTargetSolutionSpecified;
  
  const Plato::SpatialModel & mSpatialModel;
  solution_type_t mSolutionType;
  
  /******************************************************************************//**
   * \brief Initialization of Solution Function
   * \param [in] aProblemParams input parameters database
  **********************************************************************************/
  void
  initialize(
    Teuchos::ParameterList & aProblemParams
  );

  void 
  initialize_target_vector(
    Teuchos::ParameterList &aFunctionParams
  );

  void 
  initialize_normal_vector(
    Teuchos::ParameterList &aFunctionParams
  );

public:
  /******************************************************************************//**
   * \brief Primary solution function constructor
   * \param [in] aMesh mesh database
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aDataMap Plato Analyze data map
   * \param [in] aProblemParams input parameters database
   * \param [in] aName user defined function name
  **********************************************************************************/
  CriterionEvaluatorSolutionFunction(
      const Plato::SpatialModel    & aSpatialModel,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aProblemParams,
            std::string            & aName
  );

  /// @fn isLinear
  /// @brief return true if scalar function is linear
  /// @return boolean
  bool 
  isLinear() 
  const;

  /// @fn value
  /// @brief evaluate solution criterion
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return scalar
  Plato::Scalar
  value(const Plato::Database & aDatabase,
        const Plato::Scalar   & aCycle
  ) const;

  /// @fn gradientConfig
  /// @brief compute partial derivative of the solution function with respect to the configuration
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientConfig(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn gradientState
  /// @brief compute partial derivative of the solution function with respect to the states
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
  /// @brief compute partial derivative of the solution function with respect to the controls
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientControl(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  void 
  updateProblem(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /******************************************************************************//**
   * \brief Set user defined function name
   * \param [in] function name
  **********************************************************************************/
  void 
  setFunctionName(
    const std::string aFunctionName
  );

  /******************************************************************************//**
   * \brief Return user defined function name
   * \return User defined function name
  **********************************************************************************/
  std::string 
  name() 
  const;
};
// class CriterionEvaluatorSolutionFunction

} // namespace Elliptic

} // namespace Plato
