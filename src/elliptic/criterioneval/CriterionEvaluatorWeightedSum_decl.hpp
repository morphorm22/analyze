#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include "base/WorksetBase.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Weighted sum function class \f$ F(x) = \sum_{i = 1}^{n} w_i * f_i(x) \f$
 **********************************************************************************/
template<typename PhysicsType>
class CriterionEvaluatorWeightedSum :
  public Plato::Elliptic::CriterionEvaluatorBase
{
private:
  using ElementType = typename PhysicsType::ElementType;

  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  static constexpr auto mNumNodesPerFace = ElementType::mNumNodesPerFace;
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  static constexpr auto mNumControl      = ElementType::mNumControl;

  std::vector<std::string> mFunctionNames;
  std::vector<Plato::Scalar> mFunctionWeights;
  std::vector<std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>> mScalarFunctionBaseContainer;
  
  const Plato::SpatialModel & mSpatialModel;
  Plato::DataMap& mDataMap;
  std::string mFunctionName;

	/******************************************************************************//**
   * \brief Initialization of Weighted Sum Function
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aProblemParams input parameters database
  **********************************************************************************/
  void
  initialize(
    Teuchos::ParameterList & aProblemParams
  );

public:
    /******************************************************************************//**
     * \brief Primary weight sum function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    CriterionEvaluatorWeightedSum(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    );

    /******************************************************************************//**
     * \brief Secondary weight sum function constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
    **********************************************************************************/
    CriterionEvaluatorWeightedSum(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    );

    /******************************************************************************//**
     * \brief Add function weight
     * \param [in] aWeight function weight
    **********************************************************************************/
    void appendFunctionWeight(Plato::Scalar aWeight);

    /******************************************************************************//**
     * \brief Add function name to list of function names
     * \param [in] aName function weight
    **********************************************************************************/
    void appendFunctionName(const std::string & aName);

    /******************************************************************************//**
     * \brief Allocate scalar function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void allocateScalarFunctionBase(const std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>& aInput);

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
    virtual 
    void 
    updateProblem(
      const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
    ) const;

    /// @fn value
    /// @brief evaluate weighted sum criterion
    /// @param [in] aDatabase function domain and range database
    /// @param [in] aCycle    scalar, e.g.; time step
    /// @return scalar
    virtual 
    Plato::Scalar
    value(const Plato::Database & aDatabase,
          const Plato::Scalar   & aCycle
    ) const;

    /// @fn gradientConfig
    /// @brief compute partial derivative of the weighted sum function with respect to the configuration
    /// @param [in] aDatabase function domain and range database
    /// @param [in] aCycle    scalar, e.g.; time step
    /// @return plato scalar vector
    virtual 
    Plato::ScalarVector
    gradientConfig(
      const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
    ) const;

    /// @fn gradientState
    /// @brief compute partial derivative of the weighted sum function with respect to the states
    /// @param [in] aDatabase function domain and range database
    /// @param [in] aCycle    scalar, e.g.; time step
    /// @return plato scalar vector
    virtual 
    Plato::ScalarVector
    gradientState(
      const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
    ) const;

    /// @fn gradientControl
    /// @brief compute partial derivative of the weighted sum function with respect to the controls
    /// @param [in] aDatabase function domain and range database
    /// @param [in] aCycle    scalar, e.g.; time step
    /// @return plato scalar vector
    virtual 
    Plato::ScalarVector
    gradientControl(
      const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
    ) const;

    /******************************************************************************//**
     * \brief Set user defined function name
     * \param [in] function name
    **********************************************************************************/
    void setFunctionName(const std::string aFunctionName);

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    std::string name() const;
};
// class CriterionEvaluatorWeightedSum

} // namespace Elliptic

} // namespace Plato
