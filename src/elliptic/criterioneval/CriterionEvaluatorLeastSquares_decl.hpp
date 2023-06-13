#pragma once

#include "base/WorksetBase.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Least Squares function class \f$ F(x) = \sum_{i = 1}^{n} w_i * (f_i(x) - gold_i(x))^2 \f$
 **********************************************************************************/
template<typename PhysicsType>
class CriterionEvaluatorLeastSquares :
    public Plato::Elliptic::CriterionEvaluatorBase
{
private:
  using ElementType = typename PhysicsType::ElementType;
  
  static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

  std::vector<Plato::Scalar> mFunctionWeights;
  std::vector<Plato::Scalar> mFunctionGoldValues;
  std::vector<Plato::Scalar> mFunctionNormalization;
  std::vector<std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>> mScalarFunctionBaseContainer;
  
  const Plato::SpatialModel & mSpatialModel;
  Plato::DataMap& mDataMap;
  std::string mFunctionName;
  bool mGradientWRTStateIsZero = false;
  
  /*!< if (|GoldValue| > 0.1) then ((f - f_gold) / f_gold)^2 ; otherwise  (f - f_gold)^2 */
  const Plato::Scalar mFunctionNormalizationCutoff = 0.1;
  
  /******************************************************************************//**
   * \brief Initialization of Least Squares Function
   * \param [in] aProblemParams input parameters database
  **********************************************************************************/
  void 
  initialize(
    Teuchos::ParameterList & aProblemParams
  );

public:
    /******************************************************************************//**
     * \brief Primary least squares function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    CriterionEvaluatorLeastSquares(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    );

    /******************************************************************************//**
     * \brief Secondary least squares function constructor, used for unit testing / mass properties
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    CriterionEvaluatorLeastSquares(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    );

    /******************************************************************************//**
     * \brief Add function weight
     * \param [in] aWeight function weight
    **********************************************************************************/
    void appendFunctionWeight(Plato::Scalar aWeight);

    /******************************************************************************//**
     * \brief Add function gold value
     * \param [in] aGoldValue function gold value
     * \param [in] aUseAsNormalization use gold value as normalization
    **********************************************************************************/
    void appendGoldFunctionValue(Plato::Scalar aGoldValue, bool aUseAsNormalization = true);

    /******************************************************************************//**
     * \brief Add function normalization
     * \param [in] aFunctionNormalization function normalization value
    **********************************************************************************/
    void appendFunctionNormalization(Plato::Scalar aFunctionNormalization);

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
    /// @brief update criterion parameters of the least square function at runtime
    /// @param [in] aDatabase function domain and range database
    /// @param [in] aCycle    scalar, e.g.; time step
    void 
    updateProblem(
      const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
    ) const;

    /// @fn value
    /// @brief Evaluate least square function 
    /// @param [in] aDatabase function domain and range database
    /// @param [in] aCycle    scalar, e.g.; time step
    /// @return scalar
    Plato::Scalar
    value(const Plato::Database & aDatabase,
        const Plato::Scalar   & aCycle
    ) const;

    /// @fn gradientConfig
    /// @brief Evaluate partial derivative of the least square function with respect to the configuration
    /// @param [in] aDatabase function domain and range database
    /// @param [in] aCycle    scalar, e.g.; time step
    /// @return plato scalar vector
    Plato::ScalarVector
    gradientConfig(
      const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
    ) const ;

    /// @fn gradientState
    /// @brief Evaluate partial derivative of the least squares function with respect to the states
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
    /// @brief compute partial derivative of the least squares function with respect to the controls
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
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    std::string name() const;

    /******************************************************************************//**
     * \brief Set gradient wrt state flag
     * \return Gradient WRT State is zero flag
    **********************************************************************************/
    void setGradientWRTStateIsZeroFlag(bool aGradientWRTStateIsZero);
};
// class CriterionEvaluatorLeastSquares

} // namespace Elliptic

} // namespace Plato
