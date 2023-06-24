#pragma once

#include "base/WorksetBase.hpp"
#include "elliptic/evaluators/criterion/FactoryCriterionEvaluator.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorDivision.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorScalarFunction.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Volume average criterion class
 **********************************************************************************/
template<typename PhysicsType>
class CriterionEvaluatorVolumeAverage :
  public Plato::Elliptic::CriterionEvaluatorBase
{
private:
  /// @brief topological element type
  using ElementType = typename PhysicsType::ElementType;
  /// @brief scalar types associated with the automatic differentation evaluation types
  using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using GradientU = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
  using GradientX = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
  using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
  using GradientN = typename Plato::Elliptic::Evaluation<ElementType>::GradientN;
  /// @brief division function evaluator
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorDivision<PhysicsType>> mDivisionFunction;
  /// @brief contains mesh and model information
  const Plato::SpatialModel & mSpatialModel;
  /// @brief output database
  Plato::DataMap& mDataMap;
  /// @brief criterion function name
  std::string mFunctionName;
  /// @brief spatial weighting function string of x, y, z coordinates  
  std::string mSpatialWeightingFunctionString = "1.0";

  /******************************************************************************//**
   * \brief Initialization of Volume Average Criterion
   * \param [in] aInputParams input parameters database
  **********************************************************************************/
  void 
  initialize(
    Teuchos::ParameterList & aInputParams
  );

  /******************************************************************************//**
   * \brief Create the volume function only
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aInputParams parameter list
   * \return physics scalar function
  **********************************************************************************/
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>>
  getVolumeFunction(
      const Plato::SpatialModel & aSpatialModel,
      Teuchos::ParameterList & aInputParams
  );

  /******************************************************************************//**
   * \brief Create the division function
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aInputParams parameter list
  **********************************************************************************/
  void
  createDivisionFunction(
      const Plato::SpatialModel & aSpatialModel,
      Teuchos::ParameterList & aInputParams
  );

public:
  /******************************************************************************//**
   * \brief Primary volume average criterion constructor
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aDataMap PLATO Engine and Analyze data map
   * \param [in] aInputParams input parameters database
   * \param [in] aName user defined function name
  **********************************************************************************/
  CriterionEvaluatorVolumeAverage(
      const Plato::SpatialModel    & aSpatialModel,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aInputParams,
            std::string            & aName
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
  /// @brief evaluate criterion
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return scalar
  Plato::Scalar
  value(const Plato::Database & aDatabase,
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
  /// @brief compute partial derivative with respect to the controls
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientControl(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /******************************************************************************//**
   * \brief Return user defined function name
   * \return User defined function name
  **********************************************************************************/
  std::string 
  name() const;
};
// class CriterionEvaluatorVolumeAverage

} // namespace Elliptic

} // namespace Plato
