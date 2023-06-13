#pragma once

#include "base/WorksetBase.hpp"
#include "elliptic/criterioneval/FactoryCriterionEvaluator.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorDivision.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorScalarFunction.hpp"

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
    using ElementType = typename PhysicsType::ElementType;

    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientU = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using GradientX = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;

    std::shared_ptr<Plato::Elliptic::CriterionEvaluatorDivision<PhysicsType>> mDivisionFunction;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

    std::string mSpatialWeightingFunctionString = "1.0"; /*!< Spatial weighting function string of x, y, z coordinates  */

    /******************************************************************************//**
     * \brief Initialization of Volume Average Criterion
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams);

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
