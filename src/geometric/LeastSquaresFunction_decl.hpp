#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include "geometric/WorksetBase.hpp"
#include "geometric/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * \brief Least Squares function class \f$ F(x) = \sum_{i = 1}^{n} w_i * (f_i(x) - gold_i(x))^2 \f$
 **********************************************************************************/
template<typename PhysicsType>
class LeastSquaresFunction :
    public Plato::Geometric::ScalarFunctionBase,
    public Plato::Geometric::WorksetBase<typename PhysicsType::ElementType>
{
private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::Geometric::WorksetBase<ElementType>::mNumSpatialDims;
    using Plato::Geometric::WorksetBase<ElementType>::mNumNodes;

    std::vector<Plato::Scalar> mFunctionWeights;
    std::vector<Plato::Scalar> mFunctionGoldValues;
    std::vector<Plato::Scalar> mFunctionNormalization;
    std::vector<std::shared_ptr<Plato::Geometric::ScalarFunctionBase>> mScalarFunctionBaseContainer;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;

    std::string mFunctionName;

    /*!< if (|GoldValue| > 0.1) then ((f - f_gold) / f_gold)^2 ; otherwise  (f - f_gold)^2 */
    const Plato::Scalar mFunctionNormalizationCutoff = 0.1;

    /******************************************************************************//**
     * \brief Initialization of Least Squares Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void
    initialize(Teuchos::ParameterList & aProblemParams);

public:
    /******************************************************************************//**
     * \brief Primary least squares function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    LeastSquaresFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
        const std::string            & aName
    );

    /******************************************************************************//**
     * \brief Secondary least squares function constructor, used for unit testing / mass properties
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    LeastSquaresFunction(
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
    void allocateScalarFunctionBase(const std::shared_ptr<Plato::Geometric::ScalarFunctionBase>& aInput);

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Evaluate least squares function
     * \param [in] aControl 1D view of control variables
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the least squares function with respect to (wrt) the configuration parameters
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the least squares function with respect to (wrt) the control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    std::string name() const;
};
// class LeastSquaresFunction

} // namespace Geometric

} // namespace Plato
