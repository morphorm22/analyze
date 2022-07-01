#pragma once

#include "PlatoStaticsTypes.hpp"
#include "geometric/WorksetBase.hpp"
#include "geometric/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * \brief Weighted sum function class \f$ F(x) = \sum_{i = 1}^{n} w_i * f_i(x) \f$
 **********************************************************************************/
template<typename PhysicsType>
class WeightedSumFunction :
    public Plato::Geometric::ScalarFunctionBase,
    public Plato::Geometric::WorksetBase<typename PhysicsType::ElementType>
{
private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::Geometric::WorksetBase<ElementType>::mNumNodesPerCell;
    using Plato::Geometric::WorksetBase<ElementType>::mNumSpatialDims;
    using Plato::Geometric::WorksetBase<ElementType>::mNumNodes;
    using Plato::Geometric::WorksetBase<ElementType>::mNumCells;

    std::vector<Plato::Scalar> mFunctionWeights;
    std::vector<std::shared_ptr<Plato::Geometric::ScalarFunctionBase>> mScalarFunctionBaseContainer;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;

    std::string mFunctionName;

	/******************************************************************************//**
     * \brief Initialization of Weighted Sum Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aProblemParams);

public:
    /******************************************************************************//**
     * \brief Primary weight sum function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    WeightedSumFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    );

    /******************************************************************************//**
     * \brief Secondary weight sum function constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    WeightedSumFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    );

    /******************************************************************************//**
     * \brief Add function weight
     * \param [in] aWeight function weight
    **********************************************************************************/
    void appendFunctionWeight(Plato::Scalar aWeight);

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
     * \brief Evaluate weight sum function
     * \param [in] aControl 1D view of control variables
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the weight sum function with respect to (wrt) the configuration parameters
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the weight sum function with respect to (wrt) the control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aControl) const override;

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
// class WeightedSumFunction

} // namespace Geometric

} // namespace Plato
