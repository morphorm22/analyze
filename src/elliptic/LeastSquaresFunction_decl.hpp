#pragma once

#include "WorksetBase.hpp"
#include "elliptic/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Least Squares function class \f$ F(x) = \sum_{i = 1}^{n} w_i * (f_i(x) - gold_i(x))^2 \f$
 **********************************************************************************/
template<typename PhysicsType>
class LeastSquaresFunction :
    public Plato::Elliptic::ScalarFunctionBase,
    public Plato::WorksetBase<typename PhysicsType::ElementType>
{
private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumDofsPerNode;
    using Plato::WorksetBase<ElementType>::mNumSpatialDims;
    using Plato::WorksetBase<ElementType>::mNumNodes;

    std::vector<Plato::Scalar> mFunctionWeights;
    std::vector<Plato::Scalar> mFunctionGoldValues;
    std::vector<Plato::Scalar> mFunctionNormalization;
    std::vector<std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>> mScalarFunctionBaseContainer;

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
    void initialize (Teuchos::ParameterList & aProblemParams);

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
              std::string            & aName
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
    void allocateScalarFunctionBase(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aInput);

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Evaluate least squares function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::Solutions    & aSolution,
                        const Plato::ScalarVector & aControl,
                              Plato::Scalar         aTimeStep = 0.0) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the least squares function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::Solutions    & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::Scalar         aTimeStep = 0.0) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the least squares function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::Solutions    & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::OrdinalType    aStepIndex,
                                         Plato::Scalar         aTimeStep = 0.0) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the least squares function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::Solutions    & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::Scalar         aTimeStep = 0.0) const override;

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
// class LeastSquaresFunction

} // namespace Elliptic

} // namespace Plato
