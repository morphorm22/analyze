#pragma once

#include "Solutions.hpp"
#include "WorksetBase.hpp"
#include "PlatoSequence.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/hatching/EvaluationTypes.hpp"
#include "elliptic/hatching/ScalarFunctionBase.hpp"
#include "elliptic/hatching/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

/******************************************************************************//**
 * \brief Physics scalar function class
 **********************************************************************************/
template<typename PhysicsType>
class PhysicsScalarFunction :
    public ScalarFunctionBase,
    public Plato::WorksetBase<typename PhysicsType::ElementType>
{
private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumDofsPerCell;
    using Plato::WorksetBase<ElementType>::mNumLocalDofsPerCell;
    using Plato::WorksetBase<ElementType>::mNumLocalStatesPerGP;
    using Plato::WorksetBase<ElementType>::mNumNodesPerCell;
    using Plato::WorksetBase<ElementType>::mNumDofsPerNode;
    using Plato::WorksetBase<ElementType>::mNumSpatialDims;
    using Plato::WorksetBase<ElementType>::mNumGaussPoints;
    using Plato::WorksetBase<ElementType>::mNumNodes;
    using Plato::WorksetBase<ElementType>::mNumCells;

    using Plato::WorksetBase<ElementType>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<ElementType>::mControlEntryOrdinal;
    using Plato::WorksetBase<ElementType>::mConfigEntryOrdinal;

    using Residual  = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::Residual;
    using GradientU = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::Jacobian;
    using GradientC = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::GradientC;
    using GradientX = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::GradientZ;

    using ValueFunction     = std::shared_ptr<Plato::Elliptic::Hatching::AbstractScalarFunction<Residual>>;
    using GradientUFunction = std::shared_ptr<Plato::Elliptic::Hatching::AbstractScalarFunction<GradientU>>;
    using GradientCFunction = std::shared_ptr<Plato::Elliptic::Hatching::AbstractScalarFunction<GradientC>>;
    using GradientXFunction = std::shared_ptr<Plato::Elliptic::Hatching::AbstractScalarFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Elliptic::Hatching::AbstractScalarFunction<GradientZ>>;

    std::map<std::string, ValueFunction>     mValueFunctions;
    std::map<std::string, GradientUFunction> mGradientUFunctions;
    std::map<std::string, GradientCFunction> mGradientCFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    Plato::SpatialModel & mSpatialModel;

    const Plato::Sequence<ElementType> & mSequence;

    Plato::DataMap& mDataMap;
    std::string mFunctionName;

private:
    /******************************************************************************//**
     * \brief Initialization of Physics Scalar Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList & aProblemParams
    );

public:
    /******************************************************************************//**
     * \brief Primary physics scalar function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    PhysicsScalarFunction(
              Plato::SpatialModel          & aSpatialModel,
        const Plato::Sequence<ElementType> & aSequence,
              Plato::DataMap               & aDataMap,
              Teuchos::ParameterList       & aProblemParams,
              std::string                  & aName
    );

    /******************************************************************************//**
     * \brief Secondary physics scalar function constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    PhysicsScalarFunction(
              Plato::SpatialModel          & aSpatialModel,
        const Plato::Sequence<ElementType> & aSequence,
              Plato::DataMap               & aDataMap
    );

    /******************************************************************************//**
     * \brief Allocate scalar function using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const ValueFunction & aInput,
              std::string     aName
    );

    /******************************************************************************//**
     * \brief Allocate scalar function using the Jacobian automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientUFunction & aInput,
              std::string         aName
    );

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientC automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientCFunction & aInput,
              std::string         aName
    );

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientZ automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientZFunction & aInput,
              std::string         aName
    );

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientX automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientXFunction & aInput,
              std::string         aName
    );

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate physics scalar function
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aLocalState local state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::Solutions     & aSolution,
        const Plato::ScalarArray4D & aLocalStates,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::Solutions     & aSolution,
        const Plato::ScalarArray4D & aLocalStates,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(
        const Plato::Solutions     & aSolution,
        const Plato::ScalarArray4D & aLocalStates,
        const Plato::ScalarVector  & aControl,
              Plato::OrdinalType     aStepIndex,
              Plato::Scalar          aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_c(
        const Plato::Solutions     & aSolution,
        const Plato::ScalarArray4D & aLocalStates,
        const Plato::ScalarVector  & aControl,
              Plato::OrdinalType     aStepIndex,
              Plato::Scalar          aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::Solutions     & aSolution,
        const Plato::ScalarArray4D & aLocalStates,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep = 0.0
    ) const override;

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
//class PhysicsScalarFunction

} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
