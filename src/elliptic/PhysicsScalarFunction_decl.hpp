#pragma once

#include "WorksetBase.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Physics scalar function class
 **********************************************************************************/
template<typename PhysicsType>
class PhysicsScalarFunction : public ScalarFunctionBase, public Plato::WorksetBase<typename PhysicsType::ElementType>
{
private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumDofsPerCell; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<ElementType>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<ElementType>::mNumDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<ElementType>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<ElementType>::mNumNodes; /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<ElementType>::mNumCells; /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<ElementType>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<ElementType>::mControlEntryOrdinal;
    using Plato::WorksetBase<ElementType>::mConfigEntryOrdinal;

    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using Jacobian  = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using GradientX = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;

    using ValueFunction     = std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<Residual>>;
    using GradientUFunction = std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<Jacobian>>;
    using GradientXFunction = std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientZ>>;

    std::map<std::string, ValueFunction>     mValueFunctions;     /*!< scalar function value interface */
    std::map<std::string, GradientUFunction> mGradientUFunctions; /*!< scalar function value partial wrt states */
    std::map<std::string, GradientXFunction> mGradientXFunctions; /*!< scalar function value partial wrt configuration */
    std::map<std::string, GradientZFunction> mGradientZFunctions; /*!< scalar function value partial wrt controls */

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;   /*!< output data map */
    std::string mFunctionName;  /*!< User defined function name */

private:
    /******************************************************************************//**
     * \brief Initialization of Physics Scalar Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aProblemParams);

public:
    /******************************************************************************//**
     * \brief Primary physics scalar function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    PhysicsScalarFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    );

    /******************************************************************************//**
     * \brief Secondary physics scalar function constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    PhysicsScalarFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    );

    /******************************************************************************//**
     * \brief Allocate scalar function using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator( const ValueFunction & aInput, std::string aName);

    /******************************************************************************//**
     * \brief Allocate scalar function using the Jacobian automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator( const GradientUFunction & aInput, std::string aName);

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientZ automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator( const GradientZFunction & aInput, std::string aName);

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientX automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator(const GradientXFunction & aInput, std::string aName);

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
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
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

} // namespace Elliptic

} // namespace Plato
