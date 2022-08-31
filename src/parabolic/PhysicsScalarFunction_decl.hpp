#pragma once

#include "WorksetBase.hpp"
#include "parabolic/EvaluationTypes.hpp"
#include "parabolic/ScalarFunctionBase.hpp"
#include "parabolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************//**
 * \brief Physics scalar function inc class
 **********************************************************************************/
template<typename PhysicsType>
class PhysicsScalarFunction :
  public Plato::Parabolic::ScalarFunctionBase,
  public Plato::WorksetBase<typename PhysicsType::ElementType>
{
private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumDofsPerCell; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<ElementType>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<ElementType>::mNumDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<ElementType>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<ElementType>::mNumControl; /*!< number of control variables */
    using Plato::WorksetBase<ElementType>::mNumNodes; /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<ElementType>::mNumCells; /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<ElementType>::mGlobalStateEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<ElementType>::mControlEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<ElementType>::mConfigEntryOrdinal; /*!< number of degree of freedom per cell/element */

    using Residual  = typename Plato::Parabolic::Evaluation<ElementType>::Residual;
    using GradientU = typename Plato::Parabolic::Evaluation<ElementType>::GradientU;
    using GradientV = typename Plato::Parabolic::Evaluation<ElementType>::GradientV;
    using GradientX = typename Plato::Parabolic::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Parabolic::Evaluation<ElementType>::GradientZ;

    using ValueFunction     = std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<Residual>>;
    using GradientUFunction = std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientU>>;
    using GradientVFunction = std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientV>>;
    using GradientXFunction = std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientZ>>;

    std::map<std::string, ValueFunction>     mValueFunctions;
    std::map<std::string, GradientUFunction> mGradientUFunctions;
    std::map<std::string, GradientVFunction> mGradientVFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName;/*!< User defined function name */

    /******************************************************************************//**
     * \brief Initialization of parabolic Physics Scalar Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList & aInputParams
    );

public:
    /******************************************************************************//**
     * \brief Primary physics scalar function inc constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    PhysicsScalarFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string            & aName
    );

    /******************************************************************************//**
     * \brief Secondary physics scalar function inc constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    PhysicsScalarFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap& aDataMap
    );

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
     * \param [in] aStepIndex step index
     * \param [in] aTimeStep time step
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aStepIndex step index
     * \param [in] aTimeStep time step
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_v(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep
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

} // namespace Parabolic

} // namespace Plato
