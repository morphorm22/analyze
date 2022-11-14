#pragma once

#include "PlatoStaticsTypes.hpp"
#include "geometric/WorksetBase.hpp"
#include "geometric/EvaluationTypes.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * \brief Geometry scalar function class
 **********************************************************************************/
template<typename GeometryT>
class GeometryScalarFunction :
    public Plato::Geometric::ScalarFunctionBase,
    public Plato::Geometric::WorksetBase<typename GeometryT::ElementType>
{
private:
    using ElementType = typename GeometryT::ElementType;

    using Plato::Geometric::WorksetBase<ElementType>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::Geometric::WorksetBase<ElementType>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::Geometric::WorksetBase<ElementType>::mNumControl; /*!< number of control variables */
    using Plato::Geometric::WorksetBase<ElementType>::mNumNodes; /*!< total number of nodes in the mesh */
    using Plato::Geometric::WorksetBase<ElementType>::mNumCells; /*!< total number of cells/elements in the mesh */

    using Plato::Geometric::WorksetBase<ElementType>::mControlEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::Geometric::WorksetBase<ElementType>::mConfigEntryOrdinal; /*!< number of degree of freedom per cell/element */

    using Residual  = typename Plato::Geometric::Evaluation<ElementType>::Residual;
    using GradientX = typename Plato::Geometric::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Geometric::Evaluation<ElementType>::GradientZ;

    using ValueFunction     = std::shared_ptr<Plato::Geometric::AbstractScalarFunction<Residual>>;
    using GradientXFunction = std::shared_ptr<Plato::Geometric::AbstractScalarFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Geometric::AbstractScalarFunction<GradientZ>>;

    std::map<std::string, ValueFunction>     mValueFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;   /*!< output data map */
    std::string mFunctionName;  /*!< User defined function name */

// private access functions
private:
    /******************************************************************************//**
     * \brief Initialization of Geometry Scalar Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aProblemParams);

public:
    /******************************************************************************//**
     * \brief Primary physics scalar function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    GeometryScalarFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
        const std::string            & aName
    );

    /******************************************************************************//**
     * \brief Secondary physics scalar function constructor, used for unit testing
     * \param [in] aMesh mesh database
    **********************************************************************************/
    GeometryScalarFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    );

    /******************************************************************************//**
     * \brief Set scalar function using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const ValueFunction & aInput,
              std::string     aName
    );

    /******************************************************************************//**
     * \brief Set scalar function using the GradientZ automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientZFunction & aInput,
              std::string         aName
    );

    /******************************************************************************//**
     * \brief Set scalar function using the GradientX automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientXFunction & aInput,
              std::string         aName
    );

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarVector & aControl
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate physics scalar function
     * \param [in] aControl 1D view of control variables
     * \return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::ScalarVector & aControl
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::ScalarVector & aControl
    ) const override;


    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::ScalarVector & aControl
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
//class GeometryScalarFunction

} // namespace Geometric

} // namespace Plato
