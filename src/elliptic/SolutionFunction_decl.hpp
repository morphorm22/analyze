#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include "WorksetBase.hpp"
#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Solution function class
 **********************************************************************************/
template<typename PhysicsType>
class SolutionFunction :
    public Plato::Elliptic::ScalarFunctionBase,
    public Plato::WorksetBase<typename PhysicsType::ElementType>
{
    enum solution_type_t
    {
        UNKNOWN_TYPE = 0,
        SOLUTION_IN_DIRECTION = 1,
        SOLUTION_MAG_IN_DIRECTION = 2,
        DIFF_BETWEEN_SOLUTION_MAG_IN_DIRECTION_AND_TARGET = 3,
        DIFF_BETWEEN_SOLUTION_VECTOR_AND_TARGET_VECTOR = 4,
        DIFF_BETWEEN_SOLUTION_IN_DIRECTION_AND_TARGET_SOLUTION_IN_DIRECTION = 5
    };

private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumDofsPerCell;  /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<ElementType>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<ElementType>::mNumDofsPerNode;  /*!< number of degree of freedom per node */
    using Plato::WorksetBase<ElementType>::mNumSpatialDims;  /*!< number of spatial dimensions */
    using Plato::WorksetBase<ElementType>::mNumControl;      /*!< number of control variables */
    using Plato::WorksetBase<ElementType>::mNumNodes;        /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<ElementType>::mNumCells;        /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<ElementType>::mGlobalStateEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<ElementType>::mControlEntryOrdinal;     /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<ElementType>::mConfigEntryOrdinal;      /*!< number of degree of freedom per cell/element */

    std::string mFunctionName; /*!< User defined function name */
    std::string mDomainName;   /*!< Name of the node set that represents the domain of interest */

    Plato::Array<mNumDofsPerNode> mNormal;  /*!< Direction of solution criterion */
    Plato::Array<mNumDofsPerNode> mTargetSolutionVector;  /*!< Target solution vector */
    Plato::Scalar mTargetMagnitude; /*!< Target magnitude */
    Plato::Scalar mTargetSolution; /*!< Target solution */
    bool mMagnitudeSpecified;
    bool mNormalSpecified;
    bool mTargetSolutionVectorSpecified;
    bool mTargetMagnitudeSpecified;
    bool mTargetSolutionSpecified;

    const Plato::SpatialModel & mSpatialModel;
    solution_type_t mSolutionType;

    /******************************************************************************//**
     * \brief Initialization of Solution Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void
    initialize (Teuchos::ParameterList & aProblemParams);
  
    void initialize_target_vector(Teuchos::ParameterList &aFunctionParams);

    void initialize_normal_vector(Teuchos::ParameterList &aFunctionParams);

public:
    /******************************************************************************//**
     * \brief Primary solution function constructor
     * \param [in] aMesh mesh database
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    SolutionFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    );

    /******************************************************************************//**
     * \brief Evaluate solution function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the solution function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the solution function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the solution function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the control variables

       NOTE:  Currently, no penalty is applied, so the gradient wrt z is zero.

    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \fn virtual void updateProblem(const Plato::ScalarVector & aState,
                                      const Plato::ScalarVector & aControl) const
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
// class SolutionFunction

} // namespace Elliptic

} // namespace Plato
