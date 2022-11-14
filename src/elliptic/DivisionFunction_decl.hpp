#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include "WorksetBase.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Division function class \f$ F(x) = numerator(x) / denominator(x) \f$
 **********************************************************************************/
template<typename PhysicsType>
class DivisionFunction :
    public Plato::Elliptic::ScalarFunctionBase,
    public Plato::WorksetBase<typename PhysicsType::ElementType>
{
private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<ElementType>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<ElementType>::mNumNodes;       /*!< total number of nodes in the mesh */

    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> mScalarFunctionBaseNumerator; /*!< numerator function */
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> mScalarFunctionBaseDenominator; /*!< denominator function */

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

    /******************************************************************************//**
     * \brief Initialization of Division Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aProblemParams);

public:
    /******************************************************************************//**
     * \brief Primary division function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    DivisionFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
        const std::string            & aName
    );

    /******************************************************************************//**
     * \brief Secondary division function constructor, used for unit testing
     * \param [in] aMesh mesh database
    **********************************************************************************/
    DivisionFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    );

    /******************************************************************************//**
     * \brief Allocate numerator function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void allocateNumeratorFunction(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aInput);

    /******************************************************************************//**
     * \brief Allocate denominator function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void allocateDenominatorFunction(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aInput);

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const;

    /******************************************************************************//**
     * \brief Evaluate division function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(const Plato::Solutions    & aSolution,
          const Plato::ScalarVector & aControl,
                Plato::Scalar         aTimeStep = 0.0) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar         aTimeStep = 0.0) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::OrdinalType    aStepIndex,
                     Plato::Scalar         aTimeStep = 0.0) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar aTimeStep = 0.0) const override;

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
// class DivisionFunction

} // namespace Elliptic

} // namespace Plato
