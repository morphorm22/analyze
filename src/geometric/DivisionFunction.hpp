#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include "PlatoStaticsTypes.hpp"
#include "geometric/WorksetBase.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/ScalarFunctionBaseFactory.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * \brief Division function class \f$ F(x) = numerator(x) / denominator(x) \f$
 **********************************************************************************/
template<typename PhysicsType>
class DivisionFunction :
    public Plato::Geometric::ScalarFunctionBase,
    public Plato::Geometric::WorksetBase<typename PhysicsType::ElementType>
{
private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::Geometric::WorksetBase<ElementType>::mNumSpatialDims;
    using Plato::Geometric::WorksetBase<ElementType>::mNumNodes;

    std::shared_ptr<Plato::Geometric::ScalarFunctionBase> mScalarFunctionBaseNumerator;
    std::shared_ptr<Plato::Geometric::ScalarFunctionBase> mScalarFunctionBaseDenominator;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

	/******************************************************************************//**
     * \brief Initialization of Division Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList & aProblemParams
    )
    {
        Plato::Geometric::ScalarFunctionBaseFactory<PhysicsType> tFactory;

        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);

        auto tNumeratorFunctionName = tFunctionParams.get<std::string>("Numerator");
        auto tDenominatorFunctionName = tFunctionParams.get<std::string>("Denominator");

        mScalarFunctionBaseNumerator = 
             tFactory.create(mSpatialModel, mDataMap, aProblemParams, tNumeratorFunctionName);

        mScalarFunctionBaseDenominator = 
             tFactory.create(mSpatialModel, mDataMap, aProblemParams, tDenominatorFunctionName);
    }

public:
    /******************************************************************************//**
     * \brief Primary division function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    DivisionFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
        const std::string            & aName
    ) :
        Plato::Geometric::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Secondary division function constructor, used for unit testing
     * \param [in] aMesh mesh database
    **********************************************************************************/
    DivisionFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::Geometric::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName ("Division Function")
    {
    }

    /******************************************************************************//**
     * \brief Allocate numerator function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void allocateNumeratorFunction(const std::shared_ptr<Plato::Geometric::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseNumerator = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate denominator function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void allocateDenominatorFunction(const std::shared_ptr<Plato::Geometric::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseDenominator = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl) const override
    {
        mScalarFunctionBaseNumerator->updateProblem(aControl);
        mScalarFunctionBaseDenominator->updateProblem(aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate division function
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aControl) const override
    {
        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aControl);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aControl);
        Plato::Scalar tResult = tNumeratorValue / tDenominatorValue;
        if (tDenominatorValue == 0.0)
        {
            ANALYZE_THROWERR("Denominator of division function evaluated to 0!")
        }
        
        return tResult;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the configuration parameters
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aControl) const override
    {
        const Plato::OrdinalType tNumDofs = mNumSpatialDims * mNumNodes;
        Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);

        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aControl);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aControl);

        Plato::ScalarVector tNumeratorGradX = mScalarFunctionBaseNumerator->gradient_x(aControl);
        Plato::ScalarVector tDenominatorGradX = mScalarFunctionBaseDenominator->gradient_x(aControl);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientX(tDof) = (tNumeratorGradX(tDof) * tDenominatorValue - 
                                tDenominatorGradX(tDof) * tNumeratorValue) 
                               / (pow(tDenominatorValue, 2));
        },"Division Function Grad X");
        return tGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aControl) const override
    {
        const Plato::OrdinalType tNumDofs = mNumNodes;
        Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
        
        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aControl);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aControl);

        Plato::ScalarVector tNumeratorGradZ = mScalarFunctionBaseNumerator->gradient_z(aControl);
        Plato::ScalarVector tDenominatorGradZ = mScalarFunctionBaseDenominator->gradient_z(aControl);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientZ(tDof) = (tNumeratorGradZ(tDof) * tDenominatorValue - 
                                tDenominatorGradZ(tDof) * tNumeratorValue) 
                               / (pow(tDenominatorValue, 2));
        },"Division Function Grad Z");

        return tGradientZ;
    }

    /******************************************************************************//**
     * \brief Set user defined function name
     * \param [in] function name
    **********************************************************************************/
    void setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    std::string name() const
    {
        return mFunctionName;
    }
};
// class DivisionFunction

} // namespace Geometric

} // namespace Plato

#include "Geometrical.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEC(Plato::Geometric::DivisionFunction, Plato::Geometrical)
