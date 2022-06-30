#pragma once

#include "elliptic/DivisionFunction_decl.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Initialization of Division Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    template<typename PhysicsType>
    void DivisionFunction<PhysicsType>::initialize(
        Teuchos::ParameterList & aProblemParams
    )
    {
        Plato::Elliptic::ScalarFunctionBaseFactory<PhysicsType> tFactory;

        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);

        auto tNumeratorFunctionName = tFunctionParams.get<std::string>("Numerator");
        auto tDenominatorFunctionName = tFunctionParams.get<std::string>("Denominator");

        mScalarFunctionBaseNumerator = 
             tFactory.create(mSpatialModel, mDataMap, aProblemParams, tNumeratorFunctionName);

        mScalarFunctionBaseDenominator = 
             tFactory.create(mSpatialModel, mDataMap, aProblemParams, tDenominatorFunctionName);
    }

    /******************************************************************************//**
     * \brief Primary division function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    template<typename PhysicsType>
    DivisionFunction<PhysicsType>::DivisionFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
        const std::string            & aName
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
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
    template<typename PhysicsType>
    DivisionFunction<PhysicsType>::DivisionFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName ("Division Function")
    {
    }

    /******************************************************************************//**
     * \brief Allocate numerator function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    template<typename PhysicsType>
    void DivisionFunction<PhysicsType>::allocateNumeratorFunction(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseNumerator = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate denominator function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    template<typename PhysicsType>
    void DivisionFunction<PhysicsType>::allocateDenominatorFunction(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseDenominator = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    template<typename PhysicsType>
    void DivisionFunction<PhysicsType>::updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    {
        mScalarFunctionBaseNumerator->updateProblem(aState, aControl);
        mScalarFunctionBaseDenominator->updateProblem(aState, aControl);
    }


    /******************************************************************************//**
     * \brief Evaluate division function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::Scalar
    DivisionFunction<PhysicsType>::value(const Plato::Solutions    & aSolution,
          const Plato::ScalarVector & aControl,
                Plato::Scalar         aTimeStep) const
    {
        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tResult = tNumeratorValue / tDenominatorValue;
        if (tDenominatorValue == 0.0)
        {
            ANALYZE_THROWERR("Denominator of division function evaluated to 0!")
        }
        
        return tResult;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    DivisionFunction<PhysicsType>::gradient_x(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar         aTimeStep) const
    {
        const Plato::OrdinalType tNumDofs = mNumSpatialDims * mNumNodes;
        Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);

        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValueSquared = tDenominatorValue * tDenominatorValue;

        Plato::ScalarVector tNumeratorGradX = mScalarFunctionBaseNumerator->gradient_x(aSolution, aControl, aTimeStep);
        Plato::ScalarVector tDenominatorGradX = mScalarFunctionBaseDenominator->gradient_x(aSolution, aControl, aTimeStep);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientX(tDof) = (tNumeratorGradX(tDof) * tDenominatorValue - 
                                tDenominatorGradX(tDof) * tNumeratorValue) 
                               / (tDenominatorValueSquared);
        },"Division Function Grad X");
        return tGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    DivisionFunction<PhysicsType>::gradient_u(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::OrdinalType    aStepIndex,
                     Plato::Scalar         aTimeStep) const
    {
        const Plato::OrdinalType tNumDofs = mNumDofsPerNode * mNumNodes;
        Plato::ScalarVector tGradientU ("gradient state", tNumDofs);

        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValueSquared = tDenominatorValue * tDenominatorValue;

        Plato::ScalarVector tNumeratorGradU = mScalarFunctionBaseNumerator->gradient_u(aSolution, aControl, aTimeStep);
        Plato::ScalarVector tDenominatorGradU = mScalarFunctionBaseDenominator->gradient_u(aSolution, aControl, aTimeStep);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientU(tDof) = (tNumeratorGradU(tDof) * tDenominatorValue - 
                                tDenominatorGradU(tDof) * tNumeratorValue) 
                               / (tDenominatorValueSquared);
        },"Division Function Grad U");

        return tGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the division function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    DivisionFunction<PhysicsType>::gradient_z(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar aTimeStep) const
    {
        const Plato::OrdinalType tNumDofs = mNumNodes;
        Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
        
        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aSolution, aControl, aTimeStep);
        Plato::Scalar tDenominatorValueSquared = tDenominatorValue * tDenominatorValue;

        Plato::ScalarVector tNumeratorGradZ = mScalarFunctionBaseNumerator->gradient_z(aSolution, aControl, aTimeStep);
        Plato::ScalarVector tDenominatorGradZ = mScalarFunctionBaseDenominator->gradient_z(aSolution, aControl, aTimeStep);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientZ(tDof) = (tNumeratorGradZ(tDof) * tDenominatorValue - 
                                tDenominatorGradZ(tDof) * tNumeratorValue) 
                               / (tDenominatorValueSquared);
        },"Division Function Grad Z");

        return tGradientZ;
    }

    /******************************************************************************//**
     * \brief Set user defined function name
     * \param [in] function name
    **********************************************************************************/
    template<typename PhysicsType>
    void DivisionFunction<PhysicsType>::setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    template<typename PhysicsType>
    std::string DivisionFunction<PhysicsType>::name() const
    {
        return mFunctionName;
    }
} // namespace Elliptic

} // namespace Plato
