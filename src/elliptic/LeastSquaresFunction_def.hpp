#pragma once

#include "elliptic/LeastSquaresFunction_decl.hpp"

#include "BLAS1.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Initialization of Least Squares Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    template<typename PhysicsType>
    void LeastSquaresFunction<PhysicsType>::initialize (
        Teuchos::ParameterList & aProblemParams
    )
    {
        Plato::Elliptic::ScalarFunctionBaseFactory<PhysicsType> tFactory;

        mScalarFunctionBaseContainer.clear();
        mFunctionWeights.clear();
        mFunctionGoldValues.clear();
        mFunctionNormalization.clear();

        auto tFunctionParams = aProblemParams.sublist("Criterial").sublist(mFunctionName);

        auto tFunctionNamesArray = tFunctionParams.get<Teuchos::Array<std::string>>("Functions");
        auto tFunctionWeightsArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Weights");
        auto tFunctionGoldValuesArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Gold Values");

        auto tFunctionNames      = tFunctionNamesArray.toVector();
        auto tFunctionWeights    = tFunctionWeightsArray.toVector();
        auto tFunctionGoldValues = tFunctionGoldValuesArray.toVector();

        if (tFunctionNames.size() != tFunctionWeights.size())
        {
            const std::string tErrorString = std::string("Number of 'Functions' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Weights'";
            ANALYZE_THROWERR(tErrorString)
        }

        if (tFunctionNames.size() != tFunctionGoldValues.size())
        {
            const std::string tErrorString = std::string("Number of 'Gold Values' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Functions'";
            ANALYZE_THROWERR(tErrorString)
        }

        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
        {
            mScalarFunctionBaseContainer.push_back(
                tFactory.create(
                    mSpatialModel, mDataMap, aProblemParams, tFunctionNames[tFunctionIndex]));
            mFunctionWeights.push_back(tFunctionWeights[tFunctionIndex]);

            appendGoldFunctionValue(tFunctionGoldValues[tFunctionIndex]);
        }

    }

    /******************************************************************************//**
     * \brief Primary least squares function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    template<typename PhysicsType>
    LeastSquaresFunction<PhysicsType>::LeastSquaresFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    ) :
        Plato::WorksetBase<typename PhysicsType::ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Secondary least squares function constructor, used for unit testing / mass properties
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    template<typename PhysicsType>
    LeastSquaresFunction<PhysicsType>::LeastSquaresFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::WorksetBase<typename PhysicsType::ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName ("Least Squares")
    {
    }

    /******************************************************************************//**
     * \brief Add function weight
     * \param [in] aWeight function weight
    **********************************************************************************/
    template<typename PhysicsType>
    void LeastSquaresFunction<PhysicsType>::appendFunctionWeight(Plato::Scalar aWeight)
    {
        mFunctionWeights.push_back(aWeight);
    }

    /******************************************************************************//**
     * \brief Add function gold value
     * \param [in] aGoldValue function gold value
     * \param [in] aUseAsNormalization use gold value as normalization
    **********************************************************************************/
    template<typename PhysicsType>
    void LeastSquaresFunction<PhysicsType>::
    appendGoldFunctionValue(Plato::Scalar aGoldValue, bool aUseAsNormalization)
    {
        mFunctionGoldValues.push_back(aGoldValue);

        if (aUseAsNormalization)
        {
            if (std::abs(aGoldValue) > mFunctionNormalizationCutoff)
                mFunctionNormalization.push_back(std::abs(aGoldValue));
            else
                mFunctionNormalization.push_back(1.0);
        }
    }

    /******************************************************************************//**
     * \brief Add function normalization
     * \param [in] aFunctionNormalization function normalization value
    **********************************************************************************/
    template<typename PhysicsType>
    void LeastSquaresFunction<PhysicsType>::
    appendFunctionNormalization(Plato::Scalar aFunctionNormalization)
    {
        // Dont allow the function normalization to be "too small"
        if (std::abs(aFunctionNormalization) > mFunctionNormalizationCutoff)
            mFunctionNormalization.push_back(std::abs(aFunctionNormalization));
        else
            mFunctionNormalization.push_back(mFunctionNormalizationCutoff);
    }

    /******************************************************************************//**
     * \brief Allocate scalar function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    template<typename PhysicsType>
    void LeastSquaresFunction<PhysicsType>::
    allocateScalarFunctionBase(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseContainer.push_back(aInput);
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    template<typename PhysicsType>
    void LeastSquaresFunction<PhysicsType>::
    updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    {
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            mScalarFunctionBaseContainer[tFunctionIndex]->updateProblem(aState, aControl);
        }
    }


    /******************************************************************************//**
     * \brief Evaluate least squares function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::Scalar LeastSquaresFunction<PhysicsType>::
    value(const Plato::Solutions    & aSolution,
                        const Plato::ScalarVector & aControl,
                              Plato::Scalar         aTimeStep) const
    {
        assert(mFunctionWeights.size() == mScalarFunctionBaseContainer.size());
        assert(mFunctionGoldValues.size() == mScalarFunctionBaseContainer.size());
        assert(mFunctionNormalization.size() == mScalarFunctionBaseContainer.size());

        Plato::Scalar tResult = 0.0;
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
            const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aSolution, aControl, aTimeStep);
            tResult += tFunctionWeight * 
                       std::pow((tFunctionValue - tFunctionGoldValue) / tFunctionScale, 2);

            Plato::Scalar tPercentDiff = std::abs(tFunctionGoldValue) > 0.0 ? 
                                         100.0 * (tFunctionValue - tFunctionGoldValue) / tFunctionGoldValue :
                                         (tFunctionValue - tFunctionGoldValue);
            printf("%20s = %12.4e * ((%12.4e - %12.4e) / %12.4e)^2 =  %12.4e (PercDiff = %10.1f)\n", 
                   mScalarFunctionBaseContainer[tFunctionIndex]->name().c_str(),
                   tFunctionWeight,
                   tFunctionValue, 
                   tFunctionGoldValue,
                   tFunctionScale,
                   tFunctionWeight * 
                             std::pow((tFunctionValue - tFunctionGoldValue) / tFunctionScale, 2),
                   tPercentDiff);
        }
        return tResult;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the least squares function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector LeastSquaresFunction<PhysicsType>::
    gradient_x(const Plato::Solutions    & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::Scalar         aTimeStep) const
    {
        const Plato::OrdinalType tNumDofs = mNumSpatialDims * mNumNodes;
        Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
            const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aSolution, aControl, aTimeStep);
            Plato::ScalarVector tFunctionGradX = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_x(aSolution, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientX(tDof) += 2.0 * tFunctionWeight * (tFunctionValue - tFunctionGoldValue) 
                                        * tFunctionGradX(tDof) / (tFunctionScale * tFunctionScale);
            },"Least Squares Function Summation Grad X");
        }
        return tGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the least squares function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector LeastSquaresFunction<PhysicsType>::
    gradient_u(const Plato::Solutions    & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::OrdinalType    aStepIndex,
                                         Plato::Scalar         aTimeStep) const
    {
        const Plato::OrdinalType tNumDofs = mNumDofsPerNode * mNumNodes;
        Plato::ScalarVector tGradientU ("gradient state", tNumDofs);
        if (mGradientWRTStateIsZero)
        {
            Plato::blas1::fill(0.0, tGradientU);
        }
        else
        {
            for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
            {
                const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
                const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
                const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
                Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aSolution, aControl, aTimeStep);
                Plato::ScalarVector tFunctionGradU = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_u(aSolution, aControl, aTimeStep);
                Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
                {
                    tGradientU(tDof) += 2.0 * tFunctionWeight * (tFunctionValue - tFunctionGoldValue) 
                                            * tFunctionGradU(tDof) / (tFunctionScale * tFunctionScale);
                },"Least Squares Function Summation Grad U");
            }
        }
        return tGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the least squares function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector LeastSquaresFunction<PhysicsType>::
    gradient_z(const Plato::Solutions    & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::Scalar         aTimeStep) const
    {
        const Plato::OrdinalType tNumDofs = mNumNodes;
        Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            const Plato::Scalar tFunctionGoldValue = mFunctionGoldValues[tFunctionIndex];
            const Plato::Scalar tFunctionScale = mFunctionNormalization[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aSolution, aControl, aTimeStep);
            Plato::ScalarVector tFunctionGradZ = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_z(aSolution, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientZ(tDof) += 2.0 * tFunctionWeight * (tFunctionValue - tFunctionGoldValue) 
                                        * tFunctionGradZ(tDof) / (tFunctionScale * tFunctionScale);
            },"Least Squares Function Summation Grad Z");
        }
        return tGradientZ;
    }

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    template<typename PhysicsType>
    std::string LeastSquaresFunction<PhysicsType>::
    name() const
    {
        return mFunctionName;
    }

    /******************************************************************************//**
     * \brief Set gradient wrt state flag
     * \return Gradient WRT State is zero flag
    **********************************************************************************/
    template<typename PhysicsType>
    void LeastSquaresFunction<PhysicsType>::
    setGradientWRTStateIsZeroFlag(bool aGradientWRTStateIsZero)
    {
        mGradientWRTStateIsZero = aGradientWRTStateIsZero;
    }
} // namespace Elliptic

} // namespace Plato
