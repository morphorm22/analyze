#pragma once

#include "AnalyzeMacros.hpp"
#include "geometric/ScalarFunctionBaseFactory.hpp"

namespace Plato
{

namespace Geometric
{

    /******************************************************************************//**
     * \brief Initialization of Weighted Sum Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename PhysicsType>
    void
    WeightedSumFunction<PhysicsType>::
    initialize(
        Teuchos::ParameterList & aProblemParams
    )
    {
        Plato::Geometric::ScalarFunctionBaseFactory<PhysicsType> tFactory;

        mScalarFunctionBaseContainer.clear();
        mFunctionWeights.clear();

        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);

        auto tFunctionNamesArray = tFunctionParams.get<Teuchos::Array<std::string>>("Functions");
        auto tFunctionWeightsArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Weights");

        auto tFunctionNames = tFunctionNamesArray.toVector();
        auto tFunctionWeights = tFunctionWeightsArray.toVector();

        if (tFunctionNames.size() != tFunctionWeights.size())
        {
            const std::string tErrorString = std::string("Number of 'Functions' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Weights'";
            ANALYZE_THROWERR(tErrorString)
        }

        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
        {
            mScalarFunctionBaseContainer.push_back(
                tFactory.create(
                    mSpatialModel, mDataMap, aProblemParams, tFunctionNames[tFunctionIndex]));
            mFunctionWeights.push_back(tFunctionWeights[tFunctionIndex]);
        }

    }

    /******************************************************************************//**
     * \brief Primary weight sum function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    template<typename PhysicsType>
    WeightedSumFunction<PhysicsType>::
    WeightedSumFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    ) :
        Plato::Geometric::WorksetBase<typename PhysicsType::ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Secondary weight sum function constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    template<typename PhysicsType>
    WeightedSumFunction<PhysicsType>::
    WeightedSumFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::Geometric::WorksetBase<typename PhysicsType::ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName ("Weighted Sum")
    {
    }

    /******************************************************************************//**
     * \brief Add function weight
     * \param [in] aWeight function weight
    **********************************************************************************/
    template<typename PhysicsType>
    void
    WeightedSumFunction<PhysicsType>::
    appendFunctionWeight(Plato::Scalar aWeight)
    {
        mFunctionWeights.push_back(aWeight);
    }

    /******************************************************************************//**
     * \brief Allocate scalar function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    template<typename PhysicsType>
    void
    WeightedSumFunction<PhysicsType>::
    allocateScalarFunctionBase(const std::shared_ptr<Plato::Geometric::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseContainer.push_back(aInput);
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    template<typename PhysicsType>
    void
    WeightedSumFunction<PhysicsType>::
    updateProblem(const Plato::ScalarVector & aControl) const
    {
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            mScalarFunctionBaseContainer[tFunctionIndex]->updateProblem(aControl);
        }
    }

    /******************************************************************************//**
     * \brief Evaluate weight sum function
     * \param [in] aControl 1D view of control variables
     * \return scalar function evaluation
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::Scalar
    WeightedSumFunction<PhysicsType>::
    value(const Plato::ScalarVector & aControl) const
    {
        assert(mScalarFunctionBaseContainer.size() == mFunctionWeights.size());

        Plato::Scalar tResult = 0.0;
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aControl);
            tResult += tFunctionWeight * tFunctionValue;
        }
        return tResult;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the weight sum function with respect to (wrt) the configuration parameters
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    WeightedSumFunction<PhysicsType>::
    gradient_x(const Plato::ScalarVector & aControl) const
    {
        const Plato::OrdinalType tNumDofs = mNumSpatialDims * mNumNodes;
        Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::ScalarVector tFunctionGradX = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_x(aControl);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientX(tDof) += tFunctionWeight * tFunctionGradX(tDof);
            },"Weighted Sum Function Summation Grad X");
        }
        return tGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the weight sum function with respect to (wrt) the control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    WeightedSumFunction<PhysicsType>::
    gradient_z(const Plato::ScalarVector & aControl) const
    {
        const Plato::OrdinalType tNumDofs = mNumNodes;
        Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::ScalarVector tFunctionGradZ = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_z(aControl);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientZ(tDof) += tFunctionWeight * tFunctionGradZ(tDof);
            },"Weighted Sum Function Summation Grad Z");
        }
        return tGradientZ;
    }

    /******************************************************************************//**
     * \brief Set user defined function name
     * \param [in] function name
    **********************************************************************************/
    template<typename PhysicsType>
    void
    WeightedSumFunction<PhysicsType>::
    setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    template<typename PhysicsType>
    std::string
    WeightedSumFunction<PhysicsType>::
    name() const
    {
        return mFunctionName;
    }
} // namespace Geometric

} // namespace Plato
