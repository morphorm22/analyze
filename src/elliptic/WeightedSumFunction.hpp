#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include "WorksetBase.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include "AnalyzeMacros.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Weighted sum function class \f$ F(x) = \sum_{i = 1}^{n} w_i * f_i(x) \f$
 **********************************************************************************/
template<typename PhysicsT>
class WeightedSumFunction : public Plato::Elliptic::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<PhysicsT>::mNumControl; /*!< number of control variables */
    using Plato::WorksetBase<PhysicsT>::mNumNodes; /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<PhysicsT>::mNumCells; /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mConfigEntryOrdinal;

    std::vector<Plato::Scalar> mFunctionWeights; /*!< Vector of function weights */
    std::vector<std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>> mScalarFunctionBaseContainer; /*!< Vector of ScalarFunctionBase objects */

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

	/******************************************************************************//**
     * \brief Initialization of Weighted Sum Function
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList & aProblemParams
    )
    {
        Plato::Elliptic::ScalarFunctionBaseFactory<PhysicsT> tFactory;

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

public:
    /******************************************************************************//**
     * \brief Primary weight sum function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    WeightedSumFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Secondary weight sum function constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
    **********************************************************************************/
    WeightedSumFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName ("Weighted Sum")
    {
    }

    /******************************************************************************//**
     * \brief Add function weight
     * \param [in] aWeight function weight
    **********************************************************************************/
    void appendFunctionWeight(Plato::Scalar aWeight)
    {
        mFunctionWeights.push_back(aWeight);
    }

    /******************************************************************************//**
     * \brief Allocate scalar function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void allocateScalarFunctionBase(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseContainer.push_back(aInput);
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const override
    {
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            mScalarFunctionBaseContainer[tFunctionIndex]->updateProblem(aState, aControl);
        }
    }


    /******************************************************************************//**
     * \brief Evaluate weight sum function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::Solutions    & aSolution,
                        const Plato::ScalarVector & aControl,
                              Plato::Scalar         aTimeStep = 0.0) const override
    {
        assert(mScalarFunctionBaseContainer.size() == mFunctionWeights.size());

        Plato::Scalar tResult = 0.0;
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aSolution, aControl, aTimeStep);
            tResult += tFunctionWeight * tFunctionValue;
        }
        return tResult;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the weight sum function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::Solutions    & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::Scalar         aTimeStep = 0.0) const override
    {
        const Plato::OrdinalType tNumDofs = mNumSpatialDims * mNumNodes;
        Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::ScalarVector tFunctionGradX = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_x(aSolution, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
            {
                tGradientX(tDof) += tFunctionWeight * tFunctionGradX(tDof);
            },"Weighted Sum Function Summation Grad X");
        }
        return tGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the weight sum function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::Solutions    & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::OrdinalType    aStepIndex,
                                         Plato::Scalar         aTimeStep = 0.0) const override
    {
        const Plato::OrdinalType tNumDofs = mNumDofsPerNode * mNumNodes;
        Plato::ScalarVector tGradientU ("gradient state", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::ScalarVector tFunctionGradU = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_u(aSolution, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
            {
                tGradientU(tDof) += tFunctionWeight * tFunctionGradU(tDof);
            },"Weighted Sum Function Summation Grad U");
        }
        return tGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the weight sum function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::Solutions    & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::Scalar         aTimeStep = 0.0) const override
    {
        const Plato::OrdinalType tNumDofs = mNumNodes;
        Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::ScalarVector tFunctionGradZ = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_z(aSolution, aControl, aTimeStep);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
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
// class WeightedSumFunction

} // namespace Elliptic

} // namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermal<1>>;
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Mechanics<1>>;
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Electromechanics<1>>;
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermal<2>>;
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Mechanics<2>>;
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Electromechanics<2>>;
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermal<3>>;
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Mechanics<3>>;
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Electromechanics<3>>;
extern template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermomechanics<3>>;
#endif
