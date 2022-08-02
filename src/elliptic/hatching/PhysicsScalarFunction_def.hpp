#pragma once

#include "PlatoUtilities.hpp"


#include <memory>
#include <cassert>
#include <vector>

#include "Solutions.hpp"
#include "WorksetBase.hpp"
#include "PlatoSequence.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/hatching/EvaluationTypes.hpp"
#include "elliptic/hatching/ScalarFunctionBase.hpp"
#include "elliptic/hatching/AbstractScalarFunction.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

    /******************************************************************************//**
     * \brief Initialization of Physics Scalar Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    template<typename PhysicsType>
    void
    PhysicsScalarFunction<PhysicsType>::
    initialize(
        Teuchos::ParameterList & aProblemParams
    )
    {
        typename PhysicsType::FunctionFactory tFactory;

        auto tProblemDefault = aProblemParams.sublist("Criteria").sublist(mFunctionName);
        auto tFunctionType = tProblemDefault.get<std::string>("Scalar Function Type", "");


        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mValueFunctions[tName]     = tFactory.template createScalarFunction<Residual> 
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
            mGradientUFunctions[tName] = tFactory.template createScalarFunction<GradientU> 
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
            mGradientCFunctions[tName] = tFactory.template createScalarFunction<GradientC> 
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
            mGradientXFunctions[tName] = tFactory.template createScalarFunction<GradientX>
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
            mGradientZFunctions[tName] = tFactory.template createScalarFunction<GradientZ>
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
        }
    }

    /******************************************************************************//**
     * \brief Primary physics scalar function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    template<typename PhysicsType>
    PhysicsScalarFunction<PhysicsType>::
    PhysicsScalarFunction(
              Plato::SpatialModel          & aSpatialModel,
        const Plato::Sequence<ElementType> & aSequence,
              Plato::DataMap               & aDataMap,
              Teuchos::ParameterList       & aProblemParams,
              std::string                  & aName
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mSequence     (aSequence),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Secondary physics scalar function constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    template<typename PhysicsType>
    PhysicsScalarFunction<PhysicsType>::
    PhysicsScalarFunction(
              Plato::SpatialModel          & aSpatialModel,
        const Plato::Sequence<ElementType> & aSequence,
              Plato::DataMap               & aDataMap
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mSequence     (aSequence),
        mDataMap      (aDataMap),
        mFunctionName ("Undefined Name")
    {
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    template<typename PhysicsType>
    void
    PhysicsScalarFunction<PhysicsType>::
    setEvaluator(
        const ValueFunction & aInput,
              std::string     aName
    )
    {
        mValueFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mValueFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the Jacobian automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    template<typename PhysicsType>
    void
    PhysicsScalarFunction<PhysicsType>::
    setEvaluator(
        const GradientUFunction & aInput,
              std::string         aName
    )
    {
        mGradientUFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mGradientUFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientC automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    template<typename PhysicsType>
    void
    PhysicsScalarFunction<PhysicsType>::
    setEvaluator(
        const GradientCFunction & aInput,
              std::string         aName
    )
    {
        mGradientCFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mGradientCFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientZ automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    template<typename PhysicsType>
    void
    PhysicsScalarFunction<PhysicsType>::
    setEvaluator(
        const GradientZFunction & aInput,
              std::string         aName
    )
    {
        mGradientZFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mGradientZFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientX automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    template<typename PhysicsType>
    void
    PhysicsScalarFunction<PhysicsType>::
    setEvaluator(
        const GradientXFunction & aInput,
              std::string         aName
    )
    {
        mGradientXFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mGradientXFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    template<typename PhysicsType>
    void
    PhysicsScalarFunction<PhysicsType>::
    updateProblem(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl
    ) const {}

    /******************************************************************************//**
     * \brief Evaluate physics scalar function
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aLocalState local state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar physics function evaluation
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::Scalar
    PhysicsScalarFunction<PhysicsType>::
    value(
        const Plato::Solutions     & aSolution,
        const Plato::ScalarArray4D & aLocalStates,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep
    ) const
    {
        using ConfigScalar      = typename Residual::ConfigScalarType;
        using GlobalStateScalar = typename Residual::GlobalStateScalarType;
        using LocalStateScalar  = typename Residual::LocalStateScalarType;
        using ControlScalar     = typename Residual::ControlScalarType;
        using ResultScalar      = typename Residual::ResultScalarType;

        Plato::Scalar tReturnVal(0.0);

        auto tGlobalStates = aSolution.get("State");
        auto tNumStates = tGlobalStates.extent(0);

        const auto& tSequenceSteps = mSequence.getSteps();
        auto tNumSequenceSteps = tSequenceSteps.size();

        assert(tNumStates == tNumSequenceSteps);

        for (Plato::OrdinalType tStepIndex=0; tStepIndex<tNumSequenceSteps; tStepIndex++)
        {

            const auto& tSequenceStep = tSequenceSteps[tStepIndex];

            mSpatialModel.applyMask(tSequenceStep.getMask());

            for(const auto& tDomain : mSpatialModel.Domains)
            {
                auto tNumCells = tDomain.numCells();
                auto tName     = tDomain.getDomainName();

                // workset control
                //
                Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
                Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

                // workset config
                //
                Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
                Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

                // create result view
                //
                Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);
                mDataMap.scalarVectors[mValueFunctions.at(tName)->getName()] = tResult;

                Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("global state workset", tNumCells, mNumDofsPerCell);
                Plato::ScalarArray3DT    <LocalStateScalar>  tLocalStateWS ("local state workset",  tNumCells, mNumGaussPoints, mNumLocalStatesPerGP);

                // workset global state
                //
                auto tGlobalState = Kokkos::subview(tGlobalStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tGlobalState, tGlobalStateWS, tDomain);

                // workset local state
                //
                Plato::ScalarArray3D tLocalState;
                if (tStepIndex > 0)
                {
                    tLocalState  = Kokkos::subview(aLocalStates, tStepIndex-1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()); 
                }
                else
                {
                    tLocalState = Plato::ScalarArray3D("initial local state", aLocalStates.extent(1), aLocalStates.extent(2), aLocalStates.extent(3));
                }
                Plato::WorksetBase<ElementType>::worksetLocalState(tLocalState, tLocalStateWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mValueFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

                // sum across elements
                //
                tReturnVal += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mValueFunctions.at(tName)->postEvaluate(tReturnVal);

        return tReturnVal;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    PhysicsScalarFunction<PhysicsType>::
    gradient_x(
        const Plato::Solutions     & aSolution,
        const Plato::ScalarArray4D & aLocalStates,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep
    ) const
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using GlobalStateScalar = typename GradientX::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientX::LocalStateScalarType;
        using ControlScalar     = typename GradientX::ControlScalarType;
        using ResultScalar      = typename GradientX::ResultScalarType;

        // create return view
        //
        Plato::Scalar tValue(0.0);
        Plato::ScalarVector tObjGradientX("objective gradient configuration", mNumSpatialDims * mNumNodes);

        auto tGlobalStates = aSolution.get("State");
        auto tNumSteps = tGlobalStates.extent(0);

        auto& tSequenceSteps = mSequence.getSteps();
        auto tNumSequenceSteps = tSequenceSteps.size();

        assert(tNumSteps == tNumSequenceSteps);

        for (Plato::OrdinalType tStepIndex=0; tStepIndex<tNumSequenceSteps; tStepIndex++)
        {
            const auto& tSequenceStep = tSequenceSteps[tStepIndex];

            mSpatialModel.applyMask(tSequenceStep.getMask());

            for(const auto& tDomain : mSpatialModel.Domains)
            {
                auto tNumCells = tDomain.numCells();
                auto tName     = tDomain.getDomainName();

                Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("global state workset", tNumCells, mNumDofsPerCell);
                Plato::ScalarArray3DT    <LocalStateScalar>  tLocalStateWS ("local state workset",  tNumCells, mNumGaussPoints, mNumLocalStatesPerGP);

                // workset control
                //
                Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
                Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

                // workset config
                //
                Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
                Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

                Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);

                // workset global state
                //
                auto tGlobalState = Kokkos::subview(tGlobalStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tGlobalState, tGlobalStateWS, tDomain);

                // workset local state
                //
                Plato::ScalarArray3D tLocalState;
                if (tStepIndex > 0)
                {
                    tLocalState  = Kokkos::subview(aLocalStates, tStepIndex-1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()); 
                }
                else
                {
                    tLocalState = Plato::ScalarArray3D("initial local state", aLocalStates.extent(1), aLocalStates.extent(2), aLocalStates.extent(3));
                }
                Plato::WorksetBase<ElementType>::worksetLocalState(tLocalState, tLocalStateWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mGradientXFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

                // create and assemble to return view
                //
                Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                    (tDomain, mConfigEntryOrdinal, tResult, tObjGradientX);

                tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientXFunctions.at(tName)->postEvaluate(tObjGradientX, tValue);

        return tObjGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    PhysicsScalarFunction<PhysicsType>::
    gradient_u(
        const Plato::Solutions     & aSolution,
        const Plato::ScalarArray4D & aLocalStates,
        const Plato::ScalarVector  & aControl,
              Plato::OrdinalType     aStepIndex,
              Plato::Scalar          aTimeStep
    ) const
    {
        using ConfigScalar      = typename GradientU::ConfigScalarType;
        using GlobalStateScalar = typename GradientU::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientU::LocalStateScalarType;
        using ControlScalar     = typename GradientU::ControlScalarType;
        using ResultScalar      = typename GradientU::ResultScalarType;

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state", mNumDofsPerNode * mNumNodes);

        const auto& tSequenceStep = mSequence.getSteps()[aStepIndex];

        mSpatialModel.applyMask(tSequenceStep.getMask());

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            auto tGlobalStates = aSolution.get("State");
            auto tGlobalState = Kokkos::subview(tGlobalStates, aStepIndex, Kokkos::ALL());

            // workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("sacado-ized state", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(tGlobalState, tGlobalStateWS, tDomain);

            // workset local state
            //
            // workset local state
            //
            Plato::ScalarArray3D tLocalState;
            if (aStepIndex > 0)
            {
                tLocalState  = Kokkos::subview(aLocalStates, aStepIndex-1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()); 
            }
            else
            {
                tLocalState = Plato::ScalarArray3D("initial local state", aLocalStates.extent(1), aLocalStates.extent(2), aLocalStates.extent(3));
            }
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("sacado-ized state", tNumCells, mNumGaussPoints, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(tLocalState, tLocalStateWS, tDomain);

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);

            // evaluate function
            //
            mGradientUFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>
                (tDomain, mGlobalStateEntryOrdinal, tResult, tObjGradientU);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientUFunctions.at(tName)->postEvaluate(tObjGradientU, tValue);

        return tObjGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    PhysicsScalarFunction<PhysicsType>::
    gradient_c(
        const Plato::Solutions     & aSolution,
        const Plato::ScalarArray4D & aLocalStates,
        const Plato::ScalarVector  & aControl,
              Plato::OrdinalType     aStepIndex,
              Plato::Scalar          aTimeStep
    ) const
    {
        using ConfigScalar      = typename GradientC::ConfigScalarType;
        using GlobalStateScalar = typename GradientC::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientC::LocalStateScalarType;
        using ControlScalar     = typename GradientC::ControlScalarType;
        using ResultScalar      = typename GradientC::ResultScalarType;

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientC("objective gradient state", mNumLocalDofsPerCell * mNumCells);

        auto tLastStepIndex = aLocalStates.extent(0) - 1;
        if (aStepIndex != tLastStepIndex)
        {
            const auto& tSequenceStep = mSequence.getSteps()[aStepIndex+1];

            mSpatialModel.applyMask(tSequenceStep.getMask());

            Plato::Scalar tValue(0.0);
            for(const auto& tDomain : mSpatialModel.Domains)
            {
                auto tNumCells = tDomain.numCells();
                auto tName     = tDomain.getDomainName();

                auto tGlobalStates = aSolution.get("State");
                auto tGlobalState = Kokkos::subview(tGlobalStates, aStepIndex+1, Kokkos::ALL());

                // workset global state
                //
                Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("sacado-ized state", tNumCells, mNumDofsPerCell);
                Plato::WorksetBase<ElementType>::worksetState(tGlobalState, tGlobalStateWS, tDomain);

                // workset local state
                //
                auto tLocalState = Kokkos::subview(aLocalStates, aStepIndex, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
                Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("sacado-ized state", tNumCells, mNumGaussPoints, mNumLocalStatesPerGP);
                Plato::WorksetBase<ElementType>::worksetLocalState(tLocalState, tLocalStateWS, tDomain);

                // workset control
                //
                Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
                Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

                // workset config
                //
                Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
                Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

                // create return view
                //
                Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);

                // evaluate function
                //
                mGradientCFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

                Plato::transform_ad_type_to_pod_1Dview<mNumLocalDofsPerCell>(tDomain, tResult, tObjGradientC);

                tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
            }
            auto tName = mSpatialModel.Domains[0].getDomainName();
            mGradientCFunctions.at(tName)->postEvaluate(tObjGradientC, tValue);
        }
        return tObjGradientC;
    }
    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    PhysicsScalarFunction<PhysicsType>::
    gradient_z(
        const Plato::Solutions     & aSolution,
        const Plato::ScalarArray4D & aLocalStates,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep
    ) const
    {
        using ConfigScalar      = typename GradientZ::ConfigScalarType;
        using GlobalStateScalar = typename GradientZ::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientZ::LocalStateScalarType;
        using ControlScalar     = typename GradientZ::ControlScalarType;
        using ResultScalar      = typename GradientZ::ResultScalarType;

        // create return vector
        //
        Plato::Scalar tValue(0.0);
        Plato::ScalarVector tObjGradientZ("objective gradient control", mNumNodes);

        auto tGlobalStates = aSolution.get("State");
        auto tNumSteps = tGlobalStates.extent(0);

        auto& tSequenceSteps = mSequence.getSteps();
        auto tNumSequenceSteps = tSequenceSteps.size();

        assert(tNumSteps == tNumSequenceSteps);

        for (Plato::OrdinalType tStepIndex=0; tStepIndex<tNumSequenceSteps; tStepIndex++)
        {
            const auto& tSequenceStep = tSequenceSteps[tStepIndex];

            mSpatialModel.applyMask(tSequenceStep.getMask());

            for(const auto& tDomain : mSpatialModel.Domains)
            {
                auto tNumCells = tDomain.numCells();
                auto tName     = tDomain.getDomainName();

                Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("global state workset", tNumCells, mNumDofsPerCell);
                Plato::ScalarArray3DT    <LocalStateScalar>  tLocalStateWS ("local state workset",  tNumCells, mNumGaussPoints, mNumLocalStatesPerGP);

                // workset control
                //
                Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
                Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

                // workset config
                //
                Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
                Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

                // create result
                //
                Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);

                // global workset state
                //
                auto tGlobalState = Kokkos::subview(tGlobalStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tGlobalState, tGlobalStateWS, tDomain);

                // workset local state
                //
                Plato::ScalarArray3D tLocalState;
                if (tStepIndex > 0)
                {
                    tLocalState  = Kokkos::subview(aLocalStates, tStepIndex-1, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()); 
                }
                else
                {
                    tLocalState = Plato::ScalarArray3D("initial local state", aLocalStates.extent(1), aLocalStates.extent(2), aLocalStates.extent(3));
                }
                Plato::WorksetBase<ElementType>::worksetLocalState(tLocalState, tLocalStateWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mGradientZFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

                Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>
                    (tDomain, mControlEntryOrdinal, tResult, tObjGradientZ);

                tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientZFunctions.at(tName)->postEvaluate(tObjGradientZ, tValue);

        return tObjGradientZ;
    }

    /******************************************************************************//**
     * \brief Set user defined function name
     * \param [in] function name
    **********************************************************************************/
    template<typename PhysicsType>
    void
    PhysicsScalarFunction<PhysicsType>::
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
    PhysicsScalarFunction<PhysicsType>::
    name() const
    {
        return mFunctionName;
    }
} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
