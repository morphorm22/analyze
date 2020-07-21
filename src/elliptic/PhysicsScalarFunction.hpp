#pragma once

#include "PlatoUtilities.hpp"


#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "WorksetBase.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Physics scalar function class
 **********************************************************************************/
template<typename PhysicsT>
class PhysicsScalarFunction : public ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<PhysicsT>::mNumControl; /*!< number of control variables */
    using Plato::WorksetBase<PhysicsT>::mNumNodes; /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<PhysicsT>::mNumCells; /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mConfigEntryOrdinal; /*!< number of degree of freedom per cell/element */

    using Residual = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual; /*!< result variables automatic differentiation type */
    using Jacobian = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian; /*!< state variables automatic differentiation type */
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX; /*!< configuration variables automatic differentiation type */
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ; /*!< control variables automatic differentiation type */

    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<Residual>> mScalarFunctionValue; /*!< scalar function value interface */
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<Jacobian>> mScalarFunctionGradientU; /*!< scalar function value partial wrt states */
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientX>> mScalarFunctionGradientX; /*!< scalar function value partial wrt configuration */
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientZ>> mScalarFunctionGradientZ; /*!< scalar function value partial wrt controls */

    Plato::DataMap& mDataMap;   /*!< output data map */
    std::string mFunctionName;  /*!< User defined function name */

// private access functions
private:
    /******************************************************************************//**
     * \brief Initialization of Physics Scalar Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aInputParams)
    {
        typename PhysicsT::FunctionFactory tFactory;

        auto tProblemDefault = aInputParams.sublist(mFunctionName);
        // tFunctionType must be the hard-coded type name (e.g. Volume)
        auto tFunctionType = tProblemDefault.get<std::string>("Scalar Function Type", "");

        mScalarFunctionValue =
            tFactory.template createScalarFunction<Residual>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientU =
            tFactory.template createScalarFunction<Jacobian>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientX =
            tFactory.template createScalarFunction<GradientX>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientZ =
            tFactory.template createScalarFunction<GradientZ>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
    }

public:
    /******************************************************************************//**
     * \brief Primary physics scalar function constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    PhysicsScalarFunction(Omega_h::Mesh& aMesh,
            Omega_h::MeshSets& aMeshSets,
            Plato::DataMap & aDataMap,
            Teuchos::ParameterList& aInputParams,
            std::string& aName) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap),
            mFunctionName(aName)
    {
        initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * \brief Secondary physics scalar function constructor, used for unit testing
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
    **********************************************************************************/
    PhysicsScalarFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mScalarFunctionValue(),
            mScalarFunctionGradientU(),
            mScalarFunctionGradientX(),
            mScalarFunctionGradientZ(),
            mDataMap(aDataMap),
            mFunctionName("Undefined Name")
    {
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator(const std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<Residual>>& aInput)
    {
        mScalarFunctionValue = nullptr; // ensures shared_ptr is decremented
        mScalarFunctionValue = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the Jacobian automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator(const std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<Jacobian>>& aInput)
    {
        mScalarFunctionGradientU = nullptr; // ensures shared_ptr is decremented
        mScalarFunctionGradientU = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientZ automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator(const std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientZ>>& aInput)
    {
        mScalarFunctionGradientZ = nullptr; // ensures shared_ptr is decremented
        mScalarFunctionGradientZ = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientX automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator(const std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<GradientX>>& aInput)
    {
        mScalarFunctionGradientX = nullptr; // ensures shared_ptr is decremented
        mScalarFunctionGradientX = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const override
    {
        Plato::ScalarMultiVector tStateWS("state workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        Plato::ScalarMultiVector tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        Plato::ScalarArray3D tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        mScalarFunctionValue->updateProblem(tStateWS, tControlWS, tConfigWS);
        mScalarFunctionGradientU->updateProblem(tStateWS, tControlWS, tConfigWS);
        mScalarFunctionGradientZ->updateProblem(tStateWS, tControlWS, tConfigWS);
        mScalarFunctionGradientX->updateProblem(tStateWS, tControlWS, tConfigWS);
    }

    /******************************************************************************//**
     * \brief Evaluate physics scalar function
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::Solution     & aSolution,
                        const Plato::ScalarVector & aControl,
                              Plato::Scalar         aTimeStep = 0.0) const override
    {
        using ConfigScalar  = typename Residual::ConfigScalarType;
        using StateScalar   = typename Residual::StateScalarType;
        using ControlScalar = typename Residual::ControlScalarType;
        using ResultScalar  = typename Residual::ResultScalarType;

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", mNumCells);
        mDataMap.scalarVectors[mScalarFunctionValue->getName()] = tResult;

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", mNumCells, mNumDofsPerCell);

        ResultScalar tReturnVal(0.0);

        auto tStates = aSolution.State;
        auto tNumSteps = tStates.extent(0);
        for( decltype(tNumSteps) tStepIndex=0; tStepIndex < tNumSteps; ++tStepIndex )
        {
            // workset state
            //
            auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

            // evaluate function
            //
            Kokkos::deep_copy(tResult, 0.0);
            mScalarFunctionValue->evaluate(tStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

            // sum across elements
            //
            tReturnVal += Plato::local_result_sum<Plato::Scalar>(mNumCells, tResult);
        }
        printf("%s value = %12.4e\n", (mScalarFunctionValue->getName()).c_str(), tReturnVal);
        mScalarFunctionValue->postEvaluate(tReturnVal);

        return tReturnVal;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::Solution     & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::Scalar         aTimeStep = 0.0) const override
    {
        using ConfigScalar  = typename GradientX::ConfigScalarType;
        using StateScalar   = typename GradientX::StateScalarType;
        using ControlScalar = typename GradientX::ControlScalarType;
        using ResultScalar  = typename GradientX::ResultScalarType;

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", mNumCells, mNumDofsPerCell);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::Scalar tObjectiveValue(0.0);
        Plato::ScalarVector tObjGradientX("objective gradient configuration", mNumSpatialDims * mNumNodes);
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", mNumCells);

        auto tStates = aSolution.State;

        auto tNumSteps = tStates.extent(0);
        for( decltype(tNumSteps) tStepIndex=0; tStepIndex < tNumSteps; ++tStepIndex )
        {

            // workset state
            //
            auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

            // evaluate function
            //
            Kokkos::deep_copy(tResult, 0.0);
            mScalarFunctionGradientX->evaluate(tStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

            // create and assemble to return view
            //
            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>(mNumCells,
                                                                                   mConfigEntryOrdinal,
                                                                                   tResult,
                                                                                   tObjGradientX);
            tObjectiveValue += Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);
        }
        mScalarFunctionGradientX->postEvaluate(tObjGradientX, tObjectiveValue);

        return tObjGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::Solution     & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::OrdinalType    aStepIndex,
                                         Plato::Scalar         aTimeStep = 0.0) const override
    {
        using ConfigScalar  = typename Jacobian::ConfigScalarType;
        using StateScalar   = typename Jacobian::StateScalarType;
        using ControlScalar = typename Jacobian::ControlScalarType;
        using ResultScalar  = typename Jacobian::ResultScalarType;

        auto tStates = aSolution.State;
        auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("sacado-ized state", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", mNumCells);

        // evaluate function
        //
        mScalarFunctionGradientU->evaluate(tStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state", mNumDofsPerNode * mNumNodes);
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>(mNumCells, mGlobalStateEntryOrdinal, tResult, tObjGradientU);
        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);
        mScalarFunctionGradientU->postEvaluate(tObjGradientU, tObjectiveValue);
        return tObjGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::Solution     & aSolution,
                                   const Plato::ScalarVector & aControl,
                                         Plato::Scalar         aTimeStep = 0.0) const override
    {        
        using ConfigScalar  = typename GradientZ::ConfigScalarType;
        using StateScalar   = typename GradientZ::StateScalarType;
        using ControlScalar = typename GradientZ::ControlScalarType;
        using ResultScalar  = typename GradientZ::ResultScalarType;

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", mNumCells, mNumDofsPerCell);

        Plato::Scalar tObjectiveValue(0.0);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", mNumCells);
 
        // create return vector
        //
        Plato::ScalarVector tObjGradientZ("objective gradient control", mNumNodes);

        auto tStates = aSolution.State;

        auto tNumSteps = tStates.extent(0);
        for( decltype(tNumSteps) tStepIndex=0; tStepIndex < tNumSteps; ++tStepIndex )
        {

            // workset state
            //
            auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

            // evaluate function
            //
            Kokkos::deep_copy(tResult, 0.0);
            mScalarFunctionGradientZ->evaluate(tStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

            Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>(mNumCells, mControlEntryOrdinal, tResult, tObjGradientZ);

            tObjectiveValue += Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);
        }

        mScalarFunctionGradientZ->postEvaluate(tObjGradientZ, tObjectiveValue);

        return tObjGradientZ;
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
    decltype(mFunctionName) name() const
    {
        return mFunctionName;
    }
};
//class PhysicsScalarFunction

} // namespace Elliptic

} // namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermal<1>>;
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<1>>;
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Electromechanics<1>>;
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermal<2>>;
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<2>>;
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Electromechanics<2>>;
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermal<3>>;
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<3>>;
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Electromechanics<3>>;
extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<3>>;
#endif
