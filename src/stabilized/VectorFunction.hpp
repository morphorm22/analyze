#pragma once


#include <memory>

#include "WorksetBase.hpp"
#include "stabilized/AbstractVectorFunction.hpp"
#include "stabilized/EvaluationTypes.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************/
/*! Stabilized Partial Differential Equation (PDE) constraint workset manager

   This class takes as a template argument a vector function in the form:

   and manages the evaluation of the function and Jacobians wrt state, node
   state, control, and configuration.

   NOTES:
   1. The use case will define the node state: a) If the stabilized
      mechanics residual is used, the node state is denoted by the projected
      pressure gradient. 2) If the projected gradient residual is used, the
      node state is denoted by the projected pressure field.
   2. The use case will define the state: a) If the stabilized mechanics
      residual is used, the states are displacement+pressure. 2) If the
      projected gradient residual is used, the state is denoted by the
      projected pressure gradient.
*/
/******************************************************************************/
template<typename PhysicsType>
class VectorFunction : public Plato::WorksetBase<typename PhysicsType::ElementType>
{
private:

    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumDofsPerCell;
    using Plato::WorksetBase<ElementType>::mNumNodesPerCell;
    using Plato::WorksetBase<ElementType>::mNumDofsPerNode;
    using Plato::WorksetBase<ElementType>::mNumSpatialDims;
    using Plato::WorksetBase<ElementType>::mNumControl;
    using Plato::WorksetBase<ElementType>::mNumNodes;
    using Plato::WorksetBase<ElementType>::mNumCells;

    using ElementType::mNumNodeStatePerNode;
    using ElementType::mNumNodeStatePerCell;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;

    using Residual  = typename Plato::Stabilized::Evaluation<ElementType>::Residual;
    using Jacobian  = typename Plato::Stabilized::Evaluation<ElementType>::Jacobian;
    using JacobianN = typename Plato::Stabilized::Evaluation<ElementType>::JacobianN;
    using GradientX = typename Plato::Stabilized::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Stabilized::Evaluation<ElementType>::GradientZ;

    using ResidualFunction  = std::shared_ptr<Plato::Stabilized::AbstractVectorFunction<Residual>>;
    using JacobianUFunction = std::shared_ptr<Plato::Stabilized::AbstractVectorFunction<Jacobian>>;
    using JacobianNFunction = std::shared_ptr<Plato::Stabilized::AbstractVectorFunction<JacobianN>>;
    using JacobianXFunction = std::shared_ptr<Plato::Stabilized::AbstractVectorFunction<GradientX>>;
    using JacobianZFunction = std::shared_ptr<Plato::Stabilized::AbstractVectorFunction<GradientZ>>;

    std::map<std::string, ResidualFunction>  mResidualFunctions;
    std::map<std::string, JacobianUFunction> mJacobianUFunctions;
    std::map<std::string, JacobianNFunction> mJacobianNFunctions;
    std::map<std::string, JacobianXFunction> mJacobianXFunctions;
    std::map<std::string, JacobianZFunction> mJacobianZFunctions;

    ResidualFunction  mBoundaryLoadsResidualFunction;
    JacobianUFunction mBoundaryLoadsJacobianUFunction;
    JacobianNFunction mBoundaryLoadsJacobianNFunction;
    JacobianXFunction mBoundaryLoadsJacobianXFunction;
    JacobianZFunction mBoundaryLoadsJacobianZFunction;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< output data map */

public:
    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aMesh mesh data base
    * \param [in] aDataMap problem-specific data map
    * \param [in] aParamList Teuchos parameter list with input data
    * \param [in] aProblemType problem type
    *
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
        const std::string            & aProblemType
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap)
    {
        typename PhysicsType::FunctionFactory tFunctionFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFunctions[tName]  = tFunctionFactory.template createVectorFunction<Residual> (tDomain, aDataMap, aParamList, aProblemType);
            mJacobianUFunctions[tName] = tFunctionFactory.template createVectorFunction<Jacobian> (tDomain, aDataMap, aParamList, aProblemType);
            mJacobianNFunctions[tName] = tFunctionFactory.template createVectorFunction<JacobianN>(tDomain, aDataMap, aParamList, aProblemType);
            mJacobianZFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientZ>(tDomain, aDataMap, aParamList, aProblemType);
            mJacobianXFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientX>(tDomain, aDataMap, aParamList, aProblemType);
        }

        // any block can compute the boundary terms for the entire mesh.  We'll use the first block.
        auto tFirstBlockName = aSpatialModel.Domains.front().getDomainName();

        mBoundaryLoadsResidualFunction  = mResidualFunctions[tFirstBlockName];
        mBoundaryLoadsJacobianUFunction = mJacobianUFunctions[tFirstBlockName];
        mBoundaryLoadsJacobianNFunction = mJacobianNFunctions[tFirstBlockName];
        mBoundaryLoadsJacobianZFunction = mJacobianZFunctions[tFirstBlockName];
        mBoundaryLoadsJacobianXFunction = mJacobianXFunctions[tFirstBlockName];
    }


    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aMesh mesh data base
    * \param [in] aDataMap problem-specific data map
    *
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel  (aSpatialModel),
        mDataMap(aDataMap)
    {
    }

    /**************************************************************************//**
    * \brief Return dof names for Physics
    * \return dof names
    ******************************************************************************/
    std::vector<std::string> getDofNames() const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        return mResidualFunctions.at(tFirstBlockName)->getDofNames();
    }

    /**************************************************************************//**
    *
    * \brief Return number of degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return (mNumNodes * mNumDofsPerNode);
    }

    /**************************************************************************//**
     *
     * \brief Return total number of nodes
     * \return total number of nodes
     *
     ******************************************************************************/
    decltype(mNumNodes) numNodes() const
    {
        return mNumNodes;
    }

    /**************************************************************************//**
     *
     * \brief Return total number of cells
     * \return total number of cells
     *
     ******************************************************************************/
    decltype(mNumCells) numCells() const
    {
        return mNumCells;
    }

    /***********************************************************************//**
     * \brief Return number of spatial dimensions.
     * \return number of spatial dimensions
    ***************************************************************************/
    decltype(mNumSpatialDims) numSpatialDims() const
    {
        return mNumSpatialDims;
    }

    /***********************************************************************//**
     * \brief Return number of nodes per cell.
     * \return number of nodes per cell
    ***************************************************************************/
    decltype(mNumNodesPerCell) numNodesPerCell() const
    {
        return mNumNodesPerCell;
    }

    /***********************************************************************//**
     * \brief Return number of projected pressure gradient degrees of freedom per node.
     * \return number of projected pressure gradient degrees of freedom per node
    ***************************************************************************/
    decltype(mNumDofsPerNode) numDofsPerNode() const
    {
        return mNumDofsPerNode;
    }

    /***********************************************************************//**
     * \brief Return number of projected pressure gradient degrees of freedom per cell.
     * \return number of projected pressure gradient degrees of freedom per cell
    ***************************************************************************/
    decltype(mNumDofsPerCell) numDofsPerCell() const
    {
        return mNumDofsPerCell;
    }

    /***********************************************************************//**
     * \brief Return number of pressure degrees of freedom per node.
     * \return number of pressure degrees of freedom per node
    ***************************************************************************/
    decltype(mNumNodeStatePerNode) numNodeStatePerNode() const
    {
        return mNumNodeStatePerNode;
    }

    /***********************************************************************//**
     * \brief Return number of pressure degrees of freedom per cell.
     * \return number of pressure degrees of freedom per cell
    ***************************************************************************/
    decltype(mNumNodeStatePerCell) numNodeStatePerCell() const
    {
        return mNumNodeStatePerCell;
    }

    /***********************************************************************//**
     * \brief Return number of configuration degrees of freedom per cell.
     * \return number of configuration degrees of freedom per cell
    ***************************************************************************/
    decltype(mNumConfigDofsPerCell) numConfigDofsPerCell() const
    {
        return mNumConfigDofsPerCell;
    }

    /****************************************************************************//**
    * \brief Function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        auto tItr = mResidualFunctions.find(tFirstBlockName);
        if(tItr == mResidualFunctions.end())
            { ANALYZE_THROWERR(std::string("Element block with name '") + tFirstBlockName + "is not defined in residual function to element block map.") }
        return tItr->second->getSolutionStateOutputData(aSolutions);
    }

    /***************************************************************************//**
     * \brief Evaluate and assemble residual
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled residual
    *******************************************************************************/
    Plato::ScalarVector
    value(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename Residual::ConfigScalarType;
        using StateScalar     = typename Residual::StateScalarType;
        using NodeStateScalar = typename Residual::NodeStateScalarType;
        using ControlScalar   = typename Residual::ControlScalarType;
        using ResultScalar    = typename Residual::ResultScalarType;

        Plato::ScalarVector tReturnValue("Assembled Residual", mNumDofsPerNode * mNumNodes);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mResidualFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // assemble to return view
            //
            Plato::WorksetBase<ElementType>::assembleResidual(tResidual, tReturnValue, tDomain);
        }

        {
            auto tNumCells = mSpatialModel.Mesh->NumElements();

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsResidualFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // assemble to return view
            //
            Plato::WorksetBase<ElementType>::assembleResidual(tResidual, tReturnValue);
        }

        return tReturnValue;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to configuration degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename GradientX::ConfigScalarType;
        using StateScalar     = typename GradientX::StateScalarType;
        using NodeStateScalar = typename GradientX::NodeStateScalarType;
        using ControlScalar   = typename GradientX::ControlScalarType;
        using ResultScalar    = typename GradientX::ResultScalarType;

        // Allocate Jacobian
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumDofsPerNode>(tMesh);

        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode> tMatEntryOrdinal(tJacobianMat, tMesh);

        auto tMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Configuration", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianXFunctions.at(tName)->evaluate(tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

            // Assemble Jacobian
            //
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian(mNumDofsPerCell, mNumConfigDofsPerCell, tMatEntryOrdinal, tJacobian, tMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Configuration", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianXFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

            // Assemble Jacobian
            //
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian(mNumDofsPerCell, mNumConfigDofsPerCell, tMatEntryOrdinal, tJacobian, tMatEntries);
        }

        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to configuration degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return Workset of Jacobian with respect to configuration degrees of freedom
    *******************************************************************************/
    Plato::ScalarArray3D
    gradient_x_workset(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename GradientX::ConfigScalarType;
        using StateScalar     = typename GradientX::StateScalarType;
        using NodeStateScalar = typename GradientX::NodeStateScalarType;
        using ControlScalar   = typename GradientX::ControlScalarType;
        using ResultScalar    = typename GradientX::ResultScalarType;

        Plato::ScalarArray3D tOutputJacobian("Jacobian WRT Configuration", mNumCells, mNumDofsPerCell, mNumConfigDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Configuration", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianXFunctions.at(tName)->evaluate(tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

            // assemble
            //
            Plato::transform_ad_type_to_pod_3Dview<mNumDofsPerCell, mNumConfigDofsPerCell>(tDomain, tJacobianWS, tOutputJacobian);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Configuration", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianXFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

            // assemble
            //
            Plato::transform_ad_type_to_pod_3Dview<mNumDofsPerCell, mNumConfigDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        }

        return tOutputJacobian;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian transpose
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u_T(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename Jacobian::ConfigScalarType;
        using StateScalar     = typename Jacobian::StateScalarType;
        using NodeStateScalar = typename Jacobian::NodeStateScalarType;
        using ControlScalar   = typename Jacobian::ControlScalarType;
        using ResultScalar    = typename Jacobian::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( tMesh );

        Plato::BlockMatrixTransposeEntryOrdinal<mNumSpatialDims, mNumDofsPerNode> tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

        auto tJacobianMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian State", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianUFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian State", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianUFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }

        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename Jacobian::ConfigScalarType;
        using StateScalar     = typename Jacobian::StateScalarType;
        using NodeStateScalar = typename Jacobian::NodeStateScalarType;
        using ControlScalar   = typename Jacobian::ControlScalarType;
        using ResultScalar    = typename Jacobian::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( tMesh );

        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode> tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

        auto tJacobianMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian State", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianUFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assemble to return matrix
            //
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian State", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianUFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assemble to return matrix
            //
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to node state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return workset of Jacobian with respect to node state degrees of freedom
    *******************************************************************************/
    Plato::ScalarArray3D
    gradient_n_workset(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename JacobianN::ConfigScalarType;
        using StateScalar     = typename JacobianN::StateScalarType;
        using NodeStateScalar = typename JacobianN::NodeStateScalarType;
        using ControlScalar   = typename JacobianN::ControlScalarType;
        using ResultScalar    = typename JacobianN::ResultScalarType;

        // create return array
        //
        Plato::ScalarArray3D tOutJacobian("POD Jacobian Node State", mNumCells, mNumDofsPerCell, mNumNodeStatePerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Node State", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianNFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep );

            Plato::transform_ad_type_to_pod_3Dview<mNumDofsPerCell, mNumNodeStatePerCell>(tDomain, tJacobianWS, tOutJacobian);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Node State", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianNFunction->evaluate_boundary( mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep );
  
            Plato::transform_ad_type_to_pod_3Dview<mNumDofsPerCell, mNumNodeStatePerCell>(mNumCells, tJacobianWS, tOutJacobian);
        }
        return (tOutJacobian);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to node state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_n(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename JacobianN::ConfigScalarType;
        using StateScalar     = typename JacobianN::StateScalarType;
        using NodeStateScalar = typename JacobianN::NodeStateScalarType;
        using ControlScalar   = typename JacobianN::ControlScalarType;
        using ResultScalar    = typename JacobianN::ResultScalarType;

        // tJacobian has shape (Nc, (Nv x Nd), (Nv x Nn))
        //   Nc: number of cells
        //   Nv: number of vertices per cell
        //   Nd: dimensionality of the vector function
        //   Nn: dimensionality of the node state
        //   (I x J) is a strided 1D array indexed by i*J+j where i /in I and j /in J.
        //
        // (tJacobian is (Nc, (Nv x Nd)) and the third dimension, (Nv x Nn), is in the AD type)

        // create matrix with block size (Nd, Nn).
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumNodeStatePerNode>( tMesh );

        // create entry ordinal functor:
        // tJacobianMatEntryOrdinal(e, k, l) => G
        //   e: cell index
        //   k: row index /in (Nv x Nd)
        //   l: col index /in (Nv x Nn)
        //   G: entry index into CRS matrix
        //
        // Template parameters:
        //   mNumSpatialDims: Nv-1
        //   mNumSpatialDims: Nd
        //   mNumNodeStatePerNode:   Nn
        //
        // Note that the second two template parameters must match the block shape of the destination matrix, tJacobianMat
        //
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumNodeStatePerNode>
            tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

        // Assemble from the AD-typed result, tJacobian, into the POD-typed global matrix, tJacobianMat.
        //
        // Arguments 1 and 2 below correspond to the size of tJacobian ((Nv x Nd), (Nv x Nn)) and the size of
        // tJacobianMat (Nd, Nn).
        //
        auto tJacobianMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Node State", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianNFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            //
            Plato::WorksetBase<ElementType>::assembleJacobianFad(
              mNumDofsPerCell,     /* (Nv x Nd) */
              mNumNodeStatePerCell, /* (Nv x Nn) */
              tJacobianMatEntryOrdinal, /* entry ordinal functor */
              tJacobian,                /* source data */
              tJacobianMatEntries,      /* destination */
              tDomain
            );
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Node State", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianNFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            //
            Plato::WorksetBase<ElementType>::assembleJacobianFad(
              mNumDofsPerCell,     /* (Nv x Nd) */
              mNumNodeStatePerCell, /* (Nv x Nn) */
              tJacobianMatEntryOrdinal, /* entry ordinal functor */
              tJacobian,                /* source data */
              tJacobianMatEntries       /* destination */
            );
        }

        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to node state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian transpose
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_n_T(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename JacobianN::ConfigScalarType;
        using StateScalar     = typename JacobianN::StateScalarType;
        using NodeStateScalar = typename JacobianN::NodeStateScalarType;
        using ControlScalar   = typename JacobianN::ControlScalarType;
        using ResultScalar    = typename JacobianN::ResultScalarType;

        // tJacobian has shape (Nc, (Nv x Nd), (Nv x Nn))
        //   Nc: number of cells
        //   Nv: number of vertices per cell
        //   Nd: dimensionality of the vector function
        //   Nn: dimensionality of the node state
        //   (I x J) is a strided 1D array indexed by i*J+j where i /in I and j /in J.
        //
        // (tJacobian is (Nc, (Nv x Nd)) and the third dimension, (Nv x Nn), is in the AD type)

        // create *transpose* matrix with block size (Nn, Nd).
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumNodeStatePerNode, mNumDofsPerNode>( tMesh );

        // create entry ordinal functor:
        // tJacobianMatEntryOrdinal(e, k, l) => G
        //   e: cell index
        //   k: row index /in (Nv x Nd)
        //   l: col index /in (Nv x Nn)
        //   G: entry index into CRS matrix
        //
        // Template parameters:
        //   mNumSpatialDims: Nv-1
        //   mNumNodeStatePerNode:   Nn
        //   mNumDofsPerNode: Nd
        //
        // Note that the second two template parameters must match the block shape of the destination matrix, tJacobianMat
        //
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumNodeStatePerNode, mNumDofsPerNode>
            tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

        // Assemble from the AD-typed result, tJacobian, into the POD-typed global matrix, tJacobianMat.
        //
        // The transpose is being assembled, (i.e., tJacobian is transposed before assembly into tJacobianMat), so
        // arguments 1 and 2 below correspond to the size of tJacobian ((Nv x Nd), (Nv x Nn)) and the size of the
        // *transpose* of tJacobianMat (Transpose(Nn, Nd) => (Nd, Nn)).
        //
        auto tJacobianMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Node State", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianNFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            Plato::WorksetBase<ElementType>::assembleTransposeJacobian(
                mNumDofsPerCell,     /* (Nv x Nd) */
                mNumNodeStatePerCell, /* (Nv x Nn) */
                tJacobianMatEntryOrdinal, /* entry ordinal functor */
                tJacobian,                /* source data */
                tJacobianMatEntries,      /* destination */
                tDomain
            );
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Node State", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianNFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            Plato::WorksetBase<ElementType>::assembleTransposeJacobian(
                mNumDofsPerCell,     /* (Nv x Nd) */
                mNumNodeStatePerCell, /* (Nv x Nn) */
                tJacobianMatEntryOrdinal, /* entry ordinal functor */
                tJacobian,                /* source data */
                tJacobianMatEntries       /* destination */
            );
        }
        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to control degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(
        const Plato::ScalarVectorT<Plato::Scalar> & aState,
        const Plato::ScalarVectorT<Plato::Scalar> & aNodeState,
        const Plato::ScalarVectorT<Plato::Scalar> & aControl,
              Plato::Scalar                         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename GradientZ::ConfigScalarType;
        using StateScalar     = typename GradientZ::StateScalarType;
        using NodeStateScalar = typename GradientZ::NodeStateScalarType;
        using ControlScalar   = typename GradientZ::ControlScalarType;
        using ResultScalar    = typename GradientZ::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( tMesh );

        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode> tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

        auto tJacobianMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianZFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            Plato::WorksetBase<ElementType>::assembleTransposeJacobian(
                mNumDofsPerCell,
                mNumNodesPerCell,
                tJacobianMatEntryOrdinal,
                tJacobian,
                tJacobianMatEntries,
                tDomain
            );
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianZFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            Plato::WorksetBase<ElementType>::assembleTransposeJacobian(
                mNumDofsPerCell,
                mNumNodesPerCell,
                tJacobianMatEntryOrdinal,
                tJacobian,
                tJacobianMatEntries
            );
        }
        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to control degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return Workset of Jacobian with respect to control degrees of freedom
    *******************************************************************************/
    Plato::ScalarArray3D
    gradient_z_workset(
        const Plato::ScalarVectorT<Plato::Scalar> & aState,
        const Plato::ScalarVectorT<Plato::Scalar> & aNodeState,
        const Plato::ScalarVectorT<Plato::Scalar> & aControl,
              Plato::Scalar                         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename GradientZ::ConfigScalarType;
        using StateScalar     = typename GradientZ::StateScalarType;
        using NodeStateScalar = typename GradientZ::NodeStateScalarType;
        using ControlScalar   = typename GradientZ::ControlScalarType;
        using ResultScalar    = typename GradientZ::ResultScalarType;

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian WRT Control", mNumCells, mNumDofsPerCell, mNumNodesPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("JacobianControl", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianZFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep );

            Plato::transform_ad_type_to_pod_3Dview<mNumDofsPerCell, mNumNodesPerCell>(tDomain, tJacobianWS, tOutputJacobian);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
            Plato::WorksetBase<ElementType>::worksetNodeState(aNodeState, tNodeStateWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("JacobianControl", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianZFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep );

            Plato::transform_ad_type_to_pod_3Dview<mNumDofsPerCell, mNumNodesPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        }
        return tOutputJacobian;
    }

    /***************************************************************************//**
     * \brief Update physics-based parameters within a frequency of optimization iterations
     * \param [in] aStates     global states for all time steps
     * \param [in] aControls   current controls, i.e. design variables
     * \param [in] aTimeStep   current time step increment
    *******************************************************************************/
    void
    updateProblem(
        const Plato::ScalarMultiVector & aStates,
        const Plato::ScalarVector      & aControls,
              Plato::Scalar              aTimeStep = 0.0
    ) const
    {
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            mResidualFunctions.at(tName)->updateProblem(aStates, aControls, aTimeStep);
            mJacobianUFunctions.at(tName)->updateProblem(aStates, aControls, aTimeStep);
            mJacobianNFunctions.at(tName)->updateProblem(aStates, aControls, aTimeStep);
            mJacobianXFunctions.at(tName)->updateProblem(aStates, aControls, aTimeStep);
            mJacobianZFunctions.at(tName)->updateProblem(aStates, aControls, aTimeStep);
        }

        mBoundaryLoadsResidualFunction->updateProblem(aStates, aControls, aTimeStep);
        mBoundaryLoadsJacobianUFunction->updateProblem(aStates, aControls, aTimeStep);
        mBoundaryLoadsJacobianNFunction->updateProblem(aStates, aControls, aTimeStep);
        mBoundaryLoadsJacobianXFunction->updateProblem(aStates, aControls, aTimeStep);
        mBoundaryLoadsJacobianZFunction->updateProblem(aStates, aControls, aTimeStep);
    }
};
// class VectorFunction

} // namespace Stabilized
} // namespace Plato
