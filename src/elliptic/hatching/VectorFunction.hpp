#pragma once

#include <memory>

#include "WorksetBase.hpp"
#include "NaturalBCs.hpp"
#include "elliptic/hatching/AbstractVectorFunction.hpp"
#include "elliptic/hatching/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

/******************************************************************************/
/*! VectorFunction class

   This class defines a vector function of the form:

   \f[
     F(\phi, U^k, c^{k-1})
   \f]

   where \f$ \phi \f$ is the control, \f$ U^k \f$ is the nodal state at the
   current step, and \f$ c^{k-1} \f$ is the element state at the previous step.

   This class is intended for use in an updated lagrangian formulation where
   the reference configuration is updated by the displacement at each step.

*/
/******************************************************************************/
template<typename PhysicsType>
class VectorFunction : public Plato::WorksetBase<typename PhysicsType::ElementType>
{
  private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumDofsPerCell;
    using Plato::WorksetBase<ElementType>::mNumNodesPerCell;
    using Plato::WorksetBase<ElementType>::mNumLocalDofsPerCell;
    using Plato::WorksetBase<ElementType>::mNumLocalStatesPerGP;
    using Plato::WorksetBase<ElementType>::mNumDofsPerNode;
    using Plato::WorksetBase<ElementType>::mNumSpatialDims;
    using Plato::WorksetBase<ElementType>::mNumControl;
    using Plato::WorksetBase<ElementType>::mNumNodes;
    using Plato::WorksetBase<ElementType>::mNumCells;

// NEEDED?    using Plato::WorksetBase<ElementType>::mGlobalStateEntryOrdinal;
// NEEDED?    using Plato::WorksetBase<ElementType>::mControlEntryOrdinal;

    using Residual  = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::Residual;
    using Jacobian  = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::Jacobian;
    using GradientC = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::GradientC;
    using GradientX = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::GradientZ;

    using ResidualFunction  = std::shared_ptr<Plato::Elliptic::Hatching::AbstractVectorFunction<Residual>>;
    using JacobianFunction  = std::shared_ptr<Plato::Elliptic::Hatching::AbstractVectorFunction<Jacobian>>;
    using GradientCFunction = std::shared_ptr<Plato::Elliptic::Hatching::AbstractVectorFunction<GradientC>>;
    using GradientXFunction = std::shared_ptr<Plato::Elliptic::Hatching::AbstractVectorFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Elliptic::Hatching::AbstractVectorFunction<GradientZ>>;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    std::map<std::string, ResidualFunction>  mResidualFunctions;
    std::map<std::string, JacobianFunction>  mJacobianFunctions;
    std::map<std::string, GradientCFunction> mGradientCFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap & mDataMap;

  public:

    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap problem-specific data map
    * \param [in] aProblemParams Teuchos parameter list with input data
    * \param [in] aProblemType problem type
    *
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aProblemType
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel  (aSpatialModel),
        mDataMap       (aDataMap)
    {
        typename PhysicsType::FunctionFactory tFunctionFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
          auto tName = tDomain.getDomainName();
          mResidualFunctions [tName] = tFunctionFactory.template createVectorFunction<Residual> (tDomain, aDataMap, aProblemParams, aProblemType);
          mJacobianFunctions [tName] = tFunctionFactory.template createVectorFunction<Jacobian> (tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientCFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientC>(tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientZFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientZ>(tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientXFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientX>(tDomain, aDataMap, aProblemParams, aProblemType);
        }
    }

    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap problem-specific data map
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel  (aSpatialModel),
        mDataMap       (aDataMap)
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
    * \brief Return number of nodes on the mesh
    * \return number of nodes
    ******************************************************************************/
    Plato::OrdinalType numNodes() const
    {
        return (mNumNodes);
    }

    /**************************************************************************//**
    * \brief Return number of elements/cells on the mesh
    * \return number of elements
    ******************************************************************************/
    Plato::OrdinalType numCells() const
    {
        return (mNumCells);
    }

    /**************************************************************************//**
    * \brief Return total number of global degrees of freedom
    * \return total number of global degrees of freedom
    ******************************************************************************/
    Plato::OrdinalType numDofsPerCell() const
    {
        return (mNumDofsPerCell);
    }

    /**************************************************************************//**
    * \brief Return total number of nodes per cell/element
    * \return total number of nodes per cell/element
    ******************************************************************************/
    Plato::OrdinalType numNodesPerCell() const
    {
        return (mNumNodesPerCell);
    }

    /**************************************************************************//**
    * \brief Return number of degrees of freedom per node
    * \return number of degrees of freedom per node
    ******************************************************************************/
    Plato::OrdinalType numDofsPerNode() const
    {
        return (mNumDofsPerNode);
    }

    /**************************************************************************//**
    * \brief Return number of control vectors/fields, e.g. number of materials.
    * \return number of control vectors
    ******************************************************************************/
    Plato::OrdinalType numControlsPerNode() const
    {
        return (mNumControl);
    }

    /**************************************************************************//**
    *
    * \brief Allocate residual evaluator
    * \param [in] aResidual residual evaluator
    * \param [in] aJacobian Jacobian evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const ResidualFunction & aResidual,
        const JacobianFunction & aJacobian,
              std::string        aName
    )
    {
        mResidualFunctions[aName] = aResidual;
        mJacobianFunctions[aName] = aJacobian;
    }

    /**************************************************************************//**
    *
    * \brief Allocate partial derivative with respect to control evaluator
    * \param [in] aGradientC partial derivative with respect to control evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const GradientCFunction & aGradientC,
              std::string         aName
    )
    {
        mGradientCFunctions[aName] = aGradientC;
    }

    /**************************************************************************//**
    *
    * \brief Allocate partial derivative with respect to control evaluator
    * \param [in] aGradientZ partial derivative with respect to control evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const GradientZFunction & aGradientZ,
              std::string         aName
    )
    {
        mGradientZFunctions[aName] = aGradientZ;
    }

    /**************************************************************************//**
    *
    * \brief Allocate partial derivative with respect to configuration evaluator
    * \param [in] GradientX partial derivative with respect to configuration evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const GradientXFunction & aGradientX,
              std::string         aName
    )
    {
        mGradientXFunctions[aName] = aGradientX;
    }

    /**************************************************************************//**
    *
    * \brief Return number of global state degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return mNumNodes*mNumDofsPerNode;
    }

    /**************************************************************************//**
    *
    * \brief Return number of local state degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType stateSize() const
    {
      return mNumCells*mNumLocalDofsPerCell;
    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
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
    
    /**************************************************************************/
    Plato::ScalarVector
    value(
        const Plato::ScalarVector  & aGlobalState,
        const Plato::ScalarArray3D & aLocalState,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename Residual::ConfigScalarType;
        using GlobalStateScalar = typename Residual::GlobalStateScalarType;
        using LocalStateScalar  = typename Residual::LocalStateScalarType;
        using ControlScalar     = typename Residual::ControlScalarType;
        using ResultScalar      = typename Residual::ResultScalarType;

        constexpr auto tNumGP = ElementType::mNumGaussPoints;

        Plato::ScalarVector  tReturnValue("Assembled Residual", mNumDofsPerNode*mNumNodes);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

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
            mResidualFunctions.at(tName)->evaluate( tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // assemble to return view
            //
            Plato::WorksetBase<ElementType>::assembleResidual( tResidual, tReturnValue, tDomain );

        }

        {
            auto tNumCells = mSpatialModel.Mesh->NumElements();

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS);

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
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mResidualFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // create and assemble to return view
            //
            Plato::WorksetBase<ElementType>::assembleResidual( tResidual, tReturnValue);
        }

        return tReturnValue;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(
        const Plato::ScalarVector  & aGlobalState,
        const Plato::ScalarArray3D & aLocalState,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using GlobalStateScalar = typename GradientX::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientX::LocalStateScalarType;
        using ControlScalar     = typename GradientX::ControlScalarType;
        using ResultScalar      = typename GradientX::ResultScalarType;

        constexpr auto tNumGP = ElementType::mNumGaussPoints;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumDofsPerNode>(tMesh);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientXFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal(tJacobianMat, tMesh);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);

        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mGradientXFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal(tJacobianMat, tMesh);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }

        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u_T(
        const Plato::ScalarVector  & aGlobalState,
        const Plato::ScalarArray3D & aLocalState,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename Jacobian::ConfigScalarType;
        using GlobalStateScalar = typename Jacobian::GlobalStateScalarType;
        using LocalStateScalar  = typename Jacobian::LocalStateScalarType;
        using ControlScalar     = typename Jacobian::ControlScalarType;
        using ResultScalar      = typename Jacobian::ResultScalarType;

        constexpr auto tNumGP = ElementType::mNumGaussPoints;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianFunctions.at(tName)->evaluate( tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixTransposeEntryOrdinal<mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mJacobianFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixTransposeEntryOrdinal<mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }

        return tJacobianMat;
    }
    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(
        const Plato::ScalarVector  & aGlobalState,
        const Plato::ScalarArray3D & aLocalState,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        using ConfigScalar      = typename Jacobian::ConfigScalarType;
        using GlobalStateScalar = typename Jacobian::GlobalStateScalarType;
        using LocalStateScalar  = typename Jacobian::LocalStateScalarType;
        using ControlScalar     = typename Jacobian::ControlScalarType;
        using ResultScalar      = typename Jacobian::ResultScalarType;

        constexpr auto tNumGP = ElementType::mNumGaussPoints;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianFunctions.at(tName)->evaluate( tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mJacobianFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }
    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_cp_T(
        const Plato::ScalarVector  & aGlobalState,
        const Plato::ScalarArray3D & aLocalState,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        // Note that aLocalState is the previous state in this function.

        using ConfigScalar      = typename GradientC::ConfigScalarType;
        using GlobalStateScalar = typename GradientC::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientC::LocalStateScalarType;
        using ControlScalar     = typename GradientC::ControlScalarType;
        using ResultScalar      = typename GradientC::ResultScalarType;

        constexpr auto tNumGP = ElementType::mNumGaussPoints;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateLocalByGlobalBlockMatrix<Plato::CrsMatrixType, mNumNodesPerCell, mNumLocalDofsPerCell, mNumDofsPerNode>( tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientCFunctions.at(tName)->evaluate( tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::LocalByGlobalEntryFunctor<mNumSpatialDims, mNumLocalDofsPerCell, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumLocalDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mGradientCFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::LocalByGlobalEntryFunctor<mNumSpatialDims, mNumLocalDofsPerCell, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumLocalDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(
        const Plato::ScalarVector  & aGlobalState,
        const Plato::ScalarArray3D & aLocalState,
        const Plato::ScalarVector  & aControl,
              Plato::Scalar          aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientZ::ConfigScalarType;
        using GlobalStateScalar = typename GradientZ::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientZ::LocalStateScalarType;
        using ControlScalar     = typename GradientZ::ControlScalarType;
        using ResultScalar      = typename GradientZ::ResultScalarType;

        constexpr auto tNumGP = ElementType::mNumGaussPoints;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar>
                tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientZFunctions.at(tName)->evaluate( tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
              tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar>
                tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mGradientZFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
              tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }
};
// class VectorFunction

} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
