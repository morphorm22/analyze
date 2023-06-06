#pragma once

#include <memory>

#include "SpatialModel.hpp"
#include "base/WorksetBase.hpp"
#include "hyperbolic/AbstractVectorFunction.hpp"
#include "hyperbolic/EvaluationTypes.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   F = F(\phi, U^k, V^k, A^k, X)

   and manages the evaluation of the function and derivatives with respect to
   state, U^k, state dot, V^k, state dot dot, V^k, and control, X.

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

    using Plato::WorksetBase<ElementType>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<ElementType>::mControlEntryOrdinal;

    using Residual  = typename Plato::Hyperbolic::Evaluation<ElementType>::Residual;
    using GradientU = typename Plato::Hyperbolic::Evaluation<ElementType>::GradientU;
    using GradientV = typename Plato::Hyperbolic::Evaluation<ElementType>::GradientV;
    using GradientA = typename Plato::Hyperbolic::Evaluation<ElementType>::GradientA;
    using GradientX = typename Plato::Hyperbolic::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Hyperbolic::Evaluation<ElementType>::GradientZ;

    using ResidualFunction  = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<Residual>>;
    using GradientUFunction = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientU>>;
    using GradientVFunction = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientV>>;
    using GradientAFunction = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientA>>;
    using GradientXFunction = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientZ>>;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    std::map<std::string, ResidualFunction>  mResidualFunctions;
    std::map<std::string, GradientUFunction> mGradientUFunctions;
    std::map<std::string, GradientVFunction> mGradientVFunctions;
    std::map<std::string, GradientAFunction> mGradientAFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    ResidualFunction  mBoundaryLoadsResidualFunction;
    GradientUFunction mBoundaryLoadsGradientUFunction;
    GradientVFunction mBoundaryLoadsGradientVFunction;
    GradientAFunction mBoundaryLoadsGradientAFunction;
    GradientXFunction mBoundaryLoadsGradientXFunction;
    GradientZFunction mBoundaryLoadsGradientZFunction;

    const Plato::SpatialModel& mSpatialModel;

    Plato::DataMap& mDataMap;

  public:

    VectorFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string            & aProblemType
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap)
    {
        typename PhysicsType::FunctionFactory tFunctionFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFunctions[tName]  = tFunctionFactory.template createVectorFunction<Residual>
                (tDomain, aDataMap, aParamList, aProblemType);
            mGradientUFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientU>
                (tDomain, aDataMap, aParamList, aProblemType);
            mGradientVFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientV>
                (tDomain, aDataMap, aParamList, aProblemType);
            mGradientAFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientA>
                (tDomain, aDataMap, aParamList, aProblemType);
            mGradientZFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientZ>
                (tDomain, aDataMap, aParamList, aProblemType);
            mGradientXFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientX>
                (tDomain, aDataMap, aParamList, aProblemType);
        }
        // any block can compute the boundary terms for the entire mesh.  We'll use the first block.
        auto tFirstBlockName = aSpatialModel.Domains.front().getDomainName();

        mBoundaryLoadsResidualFunction  = mResidualFunctions[tFirstBlockName];
        mBoundaryLoadsGradientUFunction = mGradientUFunctions[tFirstBlockName];
        mBoundaryLoadsGradientVFunction = mGradientVFunctions[tFirstBlockName];
        mBoundaryLoadsGradientAFunction = mGradientAFunctions[tFirstBlockName];
        mBoundaryLoadsGradientZFunction = mGradientZFunctions[tFirstBlockName];
        mBoundaryLoadsGradientXFunction = mGradientXFunctions[tFirstBlockName];

    }

    VectorFunction(Plato::Mesh aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<ElementType>(aMesh),
            mDataMap(aDataMap)
    {
    }

    Plato::OrdinalType size() const
    {
      return mNumNodes*mNumDofsPerNode;
    }

    std::vector<std::string> getDofNames() const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        return mResidualFunctions.at(tFirstBlockName)->getDofNames();
    }

    std::vector<std::string> getDofDotNames() const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        return mResidualFunctions.at(tFirstBlockName)->getDofDotNames();
    }

    std::vector<std::string> getDofDotDotNames() const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        return mResidualFunctions.at(tFirstBlockName)->getDofDotDotNames();
    }

    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
        return mBoundaryLoadsResidualFunction->getSolutionStateOutputData(aSolutions);
    }

    Plato::Scalar getMaxEigenvalue() const
    {
        Plato::Scalar tMaxEigenvalue(0.0);

        using ConfigScalar = typename Residual::ConfigScalarType;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            auto tDomainMax = mResidualFunctions.at(tName)->getMaxEigenvalue(tConfigWS);

            if( tDomainMax > tMaxEigenvalue )
            {
                tMaxEigenvalue = tDomainMax;
            }
        }
        return tMaxEigenvalue;
    }

    Plato::ScalarVector
    value(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar         aTimeStep,
            Plato::Scalar         aCurrentTime = 0.0
    ) const
    {
        using ConfigScalar      = typename Residual::ConfigScalarType;
        using StateScalar       = typename Residual::StateScalarType;
        using StateDotScalar    = typename Residual::StateDotScalarType;
        using StateDotDotScalar = typename Residual::StateDotDotScalarType;
        using ControlScalar     = typename Residual::ControlScalarType;
        using ResultScalar      = typename Residual::ResultScalarType;

        Plato::ScalarVector tReturnValue("Assembled Residual", mNumDofsPerNode * mNumNodes);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", tNumCells, mNumDofsPerCell);

            mResidualFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResidual, aTimeStep, aCurrentTime );

            Plato::WorksetBase<ElementType>::assembleResidual( tResidual, tReturnValue, tDomain );
        }

        {
            auto tNumCells = mSpatialModel.Mesh->NumElements();

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", tNumCells, mNumDofsPerCell);

            mBoundaryLoadsResidualFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResidual, aTimeStep, aCurrentTime );

            Plato::WorksetBase<ElementType>::assembleResidual( tResidual, tReturnValue );
        }

        return tReturnValue;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar         aTimeStep,
            Plato::Scalar         aCurrentTime = 0.0
    ) const
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using StateScalar       = typename GradientX::StateScalarType;
        using StateDotScalar    = typename GradientX::StateDotScalarType;
        using StateDotDotScalar = typename GradientX::StateDotDotScalarType;
        using ControlScalar     = typename GradientX::ControlScalarType;
        using ResultScalar      = typename GradientX::ResultScalarType;

        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumDofsPerNode>(tMesh);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", tNumCells, mNumDofsPerCell);

            mGradientXFunctions.at(tName)->evaluate(tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime);

            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal(tJacobianMat, tMesh);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", mNumCells, mNumDofsPerCell);

            mBoundaryLoadsGradientXFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime);

            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal(tJacobianMat, tMesh);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar         aTimeStep,
            Plato::Scalar         aCurrentTime = 0.0
    ) const
    {
        using ConfigScalar      = typename GradientU::ConfigScalarType;
        using StateScalar       = typename GradientU::StateScalarType;
        using StateDotScalar    = typename GradientU::StateDotScalarType;
        using StateDotDotScalar = typename GradientU::StateDotDotScalarType;
        using ControlScalar     = typename GradientU::ControlScalarType;
        using ResultScalar      = typename GradientU::ResultScalarType;

        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            mGradientUFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            mBoundaryLoadsGradientUFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_v(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar         aTimeStep,
            Plato::Scalar         aCurrentTime = 0.0
    ) const
    {
        using ConfigScalar      = typename GradientV::ConfigScalarType;
        using StateScalar       = typename GradientV::StateScalarType;
        using StateDotScalar    = typename GradientV::StateDotScalarType;
        using StateDotDotScalar = typename GradientV::StateDotDotScalarType;
        using ControlScalar     = typename GradientV::ControlScalarType;
        using ResultScalar      = typename GradientV::ResultScalarType;

        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
             Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            mGradientVFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            mBoundaryLoadsGradientVFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_a(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar         aTimeStep,
            Plato::Scalar         aCurrentTime = 0.0
    ) const
    {
        using ConfigScalar      = typename GradientA::ConfigScalarType;
        using StateScalar       = typename GradientA::StateScalarType;
        using StateDotScalar    = typename GradientA::StateDotScalarType;
        using StateDotDotScalar = typename GradientA::StateDotDotScalarType;
        using ControlScalar     = typename GradientA::ControlScalarType;
        using ResultScalar      = typename GradientA::ResultScalarType;

        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
             Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            mGradientAFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            mBoundaryLoadsGradientAFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }

    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar       aTimeStep,
            Plato::Scalar       aCurrentTime = 0.0
    ) const
    {
        using ConfigScalar      = typename GradientZ::ConfigScalarType;
        using StateScalar       = typename GradientZ::StateScalarType;
        using StateDotScalar    = typename GradientZ::StateDotScalarType;
        using StateDotDotScalar = typename GradientZ::StateDotDotScalarType;
        using ControlScalar     = typename GradientZ::ControlScalarType;
        using ResultScalar      = typename GradientZ::ResultScalarType;

        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS, tDomain);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", tNumCells, mNumDofsPerCell);

            mGradientZFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumControl, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDot, tStateDotWS);

            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aStateDotDot, tStateDotDotWS);

            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", mNumCells, mNumDofsPerCell);

            mBoundaryLoadsGradientZFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

            Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumControl, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }
}; // class VectorFunction

} // namespace Hyperbolic

} // namespace Plato
