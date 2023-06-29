/*
 * VectorFunction_def.hpp
 *
 *  Created on: June 7, 2023
 */

#pragma once

#include "WorkSets.hpp"
#include "ImplicitFunctors.hpp"
#include "elliptic/base/WorksetBuilder.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
VectorFunction<PhysicsType>::
VectorFunction(
  const std::string            & aType,
  const Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProbParams
) :
  mSpatialModel(aSpatialModel),
  mWorksetFuncs(aSpatialModel.Mesh),
  mDataMap     (aDataMap)
{
  typename PhysicsType::FunctionFactory tFactoryResidual;
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    auto tName = tDomain.getDomainName();
    mResiduals [tName] = 
      tFactoryResidual.template createVectorFunction<ResidualEvalType> (tDomain, aDataMap, aProbParams, aType);
    mJacobiansU[tName] = 
      tFactoryResidual.template createVectorFunction<JacobianUEvalType>(tDomain, aDataMap, aProbParams, aType);
    mJacobiansZ[tName] = 
      tFactoryResidual.template createVectorFunction<JacobianZEvalType>(tDomain, aDataMap, aProbParams, aType);
    mJacobiansX[tName] = 
      tFactoryResidual.template createVectorFunction<JacobianXEvalType>(tDomain, aDataMap, aProbParams, aType);
    mJacobiansN[tName] = 
      tFactoryResidual.template createVectorFunction<JacobianNEvalType>(tDomain, aDataMap, aProbParams, aType);
  }
}

template<typename PhysicsType>
Plato::Elliptic::residual_t
VectorFunction<PhysicsType>::
type() 
const
{
  auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
  return ( mResiduals.at(tFirstBlockName)->type() );
}

template<typename PhysicsType>
Plato::OrdinalType 
VectorFunction<PhysicsType>::
numDofs() 
const
{
  auto tNumNodes = mSpatialModel.Mesh->NumNodes();
  return (tNumNodes*mNumStateDofsPerNode);
}

template<typename PhysicsType>
Plato::OrdinalType 
VectorFunction<PhysicsType>::
numNodes() 
const
{
  return (mSpatialModel.Mesh->NumNodes());
}

template<typename PhysicsType>
Plato::OrdinalType 
VectorFunction<PhysicsType>::
numCells() 
const
{
  return (mSpatialModel.Mesh->NumElements());
}

template<typename PhysicsType>
Plato::OrdinalType 
VectorFunction<PhysicsType>::
numDofsPerCell() 
const
{
  return (mNumStateDofsPerCell);
}

template<typename PhysicsType>
Plato::OrdinalType 
VectorFunction<PhysicsType>::
numNodesPerCell() 
const
{
  return (mNumNodesPerCell);
}

template<typename PhysicsType>
Plato::OrdinalType 
VectorFunction<PhysicsType>::
numStateDofsPerNode() 
const
{
  return (mNumStateDofsPerNode);
}

template<typename PhysicsType>
Plato::OrdinalType 
VectorFunction<PhysicsType>::
numControlDofsPerNode() 
const
{
  return (mNumControlDofsPerNode);
}

template<typename PhysicsType>
std::vector<std::string> 
VectorFunction<PhysicsType>::
getDofNames() 
const
{
  auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
  return mResiduals.at(tFirstBlockName)->getDofNames();
}

template<typename PhysicsType>
void
VectorFunction<PhysicsType>::
postProcess(
  const Plato::Solutions & aSolutions
)
{
  auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
  auto tItr = mResiduals.find(tFirstBlockName);
  if(tItr == mResiduals.end())
  { 
      ANALYZE_THROWERR(std::string("Element block with name '") + tFirstBlockName 
      + "is not defined in residual function to element block map.") 
  }
  tItr->second->postProcess(aSolutions);
}

template<typename PhysicsType>
Plato::ScalarVector
VectorFunction<PhysicsType>::
value(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
)
{
  // set local result workset scalar type
  using ResultScalarType  = typename ResidualEvalType::ResultScalarType;
  auto tNumNodes = mSpatialModel.Mesh->NumNodes();
  Plato::Elliptic::WorksetBuilder<ResidualEvalType> tWorksetBuilder(mWorksetFuncs);
  Plato::ScalarVector tResidual("Assembled Residual",mNumStateDofsPerNode*tNumNodes);
  // internal forces
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build residual domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build residual range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumStateDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate internal forces
    auto tName = tDomain.getDomainName();
    mResiduals.at(tName)->evaluate( tWorksets, aCycle );
    // assemble to return view
    mWorksetFuncs.assembleResidual(tResultWS->mData, tResidual, tDomain );
  }
  // prescribed boundary conditions
  {
    // build residual domain worksets
    Plato::WorkSets tWorksets;
    auto tNumCells = mSpatialModel.Mesh->NumElements();
    tWorksetBuilder.build(tNumCells, aDatabase, tWorksets);
    // build residual range workset
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumStateDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate prescribed forces
    auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
    mResiduals.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );
    // create and assemble to return view
    mWorksetFuncs.assembleResidual(tResultWS->mData, tResidual);
  }
  return tResidual;
}

template<typename PhysicsType>
Teuchos::RCP<Plato::CrsMatrixType>
VectorFunction<PhysicsType>::
jacobianState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle,
        bool              aTranspose
)
{
  // set local result workset scalar type
  using ResultScalarType = typename JacobianUEvalType::ResultScalarType;
  // create return Jacobian
  auto tMesh = mSpatialModel.Mesh;
  Teuchos::RCP<Plato::CrsMatrixType> tJacobianU =
          Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumStateDofsPerNode, mNumStateDofsPerNode>( tMesh );
  Plato::Elliptic::WorksetBuilder<JacobianUEvalType> tWorksetBuilder(mWorksetFuncs);
  // internal forces
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build jacobian domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build jacobian range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumStateDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate internal forces
    auto tName = tDomain.getDomainName();
    mJacobiansU.at(tName)->evaluate(tWorksets, aCycle);
    // assembly to return Jacobian
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumStateDofsPerNode, mNumStateDofsPerNode>
      tJacEntryOrdinal( tJacobianU, tMesh );
    auto tJacEntries = tJacobianU->entries();
    mWorksetFuncs.assembleJacobianFad(
      mNumStateDofsPerCell,mNumStateDofsPerCell,tJacEntryOrdinal,tResultWS->mData,tJacEntries,tDomain
    );
  }
  // prescribed forces
  {
    // build jacobian domain worksets
    Plato::WorkSets tWorksets;
    auto tNumCells = mSpatialModel.Mesh->NumElements();
    tWorksetBuilder.build(tNumCells, aDatabase, tWorksets);
    // build jacobian range workset
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumStateDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate prescribed forces
    auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
    mJacobiansU.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumStateDofsPerNode, mNumStateDofsPerNode> tJacEntryOrdinal( tJacobianU, tMesh );
    auto tJacEntries = tJacobianU->entries();
    mWorksetFuncs.assembleJacobianFad(
      mNumStateDofsPerCell,mNumStateDofsPerCell,tJacEntryOrdinal,tResultWS->mData,tJacEntries
    );
  }
  return tJacobianU;
}

template<typename PhysicsType>
Teuchos::RCP<Plato::CrsMatrixType>
VectorFunction<PhysicsType>::
jacobianNodeState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle,
        bool              aTranspose
)
{
  // set local result workset scalar type
  using ResultScalarType = typename JacobianNEvalType::ResultScalarType;
  // create return matrix
  auto tMesh = mSpatialModel.Mesh;
  Teuchos::RCP<Plato::CrsMatrixType> tJacobianN;
  if(aTranspose)
  { tJacobianN = Plato::CreateBlockMatrix<Plato::CrsMatrixType,mNumNodeStateDofsPerNode,mNumStateDofsPerNode>(tMesh); }
  else
  { tJacobianN = Plato::CreateBlockMatrix<Plato::CrsMatrixType,mNumStateDofsPerNode,mNumNodeStateDofsPerNode>(tMesh); }
  Plato::Elliptic::WorksetBuilder<JacobianNEvalType> tWorksetBuilder(mWorksetFuncs);
  // internal forces
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build jacobian domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build jacobian range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumStateDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate internal forces
    auto tName     = tDomain.getDomainName();
    mJacobiansN.at(tName)->evaluate(tWorksets, aCycle);
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumNodeStateDofsPerNode, mNumStateDofsPerNode>
      tJacEntryOrdinal(tJacobianN, tMesh);
    auto tJacEntries = tJacobianN->entries();
    if(aTranspose)
    { 
      mWorksetFuncs.assembleTransposeJacobian(
        mNumStateDofsPerCell, mNumNodeStateDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain
      ); 
    }
    else
    { 
      mWorksetFuncs.assembleJacobianFad(
        mNumStateDofsPerCell, mNumNodeStateDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain
      ); 
    }
  }
  // prescribed forces
  {
    // build jacobian domain worksets
    Plato::WorkSets tWorksets;
    auto tNumCells = mSpatialModel.Mesh->NumElements();
    tWorksetBuilder.build(tNumCells, aDatabase, tWorksets);
    // build jacobian range workset
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumStateDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate prescribed forces
    auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
    mJacobiansN.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumNodeStateDofsPerNode, mNumStateDofsPerNode>
      tJacEntryOrdinal(tJacobianN, tMesh);
    auto tJacEntries = tJacobianN->entries();
    if(aTranspose)
    { 
      mWorksetFuncs.assembleTransposeJacobian(
        mNumStateDofsPerCell, mNumNodeStateDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries
      ); 
    }
    else
    {
      mWorksetFuncs.assembleJacobianFad(
        mNumStateDofsPerCell, mNumNodeStateDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries
      ); 
    }
  }
  return tJacobianN;
}

template<typename PhysicsType>
Teuchos::RCP<Plato::CrsMatrixType>
VectorFunction<PhysicsType>::
jacobianConfig(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle,
        bool              aTranspose
)
{
  // set local result workset scalar type
  using ResultScalarType = typename JacobianXEvalType::ResultScalarType;
  // create return matrix
  auto tMesh = mSpatialModel.Mesh;
  Teuchos::RCP<Plato::CrsMatrixType> tJacobianX;
  if(aTranspose)
  { tJacobianX = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumStateDofsPerNode>(tMesh); }
  else
  { tJacobianX = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumStateDofsPerNode, mNumSpatialDims>(tMesh); }
  Plato::Elliptic::WorksetBuilder<JacobianXEvalType> tWorksetBuilder(mWorksetFuncs);
  // internal forces
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build jacobian domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build jacobian range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumStateDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate internal forces
    auto tName     = tDomain.getDomainName();
    mJacobiansX.at(tName)->evaluate(tWorksets, aCycle);
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumSpatialDims, mNumStateDofsPerNode>
      tJacEntryOrdinal(tJacobianX, tMesh);
    auto tJacEntries = tJacobianX->entries();
    if(aTranspose)
    { 
      mWorksetFuncs.assembleTransposeJacobian(
        mNumStateDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain
      ); 
    }
    else
    { 
      mWorksetFuncs.assembleJacobianFad(
        mNumStateDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain
      ); 
    }
  }
  // prescribed forces
  {
    // build jacobian domain worksets
    Plato::WorkSets tWorksets;
    auto tNumCells = mSpatialModel.Mesh->NumElements();
    tWorksetBuilder.build(tNumCells, aDatabase, tWorksets);
    // build jacobian range workset
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumStateDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate prescribed forces
    auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
    mJacobiansX.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumSpatialDims, mNumStateDofsPerNode>
      tJacEntryOrdinal(tJacobianX, tMesh);
    auto tJacEntries = tJacobianX->entries();
    if(aTranspose)
    { 
      mWorksetFuncs.assembleTransposeJacobian(
        mNumStateDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries
      ); 
    }
    else
    { 
      mWorksetFuncs.assembleJacobianFad(
        mNumStateDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries
      ); 
    }
  }
  return tJacobianX;
}

template<typename PhysicsType>
Teuchos::RCP<Plato::CrsMatrixType>
VectorFunction<PhysicsType>::
jacobianControl
(const Plato::Database & aDatabase,
 const Plato::Scalar   & aCycle,
       bool              aTranspose)
{
  // set local result workset scalar type
  using ResultScalarType = typename JacobianZEvalType::ResultScalarType;
  // create return matrix
  auto tMesh = mSpatialModel.Mesh;
  Teuchos::RCP<Plato::CrsMatrixType> tJacobianZ;
  if(aTranspose)
  { tJacobianZ = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControlDofsPerNode, mNumStateDofsPerNode>(tMesh); }
  else
  { tJacobianZ = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumStateDofsPerNode, mNumControlDofsPerNode>(tMesh); }
  Plato::Elliptic::WorksetBuilder<JacobianZEvalType> tWorksetBuilder(mWorksetFuncs);
  // internal forces
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build jacobian domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build jacobian range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
        ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumStateDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate internal forces
    auto tName = tDomain.getDomainName();
    mJacobiansZ.at(tName)->evaluate(tWorksets, aCycle);
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumControlDofsPerNode, mNumStateDofsPerNode> 
      tJacEntryOrdinal( tJacobianZ, tMesh );
    auto tJacEntries = tJacobianZ->entries();
    if(aTranspose)
    { 
      mWorksetFuncs.assembleTransposeJacobian(
        mNumStateDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain); 
    }
    else
    { 
      mWorksetFuncs.assembleJacobianFad(
        mNumStateDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain); 
    }
  }
  // prescribed forces
  {
    // build jacobian domain worksets
    Plato::WorkSets tWorksets;
    auto tNumCells = mSpatialModel.Mesh->NumElements();
    tWorksetBuilder.build(tNumCells, aDatabase, tWorksets);
    // build jacobian range workset
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumStateDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate prescribed forces
    auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
    mJacobiansZ.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumControlDofsPerNode, mNumStateDofsPerNode> 
      tJacEntryOrdinal( tJacobianZ, tMesh );
    auto tJacEntries = tJacobianZ->entries();
    if(aTranspose)
    { 
      mWorksetFuncs.assembleTransposeJacobian(
        mNumStateDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries); 
    }
    else
    { 
      mWorksetFuncs.assembleJacobianFad(
        mNumStateDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries); 
    }
  }
  return tJacobianZ;
}

} // namespace Elliptic


} // namespace Plato
