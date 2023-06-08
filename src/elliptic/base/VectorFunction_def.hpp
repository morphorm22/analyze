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
    /*
    auto tName = tDomain.getDomainName();
    mResiduals [tName] = 
      tFactoryResidual.template createVectorFunction<ResidualEvalType> (tDomain, aDataMap, aProbParams, aType);
    mJacobiansU[tName] = 
      tFactoryResidual.template createVectorFunction<JacobianUEvalType>(tDomain, aDataMap, aProbParams, aType);
    mJacobiansZ[tName] = 
      tFactoryResidual.template createVectorFunction<JacobianZEvalType>(tDomain, aDataMap, aProbParams, aType);
    mJacobiansX[tName] = 
      tFactoryResidual.template createVectorFunction<JacobianXEvalType>(tDomain, aDataMap, aProbParams, aType);
    */
  }
}

template<typename PhysicsType>
Plato::OrdinalType 
VectorFunction<PhysicsType>::
numDofs() 
const
{
  auto tNumNodes = mSpatialModel.Mesh->NumNodes();
  return (tNumNodes*mNumDofsPerNode);
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
  Plato::ScalarVector tResidual("Assembled Residual",mNumDofsPerNode*tNumNodes);
  // internal forces
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build residual domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build residual range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
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
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
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
          Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( tMesh );
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
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate internal forces
    auto tName = tDomain.getDomainName();
    mJacobiansU.at(tName)->evaluate(tWorksets, aCycle);
    // assembly to return Jacobian
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode>
      tJacEntryOrdinal( tJacobianU, tMesh );
    auto tJacEntries = tJacobianU->entries();
    mWorksetFuncs.assembleJacobianFad(
      mNumDofsPerCell,mNumDofsPerCell,tJacEntryOrdinal,tResultWS->mData,tJacEntries,tDomain
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
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate prescribed forces
    auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
    mJacobiansU.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal( tJacobianU, tMesh );
    auto tJacEntries = tJacobianU->entries();
    mWorksetFuncs.assembleJacobianFad(
      mNumDofsPerCell, mNumDofsPerCell,tJacEntryOrdinal,tResultWS->mData,tJacEntries
    );
  }
  return tJacobianU;
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
  { tJacobianX = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumDofsPerNode>(tMesh); }
  else
  { tJacobianX = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumSpatialDims>(tMesh); }
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
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate internal forces
    auto tName     = tDomain.getDomainName();
    mJacobiansX.at(tName)->evaluate(tWorksets, aCycle);
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumSpatialDims, mNumDofsPerNode>
      tJacEntryOrdinal(tJacobianX, tMesh);
    auto tJacEntries = tJacobianX->entries();
    if(aTranspose)
    { 
      mWorksetFuncs.assembleTransposeJacobian(
        mNumDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain
      ); 
    }
    else
    { 
      mWorksetFuncs.assembleJacobianFad(
        mNumDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain
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
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate prescribed forces
    auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
    mJacobiansX.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumSpatialDims, mNumDofsPerNode>
      tJacEntryOrdinal(tJacobianX, tMesh);
    auto tJacEntries = tJacobianX->entries();
    if(aTranspose)
    { 
      mWorksetFuncs.assembleTransposeJacobian(
        mNumDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries
      ); 
    }
    else
    { 
      mWorksetFuncs.assembleJacobianFad(
        mNumDofsPerCell, mNumConfigDofsPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries
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
  { tJacobianZ = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( tMesh ); }
  else
  { tJacobianZ = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumControl>( tMesh ); }
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
        ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate internal forces
    auto tName = tDomain.getDomainName();
    mJacobiansZ.at(tName)->evaluate(tWorksets, aCycle);
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumControl, mNumDofsPerNode> 
      tJacEntryOrdinal( tJacobianZ, tMesh );
    auto tJacEntries = tJacobianZ->entries();
    if(aTranspose)
    { 
      mWorksetFuncs.assembleTransposeJacobian(
        mNumDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain); 
    }
    else
    { 
      mWorksetFuncs.assembleJacobianFad(
        mNumDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries, tDomain); 
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
      ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, mNumDofsPerCell) );
    tWorksets.set("result", tResultWS);
    // evaluate prescribed forces
    auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
    mJacobiansZ.at(tFirstBlockName)->evaluateBoundary(mSpatialModel, tWorksets, aCycle );
    // assembly to return matrix
    Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumControl, mNumDofsPerNode> 
      tJacEntryOrdinal( tJacobianZ, tMesh );
    auto tJacEntries = tJacobianZ->entries();
    if(aTranspose)
    { 
      mWorksetFuncs.assembleTransposeJacobian(
        mNumDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries); 
    }
    else
    { 
      mWorksetFuncs.assembleJacobianFad(
        mNumDofsPerCell, mNumNodesPerCell, tJacEntryOrdinal, tResultWS->mData, tJacEntries); 
    }
  }
  return tJacobianZ;
}

} // namespace Elliptic


} // namespace Plato
