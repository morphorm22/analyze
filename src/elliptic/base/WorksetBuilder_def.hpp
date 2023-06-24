/*
 * WorksetBuilder_def.hpp
 *
 *  Created on: June 7, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
WorksetBuilder<EvaluationType>::
WorksetBuilder(
  const Plato::WorksetBase<ElementType> & aWorksetFuncs
) :
  mWorksetFuncs(aWorksetFuncs)
{}

template<typename EvaluationType>
void 
WorksetBuilder<EvaluationType>::
build(
  const Plato::SpatialDomain & aDomain,
  const Plato::Database      & aDatabase,
        Plato::WorkSets      & aWorkSets
) 
const
{
  // number of cells in the spatial domain
  auto tNumCells = aDomain.numCells();
  // build state workset
  using StateScalarType = typename EvaluationType::StateScalarType;
  auto tStateWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<StateScalarType> > >
    ( Plato::ScalarMultiVectorT<StateScalarType>("State Workset", tNumCells, mNumVecStateDofsPerCell) );
  mWorksetFuncs.worksetState(aDatabase.vector("states"), tStateWS->mData, aDomain);
  aWorkSets.set("states", tStateWS);
  // build control workset
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlScalarType> > >
    ( Plato::ScalarMultiVectorT<ControlScalarType>("Control Workset", tNumCells, mNumNodesPerCell) );
  mWorksetFuncs.worksetControl(aDatabase.vector("controls"), tControlWS->mData, aDomain);
  aWorkSets.set("controls", tControlWS);
  // build configuration workset
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  auto tConfigWS = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigScalarType> > >
    ( Plato::ScalarArray3DT<ConfigScalarType>("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims) );
  mWorksetFuncs.worksetConfig(tConfigWS->mData, aDomain);
  aWorkSets.set("configuration", tConfigWS);
  // if defined, build node state workset
  if( aDatabase.isScalarVectorDefined("node states") )
  {
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    auto tNodeStateWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<NodeStateScalarType> > >
      ( Plato::ScalarMultiVectorT<NodeStateScalarType>("Node State Workset", tNumCells, mNumNodeStateDofsPerCell) );
    mWorksetFuncs.worksetNodeState(aDatabase.vector("node states"), tNodeStateWS->mData, aDomain);
    aWorkSets.set("node states", tNodeStateWS);
  }
}

template<typename EvaluationType>
void 
WorksetBuilder<EvaluationType>::
build(
  const Plato::OrdinalType & aNumCells,
  const Plato::Database    & aDatabase,
        Plato::WorkSets    & aWorkSets
) const
{
  // build state workset
  using StateScalarType = typename EvaluationType::StateScalarType;
  auto tStateWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<StateScalarType> > >
    ( Plato::ScalarMultiVectorT<StateScalarType>("State Workset", aNumCells, mNumVecStateDofsPerCell) );
  mWorksetFuncs.worksetState(aDatabase.vector("states"), tStateWS->mData);
  aWorkSets.set("states", tStateWS);  
  // build control workset
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlScalarType> > >
    ( Plato::ScalarMultiVectorT<ControlScalarType>("Control Workset", aNumCells, mNumNodesPerCell) );
  mWorksetFuncs.worksetControl(aDatabase.vector("controls"), tControlWS->mData);
  aWorkSets.set("controls", tControlWS);  
  // build configuration workset
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  auto tConfigWS = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigScalarType> > >
    ( Plato::ScalarArray3DT<ConfigScalarType>("Config Workset", aNumCells, mNumNodesPerCell, mNumSpatialDims) );
  mWorksetFuncs.worksetConfig(tConfigWS->mData);
  aWorkSets.set("configuration", tConfigWS);  
  // if defined, build node state workset
  if( aDatabase.isScalarVectorDefined("node states") )
  {
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    auto tNodeStateWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<NodeStateScalarType> > >
      ( Plato::ScalarMultiVectorT<NodeStateScalarType>("Node State Workset", aNumCells, mNumNodeStateDofsPerCell) );
    mWorksetFuncs.worksetNodeState(aDatabase.vector("node states"), tNodeStateWS->mData);
    aWorkSets.set("node states", tNodeStateWS);
  }
  // if essential boundary conditions are enforced weakly, set essential states workset
  if( aDatabase.isScalarVectorDefined("dirichlet") )
  {
    auto tEssentialStateWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVector > >
      ( Plato::ScalarMultiVector("Dirichlet Workset", aNumCells, mNumVecStateDofsPerCell) );
    mWorksetFuncs.worksetState(aDatabase.vector("dirichlet"), tEssentialStateWS->mData);
    aWorkSets.set("dirichlet", tEssentialStateWS);
  }
}

} // namespace Elliptic

} // namespace Plato
