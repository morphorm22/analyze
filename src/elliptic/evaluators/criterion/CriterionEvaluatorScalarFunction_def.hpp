#pragma once

#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "PlatoUtilities.hpp"
#include "elliptic/base/WorksetBuilder.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
void
CriterionEvaluatorScalarFunction<PhysicsType>::
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
    mValueFunctions[tName]     = tFactory.template createScalarFunction<ValueEvalType> 
      (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
    mGradientUFunctions[tName] = tFactory.template createScalarFunction<GradUEvalType> 
      (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
    mGradientXFunctions[tName] = tFactory.template createScalarFunction<GradXEvalType>
      (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
    mGradientZFunctions[tName] = tFactory.template createScalarFunction<GradZEvalType>
      (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
    mGradientNFunctions[tName] = tFactory.template createScalarFunction<GradNEvalType>
      (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
  }
}

template<typename PhysicsType>
CriterionEvaluatorScalarFunction<PhysicsType>::
CriterionEvaluatorScalarFunction(
  const Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
        std::string            & aName
) :
  mSpatialModel (aSpatialModel),
  mWorksetFuncs (aSpatialModel.Mesh),
  mDataMap      (aDataMap),
  mFunctionName (aName)
{
  initialize(aProblemParams);
}

template<typename PhysicsType>
CriterionEvaluatorScalarFunction<PhysicsType>::
CriterionEvaluatorScalarFunction(
  const Plato::SpatialModel & aSpatialModel,
        Plato::DataMap      & aDataMap
) :
  mSpatialModel (aSpatialModel),
  mWorksetFuncs (aSpatialModel.Mesh),
  mDataMap      (aDataMap),
  mFunctionName ("Undefined Name")
{
}

template<typename PhysicsType>
bool 
CriterionEvaluatorScalarFunction<PhysicsType>::
isLinear() 
const
{
  auto tDomainName = mSpatialModel.Domains.front().getDomainName();
  return ( mValueFunctions.at(tDomainName)->isLinear() );
}

template<typename PhysicsType>
void
CriterionEvaluatorScalarFunction<PhysicsType>::
setEvaluator(
  const Plato::evaluation_t                   & aEvalType,
  const std::shared_ptr<Plato::CriterionBase> & aCriterion,
  const std::string                           & aDomainName
)
{
  switch(aEvalType)
  {
    case evaluation_t::VALUE:
    {
      mValueFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
      mValueFunctions[aDomainName] = aCriterion;
      break;
    }
    case evaluation_t::GRAD_U:
    {
      mGradientUFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
      mGradientUFunctions[aDomainName] = aCriterion;
      break;
    }
    case evaluation_t::GRAD_Z:
    {
      mGradientZFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
      mGradientZFunctions[aDomainName] = aCriterion;
      break;
    }
    case evaluation_t::GRAD_X:
    {
      mGradientXFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
      mGradientXFunctions[aDomainName] = aCriterion;
      break;
    }
    case evaluation_t::GRAD_N:
    {
      mGradientNFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
      mGradientNFunctions[aDomainName] = aCriterion;
      break;
    }
  }
}

template<typename PhysicsType>
void
CriterionEvaluatorScalarFunction<PhysicsType>::
updateProblem(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  // build workset builders
  Plato::Elliptic::WorksetBuilder<ValueEvalType> tWorksetBuilderValue(mWorksetFuncs);
  // call update problem function
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    auto tName = tDomain.getDomainName();

    Plato::WorkSets tMyWorkSets;
    tWorksetBuilderValue.build(tDomain,aDatabase,tMyWorkSets);
    mValueFunctions.at(tName)->updateProblem(tMyWorkSets,aCycle);
    mGradientUFunctions.at(tName)->updateProblem(tMyWorkSets,aCycle);
    mGradientZFunctions.at(tName)->updateProblem(tMyWorkSets,aCycle);
    mGradientXFunctions.at(tName)->updateProblem(tMyWorkSets,aCycle);
    mGradientNFunctions.at(tName)->updateProblem(tMyWorkSets,aCycle);
  }
}

template<typename PhysicsType>
Plato::Scalar
CriterionEvaluatorScalarFunction<PhysicsType>::
value(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  // set local result scalar type
  using ResultScalarType = typename ValueEvalType::ResultScalarType;
  // evaluate value function
  Plato::Scalar tReturnVal(0.0);
  Plato::Elliptic::WorksetBuilder<ValueEvalType> tWorksetBuilder(mWorksetFuncs);
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build criterion value domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build criterion value range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarVectorT<ResultScalarType> > >
      ( Plato::ScalarVectorT<ResultScalarType>("Result Workset", tNumCells) );
    Kokkos::deep_copy(tResultWS->mData, 0.0);
    tWorksets.set("result", tResultWS);
    // save result workset to database
    auto tDomainName = tDomain.getDomainName();
    mDataMap.scalarVectors[mValueFunctions.at(tDomainName)->getName()] = tResultWS->mData;
    // evaluate criterion
    mValueFunctions.at(tDomainName)->evaluate(tWorksets, aCycle);
    // sum across elements
    tReturnVal += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS->mData);
  }
  auto tName = mSpatialModel.Domains[0].getDomainName();
  mValueFunctions.at(tName)->postEvaluate(tReturnVal);
  return tReturnVal;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorScalarFunction<PhysicsType>::
gradientConfig(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  // set local result type
  using ResultScalarType = typename GradXEvalType::ResultScalarType;
  // create output
  auto tNumNodes = mSpatialModel.Mesh->NumNodes();
  Plato::ScalarVector tGradientX("criterion gradient configuration", mNumSpatialDims * tNumNodes);
  // evaluate gradient
  Plato::Scalar tValue(0.0);
  Plato::Elliptic::WorksetBuilder<GradXEvalType> tWorksetBuilder(mWorksetFuncs);
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build gradient domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build gradient range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarVectorT<ResultScalarType> > >
        ( Plato::ScalarVectorT<ResultScalarType>("Result Workset", tNumCells) );
    Kokkos::deep_copy(tResultWS->mData, 0.0);
    tWorksets.set("result", tResultWS);
    // evaluate gradient
    auto tName = tDomain.getDomainName();
    mGradientXFunctions.at(tName)->evaluate(tWorksets, aCycle);
    // assemble gradient
    mWorksetFuncs.assembleVectorGradientFadX(tDomain, tResultWS->mData, tGradientX);
    tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS->mData);
  }
  // apply post operation to return values, if defined
  auto tDomainName = mSpatialModel.Domains.front().getDomainName();
  mGradientXFunctions.at(tDomainName)->postEvaluate(tGradientX, tValue);
  return tGradientX;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorScalarFunction<PhysicsType>::
gradientState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  // set local result type
  using ResultScalarType = typename GradUEvalType::ResultScalarType;
  // create output
  auto tNumNodes = mSpatialModel.Mesh->NumNodes();
  Plato::ScalarVector tGradientU("criterion gradient state", mNumDofsPerNode * tNumNodes);
  // evaluate gradient
  Plato::Scalar tValue(0.0);
  Plato::Elliptic::WorksetBuilder<GradUEvalType> tWorksetBuilder(mWorksetFuncs);
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build gradient domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build gradient range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarVectorT<ResultScalarType> > >
        ( Plato::ScalarVectorT<ResultScalarType>("Result Workset", tNumCells) );
    Kokkos::deep_copy(tResultWS->mData, 0.0);
    tWorksets.set("result", tResultWS);
    // evaluate function
    auto tName = tDomain.getDomainName();
    mGradientUFunctions.at(tName)->evaluate(tWorksets, aCycle);
    // assemble gradient
    mWorksetFuncs.assembleVectorGradientFadU(tDomain, tResultWS->mData, tGradientU);
    tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS->mData);
  }
  // apply post operation to return values, if defined
  auto tDomainName = mSpatialModel.Domains.front().getDomainName();
  mGradientUFunctions.at(tDomainName)->postEvaluate(tGradientU, tValue);
  return tGradientU;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorScalarFunction<PhysicsType>::
gradientNodeState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  // set local result type
  using ResultScalarType = typename GradNEvalType::ResultScalarType;
  // create output
  auto tNumNodeStateDofs = mSpatialModel.Mesh->NumNodes() * mNumNodeStatePerNode;
  Plato::ScalarVector tGradientN("criterion gradient wrt node states", tNumNodeStateDofs);
  // evaluate gradient
  Plato::Scalar tValue(0.0);
  Plato::Elliptic::WorksetBuilder<GradNEvalType> tWorksetBuilder(mWorksetFuncs);
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build gradient domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build gradient range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarVectorT<ResultScalarType>>>
      ( Plato::ScalarVectorT<ResultScalarType>("Result Workset", tNumCells) );
    Kokkos::deep_copy(tResultWS->mData, 0.0);
    tWorksets.set("result", tResultWS);
    // evaluate gradient
    auto tName = tDomain.getDomainName();
    mGradientNFunctions.at(tName)->evaluate(tWorksets, aCycle);
    // assemble gradient
    mWorksetFuncs.assembleScalarGradientFadN(tDomain,tResultWS->mData,tGradientN);
    // assemble value
    tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells,tResultWS->mData);
  }
  // apply post operation to return values, if defined
  auto tName = mSpatialModel.Domains.front().getDomainName();
  mGradientNFunctions.at(tName)->postEvaluate(tGradientN,tValue);
  return tGradientN;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorScalarFunction<PhysicsType>::
gradientControl(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  // set local result type
  using ResultScalarType = typename GradZEvalType::ResultScalarType;
  // create output
  auto tNumNodes = mSpatialModel.Mesh->NumNodes();
  Plato::ScalarVector tGradientZ("criterion gradient control", tNumNodes);
  // evaluate gradient
  Plato::Scalar tValue(0.0);
  Plato::Elliptic::WorksetBuilder<GradZEvalType> tWorksetBuilder(mWorksetFuncs);
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    // build gradient domain worksets
    Plato::WorkSets tWorksets;
    tWorksetBuilder.build(tDomain, aDatabase, tWorksets);
    // build gradient range workset
    auto tNumCells = tDomain.numCells();
    auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarVectorT<ResultScalarType> > >
      ( Plato::ScalarVectorT<ResultScalarType>("Result Workset", tNumCells) );
    Kokkos::deep_copy(tResultWS->mData, 0.0);
    tWorksets.set("result", tResultWS);
    // evaluate gradient
    auto tName = tDomain.getDomainName();
    mGradientZFunctions.at(tName)->evaluate(tWorksets, aCycle);
    // assemble gradient
    mWorksetFuncs.assembleScalarGradientFadZ(tDomain, tResultWS->mData, tGradientZ);
    // assemble value
    tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS->mData);
  }
  // apply post operation to return values, if defined
  auto tName = mSpatialModel.Domains.front().getDomainName();
  mGradientZFunctions.at(tName)->postEvaluate(tGradientZ, tValue);
  return tGradientZ;
}

template<typename PhysicsType>
void
CriterionEvaluatorScalarFunction<PhysicsType>::
setFunctionName(
  const std::string aFunctionName
)
{
  mFunctionName = aFunctionName;
}

template<typename PhysicsType>
std::string
CriterionEvaluatorScalarFunction<PhysicsType>::
name() const
{
  return mFunctionName;
}

} // namespace Elliptic

} // namespace Plato
