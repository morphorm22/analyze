/*
 * ScalarFunction_def.hpp
 *
 *  Created on: June 7, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "Assembly.hpp"
#include "base/WorksetBase.hpp"
#include "elliptic/base/WorksetBuilder.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
ScalarFunction<PhysicsType>::
ScalarFunction(
  const std::string            & aFuncName,
  const Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProbParams
) :
  mDataMap     (aDataMap),
  mSpatialModel(aSpatialModel),
  mWorksetFuncs(aSpatialModel.Mesh),
  mName        (aFuncName)
{
  this->initialize(aProbParams);
}

template<typename PhysicsType>
void
ScalarFunction<PhysicsType>::
setEvaluator(
  const evaluator_t   & aEvalType,
  const CriterionType & aCriterion,
  const std::string   & aDomainName
)
{
  switch(aEvalType)
  {
    case evaluator_t::VALUE:
    {
      mValueFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
      mValueFunctions[aDomainName] = aCriterion;
      break;
    }
    case evaluator_t::GRAD_U:
    {
      mGradientUFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
      mGradientUFunctions[aDomainName] = aCriterion;
      break;
    }
    case evaluator_t::GRAD_Z:
    {
      mGradientZFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
      mGradientZFunctions[aDomainName] = aCriterion;
      break;
    }
    case evaluator_t::GRAD_X:
    {
      mGradientXFunctions[aDomainName] = nullptr; // ensures shared_ptr is decremented
      mGradientXFunctions[aDomainName] = aCriterion;
      break;
    }
  }
}

template<typename PhysicsType>
std::string 
ScalarFunction<PhysicsType>::
name() 
const 
{ 
  return mName; 
}

template<typename PhysicsType>
bool 
ScalarFunction<PhysicsType>::
isLinear() 
const
{
  auto tDomainName = mSpatialModel.Domains.front().getDomainName();
  return ( mValueFunctions.at(tDomainName)->isLinear() );
}

template<typename PhysicsType>
Plato::Scalar
ScalarFunction<PhysicsType>::
value(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
)
{
  // set local result scalar type
  using ResultScalarType = typename ValueEvalType::ResultScalarType;
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
  // apply post operation to return value, if defined
  auto tDomainName = mSpatialModel.Domains.front().getDomainName();
  mValueFunctions.at(tDomainName)->postEvaluate(tReturnVal);
  return tReturnVal;
}

template<typename PhysicsType>
Plato::ScalarVector
ScalarFunction<PhysicsType>::
gradientConfig(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
)
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
ScalarFunction<PhysicsType>::
gradientState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
)
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
ScalarFunction<PhysicsType>::
gradientControl(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
)
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
ScalarFunction<PhysicsType>::
initialize(
  Teuchos::ParameterList & aProbParams
)
{
  typename PhysicsType::FunctionFactory tFactory;
  auto tProblemDefaults = aProbParams.sublist("Criteria").sublist(mName);
  auto tFunType = tProblemDefaults.get<std::string>("Scalar Function Type", "");
  for(const auto& tDomain : mSpatialModel.Domains)
  {
    /*
    auto tDomainName = tDomain.getDomainName();
    mValueFunctions[tDomainName] = 
      tFactory.template createScalarFunction<ValueEvalType>(tDomain, mDataMap, aProbParams, tFunType, mName);
    mGradientUFunctions[tDomainName] = 
      tFactory.template createScalarFunction<GradUEvalType>(tDomain, mDataMap, aProbParams, tFunType, mName);
    mGradientXFunctions[tDomainName] = 
      tFactory.template createScalarFunction<GradXEvalType>(tDomain, mDataMap, aProbParams, tFunType, mName);
    mGradientZFunctions[tDomainName] = 
      tFactory.template createScalarFunction<GradZEvalType>(tDomain, mDataMap, aProbParams, tFunType, mName);
    */
  }
}

} // namespace Elliptic


} // namespace Plato
