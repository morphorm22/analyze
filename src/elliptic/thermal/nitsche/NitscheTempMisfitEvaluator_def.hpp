/*
 * NitscheTempMisfitEvaluator_def.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "PlatoMathTypes.hpp"
#include "InterpolateFromNodal.hpp"

#include "utilities/SurfaceArea.hpp"
#include "utilities/ComputeCharacteristicLength.hpp"
#include "elliptic/thermal/FactoryThermalConductionMaterial.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
NitscheTempMisfitEvaluator<EvaluationType>::
NitscheTempMisfitEvaluator(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams)
{
  // create material constitutive model
  //
  Plato::FactoryThermalConductionMaterial<EvaluationType> tFactory(aParamList);
  mMaterialModel = tFactory.create(mMaterialName);
  // parse penalty parameter
  //
  if(aParamList.isType<Plato::Scalar>("Penalty")){
    mNitschePenalty = aParamList.get<Plato::Scalar>("Penalty");
  }
}

template<typename EvaluationType>
void 
NitscheTempMisfitEvaluator<EvaluationType>::
evaluate(
  const Plato::SpatialModel & aSpatialModel,
  const Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aCycle,
        Plato::Scalar         aScale
)
{
  // unpack worksets
  //
  Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
  Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVector tDirichletWS = 
    Plato::unpack<Plato::ScalarMultiVector>(aWorkSets.get("dirichlet"));
  // get side set connectivity information
  //
  auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
  auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
  auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
  // get integration points and weights
  //
  auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
  auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
  auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
  // compute characteristic length
  //
  Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
  Plato::ComputeCharacteristicLength<EvaluationType> tComputeCharacteristicLength(mSideSetName);
  Plato::ScalarVectorT<ConfigScalarType> tCharacteristicLength("characteristic length",tNumCellsOnSideSet);
  tComputeCharacteristicLength(aSpatialModel,aWorkSets,tCharacteristicLength);
  // create local functors
  //
  Plato::SurfaceArea<BodyElementBase> tComputeSurfaceArea;
  Plato::InterpolateFromNodal<BodyElementBase,mNumDofsPerNode> tProjectFromNodes;
  // evaluate integral
  //
  auto tNitschePenalty = mNitschePenalty;
  auto tConductivityTensor = mMaterialModel->getTensorConstant("Thermal Conductivity");
  Kokkos::parallel_for("evaluate integral", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
  {
    auto tCubPointOnParentFaceElem = tCubPointsOnParentFaceElem(aPointOrdinal);
    auto tBasisGradsOnParentFaceElem = FaceElementBase::basisGrads(tCubPointOnParentFaceElem);
    // quadrature data to evaluate integral on the body surface of interest
    //
    Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
    Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
    auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
    for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
      Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
      tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
    }
    auto tCubWeightOnParentBodyElemSurface = tCubWeightsOnParentBodyElemSurface(aPointOrdinal);
    auto tBasisValuesOnParentBodyElemSurface = BodyElementBase::basisValues(tCubPointOnParentBodyElemSurface);
    Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tFaceLocalNodeOrds;
    for( Plato::OrdinalType tIndex=0; tIndex<mNumNodesPerFace; tIndex++){
      tFaceLocalNodeOrds(tIndex) = tSideLocalNodeOrds(aSideOrdinal*mNumNodesPerFace+tIndex);
    }
    // compute surface area
    //
    ConfigScalarType tSurfaceArea(0.0);
    auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
    tComputeSurfaceArea(tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tSurfaceArea);
    // project temperature from nodes to integration points
    //
    StateScalarType tProjectedTemp =
      tProjectFromNodes(tCellOrdinal,tBasisValuesOnParentBodyElemSurface,tStateWS);
    // evaluate: int_{\Gamma_D}\gamma_N^T \delta{T}\cdot(T - T_D) d\Gamma_D
    //
    for(Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
    {
      ResultScalarType tValue(0.0);
      Plato::OrdinalType tLocalDofOrdinal = tNode * mNumDofsPerNode;
      for(Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
      {
        ConfigScalarType tGamma = 
          ( tNitschePenalty * tConductivityTensor(tDimI,tDimI) ) / tCharacteristicLength(aSideOrdinal);
        tValue += aScale * tGamma * tBasisValuesOnParentBodyElemSurface(tNode)
          * ( tProjectedTemp - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) )
          * tCubWeightOnParentBodyElemSurface * tSurfaceArea;
      }
      Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal),tValue);
    }
  });
}

} // namespace Elliptic

} // namespace Plato
