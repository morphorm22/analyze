/*
 * NitscheTestHeatFluxEvaluator_def.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "ScalarGrad.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathTypes.hpp"
#include "InterpolateFromNodal.hpp"

#include "utilities/WeightedNormalVector.hpp"
#include "elliptic/thermal/ThermalFlux.hpp"
#include "elliptic/thermal/FactoryThermalConductionMaterial.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
NitscheTestHeatFluxEvaluator<EvaluationType>::
NitscheTestHeatFluxEvaluator(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams)
{
  // create material constitutive model
  //
  Plato::FactoryThermalConductionMaterial<EvaluationType> tFactory(aParamList);
  mMaterialModel = tFactory.create(mMaterialName);
}

template<typename EvaluationType>
void 
NitscheTestHeatFluxEvaluator<EvaluationType>::
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
  // create local functors
  //
  Plato::ScalarGrad<BodyElementBase> tScalarGrad;
  Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
  Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
  Plato::ThermalFlux<EvaluationType> tThermalFlux(mMaterialModel);
  Plato::InterpolateFromNodal<BodyElementBase,mNumDofsPerNode> tProjectFromNodes;
  // evaluate integral
  //
  Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
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
    // compute surface area weighted normal vector
    //
    auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
    Plato::Array<mNumSpatialDims,ConfigScalarType> tWeightedNormalVector;
    tComputeWeightedNormalVector(
      tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
    );
    // compute configuration gradient
    //
    ConfigScalarType tVolume(0.0);
    Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient(ConfigScalarType(0.));
    tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
    // compute temperature gradient
    //
    Plato::Array<mNumSpatialDims,ConfigScalarType> tVirtualTempGrad(ConfigScalarType(0.));
    tScalarGrad(tVirtualTempGrad,tGradient);
    // compute virtual heat flux
    //
    StateScalarType tProjectedTemp =
      tProjectFromNodes(tCellOrdinal,tBasisValuesOnParentBodyElemSurface,tStateWS);
    Plato::Array<mNumSpatialDims,ResultScalarType> tVirtualFlux(ResultScalarType(0.));
    tThermalFlux(tVirtualFlux,tVirtualTempGrad,tProjectedTemp);
    // evaluate: int_{\Gamma_D} \delta(q\cdot{n})\cdot(T - T_D) d\Gamma_D
    //
    for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
    {
      ResultScalarType tValue(0.0);
      Plato::OrdinalType tLocalDofOrdinal = tNode * mNumDofsPerNode;
      for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
      {
        tValue += aScale * tCubWeightOnParentBodyElemSurface
          * ( tVirtualFlux(tDimI) * tWeightedNormalVector(tDimI) )
          * ( tProjectedTemp - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) );
      }
      Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal),tValue);
    }
  });
}

} // namespace Elliptic

} // namespace Plato
