/*
 * NitscheTestElasticStressEvaluator_decl.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathTypes.hpp"

#include "utilities/WeightedNormalVector.hpp"
#include "materials/mechanical/FactoryMechanicalMaterials.hpp"
#include "elliptic/mechanical/linear/ComputeStrainTensor.hpp"
#include "elliptic/mechanical/linear/ComputeIsotropicElasticStressTensor.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
NitscheTestElasticStressEvaluator<EvaluationType>::
NitscheTestElasticStressEvaluator(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams),
  mBoundaryStressEvaluator(std::make_shared<EvaluatorClassType>(aParamList,aNitscheParams))
{}

template<typename EvaluationType>
NitscheTestElasticStressEvaluator<EvaluationType>::
~NitscheTestElasticStressEvaluator()
{}

template<typename EvaluationType>
void 
NitscheTestElasticStressEvaluator<EvaluationType>::
evaluate(
  const Plato::SpatialModel & aSpatialModel,
  const Plato::WorkSets     & aWorkSets,
      Plato::Scalar           aCycle,
      Plato::Scalar           aScale
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
  // create local functors
  //
  Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
  // get integration points and weights
  //
  auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
  auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
  auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
  // evaluate boundary test stress tensors
  //
  Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
  Plato::ScalarArray4DT<ResultScalarType> tStressTensors(
    "stress tensors",tNumCellsOnSideSet,mNumGaussPointsPerFace,mNumSpatialDims,mNumSpatialDims
  );
  mBoundaryStressEvaluator->evaluate(aSpatialModel,aWorkSets,tStressTensors,aCycle);
  // evaluate integral
  //
  Kokkos::parallel_for("nitsche stress evaluator", 
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
    Plato::Array<mNumSpatialDims, ConfigScalarType> tWeightedNormalVector;
    tComputeWeightedNormalVector(
      tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tWeightedNormalVector
    );
    // project state from nodes to quadrature/cubature point
    //
    Plato::Array<mNumSpatialDims,StateScalarType> tProjectedStates;
    for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++){
      tProjectedStates(tDof) = 0.0;
      for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++){
        Plato::OrdinalType tCellDofIndex = (mNumDofsPerNode * tNodeIndex) + tDof;
        tProjectedStates(tDof) += tStateWS(tCellOrdinal, tCellDofIndex) * 
          tBasisValuesOnParentBodyElemSurface(tNodeIndex);
      }
    }
    // evaluate: int_{\Gamma_D} \delta(\sigma\cdot{n})\cdot(u - u_D) d\Gamma_D
    //
    for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
    {
      for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
      {
        auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
        ResultScalarType tVirtualStressTimesWeightedNormal(0.);
        for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++)
        {
          tVirtualStressTimesWeightedNormal += 
            tStressTensors(tCellOrdinal,aPointOrdinal,tDimI,tDimJ) * tWeightedNormalVector[tDimJ];
        }
        ResultScalarType tValue = aScale * tCubWeightOnParentBodyElemSurface * tVirtualStressTimesWeightedNormal
          * (tProjectedStates[tDimI] - tDirichletWS(tCellOrdinal,tLocalDofOrdinal));
        Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
      }
    }
  });
}

} // namespace Elliptic

} // namespace Plato
