/*
 *  NitscheTestElasticStressEvaluator_decl.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathTypes.hpp"

#include "utilities/WeightedNormalVector.hpp"
#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"

#include "elliptic/mechanical/nonlinear/nitsche/FactoryNitscheHyperElasticStressEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
NitscheTestHyperElasticStressEvaluator<EvaluationType>::
NitscheTestHyperElasticStressEvaluator(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams)
{
  // create boundary stress evaluator
  //
  Plato::Elliptic::FactoryNitscheHyperElasticStressEvaluator<EvaluationType> tFactory;
  mBoundaryStressEvaluator = tFactory.createTestEvaluator(aParamList,aNitscheParams);
}

template<typename EvaluationType>
void 
NitscheTestHyperElasticStressEvaluator<EvaluationType>::
evaluate(
  const Plato::SpatialModel & aSpatialModel,
  const Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aCycle,
        Plato::Scalar         aScale
)
{
  // unpack worksets
  //
  Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  // get side set connectivity information
  //
  auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
  auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
  auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
  // create local functors
  //
  Plato::StateGradient<EvaluationType> tComputeStateGradient;
  Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
  Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
  Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
  // get integration points and weights
  //
  auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
  auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
  auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
  // evaluate boundary trial stress tensors
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
    // compute gradient of interpolation functions
    //
    ConfigScalarType tVolume(0.0);
    Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
    tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
    // compute state gradient
    //
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
      tStateGradient(StrainScalarType(0.));
    tComputeStateGradient(tCellOrdinal,tGradient,tStateGradient);
    // compute deformation gradient 
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
      tDefGradient(StrainScalarType(0.));
    tComputeDeformationGradient(tStateGradient,tDefGradient);
    // compute nominal stress
    Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ResultScalarType> 
      tNominalStress(ResultScalarType(0.));
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
          tNominalStress(tDimI,tDimJ) += 
            tStressTensors(tCellOrdinal,aPointOrdinal,tDimI,tDimK) * tDefGradient(tDimJ,tDimK);
        }
      }
    }
    // evaluate: int_{\Gamma_D} \delta{u}\cdot(\mathbf{P}\cdot{n}) d\Gamma_D
    //
    for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
      for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
        auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
        ResultScalarType tStressTimesSurfaceWeightedNormal(0.0);
        for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
          tStressTimesSurfaceWeightedNormal += tNominalStress(tDimI,tDimJ) * tWeightedNormalVector[tDimJ];
        }
        ResultScalarType tValue = -aScale * tBasisValuesOnParentBodyElemSurface(tNode) 
          * tCubWeightOnParentBodyElemSurface * tStressTimesSurfaceWeightedNormal;
        Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
      }
    }
  });
}

} // namespace Elliptic

} // namespace Plato