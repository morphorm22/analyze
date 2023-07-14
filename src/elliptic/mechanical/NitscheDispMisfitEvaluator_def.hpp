/*
 * NitscheDispMisfitEvaluator_decl.hpp
 *
 *  Created on: July 14, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathTypes.hpp"

#include "utilities/SurfaceArea.hpp"
#include "utilities/WeightedNormalVector.hpp"
#include "utilities/ComputeCharacteristicLength.hpp"

#include "materials/mechanical/FactoryMechanicalMaterials.hpp"

#include "elliptic/mechanical/linear/ComputeStrainTensor.hpp"
#include "elliptic/mechanical/linear/ComputeIsotropicElasticStressTensor.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
NitscheDispMisfitEvaluator<EvaluationType>::
NitscheDispMisfitEvaluator(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams)
{
  // create material constitutive model
  //
  Plato::FactoryMechanicalMaterials<EvaluationType> tFactory;
  mMaterialModel = tFactory.create(mMaterialName,aParamList);
  // parse penalty parameter
  //
  if(aParamList.isType<Plato::Scalar>("Penalty")){
    mNitschePenalty = aParamList.get<Plato::Scalar>("Penalty");
  }
}

template<typename EvaluationType>
void 
NitscheDispMisfitEvaluator<EvaluationType>::
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
  tComputeCharacteristicLength(aSpatialModel, aWorkSets, tCharacteristicLength);
  // compute numerator for nitsche's penalty parameter
  //
  auto tYoungsModulus  = mMaterialModel->getScalarConstant("youngs modulus");
  auto tNitschePenaltyTimesModulus = mNitschePenalty * tYoungsModulus;
  // evaluate integral
  //
  Plato::SurfaceArea<BodyElementBase> tComputeSurfaceArea;
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
    // compute surface area
    //
    ConfigScalarType tSurfaceArea(0.0);
    auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
    tComputeSurfaceArea(tCellOrdinal,tFaceLocalNodeOrds,tBasisGradsOnParentFaceElem,tConfigWS,tSurfaceArea);
    // project state from nodes to quadrature/cubature point
    //
    Plato::Array<mNumSpatialDims,StateScalarType> tProjectedStates;
    for(Plato::OrdinalType tDof = 0; tDof < mNumDofsPerNode; tDof++)
    {
      tProjectedStates(tDof) = 0.0;
      for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
      {
        Plato::OrdinalType tCellDofIndex = (mNumDofsPerNode * tNodeIndex) + tDof;
        tProjectedStates(tDof) += tStateWS(tCellOrdinal, tCellDofIndex) * 
          tBasisValuesOnParentBodyElemSurface(tNodeIndex);
      }
    }
    // evaluate int_{\Gamma_D}\gamma_N^u \delta{u}\cdot(u - u_D) d\Gamma_D
    //
    ConfigScalarType tGamma = tNitschePenaltyTimesModulus / tCharacteristicLength(aSideOrdinal);
    for(Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++)
    {
      for(Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
      {
        auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
        ResultScalarType tValue = aScale * tGamma * tBasisValuesOnParentBodyElemSurface(tNode)
          * ( tProjectedStates[tDimI] - tDirichletWS(tCellOrdinal,tLocalDofOrdinal) ) 
          * tSurfaceArea * tCubWeightOnParentBodyElemSurface;
        Kokkos::atomic_add(&tResultWS(tCellOrdinal,tLocalDofOrdinal), tValue);
      }
    }
  });
}

} // namespace Elliptic

} // namespace Plato