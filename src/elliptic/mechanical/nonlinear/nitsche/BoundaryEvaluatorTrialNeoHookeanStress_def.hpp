/*
 *  BoundaryEvaluatorTrialNeoHookeanStress_def.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathTypes.hpp"

#include "materials/mechanical/FactoryMechanicalMaterials.hpp"

#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"
#include "elliptic/mechanical/nonlinear/GreenLagrangeStrainTensor.hpp"
#include "elliptic/mechanical/nonlinear/NeoHookeanSecondPiolaStress.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
BoundaryEvaluatorTrialNeoHookeanStress<EvaluationType>::
BoundaryEvaluatorTrialNeoHookeanStress(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams)
{
  Plato::FactoryMechanicalMaterials<EvaluationType> tFactory;
  mMaterialModel = tFactory.create(mMaterialName,aParamList);
}

template<typename EvaluationType>
void 
BoundaryEvaluatorTrialNeoHookeanStress<EvaluationType>::
evaluate(
  const Plato::SpatialModel                     & aSpatialModel,
  const Plato::WorkSets                         & aWorkSets,
        Plato::ScalarArray4DT<ResultScalarType> & aResult,
        Plato::Scalar                             aCycle
) const
{
  // unpack worksets
  //
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
  // get side set connectivity information
  //
  auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
  auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
  // create local functors
  //
  Plato::StateGradient<EvaluationType> tComputeStateGradient;
  Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
  Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
  Plato::NeoHookeanSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterialModel);
  // get integration points and weights
  //
  auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
  auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
  auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
  // evaluate integral
  //
  Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
  Kokkos::parallel_for("boundary test stress evaluator", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCellsOnSideSet, mNumGaussPointsPerFace}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
  {
    // quadrature data to evaluate integral on the body surface of interest
    //
    Plato::Array<mNumSpatialDims> tCubPointOnParentBodyElemSurface;
    Plato::OrdinalType tLocalFaceOrdinal = tSideLocalFaceOrds(aSideOrdinal);
    auto tCubPointsOnBodyParentElemSurface = tCubPointsOnParentBodyElemSurfaces(tLocalFaceOrdinal);
    for( Plato::OrdinalType tDim=0; tDim < mNumSpatialDims; tDim++ ){
      Plato::OrdinalType tIndex = BodyElementBase::mNumGaussPointsPerFace * aPointOrdinal + tDim;
      tCubPointOnParentBodyElemSurface(tDim) = tCubPointsOnBodyParentElemSurface(tIndex);
    }
    // compute gradient of interpolation functions
    //
    ConfigScalarType tVolume(0.0);
    auto tCellOrdinal = tSideCellOrdinals(aSideOrdinal);
    Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
    tComputeGradient(tCellOrdinal,tCubPointOnParentBodyElemSurface,tConfigWS,tGradient,tVolume);
    // compute state gradient
    //
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
      tStateGradient(StrainScalarType(0.));
    tComputeStateGradient(tCellOrdinal,tStateWS,tGradient,tStateGradient);
    // compute deformation gradient 
    //
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
      tDefGradient(StrainScalarType(0.));
    tComputeDeformationGradient(tStateGradient,tDefGradient);
    // compute second piola-kirchhoff stress
    //
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> 
      tStressTensor2PK(ResultScalarType(0.));
    tComputeSecondPiolaKirchhoffStress(tDefGradient,tStressTensor2PK);
    // copy stress tensor to output workset
    //
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        aResult(tCellOrdinal,aPointOrdinal,tDimI,tDimJ) = tStressTensor2PK(tDimI,tDimJ);
      }
    }
  });
}

} // namespace Elliptic

} // namespace Plato
