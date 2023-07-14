/*
 * NitscheTrialThermalHyperElasticStressEvaluator_def.hpp
 *
 *  Created on: July 13, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathTypes.hpp"
#include "InterpolateFromNodal.hpp"

#include "utilities/WeightedNormalVector.hpp"
#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/NominalStressTensor.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"
#include "elliptic/mechanical/nonlinear/KineticPullBackOperation.hpp"

#include "elliptic/thermomechanics/nonlinear/UtilitiesThermoMechanics.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermalDeformationGradient.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermoElasticDeformationGradient.hpp"

#include "elliptic/mechanical/nonlinear/nitsche/FactoryNitscheHyperElasticStressEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
NitscheTrialThermalHyperElasticStressEvaluator<EvaluationType>::
NitscheTrialThermalHyperElasticStressEvaluator(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aNitscheParams
) : 
  BaseClassType(aNitscheParams),
  mParamList(aParamList)
{
  // create boundary stress evaluator
  //
  Plato::Elliptic::FactoryNitscheHyperElasticStressEvaluator<EvaluationType> tFactory;
  mBoundaryStressEvaluator = tFactory.createTrialEvaluator(aParamList,aNitscheParams);
}

template<typename EvaluationType>
void 
NitscheTrialThermalHyperElasticStressEvaluator<EvaluationType>::
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
  Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
  Plato::ScalarMultiVectorT<NodeStateScalarType> tTempWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<NodeStateScalarType>>(aWorkSets.get("node states"));
  // get side set connectivity information
  //
  auto tSideCellOrdinals  = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
  auto tSideLocalFaceOrds = aSpatialModel.Mesh->GetSideSetFaces(mSideSetName);
  auto tSideLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
  // create local functors
  //
  Plato::StateGradient<EvaluationType> tComputeDispGradient;
  Plato::ComputeGradientMatrix<BodyElementBase> tComputeGradient;
  Plato::NominalStressTensor<EvaluationType> tComputeNominalStressTensor;
  Plato::WeightedNormalVector<BodyElementBase> tComputeWeightedNormalVector;
  Plato::KineticPullBackOperation<EvaluationType> tApplyKineticPullBackOperation;
  Plato::DeformationGradient<EvaluationType> tComputeMechanicalDeformationGradient;
  Plato::InterpolateFromNodal<BodyElementBase,mNumNodeStatePerNode> tInterpolateFromNodal;
  Plato::ThermoElasticDeformationGradient<EvaluationType> tComputeThermoElasticDeformationGradient;
  Plato::ThermalDeformationGradient<EvaluationType> tComputeThermalDeformationGradient(mMaterialName,mParamList);
  // get integration points and weights
  //
  auto tCubPointsOnParentFaceElem = FaceElementBase::getCubPoints();
  auto tCubPointsOnParentBodyElemSurfaces = BodyElementBase::getFaceCubPoints();
  auto tCubWeightsOnParentBodyElemSurface = BodyElementBase::getFaceCubWeights();
  // evaluate boundary trial stress tensors
  //
  Plato::OrdinalType tNumCellsOnSideSet = tSideCellOrdinals.size();
  Plato::ScalarArray4DT<ResultScalarType> tWorkset2PKS(
    "stress tensors",tNumCellsOnSideSet,mNumGaussPointsPerFace,mNumSpatialDims,mNumSpatialDims
  );
  mBoundaryStressEvaluator->evaluate(aSpatialModel,aWorkSets,tWorkset2PKS,aCycle);
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
    // compute displacement gradient
    //
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> tDispGradient(StrainScalarType(0.));
    tComputeDispGradient(tCellOrdinal,tStateWS,tGradient,tDispGradient);
    // compute mechanical deformation gradient 
    //
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> tMechanicalDefGradient(StrainScalarType(0.));
    tComputeMechanicalDeformationGradient(tDispGradient,tMechanicalDefGradient);
    // interpolate temperature field from nodes to integration points on the parent body element surface
    NodeStateScalarType tTemperature = 
      tInterpolateFromNodal(tCellOrdinal,tBasisValuesOnParentBodyElemSurface,tTempWS);
    // compute thermal deformation gradient 
    //
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,NodeStateScalarType> tThermalDefGradient(NodeStateScalarType(0.));
    tComputeThermalDeformationGradient(tTemperature,tThermalDefGradient);
    // compute multiplicative decomposition of the thermo-elastic deformation gradient 
    //
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tThermoElasticDefGradient(ResultScalarType(0.));
    tComputeThermoElasticDeformationGradient(tThermalDefGradient,tMechanicalDefGradient,tThermoElasticDefGradient);
    // pull back second Piola-Kirchhoff stress from deformed to undeformed configuration
    //
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tDefConfig2PKS(ResultScalarType(0.));
    Plato::Elliptic::getCell2PKS<mNumSpatialDims>(tCellOrdinal,aPointOrdinal,tWorkset2PKS,tDefConfig2PKS);
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tUnDefConfig2PKS(ResultScalarType(0.));
    tApplyKineticPullBackOperation(tThermalDefGradient,tDefConfig2PKS,tUnDefConfig2PKS);
    // compute nominal stress
    //
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tNominalStressTensor(ResultScalarType(0.));
    tComputeNominalStressTensor(tThermoElasticDefGradient,tUnDefConfig2PKS,tNominalStressTensor);
    // evaluate: int_{\Gamma_D} \delta{u}\cdot(\mathbf{P}\cdot{n}) d\Gamma_D
    //
    for( Plato::OrdinalType tNode=0; tNode<mNumNodesPerCell; tNode++){
      for( Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++){
        auto tLocalDofOrdinal = ( tNode * mNumSpatialDims ) + tDimI;
        ResultScalarType tStressTimesSurfaceWeightedNormal(0.0);
        for( Plato::OrdinalType tDimJ=0; tDimJ<mNumSpatialDims; tDimJ++){
          tStressTimesSurfaceWeightedNormal += 
            tNominalStressTensor(tDimI,tDimJ) * tWeightedNormalVector[tDimJ];
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
