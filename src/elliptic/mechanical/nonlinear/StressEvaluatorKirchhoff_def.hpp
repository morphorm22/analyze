/*
 * StressEvaluatorKirchhoff_def.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathTypes.hpp"

#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"
#include "elliptic/mechanical/nonlinear/GreenLagrangeStrainTensor.hpp"
#include "elliptic/mechanical/nonlinear/KirchhoffSecondPiolaStress.hpp"
#include "elliptic/mechanical/nonlinear/FactoryNonlinearElasticMaterial.hpp"

namespace Plato
{

template<typename EvaluationType>
StressEvaluatorKirchhoff<EvaluationType>::
StressEvaluatorKirchhoff(
  const std::string            & aMaterialName,
        Teuchos::ParameterList & aParamList,
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap
) : 
  Plato::StressEvaluator<EvaluationType>(aSpatialDomain,aDataMap)
{
  Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
  mMaterial = tFactory.create(aMaterialName);
}

template<typename EvaluationType>
void 
StressEvaluatorKirchhoff<EvaluationType>::
evaluate(
  const Plato::WorkSets                         & aWorkSets,
        Plato::ScalarArray4DT<ResultScalarType> & aResult,
        Plato::Scalar                             aCycle
) const
{
  // unpack worksets
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<ControlScalarType> tControlWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));
  Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
  // get integration rule information
  auto tCubPoints = ElementType::getCubPoints();
  auto tNumPoints = ElementType::mNumGaussPoints;
  // compute state gradient
  Plato::StateGradient<EvaluationType> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::GreenLagrangeStrainTensor<EvaluationType> tGreenLagrangeStrainTensor;
  Plato::KirchhoffSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterial);
  // evaluate stress tensor
  auto tNumCells = mSpatialDomain.numCells();
  Kokkos::parallel_for("compute nominal stress", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // compute gradient functions
    ConfigScalarType tVolume(0.0);
    Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigScalarType> 
      tGradient(ConfigScalarType(0.));
    auto tCubPoint = tCubPoints(iGpOrdinal);
    tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
    // compute state gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tStateGradient(StrainScalarType(0.));
    tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
    // compute green-lagrange strain tensor
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tStrainTensor(StrainScalarType(0.));
    tGreenLagrangeStrainTensor(tStateGradient,tStrainTensor);
    // compute second piola-kirchhoff stress
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultScalarType> 
      tStressTensor2PK(ResultScalarType(0.));
    tComputeSecondPiolaKirchhoffStress(tStrainTensor,tStressTensor2PK);
    // copy result to output workset
    for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) = tStressTensor2PK(tDimI,tDimJ);
      }
    }
  });
}
    
}