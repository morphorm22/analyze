/*
 * StressEvaluatorNeoHookean_def.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "GradientMatrix.hpp"
#include "PlatoMathTypes.hpp"

#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"
#include "elliptic/mechanical/nonlinear/NeoHookeanSecondPiolaStress.hpp"
#include "elliptic/mechanical/nonlinear/FactoryNonlinearElasticMaterial.hpp"

namespace Plato
{

template<typename EvaluationType>
StressEvaluatorNeoHookean<EvaluationType>::
StressEvaluatorNeoHookean(
  const std::string            & aMaterialName,
        Teuchos::ParameterList & aParamList,
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap) : 
  Plato::StressEvaluator<EvaluationType>(aSpatialDomain,aDataMap)
{
  Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
  mMaterial = tFactory.create(aMaterialName);
}

template<typename EvaluationType>
void 
StressEvaluatorNeoHookean<EvaluationType>::
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
  Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
  Plato::NeoHookeanSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterial);
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
    // compute deformation gradient 
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tDefGradient(StrainScalarType(0.));
    tComputeDeformationGradient(tStateGradient,tDefGradient);
    // compute second-piola kirchhoff stress tensor
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultScalarType> 
      tStressTensor2PK(ResultScalarType(0.));
    tComputeSecondPiolaKirchhoffStress(tDefGradient,tStressTensor2PK);
    // copy result to output workset
    for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) = tStressTensor2PK(tDimI,tDimJ);
      }
    }
  });
}

} // namespace Plato
