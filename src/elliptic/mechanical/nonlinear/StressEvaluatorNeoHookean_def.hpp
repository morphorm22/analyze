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
    const Plato::ScalarMultiVectorT<StateT>   & aState,
    const Plato::ScalarMultiVectorT<ControlT> & aControl,
    const Plato::ScalarArray3DT<ConfigT>      & aConfig,
    const Plato::ScalarArray4DT<ResultT>      & aResult,
    Plato::Scalar                               aCycle
) const
{
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
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,aState,tGradient,tStateGradient);
      // compute deformation gradient 
      Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tDefGradient(StrainT(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // compute second-piola kirchhoff stress tensor
      Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> 
        tStressTensor2PK(ResultT(0.));
      tComputeSecondPiolaKirchhoffStress(tDefGradient,tStressTensor2PK);
      // compute nominal stress
      for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
          for(Plato::OrdinalType tDimK = 0; tDimK < ElementType::mNumSpatialDims; tDimK++){
            aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) += 
              tStressTensor2PK(tDimI,tDimK)*tDefGradient(tDimJ,tDimK);
          }
        }
      }
  });
}

} // namespace Plato
