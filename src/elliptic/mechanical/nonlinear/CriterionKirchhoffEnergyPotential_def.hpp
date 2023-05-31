/*
 * CriterionKirchhoffEnergyPotential_def.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "GradientMatrix.hpp"
#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/GreenLagrangeStrainTensor.hpp"
#include "elliptic/mechanical/nonlinear/KirchhoffSecondPiolaStress.hpp"
#include "elliptic/mechanical/nonlinear/FactoryNonlinearElasticMaterial.hpp"

namespace Plato
{

template<typename EvaluationType>
CriterionKirchhoffEnergyPotential<EvaluationType>::
CriterionKirchhoffEnergyPotential(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aParamList,
  const std::string            & aFuncName
) :
  FunctionBaseType(aSpatialDomain,aDataMap,aParamList,aFuncName)
{
  std::string tMaterialName = mSpatialDomain.getMaterialName();
  Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
  mMaterial = tFactory.create(tMaterialName);
}

template<typename EvaluationType>
void 
CriterionKirchhoffEnergyPotential<EvaluationType>::
evaluate_conditional(
    const Plato::ScalarMultiVectorT <StateT>   & aState,
    const Plato::ScalarMultiVectorT <ControlT> & aControl,
    const Plato::ScalarArray3DT     <ConfigT>  & aConfig,
          Plato::ScalarVectorT      <ResultT>  & aResult,
          Plato::Scalar                          aCycle
) const
{
  // get integration rule information
  auto tNumPoints  = ElementType::mNumGaussPoints;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  // compute state gradient
  Plato::StateGradient<EvaluationType> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::GreenLagrangeStrainTensor<EvaluationType> tGreenLagrangeStrainTensor;
  Plato::KirchhoffSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterial);
  // evaluate stress tensor
  auto tNumCells = mSpatialDomain.numCells();
  Kokkos::parallel_for("evaluate strain energy potential", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> 
        tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
        tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,aState,tGradient,tStateGradient);
      // compute green-lagrange strain tensor
      Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
        tStrainTensor(StrainT(0.));
      tGreenLagrangeStrainTensor(tStateGradient,tStrainTensor);
      // compute second piola-kirchhoff stress
      Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> 
        tStressTensor2PK(ResultT(0.));
      tComputeSecondPiolaKirchhoffStress(tStrainTensor,tStressTensor2PK);
      // apply integration point weight to element volume
      tVolume *= tCubWeights(iGpOrdinal);
      // evaluate elastic strain energy potential 
      ResultT tValue(0.0);
      for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
          tValue += 0.5 * tStrainTensor(tDimI,tDimJ) * tStressTensor2PK(tDimI,tDimJ) * tVolume;
        }
      }
      Kokkos::atomic_add(&aResult(iCellOrdinal), tValue);
  });
}

} // namespace Plato
