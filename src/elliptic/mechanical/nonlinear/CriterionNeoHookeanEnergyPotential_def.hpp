/*
 * CriterionNeoHookeanEnergyPotential_def.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "GradientMatrix.hpp"
#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"
#include "elliptic/mechanical/nonlinear/RightDeformationTensor.hpp"
#include "elliptic/mechanical/nonlinear/FactoryNonlinearElasticMaterial.hpp"

namespace Plato
{

template<typename EvaluationType>
CriterionNeoHookeanEnergyPotential<EvaluationType>::
CriterionNeoHookeanEnergyPotential(
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
CriterionNeoHookeanEnergyPotential<EvaluationType>::
evaluate_conditional(
    const Plato::ScalarMultiVectorT <StateT>   & aState,
    const Plato::ScalarMultiVectorT <ControlT> & aControl,
    const Plato::ScalarArray3DT     <ConfigT>  & aConfig,
          Plato::ScalarVectorT      <ResultT>  & aResult,
          Plato::Scalar                          aCycle
) const
{
  // get material properties
  auto tMu     = std::stod(mMaterial->property("lame mu").front());
  auto tLambda = std::stod(mMaterial->property("lame lambda").front());
  // get integration rule information
  auto tNumPoints  = ElementType::mNumGaussPoints;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  // compute state gradient
  Plato::StateGradient<EvaluationType> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
  Plato::RightDeformationTensor<EvaluationType> tComputeRightDeformationTensor;
  // evaluate stress tensor
  auto tNumCells = mSpatialDomain.numCells();
  Kokkos::parallel_for("evaluate stored energy potential", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      auto tCubPoint = tCubPoints(iGpOrdinal);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
        tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,aState,tGradient,tStateGradient);
      // compute green-lagrange strain tensor
      Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
        tDeformationGradient(StrainT(0.));
      tComputeDeformationGradient(tStateGradient,tDeformationGradient);
      // apply transpose to deformation gradient
      Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
        tDeformationGradientT = Plato::transpose(tDeformationGradient);
      // compute right deformation tensor
      Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
        tRightDeformationTensor(StrainT(0.));
      tComputeRightDeformationTensor(tDeformationGradientT,tDeformationGradient,tRightDeformationTensor);
      // compute trace of right deformation tensor
      StrainT tTrace = Plato::trace(tRightDeformationTensor);
      // compute determinant of deformation gradient
      StrainT tDetDeformationGradient = Plato::determinant(tDeformationGradient);
      // evaluate elastic strain energy potential
      StrainT tLogDetF = log(tDetDeformationGradient);
      ResultT tValue = 0.5*tLambda*tLogDetF*tLogDetF - tMu*tLogDetF + 0.5*tMu*(tTrace-3.0);
      Kokkos::atomic_add(&aResult(iCellOrdinal), tValue);
  });
}

} // namespace Plato
