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

namespace Elliptic
{
  
template<typename EvaluationType>
CriterionNeoHookeanEnergyPotential<EvaluationType>::
CriterionNeoHookeanEnergyPotential(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aParamList,
  const std::string            & aFuncName
) :
  FunctionBaseType(aFuncName,aSpatialDomain,aDataMap,aParamList)
{
  std::string tMaterialName = mSpatialDomain.getMaterialName();
  Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
  mMaterial = tFactory.create(tMaterialName);
}

template<typename EvaluationType>
bool 
CriterionNeoHookeanEnergyPotential<EvaluationType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType>
void 
CriterionNeoHookeanEnergyPotential<EvaluationType>::
evaluateConditional(
  const Plato::WorkSets & aWorkSets,
  const Plato::Scalar   & aCycle
) const
{
  // unpack worksets
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<ControlScalarType> tControlWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));
  Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
  Plato::ScalarVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarVectorT<ResultScalarType>>(aWorkSets.get("result"));
  // get material properties
  auto tMu     = std::stod(mMaterial->property("lame mu").front());
  auto tLambda = std::stod(mMaterial->property("lame lambda").front());
  // get integration rule information
  auto tNumPoints  = mNumGaussPoints;
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
      ConfigScalarType tVolume(0.0);
      auto tCubPoint = tCubPoints(iGpOrdinal);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStateGradient(StrainScalarType(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute green-lagrange strain tensor
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tDeformationGradient(StrainScalarType(0.));
      tComputeDeformationGradient(tStateGradient,tDeformationGradient);
      // apply transpose to deformation gradient
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tDeformationGradientT = Plato::transpose(tDeformationGradient);
      // compute right deformation tensor
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tRightDeformationTensor(StrainScalarType(0.));
      tComputeRightDeformationTensor(tDeformationGradientT,tDeformationGradient,tRightDeformationTensor);
      // compute trace of right deformation tensor
      StrainScalarType tTrace = Plato::trace(tRightDeformationTensor);
      // compute determinant of deformation gradient
      StrainScalarType tDetDeformationGradient = Plato::determinant(tDeformationGradient);
      // evaluate elastic strain energy potential
      StrainScalarType tLogDetF = log(tDetDeformationGradient);
      ResultScalarType tValue = 0.5*tLambda*tLogDetF*tLogDetF - tMu*tLogDetF + 0.5*tMu*(tTrace-3.0);
      Kokkos::atomic_add(&tResultWS(iCellOrdinal), tValue);
  });
}

} // namespace Elliptic

} // namespace Plato
