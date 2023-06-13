/*
 * CriterionKirchhoffEnergyPotential_def.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "GradientMatrix.hpp"
#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/GreenLagrangeStrainTensor.hpp"
#include "elliptic/mechanical/nonlinear/KirchhoffSecondPiolaStress.hpp"
#include "elliptic/mechanical/nonlinear/FactoryNonlinearElasticMaterial.hpp"

namespace Plato
{

namespace Elliptic
{
  
template<typename EvaluationType>
CriterionKirchhoffEnergyPotential<EvaluationType>::
CriterionKirchhoffEnergyPotential(
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
CriterionKirchhoffEnergyPotential<EvaluationType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType>
void 
CriterionKirchhoffEnergyPotential<EvaluationType>::
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
  // get integration rule information
  auto tNumPoints  = mNumGaussPoints;
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
      ConfigScalarType tVolume(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStateGradient(StrainScalarType(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute green-lagrange strain tensor
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> 
        tStrainTensor(StrainScalarType(0.));
      tGreenLagrangeStrainTensor(tStateGradient,tStrainTensor);
      // compute second piola-kirchhoff stress
      Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> 
        tStressTensor2PK(ResultScalarType(0.));
      tComputeSecondPiolaKirchhoffStress(tStrainTensor,tStressTensor2PK);
      // apply integration point weight to element volume
      tVolume *= tCubWeights(iGpOrdinal);
      // evaluate elastic strain energy potential 
      ResultScalarType tValue(0.0);
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          tValue += 0.5 * tStrainTensor(tDimI,tDimJ) * tStressTensor2PK(tDimI,tDimJ) * tVolume;
        }
      }
      Kokkos::atomic_add(&tResultWS(iCellOrdinal), tValue);
  });
}

} // namespace Elliptic

} // namespace Plato
