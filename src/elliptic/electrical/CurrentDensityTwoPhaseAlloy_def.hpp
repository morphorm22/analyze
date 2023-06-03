/*
 *  CurrentDensityTwoPhaseAlloy_def.hpp
 *
 *  Created on: June 2, 2023
 */

#include "Simp.hpp"
#include "ScalarGrad.hpp"
#include "GradientMatrix.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "elliptic/electrical/FactoryElectricalMaterial.hpp"

namespace Plato
{

template<typename EvaluationType>
CurrentDensityTwoPhaseAlloy<EvaluationType>::
CurrentDensityTwoPhaseAlloy(
  const std::string            & aMaterialName,
        Teuchos::ParameterList & aParamList,
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap
) : 
  BaseClassType(aSpatialDomain,aDataMap)
{
  Plato::FactoryElectricalMaterial<EvaluationType> tFactory(aParamList);
  mMaterial = tFactory.create(aMaterialName);
  mPenaltyExponent = std::stod( mMaterial->property("Penalty Exponent").back() );
  mMinErsatzMaterialValue = std::stod( mMaterial->property("Minimum Value").back() );
}

template<typename EvaluationType>
void 
CurrentDensityTwoPhaseAlloy<EvaluationType>::
evaluate(
  const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
  const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
  const Plato::ScalarArray3DT<ResultScalarType>      & aResult,
        Plato::Scalar                                  aCycle
) const
{
  // get material tensor for each phase
  auto tMaterialNames = mMaterial->property("material name");
  const Plato::TensorConstant<ElementType::mNumSpatialDims> tTensorOne = 
    mMaterial->getTensorConstant(tMaterialNames.front());
  const Plato::TensorConstant<ElementType::mNumSpatialDims> tTensorTwo = 
    mMaterial->getTensorConstant(tMaterialNames.back());
  // inline functors
  Plato::ScalarGrad<ElementType>            tComputeScalarGrad;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  // create material penalty model
  Plato::MSIMP tSIMP(mPenaltyExponent,mMinErsatzMaterialValue);
  // evaluate current density     
  auto tCubPoints  = ElementType::getCubPoints();
  auto tNumPoints  = ElementType::mNumGaussPoints;     
  auto tNumCells   = mSpatialDomain.numCells();
  Kokkos::parallel_for("evaluate material tensor for 2-phase alloy", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // compute gradient functions 
    ConfigScalarType tCellVolume(0.0);  
    auto tCubPoint = tCubPoints(iGpOrdinal);
    Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigScalarType> tGradient;  
    tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tCellVolume);
    // compute electric field
    Plato::Array<ElementType::mNumSpatialDims,GradScalarType> tCellElectricField(0.0);
    tComputeScalarGrad(iCellOrdinal,tCellElectricField,aState,tGradient);
    // compute material interpolation
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    ControlScalarType tDensity = 
      Plato::cell_density<ElementType::mNumNodesPerCell>(iCellOrdinal, aControl, tBasisValues);
    ControlScalarType tMaterialPenalty = tSIMP(tDensity);
    // compute penalized current density
    for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      ResultScalarType tValue = 0.;
      for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tValue += ( tTensorTwo(tDimI,tDimJ) 
          + ( ( tTensorOne(tDimI,tDimJ) - tTensorTwo(tDimI,tDimJ) ) * tMaterialPenalty ) ) 
          * tCellElectricField(tDimJ);
      }
      aResult(iCellOrdinal,iGpOrdinal,tDimI) = tValue;
    }
  });
}

} // namespace Plato
