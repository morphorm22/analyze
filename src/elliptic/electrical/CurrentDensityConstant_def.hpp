/*
 *  CurrentDensityConstant_def.hpp
 *
 *  Created on: June 2, 2023
 */

#include "ToMap.hpp"
#include "ScalarGrad.hpp"
#include "GradientMatrix.hpp"
#include "elliptic/electrical/FactoryElectricalMaterial.hpp"

namespace Plato
{
    
template<typename EvaluationType>
CurrentDensityConstant<EvaluationType>::
CurrentDensityConstant(
  const std::string            & aMaterialName,
        Teuchos::ParameterList & aParamList,
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap
) : 
  BaseClassType(aSpatialDomain,aDataMap)
{
  // create material model
  Plato::FactoryElectricalMaterial<EvaluationType> tFactory(aParamList);
  mMaterial = tFactory.create(aMaterialName);
  // parse output QoI plot table
  auto tOutputParams = aParamList.sublist("Output");
  if( tOutputParams.isType<Teuchos::Array<std::string>>("Plottable") ){
    mPlottable = tOutputParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
  }
}

template<typename EvaluationType>
void 
CurrentDensityConstant<EvaluationType>::
evaluate(
  const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
  const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
  const Plato::ScalarArray3DT<ResultScalarType>      & aResult,
        Plato::Scalar                                  aCycle
) const
{
  // get material tensor
  const Plato::TensorConstant<ElementType::mNumSpatialDims> tMaterialTensor = 
    mMaterial->getTensorConstant("material tensor");
  // inline functors
  Plato::ScalarGrad<ElementType>            tComputeScalarGrad;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  // integration rules
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = ElementType::mNumGaussPoints;
  // output quantities of interest
  auto tNumCells = mSpatialDomain.numCells();
  Plato::ScalarVectorT<ConfigScalarType>      
    tVolume("volume",tNumCells);
  Plato::ScalarMultiVectorT<GradScalarType>   
    tElectricField("electrical field",tNumCells,ElementType::mNumSpatialDims);
  Plato::ScalarMultiVectorT<ResultScalarType>   
    tCurrentDensity("current density",tNumCells,ElementType::mNumSpatialDims);
  // evaluate internal forces
  Kokkos::parallel_for("evaluate electrostatics residual", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells,tNumPoints}),
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
    // compute current density
    for (Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      aResult(iCellOrdinal,iGpOrdinal,tDimI) = 0.0;
      for (Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        aResult(iCellOrdinal,iGpOrdinal,tDimI) += 
          tMaterialTensor(tDimI,tDimJ) * tCellElectricField(tDimJ);
      }
    }
    // pre-process output quantities of interests  
    tCellVolume *= tCubWeights(iGpOrdinal);
    Kokkos::atomic_add(&tVolume(iCellOrdinal),tCellVolume);
    for(Plato::OrdinalType tDim=0; tDim<ElementType::mNumSpatialDims; tDim++){
      Kokkos::atomic_add(&tElectricField(iCellOrdinal,tDim),-1.0*tCellVolume*tCellElectricField(tDim));
      Kokkos::atomic_add(&tCurrentDensity(iCellOrdinal,tDim), -1.0*tCellVolume*aResult(iCellOrdinal,iGpOrdinal,tDim));
    }
  });
  
  // post-process output quantities of interests
  Kokkos::parallel_for("compute output quantities", 
    Kokkos::RangePolicy<>(0, tNumCells),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
  {
    for(Plato::OrdinalType tDim=0; tDim< ElementType::mNumSpatialDims; tDim++){
      tElectricField(iCellOrdinal,tDim)  /= tVolume(iCellOrdinal);
      tCurrentDensity(iCellOrdinal,tDim) /= tVolume(iCellOrdinal);
    }
  });
  // save output quantities of interest in output database
  if( std::count(mPlottable.begin(),mPlottable.end(),"electric field") ) 
  { Plato::toMap(mDataMap,tElectricField,"electric field",mSpatialDomain); }
  if( std::count(mPlottable.begin(),mPlottable.end(),"current density" ) )
  { Plato::toMap(mDataMap,tCurrentDensity,"current density",mSpatialDomain); }

}

} // namespace Plato
