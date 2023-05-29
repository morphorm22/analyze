/*
 *  ResidualSteadyStateCurrent_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "ToMap.hpp"
#include "ScalarGrad.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "GeneralFluxDivergence.hpp"
#include "elliptic/electrical/FactorySourceEvaluator.hpp"
#include "elliptic/electrical/FactoryElectricalMaterial.hpp"

namespace Plato
{

template<typename EvaluationType>
ResidualSteadyStateCurrent<EvaluationType>:: 
ResidualSteadyStateCurrent(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList
) : 
  FunctionBaseType(aSpatialDomain,aDataMap)
{
  this->initialize(aParamList);
}

template<typename EvaluationType>
ResidualSteadyStateCurrent<EvaluationType>::
~ResidualSteadyStateCurrent(){}

template<typename EvaluationType>
Plato::Solutions 
ResidualSteadyStateCurrent<EvaluationType>::
getSolutionStateOutputData(
  const Plato::Solutions &aSolutions
) const 
{ 
  return aSolutions; 
}

template<typename EvaluationType>
void
ResidualSteadyStateCurrent<EvaluationType>::
evaluate(
    const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
    const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
    const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
          Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
          Plato::Scalar                                   aCycle
) const
{
  using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
  
  // inline functors
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::GeneralFluxDivergence<ElementType> tComputeDivergence;
  Plato::ScalarGrad<ElementType>            tComputeScalarGrad;
  // interpolate nodal values to integration points
  Plato::InterpolateFromNodal<ElementType,mNumDofsPerNode> tInterpolateFromNodal;
  // integration rules
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();   
  // quantity of interests
  auto tNumCells = mSpatialDomain.numCells();
  Plato::ScalarVectorT<ConfigScalarType>      
    tVolume("InterpolateFromNodalvolume",tNumCells);
  Plato::ScalarMultiVectorT<GradScalarType>   
    tElectricField("electrical field", tNumCells, mNumSpatialDims);
  Plato::ScalarMultiVectorT<ResultScalarType> 
    tCurrentDensity("current density", tNumCells, mNumSpatialDims);
  Plato::ScalarArray4DT<ResultScalarType> 
    tMaterialTensor("material tensor", tNumCells, tNumPoints, mNumSpatialDims, mNumSpatialDims);
  // evaluate material tensor
  mMaterialModel->computeMaterialTensor(mSpatialDomain,aState,aControl,tMaterialTensor);
  // evaluate internal forces       
  Kokkos::parallel_for("evaluate electrostatics residual", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
      ConfigScalarType tCellVolume(0.0);  
      Plato::Array<mNumSpatialDims,GradScalarType>   tCellElectricField(0.0);
      Plato::Array<mNumSpatialDims,ResultScalarType> tCellCurrentDensity(0.0);
      Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;  
      auto tCubPoint = tCubPoints(iGpOrdinal);
      // compute electrical field 
      tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tCellVolume);
      tComputeScalarGrad(iCellOrdinal,tCellElectricField,aState,tGradient);
      // compute current density
      for (Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        tCellCurrentDensity(tDimI) = 0.0;
        for (Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          tCellCurrentDensity(tDimI) += tMaterialTensor(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) 
            * tCellElectricField(tDimJ);
        }
      }
      // apply divergence operator to current density
      tCellVolume *= tCubWeights(iGpOrdinal);
      tComputeDivergence(iCellOrdinal,aResult,tCellCurrentDensity,tGradient,tCellVolume,1.0);
      // compute output quantities of interests  
      for(Plato::OrdinalType tIndex=0; tIndex<mNumSpatialDims; tIndex++)
      {
        // compute the electric field E = -\nabla{\phi} (or -\phi_{,j}, where j=1,\dots,dims)
        Kokkos::atomic_add(&tElectricField(iCellOrdinal,tIndex),  -1.0*tCellVolume*tCellElectricField(tIndex));
        // Ohm constitutive law J = -\gamma_{ij}\phi_{,j}, where \phi is the scalar electric potential, 
        // \gamma is the second order electric conductivity tensor, and J is the current density
        Kokkos::atomic_add(&tCurrentDensity(iCellOrdinal,tIndex), -1.0*tCellVolume*tCellCurrentDensity(tIndex));
      }
      Kokkos::atomic_add(&tVolume(iCellOrdinal),tCellVolume);
  });
  // evaluate volume forces
  if( mSourceEvaluator != nullptr )
  {
    mSourceEvaluator->evaluate( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
  }
  Kokkos::parallel_for("compute cell quantities", 
    Kokkos::RangePolicy<>(0, tNumCells),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
  {
      for(Plato::OrdinalType tIndex=0; tIndex<mNumSpatialDims; tIndex++)
      {
          tElectricField(iCellOrdinal,tIndex)  /= tVolume(iCellOrdinal);
          tCurrentDensity(iCellOrdinal,tIndex) /= tVolume(iCellOrdinal);
      }
  });
  if( std::count(mPlottable.begin(),mPlottable.end(),"electric field") ) 
  { Plato::toMap(mDataMap, tElectricField, "electric field", mSpatialDomain); }
  if( std::count(mPlottable.begin(),mPlottable.end(),"current density" ) )
  { Plato::toMap(mDataMap, tCurrentDensity, "current density" , mSpatialDomain); }
}

template<typename EvaluationType>
void
ResidualSteadyStateCurrent<EvaluationType>::
evaluate_boundary(
    const Plato::SpatialModel                           & aSpatialModel,
    const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
    const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
    const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
          Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
          Plato::Scalar                                   aCycle
) const
{
  // add contributions from natural boundary conditions
  if( mSurfaceLoads != nullptr )
  {
      mSurfaceLoads->get(aSpatialModel,aState,aControl,aConfig,aResult,1.0);
  }
}

template<typename EvaluationType>
void 
ResidualSteadyStateCurrent<EvaluationType>::
initialize(
  Teuchos::ParameterList & aParamList
)
{
  // obligatory: define dof names in order
  mDofNames.push_back("electric_potential");
  // create material constitutive model
  auto tMaterialName = mSpatialDomain.getMaterialName();
  Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
  mMaterialModel = tMaterialFactory.create(tMaterialName);
  // create source evaluator
  Plato::FactorySourceEvaluator<EvaluationType> tFactorySourceEvaluator;
  mSourceEvaluator = tFactorySourceEvaluator.create(tMaterialName,aParamList);
  // parse output QoI plot table
  auto tResidualParams = aParamList.sublist("Output");
  if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
  {
      mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
  }
}

}