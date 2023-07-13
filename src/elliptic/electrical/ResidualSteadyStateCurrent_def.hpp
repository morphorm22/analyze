/*
 *  ResidualSteadyStateCurrent_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "BLAS1.hpp"
#include "MetaData.hpp"
#include "ScalarGrad.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "GeneralFluxDivergence.hpp"

#include "materials/MaterialModel.hpp"
#include "elliptic/electrical/FactorySourceEvaluator.hpp"
#include "elliptic/electrical/FactoryElectricalMaterial.hpp"
#include "elliptic/electrical/FactoryCurrentDensityEvaluator.hpp"

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
void
ResidualSteadyStateCurrent<EvaluationType>::
postProcess(
  const Plato::Solutions & aSolutions
) 
{ 
  Plato::ScalarMultiVector tElectricalPotential = aSolutions.get("states");
  Plato::ScalarVector tMyElectricalPotential = Kokkos::subview(tElectricalPotential,/*cycle index=*/0,Kokkos::ALL());
  if( mMaterial->scalarConstantExists("Electrical Resistance") ){
    Plato::Scalar tElectricalResistance = std::stod(mMaterial->property("Electrical Resistance").front());
    auto tOneOverElectricalResistance = 1.0 / tElectricalResistance;
    Plato::ScalarVector tElectricalCurrent("Current",tMyElectricalPotential.extent(0));
    Plato::blas1::update(tOneOverElectricalResistance,tMyElectricalPotential,0.0,tElectricalCurrent);
    mDataMap.scalarNodeFields["electrical current"] = tElectricalCurrent;
  }
}

template<typename EvaluationType>
void
ResidualSteadyStateCurrent<EvaluationType>::
evaluate(
  Plato::WorkSets & aWorkSets,
  Plato::Scalar     aCycle
) const
{
  // unpack worksets
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<ControlScalarType> tControlWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));
  Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
  Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
  // inline functors
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  // integration rules
  auto tNumPoints  = mNumGaussPoints;   
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  // evaluate current density model
  auto tNumCells = mSpatialDomain.numCells();
  Plato::ScalarArray3DT<ResultScalarType> 
    tCurrentDensity("current density",tNumCells,tNumPoints,mNumSpatialDims);
  mCurrentDensityEvaluator->evaluate(tStateWS,tControlWS,tConfigWS,tCurrentDensity);
  // evaluate internal forces       
  Kokkos::parallel_for("evaluate steady state current residual", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    //compute gradient of basis functions
    ConfigScalarType tCellVolume(0.0);  
    auto tCubPoint = tCubPoints(iGpOrdinal);
    Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;  
    tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tCellVolume);
    // apply divergence operator
    tCellVolume *= tCubWeights(iGpOrdinal);
    for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++){
      ResultScalarType tValue(0.);
      Plato::OrdinalType tLocalOrdinal = tNode * mNumDofsPerNode;
      for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++){
        tValue += tCurrentDensity(iCellOrdinal,iGpOrdinal,tDim) * tGradient(tNode, tDim) * tCellVolume;
      }
      Kokkos::atomic_add( &tResultWS(iCellOrdinal,tLocalOrdinal),tValue );
    }
  });
  // evaluate volume forces
  if( mSourceEvaluator != nullptr ){
    mSourceEvaluator->evaluate( mSpatialDomain, tStateWS, tControlWS, tConfigWS, tResultWS, -1.0 );
  }
}

template<typename EvaluationType>
void
ResidualSteadyStateCurrent<EvaluationType>::
evaluateBoundary(
  const Plato::SpatialModel & aSpatialModel,
        Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aCycle
) const
{
  // add contributions from natural boundary conditions
  if( mBoundaryForces != nullptr ){
    mBoundaryForces->get(aSpatialModel,aWorkSets,aCycle,-1.0);
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
  // create current density model
  auto tMaterialName = mSpatialDomain.getMaterialName();
  Plato::FactoryCurrentDensityEvaluator<EvaluationType> tCurrentDensityFactory(tMaterialName,aParamList);
  mCurrentDensityEvaluator = tCurrentDensityFactory.create(mSpatialDomain,mDataMap);
  // create material model
  Plato::FactoryElectricalMaterial<EvaluationType> tFactory(aParamList);
  mMaterial = tFactory.create(tMaterialName);
  // create source evaluator
  Plato::FactorySourceEvaluator<EvaluationType> tFactorySourceEvaluator;
  mSourceEvaluator = tFactorySourceEvaluator.create(tMaterialName,aParamList);
  // parse neumann boundary conditions
  if(aParamList.isSublist("Natural Boundary Conditions")){
    mBoundaryForces = std::make_shared<Plato::NeumannBCs<EvaluationType>>(
      aParamList,aParamList.sublist("Natural Boundary Conditions")
    );
  }
}

}