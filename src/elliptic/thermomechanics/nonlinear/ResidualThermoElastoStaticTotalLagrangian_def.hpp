/*
 * ResidualThermoElastoStaticTotalLagrangian_def.hpp
 *
 *  Created on: June 17, 2023
 */

#pragma once

#include "PlatoMathTypes.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"

#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/NominalStressTensor.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"
#include "elliptic/mechanical/nonlinear/FactoryStressEvaluator.hpp"
#include "elliptic/mechanical/nonlinear/KineticPullBackOperation.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermalDeformationGradient.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermoElasticDeformationGradient.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType>
ResidualThermoElastoStaticTotalLagrangian<EvaluationType>::
ResidualThermoElastoStaticTotalLagrangian(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aParamList
) : 
  FunctionBaseType(aSpatialDomain, aDataMap),
  mStressEvaluator(nullptr),
  mNaturalBCs     (nullptr),
  mBodyLoads      (nullptr),
  mParamList(aParamList)
{
  // obligatory: define dof names in order
  //
  mDofNames.push_back("displacement X");
  if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
  if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");
  // initialize member data
  //
  this->initialize(aParamList);
}

template<typename EvaluationType>
Plato::Solutions
ResidualThermoElastoStaticTotalLagrangian<EvaluationType>::
getSolutionStateOutputData(
  const Plato::Solutions & aSolutions
) const
{
  return aSolutions;
}

template<typename EvaluationType>
void
ResidualThermoElastoStaticTotalLagrangian<EvaluationType>::
evaluate(
  Plato::WorkSets & aWorkSets,
  Plato::Scalar     aCycle
) const
{
  if(mStressEvaluator == nullptr){
    ANALYZE_THROWERR("ERROR: Stress evaluator is not defined, mechanical stress tensor cannot be computed")
  }
  // compute second piola-kirchhoff stress workset in the deformed mechanical configuration
  Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
  Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
  Plato::ScalarArray4DT<ResultScalarType> t2PKS_WS(
    "2nd Piola-Kirchhoff Stress",tNumCells,tNumGaussPoints,mNumSpatialDims,mNumSpatialDims
  );
  mStressEvaluator->evaluate(aWorkSets,t2PKS_WS,aCycle);
  // unpack worksets
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<ControlScalarType> tControlWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));
  Plato::ScalarMultiVectorT<StateScalarType> tDispWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
  Plato::ScalarMultiVectorT<NodeStateScalarType> tTempWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<NodeStateScalarType>>(aWorkSets.get("node_states"));
  Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
  // create local functors
  Plato::StateGradient<EvaluationType> tComputeDispGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::NominalStressTensor<EvaluationType> tComputeNominalStressTensor;
  Plato::KineticPullBackOperation<EvaluationType> tApplyKineticPullBackOperation;
  Plato::DeformationGradient<EvaluationType> tComputeMechanicalDeformationGradient;
  Plato::ThermoElasticDeformationGradient<EvaluationType> tComputeThermoElasticDeformationGradient;
  Plato::ThermalDeformationGradient<EvaluationType> tComputeThermalDeformationGradient(
    mSpatialDomain.getMaterialName(),mParamList
  );
  Plato::InterpolateFromNodal<ElementType,mNumNodeStatePerNode> tInterpolateFromNodal;
  // get integration rule data
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  // evaluate internal forces
  Kokkos::parallel_for("compute internal forces", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,mNumGaussPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // compute gradient of interpolation functions
    ConfigScalarType tVolume(0.0);
    auto tCubPoint = tCubPoints(iGpOrdinal);
    Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient(ConfigScalarType(0.));
    tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
    // compute displacement gradient
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> tDispGradient(StrainScalarType(0.));
    tComputeDispGradient(iCellOrdinal,tDispWS,tGradient,tDispGradient);
    // compute mechanical deformation gradient 
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType> tMechanicalDefGradient(StrainScalarType(0.));
    tComputeMechanicalDeformationGradient(tDispGradient,tMechanicalDefGradient);
    // interpolate temperature field from nodes to integration point
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    NodeStateScalarType tTemperature = tInterpolateFromNodal(iCellOrdinal,tBasisValues,tTempWS);
    // compute thermal deformation gradient 
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,NodeStateScalarType> tThermalDefGradient(NodeStateScalarType(0.));
    tComputeThermalDeformationGradient(tTemperature,tThermalDefGradient);
    // compute multiplicative decomposition of the thermo-elastic deformation gradient 
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tThermoElasticDefGradient(ResultScalarType(0.));
    tComputeThermoElasticDeformationGradient(tThermalDefGradient,tMechanicalDefGradient,tThermoElasticDefGradient);
    // pull back second Piola-Kirchhoff stress from deformed to undeformed configuration
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tDefConfig2PKS(ResultScalarType(0.));
    Plato::Elliptic::get_cell_2PKS<mNumSpatialDims>(iCellOrdinal,iGpOrdinal,t2PKS_WS,tDefConfig2PKS);
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tUnDefConfig2PKS(ResultScalarType(0.));
    tApplyKineticPullBackOperation(tThermalDefGradient,tDefConfig2PKS,tUnDefConfig2PKS);
    // compute nominal stress
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType> tNominalStressTensor(ResultScalarType(0.));
    tComputeNominalStressTensor(tThermoElasticDefGradient,tUnDefConfig2PKS,tNominalStressTensor);
    // apply integration point weight to element volume
    tVolume *= tCubWeights(iGpOrdinal);
    // apply divergence operator to nominal stress tensor
    for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++){
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
        Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumSpatialDims + tDimI;
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
          ResultScalarType tVal = tNominalStressTensor(tDimI,tDimJ) * tGradient(tNodeIndex,tDimJ) * tVolume;
          Kokkos::atomic_add( &tResultWS(iCellOrdinal,tLocalOrdinal),tVal );
        }
      }
    }
  });
  // evaluate body forces
  if( mBodyLoads != nullptr )
  {
    mBodyLoads->get( mSpatialDomain,tDispWS,tControlWS,tConfigWS,tResultWS,-1.0 );
  }
}

template<typename EvaluationType>
void
ResidualThermoElastoStaticTotalLagrangian<EvaluationType>::
evaluateBoundary(
  const Plato::SpatialModel & aSpatialModel,
        Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aCycle
) const
{
  // unpack worksets
  Plato::ScalarArray3DT<ConfigScalarType> tConfigWS  = 
    Plato::unpack<Plato::ScalarArray3DT<ConfigScalarType>>(aWorkSets.get("configuration"));
  Plato::ScalarMultiVectorT<ControlScalarType> tControlWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ControlScalarType>>(aWorkSets.get("controls"));
  Plato::ScalarMultiVectorT<StateScalarType> tDispWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("state"));
  Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
  // evaluate boundary forces
  if( mNaturalBCs != nullptr )
  {
    mNaturalBCs->get(aSpatialModel, tDispWS, tControlWS, tConfigWS, tResultWS, -1.0 );
  }
}

template<typename EvaluationType>
void
ResidualThermoElastoStaticTotalLagrangian<EvaluationType>::
initialize(
  Teuchos::ParameterList & aParamList
)
{
  // create material model and get stiffness
  //
  Plato::FactoryStressEvaluator<EvaluationType> tStressEvaluatorFactory(mSpatialDomain.getMaterialName());
  mStressEvaluator = tStressEvaluatorFactory.create(aParamList,mSpatialDomain,mDataMap);
  // parse body loads
  // 
  if(aParamList.isSublist("Body Loads"))
  {
    mBodyLoads = 
      std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>(aParamList.sublist("Body Loads"));
  }
  // parse boundary Conditions
  // 
  if(aParamList.isSublist("Natural Boundary Conditions"))
  {
    mNaturalBCs = 
      std::make_shared<Plato::NaturalBCs<ElementType>>(aParamList.sublist("Natural Boundary Conditions"));
  }
  // parse plot table
  //
  auto tResidualParams = aParamList.sublist("Output");
  if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
  {
    mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
  }
}

} // namespace Elliptic


} // namespace Plato
