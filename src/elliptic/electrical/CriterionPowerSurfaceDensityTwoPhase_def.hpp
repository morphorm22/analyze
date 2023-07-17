/*
 *  CriterionPowerSurfaceDensityTwoPhase_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "AnalyzeMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "Plato_TopOptFunctors.hpp"

#include "elliptic/electrical/FactoryElectricalMaterial.hpp"
#include "elliptic/electrical/FactoryCurrentDensitySourceEvaluator.hpp"

namespace Plato
{

template<typename EvaluationType>
CriterionPowerSurfaceDensityTwoPhase<EvaluationType>::
CriterionPowerSurfaceDensityTwoPhase(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList,
    const std::string            & aFuncName
) : 
  FunctionBaseType(aFuncName, aSpatialDomain, aDataMap, aParamList),
  mCriterionFunctionName(aFuncName)
{
  this->initialize(aParamList);
}

template<typename EvaluationType>
CriterionPowerSurfaceDensityTwoPhase<EvaluationType>::
~CriterionPowerSurfaceDensityTwoPhase(){}

template<typename EvaluationType>
bool 
CriterionPowerSurfaceDensityTwoPhase<EvaluationType>::
isLinear() 
const
{
  return false;
}

template<typename EvaluationType>
void
CriterionPowerSurfaceDensityTwoPhase<EvaluationType>::
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
  // integration rule
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // out-of-plane thicknesses
  Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
  Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();
  // build local functor
  Plato::InterpolateFromNodal<ElementType,mNumDofsPerNode> tInterpolateFromNodal;
  // evaluate current density
  Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
  Plato::ScalarMultiVectorT<ResultScalarType> tCurrentDensity("current density",tNumCells,tNumPoints);
  mCurrentDensitySourceEvaluator->evaluate(tStateWS,tControlWS,tConfigWS,tCurrentDensity,1.0);
  // evaluate dark current density
  Plato::Scalar tPenaltyExponent = mPenaltyExponent;
  Kokkos::parallel_for("power surface density", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // get basis functions and weights for this integration point
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, tConfigWS, iCellOrdinal));
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    // out-of-plane thickness interpolation
    ControlScalarType tDensity =
      Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,tControlWS,tBasisValues);
    ControlScalarType tThicknessPenalty = pow(tDensity, tPenaltyExponent);
    ControlScalarType tThicknessInterpolation = tThicknessTwo + 
      ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
    // evaluate electric potential
    StateScalarType tCellElectricPotential = 
      tInterpolateFromNodal(iCellOrdinal,tBasisValues,tStateWS);
    ResultScalarType tWeightedCurrentDensity = 
      tCurrentDensity(iCellOrdinal,iGpOrdinal) * tCubWeights(iGpOrdinal) * tDetJ;
    for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
    {
      ResultScalarType tCellResult = tBasisValues(tFieldOrdinal) * 
        tCellElectricPotential * tWeightedCurrentDensity * tThicknessInterpolation;
      Kokkos::atomic_add( &tResultWS(iCellOrdinal), tCellResult );
    }
  });
}

template<typename EvaluationType>
void 
CriterionPowerSurfaceDensityTwoPhase<EvaluationType>::
initialize(
  Teuchos::ParameterList & aParamList
)
{
  // build current density function
  this->buildCurrentDensityFunction(aParamList);
  // set out-of-plane thickness 
  Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
  auto tMaterialModel = tMaterialFactory.create(mSpatialDomain.getMaterialName());
  this->setOutofPlaneThickness(tMaterialModel.operator*());
}

template<typename EvaluationType>
void 
CriterionPowerSurfaceDensityTwoPhase<EvaluationType>::
buildCurrentDensityFunction(
  Teuchos::ParameterList & aParamList
)
{
  if( !aParamList.isSublist("Criteria") ){
    auto tMsg = std::string("Parameter is not valid. Argument ('Criteria') is not a parameter list");
    ANALYZE_THROWERR(tMsg)
  }
  auto tCriteriaParamList = aParamList.sublist("Criteria");
  if( !tCriteriaParamList.isSublist(mCriterionFunctionName) ){
    auto tMsg = std::string("Parameter is not valid. Argument ('") 
      + mCriterionFunctionName + "') is not a parameter list";
    ANALYZE_THROWERR(tMsg)
  }
  auto tMyCriterionParamList = tCriteriaParamList.sublist(mCriterionFunctionName);
  if( !tMyCriterionParamList.isParameter("Function") ){
    auto tMsg = std::string("Parameter ('Function') is not defined in parameter list ('") 
      + mCriterionFunctionName + "'), power surface density cannot be evaluated";
    ANALYZE_THROWERR(tMsg)
  }
  std::string tMaterialName = mSpatialDomain.getMaterialName();
  std::string tCurrentDensityFunctionName = tMyCriterionParamList.get<std::string>("Function");
  Plato::FactoryCurrentDensitySourceEvaluator<EvaluationType> tFactoryCurrentDensitySourceEvaluator;
  mCurrentDensitySourceEvaluator = 
    tFactoryCurrentDensitySourceEvaluator.create(tMaterialName,tCurrentDensityFunctionName,aParamList
  );
}

template<typename EvaluationType>
void 
CriterionPowerSurfaceDensityTwoPhase<EvaluationType>::
setOutofPlaneThickness(
  Plato::MaterialModel<EvaluationType> & aMaterialModel
)
{
  std::vector<std::string> tThickness = aMaterialModel.property("out-of-plane thickness");
  if ( tThickness.empty() )
  {
    auto tMsg = std::string("Array of out-of-plane thicknesses is empty. ") 
      + "Dark current density cannnot be computed.";
    ANALYZE_THROWERR(tMsg)
  }
  else
  {
    mOutofPlaneThickness.clear();
    for(size_t tIndex = 0; tIndex < tThickness.size(); tIndex++)
    {
      Plato::Scalar tMyThickness = std::stod(tThickness[tIndex]);
      mOutofPlaneThickness.push_back(tMyThickness);
    }
  }
}

}