/*
 *  CriterionPowerSurfaceDensityTwoPhase_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "Plato_TopOptFunctors.hpp"

#include "elliptic/electrical/FactoryElectricalMaterial.hpp"
#include "elliptic/electrical/FactoryCurrentDensityEvaluator.hpp"

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
  FunctionBaseType(aSpatialDomain, aDataMap, aParamList, aFuncName),
  mCriterionFunctionName(aFuncName)
{
  this->initialize(aParamList);
}

template<typename EvaluationType>
CriterionPowerSurfaceDensityTwoPhase<EvaluationType>::
~CriterionPowerSurfaceDensityTwoPhase(){}

template<typename EvaluationType>
void 
CriterionPowerSurfaceDensityTwoPhase<EvaluationType>::
evaluate(
    const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType> & aControl,
    const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>  & aConfig,
          Plato::ScalarVectorT      <typename EvaluationType::ResultScalarType>  & aResult,
          Plato::Scalar                                                            aCycle
)
{ 
  this->evaluate_conditional(aState, aControl, aConfig, aResult, aCycle);
}

template<typename EvaluationType>
void
CriterionPowerSurfaceDensityTwoPhase<EvaluationType>::
evaluate_conditional(
    const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
    const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
          Plato::ScalarVectorT      <ResultScalarType>  & aResult,
          Plato::Scalar                                   aCycle
) const
{
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
  Plato::Scalar tScale = aCycle;
  Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
  Plato::ScalarMultiVectorT<ResultScalarType> tCurrentDensity("current density",tNumCells,tNumPoints);
  mCurrentDensityEvaluator->evaluate(aState,aControl,aConfig,tCurrentDensity,tScale);
  // evaluate dark current density
  Kokkos::parallel_for("power surface density", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
      // get basis functions and weights for this integration point
      auto tCubPoint = tCubPoints(iGpOrdinal);
      auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));
      auto tBasisValues = ElementType::basisValues(tCubPoint);
      // out-of-plane thickness interpolation
      ControlScalarType tDensity =
        Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
      ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
      ControlScalarType tThicknessInterpolation = tThicknessTwo + 
        ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
      // evaluate electric potential
      StateScalarType tCellElectricPotential = 
        tInterpolateFromNodal(iCellOrdinal,tBasisValues,aState);
      ResultScalarType tWeightedCurrentDensity = 
        tCurrentDensity(iCellOrdinal,iGpOrdinal) * tCubWeights(iGpOrdinal) * tDetJ;
      for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
      {
          ResultScalarType tCellResult = tBasisValues(tFieldOrdinal) * 
            tCellElectricPotential * tWeightedCurrentDensity * tThicknessInterpolation;
          Kokkos::atomic_add( &aResult(iCellOrdinal), tCellResult );
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
  Plato::FactoryCurrentDensityEvaluator<EvaluationType> tFactoryCurrentDensityEvaluator;
  mCurrentDensityEvaluator = 
    tFactoryCurrentDensityEvaluator.create(tMaterialName,tCurrentDensityFunctionName,aParamList
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