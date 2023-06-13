/*
 *  CriterionVolumeTwoPhase_decl.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "MetaData.hpp"
#include "AnalyzeMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "Plato_TopOptFunctors.hpp"

#include "elliptic/electrical/FactoryElectricalMaterial.hpp"

namespace Plato
{

template<typename EvaluationType>
CriterionVolumeTwoPhase<EvaluationType>::
CriterionVolumeTwoPhase(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aParamList,
  const std::string            & aFuncName
) :
    FunctionBaseType(aFuncName, aSpatialDomain, aDataMap, aParamList)
{
    this->initialize(aParamList);
}

template<typename EvaluationType>
CriterionVolumeTwoPhase<EvaluationType>::
~CriterionVolumeTwoPhase(){}

template<typename EvaluationType>
bool 
CriterionVolumeTwoPhase<EvaluationType>::
isLinear() 
const
{
  return true;
}

template<typename EvaluationType>
void
CriterionVolumeTwoPhase<EvaluationType>::
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
  Plato::ScalarVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarVectorT<ResultScalarType>>(aWorkSets.get("result"));
  // out-of-plane thicknesses for two-phase electrical material
  Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
  Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();
  // get basis functions and weights associated with the integration rule
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // evaluate volume for a two-phase electrical material
  auto tNumCells = mSpatialDomain.numCells();
  Kokkos::parallel_for("compute volume", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // evaluate cell jacobian
    auto tCubPoint  = tCubPoints(iGpOrdinal);
    auto tCubWeight = tCubWeights(iGpOrdinal);
    auto tJacobian = ElementType::jacobian(tCubPoint, tConfigWS, iCellOrdinal);
    // compute cell volume
    ResultScalarType tCellVolume = Plato::determinant(tJacobian);
    tCellVolume *= tCubWeight;
    // evaluate out-of-plane thickness interpolation function
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    ControlScalarType tDensity = 
        Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,tControlWS,tBasisValues);
    ControlScalarType tThicknessPenalty = pow(tDensity, mPenaltyExponent);
    ControlScalarType tThicknessInterpolation = tThicknessTwo + 
      ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
    // apply penalty to volume
    ResultScalarType tPenalizedVolume = tThicknessInterpolation * tCellVolume;
    Kokkos::atomic_add(&tResultWS(iCellOrdinal), tPenalizedVolume);
  });
}

template<typename EvaluationType>
void 
CriterionVolumeTwoPhase<EvaluationType>::
initialize(
  Teuchos::ParameterList & aParamList
)
{
  Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
  auto tMaterialModel = tMaterialFactory.create(mSpatialDomain.getMaterialName());
  this->setOutofPlaneThickness(tMaterialModel.operator*());
  mPenaltyExponent = aParamList.get<Plato::Scalar>("Penalty Exponent", 3.0);
}

template<typename EvaluationType>
void 
CriterionVolumeTwoPhase<EvaluationType>::
setOutofPlaneThickness(
  Plato::MaterialModel<EvaluationType> & aMaterialModel
)
{
  std::vector<std::string> tThickness = aMaterialModel.property("out-of-plane thickness");
  if ( tThickness.empty() )
  {
      auto tMsg = std::string("Array of out-of-plane thicknesses is empty. ") 
        + "Volume criterion for two-phase material alloy cannnot be computed.";
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