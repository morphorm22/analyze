/*
 * DarkCurrentDensityTwoPhaseAlloy_def.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "elliptic/electrical/FactoryElectricalMaterial.hpp"

namespace Plato
{

template<typename EvaluationType>
DarkCurrentDensityTwoPhaseAlloy<EvaluationType>::
DarkCurrentDensityTwoPhaseAlloy(
  const std::string            & aMaterialName,
  const std::string            & aCurrentDensityName,
        Teuchos::ParameterList & aParamList
) : 
  mMaterialName(aMaterialName),
  mCurrentDensityName(aCurrentDensityName)
{
    this->initialize(aParamList);
}

template<typename EvaluationType>
DarkCurrentDensityTwoPhaseAlloy<EvaluationType>::
~DarkCurrentDensityTwoPhaseAlloy(){}

template<typename EvaluationType>
void
DarkCurrentDensityTwoPhaseAlloy<EvaluationType>::
evaluate(
    const Plato::SpatialDomain                         & aSpatialDomain,
    const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
    const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
    const Plato::Scalar                                & aScale
) const
{
  // compute current density
  Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
  Plato::ScalarMultiVectorT<StateScalarType> 
    tCurrentDensity("current density",tNumCells,mNumGaussPoints);
  mDarkCurrentDensity->evaluate(aState,tCurrentDensity);
  // integration rule
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  // out-of-plane thicknesses
  Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
  Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();
  // evaluate dark current density
  Plato::Scalar tPenaltyExponent = mPenaltyExponent;
  Kokkos::parallel_for("evaluate dark current density", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, mNumGaussPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // get basis functions and weights for this integration point
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tDetJ = Plato::determinant(ElementType::jacobian(tCubPoint, aConfig, iCellOrdinal));
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    // out-of-plane thickness interpolation
    ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
    ControlScalarType tThicknessPenalty = pow(tDensity, tPenaltyExponent);
    ControlScalarType tThicknessInterpolation = tThicknessTwo + 
      ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
    auto tWeight = aScale * tCubWeights(iGpOrdinal) * tDetJ;
    for (Plato::OrdinalType tFieldOrdinal = 0; tFieldOrdinal < mNumNodesPerCell; tFieldOrdinal++)
    {
      ResultScalarType tCellResult = ( tBasisValues(tFieldOrdinal) * 
        tCurrentDensity(iCellOrdinal,iGpOrdinal) * tWeight ) / tThicknessInterpolation;
      Kokkos::atomic_add( &aResult(iCellOrdinal,tFieldOrdinal), tCellResult );
    }
  });
}

template<typename EvaluationType>
void
DarkCurrentDensityTwoPhaseAlloy<EvaluationType>::
evaluate(
    const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
    const Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
    const Plato::Scalar                                & aScale
) 
const
{
  // compute current density
  Plato::OrdinalType tNumCells = aResult.extent(0);
  Plato::ScalarMultiVectorT<StateScalarType> 
    tCurrentDensity("current density",tNumCells,mNumGaussPoints);
  mDarkCurrentDensity->evaluate(aState,tCurrentDensity);
  // integration rule
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  // out-of-plane thicknesses
  Plato::Scalar tThicknessOne = mOutofPlaneThickness.front();
  Plato::Scalar tThicknessTwo = mOutofPlaneThickness.back();
  // evaluate dark current density
  Plato::Scalar tPenaltyExponent = mPenaltyExponent;
  Kokkos::parallel_for("evaluate dark current density", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, mNumGaussPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // get basis functions and weights for this integration point
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    // out-of-plane thickness interpolation
    ControlScalarType tDensity = Plato::cell_density<mNumNodesPerCell>(iCellOrdinal,aControl,tBasisValues);
    ControlScalarType tThicknessPenalty = pow(tDensity, tPenaltyExponent);
    ControlScalarType tThicknessInterpolation = tThicknessTwo + 
      ( ( tThicknessOne - tThicknessTwo) * tThicknessPenalty );
    aResult(iCellOrdinal,iGpOrdinal) = 
      ( aScale * tCurrentDensity(iCellOrdinal,iGpOrdinal) ) / tThicknessInterpolation;
  });
}

template<typename EvaluationType>
void
DarkCurrentDensityTwoPhaseAlloy<EvaluationType>::
initialize(
    Teuchos::ParameterList &aParamList
)
{
    if( !aParamList.isSublist("Source Terms") ){
      auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
      ANALYZE_THROWERR(tMsg)
    }
    auto tSourceTermsSublist = aParamList.sublist("Source Terms");
    if( !tSourceTermsSublist.isSublist(mCurrentDensityName) ){
      auto tMsg = std::string("Parameter is not valid. Argument ('") + mCurrentDensityName 
        + "') is not a parameter list";
      ANALYZE_THROWERR(tMsg)
    }
    Teuchos::ParameterList& tSublist = aParamList.sublist(mCurrentDensityName);
    mPenaltyExponent = tSublist.get<Plato::Scalar>("Penalty Exponent", 3.0);
    mMinErsatzMaterialValue = tSublist.get<Plato::Scalar>("Minimum Value", 0.0);
    // set current density model
    mCurrentDensityModel = tSublist.get<std::string>("Model","Quadratic");
    this->buildCurrentDensityModel(aParamList);
    // set out-of-plane thickness array
    Plato::FactoryElectricalMaterial<EvaluationType> tMaterialFactory(aParamList);
    auto tMaterialModel = tMaterialFactory.create(mMaterialName);
    this->setOutofPlaneThickness(tMaterialModel.operator*());
}

template<typename EvaluationType>
void
DarkCurrentDensityTwoPhaseAlloy<EvaluationType>::
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

template<typename EvaluationType>
void
DarkCurrentDensityTwoPhaseAlloy<EvaluationType>::
buildCurrentDensityModel(
  Teuchos::ParameterList &aParamList
)
{
  mDarkCurrentDensity = 
    std::make_shared<Plato::DarkCurrentDensityQuadratic<EvaluationType>>(mCurrentDensityName,aParamList);
}

}
// namespace Plato