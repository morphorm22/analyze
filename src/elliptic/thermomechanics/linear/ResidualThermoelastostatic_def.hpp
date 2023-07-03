#pragma once

#include "ToMap.hpp"
#include "BLAS2.hpp"
#include "MetaData.hpp"
#include "PlatoTypes.hpp"
#include "TMKinematics.hpp"
#include "GradientMatrix.hpp"
#include "TMKineticsFactory.hpp"
#include "InterpolateFromNodal.hpp"
#include "GeneralFluxDivergence.hpp"
#include "GeneralStressDivergence.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
ResidualThermoelastostatic<EvaluationType, IndicatorFunctionType>::
ResidualThermoelastostatic(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
        Teuchos::ParameterList & aPenaltyParams
) :
  FunctionBaseType      (aSpatialDomain, aDataMap),
  mIndicatorFunction    (aPenaltyParams),
  mApplyStressWeighting (mIndicatorFunction),
  mApplyFluxWeighting   (mIndicatorFunction),
  mBodyLoads            (nullptr),
  mBoundaryLoads        (nullptr),
  mBoundaryFluxes       (nullptr)
{
  // obligatory: define dof names in order
  mDofNames.push_back("displacement X");
  if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
  if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");
  mDofNames.push_back("temperature");
  // create material model and get stiffness
  //
  Plato::ThermoelasticModelFactory<EvaluationType> mmfactory(aProblemParams);
  mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
  // parse body loads
  // 
  if(aProblemParams.isSublist("Body Loads"))
  {
    mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>(
      aProblemParams.sublist("Body Loads")
    );
  }
  // parse mechanical boundary Conditions
  // 
  if(aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
  {
    mBoundaryLoads = std::make_shared<Plato::NeumannBCs<EvaluationType,NMechDims,MDofOffset>>(
      aProblemParams.sublist("Mechanical Natural Boundary Conditions")
    );
  }  
  // parse thermal boundary Conditions
  // 
  if(aProblemParams.isSublist("Thermal Natural Boundary Conditions"))
  {
    mBoundaryFluxes = std::make_shared<Plato::NeumannBCs<EvaluationType,NThrmDims,TDofOffset>>(
      aProblemParams.sublist("Thermal Natural Boundary Conditions")
    );
  }  
  auto tResidualParams = aProblemParams.sublist("Elliptic");
  if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
  {
    mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
  }
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
ResidualThermoelastostatic<EvaluationType, IndicatorFunctionType>::
postProcess(
  const Plato::Solutions &aSolutions
)
{ return; }

template<typename EvaluationType, typename IndicatorFunctionType>
void 
ResidualThermoelastostatic<EvaluationType, IndicatorFunctionType>::
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
  
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::TMKinematics<ElementType>          tKinematics;
  Plato::TMKineticsFactory< EvaluationType, ElementType > tTMKineticsFactory;
  auto tTMKinetics = tTMKineticsFactory.create(mMaterialModel, mSpatialDomain, mDataMap);

  Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset>    tInterpolateFromNodal;
  Plato::GeneralStressDivergence<ElementType, mNumDofsPerNode, MDofOffset> tStressDivergence;
  Plato::GeneralFluxDivergence  <ElementType, mNumDofsPerNode, TDofOffset> tFluxDivergence;

  auto tNumCells = mSpatialDomain.numCells();
  Plato::ScalarVectorT<ConfigScalarType>      tCellVolume("volume", tNumCells);
  Plato::ScalarMultiVectorT<GradScalarType>   tCellStrain("strain", tNumCells, mNumVoigtTerms);
  Plato::ScalarMultiVectorT<GradScalarType>   tCellTgrad("tgrad", tNumCells, mNumSpatialDims);
  Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
  Plato::ScalarMultiVectorT<ResultScalarType> tCellFlux("flux" , tNumCells, mNumSpatialDims);

  auto tNumPoints  = mNumGaussPoints;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();

  Plato::ScalarArray3DT<ResultScalarType> tStress("stress", tNumCells, tNumPoints, mNumVoigtTerms);
  Plato::ScalarArray3DT<ResultScalarType> tFlux  ("flux",   tNumCells, tNumPoints, mNumSpatialDims);
  Plato::ScalarArray3DT<GradScalarType>   tStrain("strain", tNumCells, tNumPoints, mNumVoigtTerms);
  Plato::ScalarArray3DT<GradScalarType>   tTGrad ("tgrad",  tNumCells, tNumPoints, mNumSpatialDims);
  Plato::ScalarArray4DT<ConfigScalarType> 
    tGradient("gradient", tNumCells, tNumPoints, mNumNodesPerCell, mNumSpatialDims);
  Plato::ScalarMultiVectorT<ConfigScalarType> tVolume("volume", tNumCells, tNumPoints);
  Plato::ScalarMultiVectorT<StateScalarType>  tTemperature("temperature", tNumCells, tNumPoints);
  
  auto& tApplyStressWeighting = mApplyStressWeighting;
  auto& tApplyFluxWeighting  = mApplyFluxWeighting;
  Kokkos::parallel_for("compute element tKinematics", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint = tCubPoints(iGpOrdinal);
    tComputeGradient(iCellOrdinal, iGpOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    tVolume(iCellOrdinal, iGpOrdinal) *= tCubWeights(iGpOrdinal);
    // compute strain and electric field
    //
    tKinematics(iCellOrdinal, iGpOrdinal, tStrain, tTGrad, tStateWS, tGradient);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tTemperature(iCellOrdinal, iGpOrdinal) = tInterpolateFromNodal(iCellOrdinal, tBasisValues, tStateWS);
  });
  // compute element state
  (*tTMKinetics)(tStress, tFlux, tStrain, tTGrad, tTemperature, tControlWS);
  Kokkos::parallel_for("compute divergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    // apply weighting
    //
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tApplyStressWeighting(iCellOrdinal, iGpOrdinal, tControlWS, tBasisValues, tStress);
    tApplyFluxWeighting  (iCellOrdinal, iGpOrdinal, tControlWS, tBasisValues, tFlux);
    // compute divergence
    //
    tStressDivergence(iCellOrdinal, iGpOrdinal, tResultWS, tStress, tGradient, tVolume);
    tFluxDivergence  (iCellOrdinal, iGpOrdinal, tResultWS, tFlux,   tGradient, tVolume);
    for(int i=0; i<mNumVoigtTerms; i++)
    {
      Kokkos::atomic_add(
        &tCellStrain(iCellOrdinal,i), tVolume(iCellOrdinal, iGpOrdinal)*tStrain(iCellOrdinal, iGpOrdinal, i)
      );
      Kokkos::atomic_add(
        &tCellStress(iCellOrdinal,i), tVolume(iCellOrdinal, iGpOrdinal)*tStress(iCellOrdinal, iGpOrdinal, i)
      );
    }
    for(int i=0; i<mNumSpatialDims; i++)
    {
      Kokkos::atomic_add(
        &tCellTgrad(iCellOrdinal,i), tVolume(iCellOrdinal, iGpOrdinal)*tTGrad(iCellOrdinal, iGpOrdinal, i)
      );
      Kokkos::atomic_add(
        &tCellFlux(iCellOrdinal,i), tVolume(iCellOrdinal, iGpOrdinal)*tFlux(iCellOrdinal, iGpOrdinal, i)
      );
    }
    Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume(iCellOrdinal, iGpOrdinal));
  });

  Kokkos::parallel_for("compute cell quantities", 
    Kokkos::RangePolicy<>(0, tNumCells),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
  {
    for(int i=0; i<mNumVoigtTerms; i++)
    {
      tCellStrain(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
      tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
    }
    for(int i=0; i<mNumSpatialDims; i++)
    {
      tCellTgrad(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
      tCellFlux(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
    }
  });
  // evaluate body forces
  if( mBodyLoads != nullptr )
  {
    mBodyLoads->get( mSpatialDomain, tStateWS, tControlWS, tConfigWS, tResultWS, -1.0 );
  }
  // populate output database
  if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) 
    { toMap(mDataMap, tCellStrain, "strain", mSpatialDomain); }
  if( std::count(mPlottable.begin(),mPlottable.end(),"tgrad" ) ) 
    { toMap(mDataMap, tCellTgrad,  "tgrad",  mSpatialDomain); }
  if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) 
    { toMap(mDataMap, tCellStress, "stress", mSpatialDomain); }
  if( std::count(mPlottable.begin(),mPlottable.end(),"flux"  ) ) 
    { toMap(mDataMap, tCellFlux,   "flux" ,  mSpatialDomain); }
}

template<typename EvaluationType, typename IndicatorFunctionType>
void 
ResidualThermoelastostatic<EvaluationType, IndicatorFunctionType>::
evaluateBoundary(
  const Plato::SpatialModel & aSpatialModel,
        Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aCycle
) const
{
  // evaluate boundary forces
  if( mBoundaryLoads != nullptr )
  {
    mBoundaryLoads->get( aSpatialModel, aWorkSets, aCycle, -1.0 );
  }
  if( mBoundaryFluxes != nullptr )
  {
    mBoundaryFluxes->get( aSpatialModel, aWorkSets, aCycle, -1.0 );
  }
}

} // namespace Elliptic

} // namespace Plato
