#pragma once

#include "ToMap.hpp"
#include "MetaData.hpp"
#include "PlatoTypes.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "GradientMatrix.hpp"
#include "GeneralStressDivergence.hpp"
#include "elliptic/mechanical/linear/VonMisesYieldFunction.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
ResidualElastostatic<EvaluationType, IndicatorFunctionType>::
ResidualElastostatic(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
        Teuchos::ParameterList & aPenaltyParams
) :
  FunctionBaseType   (aSpatialDomain, aDataMap),
  mIndicatorFunction (aPenaltyParams),
  mApplyWeighting    (mIndicatorFunction),
  mBodyLoads         (nullptr),
  mBoundaryForces     (nullptr)
{
  // obligatory: define dof names in order
  mDofNames.push_back("displacement X");
  if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
  if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");
  // create material model and get stiffness
  //
  Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
  mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());
  // parse body loads
  // 
  if(aProblemParams.isSublist("Body Loads"))
  {
    mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>(
      aProblemParams.sublist("Body Loads")
    );
  }
  // parse boundary Conditions
  // 
  if(aProblemParams.isSublist("Natural Boundary Conditions"))
  {
    mBoundaryForces = std::make_shared<Plato::NeumannBCs<EvaluationType>>(
      aProblemParams.sublist("Natural Boundary Conditions")
    );
  }
  // parse cell problem forcing
  //
  if(aProblemParams.isSublist("Cell Problem Forcing"))
  {
    Plato::OrdinalType tColumnIndex = 
      aProblemParams.sublist("Cell Problem Forcing").get<Plato::OrdinalType>("Column Index");
    mCellForcing.setCellStiffness(mMaterialModel->getStiffnessMatrix());
    mCellForcing.setColumnIndex(tColumnIndex);
  }
  // parse requested output quantities of interests
  //
  auto tResidualParams = aProblemParams.sublist("Elliptic");
  if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
  {
    mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
  }
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
ResidualElastostatic<EvaluationType, IndicatorFunctionType>::
postProcess(
  const Plato::Solutions &aSolutions
)
{ return; }

template<typename EvaluationType, typename IndicatorFunctionType>
void
ResidualElastostatic<EvaluationType, IndicatorFunctionType>::
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
  
  Plato::ComputeGradientMatrix<ElementType>        tComputeGradient;
  Plato::SmallStrain<ElementType>                  tComputeVoigtStrain;
  Plato::GeneralStressDivergence<ElementType>      tComputeStressDivergence;
  Plato::LinearStress<EvaluationType, ElementType> tComputeVoigtStress(mMaterialModel);

  auto tNumCells = mSpatialDomain.numCells();
  Plato::ScalarVectorT<ConfigScalarType>      tCellVolume("volume", tNumCells);
  Plato::ScalarMultiVectorT<StrainScalarType> tCellStrain("strain", tNumCells, mNumVoigtTerms);
  Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
  
  auto tNumPoints  = mNumGaussPoints;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  
  auto& tApplyWeighting = mApplyWeighting;
  auto& tCellForcing = mCellForcing;
  Kokkos::parallel_for("compute stress", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    ConfigScalarType tVolume(0.0);
    Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;
    Plato::Array<mNumVoigtTerms, StrainScalarType> tStrain(0.0);
    Plato::Array<mNumVoigtTerms, ResultScalarType> tStress(0.0);
    auto tCubPoint = tCubPoints(iGpOrdinal);
    tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    
    tComputeVoigtStrain(iCellOrdinal, tStrain, tStateWS, tGradient);
    tComputeVoigtStress(tStress, tStrain);
    tCellForcing(tStress);

    tVolume *= tCubWeights(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tApplyWeighting(iCellOrdinal, tControlWS, tBasisValues, tStress);
    tComputeStressDivergence(iCellOrdinal, tResultWS, tStress, tGradient, tVolume);
    for(int i=0; i<mNumVoigtTerms; i++)
    {
      Kokkos::atomic_add(&tCellStrain(iCellOrdinal,i), tVolume*tStrain(i));
      Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
    }
    Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
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
  });

  if( mBodyLoads != nullptr )
  {
    mBodyLoads->get( mSpatialDomain, tStateWS, tControlWS, tConfigWS, tResultWS, -1.0 );
  }

  if(std::count(mPlotTable.begin(), mPlotTable.end(), "strain")) 
  { Plato::toMap(mDataMap, tCellStrain, "strain", mSpatialDomain); }
  if(std::count(mPlotTable.begin(), mPlotTable.end(), "stress")) 
  { Plato::toMap(mDataMap, tCellStress, "stress", mSpatialDomain); }
  if(std::count(mPlotTable.begin(), mPlotTable.end(), "vonmises")) 
  { this->outputVonMises(tCellStress, mSpatialDomain); }
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
ResidualElastostatic<EvaluationType, IndicatorFunctionType>::
evaluateBoundary(
  const Plato::SpatialModel & aSpatialModel,
        Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aCycle
) const
{
  // evaluate boundary forces
  if( mBoundaryForces != nullptr )
  {
    mBoundaryForces->get( aSpatialModel, aWorkSets, aCycle, -1.0 );
  }
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
ResidualElastostatic<EvaluationType, IndicatorFunctionType>::
outputVonMises(
  const Plato::ScalarMultiVectorT<ResultScalarType> & aCauchyStress,
  const Plato::SpatialDomain                        & aSpatialDomain
) const
{
  auto tNumCells = aSpatialDomain.numCells();
  Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;
  Plato::ScalarVectorT<ResultScalarType> tVonMises("Von Mises", tNumCells);
  Kokkos::parallel_for("Compute VonMises Stress",
    Kokkos::RangePolicy<>(0, tNumCells), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
  {
    ResultScalarType tCellVonMises(0);
    tComputeVonMises(aCellOrdinal, aCauchyStress, tCellVonMises);
    tVonMises(aCellOrdinal) = tCellVonMises;
  });
  Plato::toMap(mDataMap, tVonMises, "vonmises", aSpatialDomain);
}

} // namespace Elliptic

} // namespace Plato
