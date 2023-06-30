#pragma once

#include "BLAS2.hpp"
#include "ToMap.hpp"
#include "MetaData.hpp"
#include "PlatoTypes.hpp"
#include "EMKinetics.hpp"
#include "EMKinematics.hpp"
#include "GradientMatrix.hpp"
#include "GeneralFluxDivergence.hpp"
#include "GeneralStressDivergence.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename EvaluationType, typename IndicatorFunctionType>
ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::
ElectroelastostaticResidual(
  const Plato::SpatialDomain   & aSpatialDomain,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
        Teuchos::ParameterList & aPenaltyParams
) :
  FunctionBaseType      (aSpatialDomain, aDataMap),
  mIndicatorFunction    (aPenaltyParams),
  mApplyStressWeighting (mIndicatorFunction),
  mApplyEDispWeighting  (mIndicatorFunction),
  mBodyLoads            (nullptr),
  mBoundaryLoads        (nullptr),
  mBoundaryCharges      (nullptr)
{
  // obligatory: define dof names in order
  mDofNames.push_back("displacement X");
  if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
  if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");
  mDofNames.push_back("electric potential");
  // create material model and get stiffness
  //
  Plato::ElectroelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
  mMaterialModel = mmfactory.create(mSpatialDomain.getMaterialName());
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
    mBoundaryLoads = std::make_shared<Plato::NeumannBCs<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>>(
      aProblemParams.sublist("Mechanical Natural Boundary Conditions")
    );
  }
  // parse electrical boundary Conditions
  // 
  if(aProblemParams.isSublist("Electrical Natural Boundary Conditions"))
  {
    mBoundaryCharges = std::make_shared<Plato::NeumannBCs<ElementType, NElecDims, mNumDofsPerNode, EDofOffset>>(
      aProblemParams.sublist("Electrical Natural Boundary Conditions")
    );
  }
  auto tResidualParams = aProblemParams.sublist("Electroelastostatics");
  if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
  {
    mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
  }
}

template<typename EvaluationType, typename IndicatorFunctionType>
void
ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::
postProcess(
  const Plato::Solutions &aSolutions
)
{ return; }

template<typename EvaluationType, typename IndicatorFunctionType>
void 
ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::
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

  using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::EMKinematics<ElementType>          tKinematics;
  Plato::EMKinetics<ElementType>            tKinetics(mMaterialModel);
  Plato::GeneralStressDivergence<ElementType, mNumDofsPerNode, MDofOffset> tStressDivergence;
  Plato::GeneralFluxDivergence  <ElementType, mNumDofsPerNode, EDofOffset> tEdispDivergence;

  auto tNumCells = mSpatialDomain.numCells();
  Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight",tNumCells);
  Plato::ScalarMultiVectorT<GradScalarType> tCellStrain("strain", tNumCells, mNumVoigtTerms);
  Plato::ScalarMultiVectorT<GradScalarType> tCellEField("efield", tNumCells, mNumSpatialDims);

  Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
  Plato::ScalarMultiVectorT<ResultScalarType> tCellEDisp ("edisp" , tNumCells, mNumSpatialDims);

  auto tNumPoints  = mNumGaussPoints;
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();

  auto& tApplyStressWeighting = mApplyStressWeighting;
  auto& tApplyEDispWeighting  = mApplyEDispWeighting;
  Kokkos::parallel_for("compute element state", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    ConfigScalarType tVolume(0.0);
    Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tGradient;
    Plato::Array<mNumVoigtTerms,  GradScalarType>   tStrain(0.0);
    Plato::Array<mNumSpatialDims, GradScalarType>   tEField(0.0);
    Plato::Array<mNumVoigtTerms,  ResultScalarType> tStress(0.0);
    Plato::Array<mNumSpatialDims, ResultScalarType> tEDisp (0.0);
    auto tCubPoint = tCubPoints(iGpOrdinal);
    tComputeGradient(iCellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    tVolume *= tCubWeights(iGpOrdinal);
    // compute strain and electric field
    //
    tKinematics(iCellOrdinal, tStrain, tEField, tStateWS, tGradient);
    // compute stress and electric displacement
    //
    tKinetics(tStress, tEDisp, tStrain, tEField);
    // apply weighting
    //
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    tApplyStressWeighting(iCellOrdinal, tControlWS, tBasisValues, tStress);
    tApplyEDispWeighting (iCellOrdinal, tControlWS, tBasisValues, tEDisp);
    // compute divergence
    //
    tStressDivergence(iCellOrdinal, tResultWS, tStress, tGradient, tVolume);
    tEdispDivergence (iCellOrdinal, tResultWS, tEDisp,  tGradient, tVolume);
    for(int i=0; i<mNumVoigtTerms; i++)
    {
      Kokkos::atomic_add(&tCellStrain(iCellOrdinal,i), tVolume*tStrain(i));
      Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
    }
    for(int i=0; i<mNumSpatialDims; i++)
    {
      Kokkos::atomic_add(&tCellEField(iCellOrdinal,i), tVolume*tEField(i));
      Kokkos::atomic_add(&tCellEDisp(iCellOrdinal,i), tVolume*tEDisp(i));
    }
    Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
  });
  Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
  {
    for(int i=0; i<mNumVoigtTerms; i++)
    {
      tCellStrain(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
      tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
    }
    for(int i=0; i<mNumSpatialDims; i++)
    {
      tCellEField(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
      tCellEDisp(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
    }
  });
  if( mBodyLoads != nullptr )
  {
    mBodyLoads->get( mSpatialDomain, tStateWS, tControlWS, tConfigWS, tResultWS, -1.0 );
  }
  if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) 
    toMap(mDataMap, tCellStrain, "strain", mSpatialDomain);
  if( std::count(mPlottable.begin(),mPlottable.end(),"efield") ) 
    toMap(mDataMap, tCellEField, "efield", mSpatialDomain);
  if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) 
    toMap(mDataMap, tCellStress, "stress", mSpatialDomain);
  if( std::count(mPlottable.begin(),mPlottable.end(),"edisp" ) ) 
    toMap(mDataMap, tCellEDisp,  "edisp",  mSpatialDomain);
}

template<typename EvaluationType, typename IndicatorFunctionType>
void 
ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::
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
  Plato::ScalarMultiVectorT<StateScalarType> tStateWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<StateScalarType>>(aWorkSets.get("states"));
  Plato::ScalarMultiVectorT<ResultScalarType> tResultWS = 
    Plato::unpack<Plato::ScalarMultiVectorT<ResultScalarType>>(aWorkSets.get("result"));
  
  if( mBoundaryLoads != nullptr )
  {
    mBoundaryLoads->get(aSpatialModel, tStateWS, tControlWS, tConfigWS, tResultWS, -1.0 );
  }
  if( mBoundaryCharges != nullptr )
  {
    mBoundaryCharges->get(aSpatialModel, tStateWS, tControlWS, tConfigWS, tResultWS, -1.0 );
  }
}

} // namespace Elliptic

} // namespace Plato
