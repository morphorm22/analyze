#pragma once

#include "ToMap.hpp"
#include "TMKinematics.hpp"
#include "ProjectToNode.hpp"
#include "ThermalContent.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathHelpers.hpp"
#include "TMKineticsFactory.hpp"
#include "ThermalMassMaterial.hpp"
#include "InterpolateFromNodal.hpp"
#include "GeneralFluxDivergence.hpp"
#include "ThermoelasticMaterial.hpp"
#include "GeneralStressDivergence.hpp"

namespace Plato
{

namespace Parabolic
{

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    TransientThermomechResidual<EvaluationType, IndicatorFunctionType>::
    TransientThermomechResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
     ) :
         FunctionBaseType      (aSpatialDomain, aDataMap),
         mIndicatorFunction    (aPenaltyParams),
         mApplyStressWeighting (mIndicatorFunction),
         mApplyFluxWeighting   (mIndicatorFunction),
         mApplyMassWeighting   (mIndicatorFunction),
         mBoundaryLoads        (nullptr),
         mBoundaryFluxes       (nullptr)
    /**************************************************************************/
    {
        // obligatory: define dof names in order
        mDofNames.push_back("displacement X");
        mDofDotNames.push_back("velocity X");
        if(mNumSpatialDims > 1)
        {
            mDofNames.push_back("displacement Y");
            mDofDotNames.push_back("velocity Y");
        }
        if(mNumSpatialDims > 2)
        {
            mDofNames.push_back("displacement Z");
            mDofDotNames.push_back("velocity Z");
        }
        mDofNames.push_back("temperature");
        mDofDotNames.push_back("temperature rate");

        {
            Plato::ThermoelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
            mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
        }

        {
            Plato::ThermalMassModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
            mThermalMassMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
        }

        // parse boundary Conditions
        // 
        if(aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>>
                (aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }
        // parse thermal boundary Conditions
        // 
        if(aProblemParams.isSublist("Thermal Natural Boundary Conditions"))
        {
            mBoundaryFluxes = std::make_shared<Plato::NaturalBCs<ElementType, NThrmDims, mNumDofsPerNode, TDofOffset>>
                (aProblemParams.sublist("Thermal Natural Boundary Conditions"));
        }

        auto tResidualParams = aProblemParams.sublist("Parabolic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }


    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Solutions
    TransientThermomechResidual<EvaluationType, IndicatorFunctionType>::
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
      return aSolutions;
    }


    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    TransientThermomechResidual<EvaluationType, IndicatorFunctionType>::
    evaluate(
        const Plato::ScalarMultiVectorT< StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<GradScalarType> tCellStrain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType> tCellTgrad("tgrad", tNumCells, mNumSpatialDims);

      Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tCellFlux("flux" , tNumCells, mNumSpatialDims);

      Plato::ComputeGradientMatrix<ElementType> tComputeGradient;

      Plato::TMKinematics<ElementType> tKinematics;

      Plato::TMKineticsFactory< EvaluationType, ElementType > tTMKineticsFactory;
      auto tTMKinetics = tTMKineticsFactory.create(mMaterialModel, mSpatialDomain, mDataMap);

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> tInterpolateFromNodal;

      Plato::GeneralFluxDivergence  <ElementType, mNumDofsPerNode, TDofOffset> tFluxDivergence;
      Plato::GeneralStressDivergence<ElementType, mNumDofsPerNode, MDofOffset> tStressDivergence;

      Plato::ThermalContent<mNumSpatialDims> tComputeHeatRate(mThermalMassMaterialModel);

      Plato::ProjectToNode<ElementType, mNumDofsPerNode, TDofOffset> tProjectHeatRate;

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      Plato::ScalarArray3DT<ResultScalarType> tStress("stress", tNumCells, tNumPoints, mNumVoigtTerms);
      Plato::ScalarArray3DT<ResultScalarType> tFlux  ("flux",   tNumCells, tNumPoints, mNumSpatialDims);
      Plato::ScalarArray3DT<GradScalarType>   tStrain("strain", tNumCells, tNumPoints, mNumVoigtTerms);
      Plato::ScalarArray3DT<GradScalarType>   tTGrad ("tgrad",  tNumCells, tNumPoints, mNumSpatialDims);

      Plato::ScalarArray4DT<ConfigScalarType> tGradient("gradient", tNumCells, tNumPoints, mNumNodesPerCell, mNumSpatialDims);

      Plato::ScalarMultiVectorT<ConfigScalarType> tVolume("volume", tNumCells, tNumPoints);
      Plato::ScalarMultiVectorT<StateScalarType> tTemperature("temperature", tNumCells, tNumPoints);
      
      auto& tApplyStressWeighting = mApplyStressWeighting;
      auto& tApplyFluxWeighting   = mApplyFluxWeighting;
      auto& tApplyMassWeighting   = mApplyMassWeighting;
      Kokkos::parallel_for("stress and flux divergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
    
          auto tCubPoint = tCubPoints(iGpOrdinal);

          tComputeGradient(iCellOrdinal, iGpOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          tVolume(iCellOrdinal, iGpOrdinal) *= tCubWeights(iGpOrdinal);
    
          // compute strain and temperature gradient
          //
          tKinematics(iCellOrdinal, iGpOrdinal, tStrain, tTGrad, aState, tGradient);

          // interpolate temperature
          //
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          tTemperature(iCellOrdinal, iGpOrdinal) = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aState);
      });

      // compute element state
      (*tTMKinetics)(tStress, tFlux, tStrain, tTGrad, tTemperature, aControl);

      Kokkos::parallel_for("compute divergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          // apply weighting
          //
          auto tCubPoint = tCubPoints(iGpOrdinal);
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          tApplyStressWeighting(iCellOrdinal, iGpOrdinal, aControl, tBasisValues, tStress);
          tApplyFluxWeighting  (iCellOrdinal, iGpOrdinal, aControl, tBasisValues, tFlux);

          // compute stress and flux divergence
          //
          tStressDivergence(iCellOrdinal, iGpOrdinal, aResult, tStress, tGradient, tVolume);
          tFluxDivergence  (iCellOrdinal, iGpOrdinal, aResult, tFlux,   tGradient, tVolume);

          // compute temperature rate at gausspoints
          //
          StateDotScalarType tTemperatureRate = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aStateDot);

          // compute the time rate of internal thermal energy
          //
          ResultScalarType tHeatRate(0.0);
          tComputeHeatRate(tHeatRate, tTemperatureRate, tTemperature(iCellOrdinal, iGpOrdinal));

          // apply weighting
          //
          tApplyMassWeighting(iCellOrdinal, aControl, tBasisValues, tHeatRate);

          // project to nodes
          //
          tProjectHeatRate(iCellOrdinal, tVolume(iCellOrdinal, iGpOrdinal), tBasisValues, tHeatRate, aResult);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              Kokkos::atomic_add(&tCellStrain(iCellOrdinal,i), tVolume(iCellOrdinal, iGpOrdinal)*tStrain(iCellOrdinal, iGpOrdinal, i));
              Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume(iCellOrdinal, iGpOrdinal)*tStress(iCellOrdinal, iGpOrdinal, i));
          }
          for(int i=0; i<ElementType::mNumSpatialDims; i++)
          {
              Kokkos::atomic_add(&tCellTgrad(iCellOrdinal,i), tVolume(iCellOrdinal, iGpOrdinal)*tTGrad(iCellOrdinal, iGpOrdinal, i));
              Kokkos::atomic_add(&tCellFlux(iCellOrdinal,i), tVolume(iCellOrdinal, iGpOrdinal)*tFlux(iCellOrdinal, iGpOrdinal, i));
          }
          Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume(iCellOrdinal, iGpOrdinal));

      });

      Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
      {
          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              tCellStrain(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
          for(int i=0; i<ElementType::mNumSpatialDims; i++)
          {
              tCellTgrad(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellFlux(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
      });

      if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tCellStrain, "strain", mSpatialDomain);
      if( std::count(mPlottable.begin(),mPlottable.end(),"tgrad" ) ) toMap(mDataMap, tCellTgrad,  "tgrad",  mSpatialDomain);
      if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tCellStress, "stress", mSpatialDomain);
      if( std::count(mPlottable.begin(),mPlottable.end(),"flux"  ) ) toMap(mDataMap, tCellFlux,   "flux" ,  mSpatialDomain);
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    TransientThermomechResidual<EvaluationType, IndicatorFunctionType>::
    evaluate_boundary(
        const Plato::SpatialModel                             & aSpatialModel,
        const Plato::ScalarMultiVectorT< StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep) const
    /**************************************************************************/
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0);
        }
        if( mBoundaryFluxes != nullptr )
        {
            mBoundaryFluxes->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0);
        }
    }
} // namespace Parabolic

} // namespace Plato
