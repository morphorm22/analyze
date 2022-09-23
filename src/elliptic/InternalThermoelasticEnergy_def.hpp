#pragma once

#include "elliptic/InternalThermoelasticEnergy_decl.hpp"

#include "ToMap.hpp"
#include "FadTypes.hpp"
#include "TMKinetics.hpp"
#include "TMKinematics.hpp"
#include "ScalarProduct.hpp"
#include "GradientMatrix.hpp"
#include "TMKineticsFactory.hpp"
#include "InterpolateFromNodal.hpp"

namespace Plato
{

namespace Elliptic
{

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    InternalThermoelasticEnergy<EvaluationType, IndicatorFunctionType>::InternalThermoelasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyFluxWeighting   (mIndicatorFunction)
    /**************************************************************************/
    {
        Teuchos::ParameterList tProblemParams(aProblemParams);

        auto tMaterialName = aSpatialDomain.getMaterialName();

        if( aProblemParams.isSublist("Material Models") == false )
        {
            ANALYZE_THROWERR("Required input list ('Material Models') is missing.");
        }

        if( aProblemParams.sublist("Material Models").isSublist(tMaterialName) == false )
        {
            std::stringstream ss;
            ss << "Specified material model ('" << tMaterialName << "') is not defined";
            ANALYZE_THROWERR(ss.str());
        }

        auto& tParams = aProblemParams.sublist(aFunctionName);
        if( tParams.get<bool>("Include Thermal Strain", true) == false )
        {
           auto tMaterialParams = tProblemParams.sublist("Material Models").sublist(tMaterialName);
           tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a11",0.0);
           tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a22",0.0);
           tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a33",0.0);
        }

        Plato::ThermoelasticModelFactory<mNumSpatialDims> mmfactory(tProblemParams);
        mMaterialModel = mmfactory.create(tMaterialName);
    }

    /**************************************************************************/
    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    InternalThermoelasticEnergy<EvaluationType, IndicatorFunctionType>::evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientMatrix<ElementType> computeGradient;
      Plato::TMKinematics<ElementType>          kinematics;

      Plato::TMKineticsFactory< EvaluationType, ElementType > tTMKineticsFactory;
      auto tTMKinetics = tTMKineticsFactory.create(mMaterialModel, mSpatialDomain, mDataMap);

      Plato::ScalarProduct<mNumVoigtTerms>      mechanicalScalarProduct;
      Plato::ScalarProduct<mNumSpatialDims>     thermalScalarProduct;

      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> interpolateFromNodal;

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

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyFluxWeighting   = mApplyFluxWeighting;
      Kokkos::parallel_for("compute internal energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          auto tCubPoint = tCubPoints(iGpOrdinal);

          computeGradient(iCellOrdinal, iGpOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          tVolume(iCellOrdinal, iGpOrdinal) *= tCubWeights(iGpOrdinal);

          // compute strain and temperature gradient
          //
          kinematics(iCellOrdinal, iGpOrdinal, tStrain, tTGrad, aState, tGradient);

          auto tBasisValues = ElementType::basisValues(tCubPoint);
          tTemperature(iCellOrdinal, iGpOrdinal) = interpolateFromNodal(iCellOrdinal, tBasisValues, aState);
      });

      // compute element state
      (*tTMKinetics)(tStress, tFlux, tStrain, tTGrad, tTemperature, aControl);

      Kokkos::parallel_for("compute product", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          // apply weighting
          //
          auto tCubPoint = tCubPoints(iGpOrdinal);
          auto tBasisValues = ElementType::basisValues(tCubPoint);
          applyStressWeighting(iCellOrdinal, iGpOrdinal, aControl, tBasisValues, tStress);
          applyFluxWeighting  (iCellOrdinal, iGpOrdinal, aControl, tBasisValues, tFlux);

          // compute element internal energy
          //
          mechanicalScalarProduct(iCellOrdinal, iGpOrdinal, aResult, tStress, tStrain, tVolume);
          thermalScalarProduct   (iCellOrdinal, iGpOrdinal, aResult, tFlux,   tTGrad,  tVolume);

        });
    }
} // namespace Elliptic

} // namespace Plato
